//! GitHub Copilot over its OpenAI-compatible chat-completions surface.
//!
//! The account's long-lived credential is the GitHub OAuth token from the
//! device flow. It is stored as the refresh token and exchanged for a
//! short-lived Copilot token (`copilot_internal/v2/token`) whenever the
//! slot needs one.

use std::time::Duration;

use axum::body::Bytes;
use reqwest::RequestBuilder;
use reqwest::header::CONTENT_TYPE;
use serde::Deserialize;
use uuid::Uuid;

use crate::clock;
use crate::config::CopilotConfig;
use crate::upstream::{Classify, IdList, SendError, classify, json};

pub const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";

const RULES: Classify = Classify {
   // A seat that lost Copilot answers 404, which no other account fixes.
   auth: &[401, 403, 404],
   reset_headers: &["retry-after", "x-ratelimit-reset", "x-quota-reset"],
   ..Classify::STRICT
};

pub struct CopilotClient {
   http: reqwest::Client,
   cfg: CopilotConfig,
}

/// The headers every Copilot endpoint checks to recognise an editor client.
pub fn editor(req: RequestBuilder) -> RequestBuilder {
   req.header("editor-version", "vscode/1.105.0")
      .header("editor-plugin-version", "copilot-chat/0.26.7")
      .header("user-agent", USER_AGENT)
      .header("x-github-api-version", "2025-04-01")
}

impl CopilotClient {
   pub fn new(cfg: CopilotConfig) -> Self {
      let http = reqwest::Client::builder()
         .connect_timeout(Duration::from_secs(30))
         .tcp_keepalive(Duration::from_secs(30))
         .build()
         .expect("building http client");
      Self { http, cfg }
   }

   pub const fn soft_utilization_limit(&self) -> f64 {
      self.cfg.soft_utilization_limit
   }

   fn base_url(&self) -> &str {
      self.cfg.base_url.trim_end_matches('/')
   }

   pub async fn quota(&self, copilot_token: &str) -> Result<QuotaReport, SendError> {
      let resp = editor(
         self
            .http
            .get("https://api.github.com/copilot_internal/user")
            .bearer_auth(copilot_token),
      )
      .send()
      .await?;
      json(resp, RULES).await
   }

   pub async fn models(&self, copilot_token: &str) -> Result<Vec<String>, SendError> {
      let resp = editor(
         self
            .http
            .get(format!("{}/models", self.base_url()))
            .bearer_auth(copilot_token),
      )
      .send()
      .await?;
      let listed: IdList = json(resp, RULES).await?;
      Ok(listed.data.into_iter().map(|model| model.id).collect())
   }

   pub async fn post(
      &self,
      copilot_token: &str,
      body: &Bytes,
      agent: bool,
      vision: bool,
   ) -> Result<reqwest::Response, SendError> {
      let mut req = editor(
         self
            .http
            .post(format!("{}/chat/completions", self.base_url()))
            .bearer_auth(copilot_token),
      )
      .header("copilot-integration-id", "vscode-chat")
      .header("openai-intent", "conversation-panel")
      .header("x-initiator", if agent { "agent" } else { "user" })
      .header("x-request-id", Uuid::new_v4().to_string())
      .header(CONTENT_TYPE, "application/json");
      if vision {
         req = req.header("copilot-vision-request", "true");
      }
      let resp = req.body(body.clone()).send().await?;
      classify(resp, RULES).await
   }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct QuotaReport {
   /// `quota_reset_date` is a bare date, only this one carries the time.
   pub quota_reset_date_utc: Option<String>,
   pub quota_snapshots: QuotaSnapshots,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct QuotaSnapshots {
   pub premium_interactions: Option<QuotaDetail>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct QuotaDetail {
   pub entitlement: i64,
   pub remaining: i64,
   pub unlimited: bool,
}

impl QuotaReport {
   pub fn resets_at(&self) -> Option<i64> {
      clock::unix_seconds(self.quota_reset_date_utc.as_deref())
   }
}

impl QuotaDetail {
   pub fn utilization(&self) -> f64 {
      if self.unlimited || self.entitlement <= 0 {
         return 0.0_f64;
      }
      ((self.entitlement - self.remaining).max(0) as f64 / self.entitlement as f64)
         .clamp(0.0_f64, 1.0_f64)
   }
}
