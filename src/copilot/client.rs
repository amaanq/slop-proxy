//! Chat-completions relay to the Copilot API. The body arrives as the
//! caller wrote it on `/v1/chat/completions` and the reply streams back in
//! the same dialect, so the proxy only meters usage out of the frames on the
//! way past.

use std::time::Duration;

use axum::body::Bytes;
use eyre::{Result, WrapErr as _, bail, eyre};
use reqwest::RequestBuilder;
use reqwest::header::CONTENT_TYPE;
use serde::Deserialize;
use serde::de::IgnoredAny;
use uuid::Uuid;

use crate::config::CopilotConfig;
use crate::copilot::{API_VERSION, EDITOR_PLUGIN_VERSION, EDITOR_VERSION, USER_AGENT};
use crate::upstream::{Classify, SendError, classify};

const RULES: Classify = Classify {
   pass: |_| false,
   // A revoked OAuth grant, an expired Copilot token, or a seat that lost
   // Copilot answers 401/403/404 and no retry on another account makes that
   // account work. 429 is the quota bucket and is retried elsewhere.
   auth: &[401, 403, 404],
   reset_headers: &["retry-after", "x-ratelimit-reset", "x-quota-reset"],
   account_faults: &[],
};

pub struct CopilotClient {
   http: reqwest::Client,
   cfg: CopilotConfig,
}

#[derive(Deserialize)]
struct GithubUser {
   #[serde(default)]
   login: Option<String>,
   #[serde(default)]
   id: Option<i64>,
}

#[derive(Default, Deserialize)]
struct Peek {
   #[serde(default)]
   messages: Vec<PeekMessage>,
}

#[derive(Deserialize)]
struct PeekMessage {
   #[serde(default)]
   role: String,
   #[serde(default)]
   content: Option<PeekContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PeekContent {
   Parts(Vec<PeekPart>),
   Other(IgnoredAny),
}

impl PeekContent {
   #[expect(
      clippy::pattern_type_mismatch,
      reason = "matching &self against owned arms is the readable form here; the deref alternative trips needless_borrow"
   )]
   fn parts(&self) -> Option<&[PeekPart]> {
      if let Self::Parts(parts) = self {
         Some(parts)
      } else {
         None
      }
   }
}

#[derive(Deserialize)]
struct PeekPart {
   #[serde(rename = "type", default)]
   kind: String,
}

impl Peek {
   fn parse(payload: &[u8]) -> Self {
      serde_json::from_slice(payload).unwrap_or_default()
   }

   /// Copilot routes agent traffic (a replayed assistant or tool turn) behind
   /// a different initiator than a fresh user turn.
   fn initiator(&self) -> &'static str {
      let agent = self
         .messages
         .iter()
         .any(|msg| matches!(msg.role.as_str(), "assistant" | "tool"));
      if agent { "agent" } else { "user" }
   }

   /// Vision requests must opt in or image turns are rejected.
   fn has_image(&self) -> bool {
      self
         .messages
         .iter()
         .filter_map(|msg| msg.content.as_ref()?.parts())
         .flatten()
         .any(|part| matches!(part.kind.as_str(), "image_url" | "input_image" | "image"))
   }
}

/// The headers every Copilot endpoint checks to recognise an editor client.
fn editor(req: RequestBuilder) -> RequestBuilder {
   req.header("editor-version", EDITOR_VERSION)
      .header("editor-plugin-version", EDITOR_PLUGIN_VERSION)
      .header("user-agent", USER_AGENT)
      .header("x-github-api-version", API_VERSION)
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

   fn base_url(&self) -> String {
      let base = self.cfg.base_url.trim_end_matches('/').to_owned();
      let account = self.cfg.account_type.trim().to_lowercase();
      if account.is_empty() || account == "individual" {
         return base;
      }
      base.strip_prefix("https://api.").map_or_else(
         || base.clone(),
         |rest| format!("https://api.{account}.{rest}"),
      )
   }

   /// The Copilot token minted from the account's GitHub OAuth grant. It
   /// expires quickly and is refreshed while the slot mutex is held, exactly
   /// like the other OAuth backends.
   pub async fn mint_token(github_token: &str) -> Result<TokenGrant, SendError> {
      let resp = editor(
         reqwest::Client::new()
            .get("https://api.github.com/copilot_internal/v2/token")
            .header("authorization", format!("token {github_token}"))
            .header("accept", "application/json"),
      )
      .send()
      .await
      .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, RULES).await?;
      let status = resp.status().as_u16();
      resp.json().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("parsing copilot token response: {err}"),
      })
   }

   /// Reads the account's login for the stored identity. GitHub tokens do not
   /// carry an account id of their own, so this runs once at login.
   pub async fn login(github_token: &str) -> Result<String> {
      let resp = reqwest::Client::new()
         .get("https://api.github.com/user")
         .header("authorization", format!("token {github_token}"))
         .header("accept", "application/json")
         .header("user-agent", USER_AGENT)
         .send()
         .await
         .wrap_err("reading github user")?;
      if !resp.status().is_success() {
         let status = resp.status();
         let text = resp.text().await.unwrap_or_default();
         bail!("reading github user: {status}: {text}");
      }
      let user: GithubUser = resp.json().await.wrap_err("parsing github user")?;
      user
         .login
         .or_else(|| user.id.map(|id| id.to_string()))
         .ok_or_else(|| eyre!("github user response has no login"))
   }

   /// Quota for the seat behind this token, without spending an inference
   /// request. `premium_interactions` is the billed budget; the `chat` and
   /// `completions` snapshots ride along for the dashboard.
   pub async fn quota(&self, copilot_token: &str) -> Result<QuotaReport, SendError> {
      let resp = editor(
         self
            .http
            .get("https://api.github.com/copilot_internal/user")
            .bearer_auth(copilot_token)
            .header("accept", "application/json"),
      )
      .send()
      .await
      .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, RULES).await?;
      let status = resp.status().as_u16();
      resp.json().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("parsing copilot quota response: {err}"),
      })
   }

   pub async fn models(&self, copilot_token: &str) -> Result<Vec<String>, SendError> {
      let resp = editor(
         self
            .http
            .get(format!("{}/models", self.base_url()))
            .bearer_auth(copilot_token)
            .header("accept", "application/json"),
      )
      .send()
      .await
      .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, RULES).await?;
      let status = resp.status().as_u16();
      let listed: ModelList = resp.json().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("parsing copilot models response: {err}"),
      })?;
      Ok(listed.data.into_iter().map(|model| model.id).collect())
   }

   /// A caller already speaking chat completions, forwarded byte for byte so
   /// the Copilot-only fields (`max_tokens`, stream options) survive.
   pub async fn post(
      &self,
      copilot_token: &str,
      payload: &Bytes,
   ) -> Result<reqwest::Response, SendError> {
      let peek = Peek::parse(payload);
      let mut req = editor(
         self
            .http
            .post(format!("{}/chat/completions", self.base_url()))
            .bearer_auth(copilot_token),
      )
      .header("copilot-integration-id", "vscode-chat")
      .header("openai-intent", "conversation-panel")
      .header("x-initiator", peek.initiator())
      .header("x-request-id", Uuid::new_v4().to_string())
      .header(CONTENT_TYPE, "application/json");
      if peek.has_image() {
         req = req.header("copilot-vision-request", "true");
      }
      let resp = req
         .body(payload.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      classify(resp, RULES).await
   }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenGrant {
   #[serde(default)]
   pub token: String,
   #[serde(default)]
   pub expires_at: Option<i64>,
   #[serde(default)]
   pub refresh_in: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuotaReport {
   pub copilot_plan: Option<String>,
   pub quota_reset_date: Option<String>,
   pub quota_snapshots: QuotaSnapshots,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuotaSnapshots {
   pub chat: Option<QuotaDetail>,
   pub completions: Option<QuotaDetail>,
   pub premium_interactions: Option<QuotaDetail>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuotaDetail {
   pub entitlement: i64,
   pub remaining: i64,
   pub percent_remaining: f64,
   pub unlimited: bool,
}

impl QuotaDetail {
   /// Fraction consumed, 0.0 to 1.0. Unlimited seats report no consumption so
   /// a dashboard ranks them as healthy rather than spent.
   pub fn utilization(&self) -> f64 {
      if self.unlimited || self.entitlement <= 0 {
         return 0.0_f64;
      }
      ((self.entitlement - self.remaining).max(0) as f64 / self.entitlement as f64)
         .clamp(0.0_f64, 1.0_f64)
   }
}

impl QuotaReport {
   /// Seconds until the quota window rolls over, when reported.
   pub fn resets_at(&self) -> Option<i64> {
      self
         .quota_reset_date
         .as_deref()?
         .parse::<jiff::Timestamp>()
         .ok()
         .map(jiff::Timestamp::as_second)
   }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelList {
   #[serde(default)]
   data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelEntry {
   #[serde(default)]
   id: String,
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn unlimited_quota_reports_no_consumption() {
      let detail = QuotaDetail {
         unlimited: true,
         ..QuotaDetail::default()
      };
      assert!(detail.utilization() < f64::EPSILON);
   }

   #[test]
   fn quota_utilization_clamps_overuse() {
      let detail = QuotaDetail {
         entitlement: 100,
         remaining: -10,
         ..QuotaDetail::default()
      };
      assert!((detail.utilization() - 1.0_f64).abs() < f64::EPSILON);
   }

   #[test]
   fn assistant_turn_marks_the_agent_initiator() {
      let agent = Peek::parse(br#"{"messages":[{"role":"assistant","content":"x"}]}"#);
      assert_eq!(agent.initiator(), "agent");
      let user = Peek::parse(br#"{"messages":[{"role":"user","content":"x"}]}"#);
      assert_eq!(user.initiator(), "user");
   }

   #[test]
   fn business_accounts_hit_their_own_host() {
      let client = CopilotClient::new(CopilotConfig {
         account_type: "business".into(),
         ..CopilotConfig::default()
      });
      assert_eq!(client.base_url(), "https://api.business.githubcopilot.com");
   }
}
