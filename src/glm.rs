//! Z.ai publishes an Anthropic-compatible endpoint, so a GLM request is the
//! one the caller already sent and the reply needs no translation.

use axum::body::Bytes;
use reqwest::RequestBuilder;
use reqwest::header::CONTENT_TYPE;
use uuid::Uuid;

use crate::anthropic::Model;
use crate::config::RelayConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify, json};

/// The client identity `ZCode` 3.14.4 sends with a coding-plan key.
fn zcode(req: RequestBuilder, key: &str) -> RequestBuilder {
   req.header("x-api-key", key)
      .header("anthropic-version", "2023-06-01")
      .header("user-agent", "ZCode/3.14.4")
      .header("x-zcode-app-version", "3.14.4")
      .header("x-title", "Z Code@cli")
      .header("x-zcode-agent", "glm")
      .header("http-referer", "https://zcode.z.ai")
}

const RULES: Classify = Classify {
   dead_key: &["Insufficient balance"],
   ..Classify::STRICT
};

#[derive(serde::Deserialize)]
struct Listing {
   data: Vec<Model>,
}

const BASE_URL: &str = "https://api.z.ai/api/anthropic";
const QUOTA_URL: &str = "https://api.z.ai/api/monitor/usage/quota/limit";

#[derive(serde::Deserialize)]
struct QuotaEnvelope {
   data: QuotaReport,
}

#[derive(serde::Deserialize)]
pub struct QuotaReport {
   pub limits: Vec<QuotaLimit>,
}

#[derive(serde::Deserialize)]
pub struct QuotaLimit {
   pub unit: QuotaUnit,
   pub number: u32,
   #[serde(rename = "usage")]
   pub allowance: f64,
   #[serde(rename = "currentValue")]
   pub used: f64,
   pub remaining: f64,
   #[serde(rename = "nextResetTime")]
   pub next_reset_ms: Option<i64>,
}

/// Z.ai's `unit` codes, as observed on a pro coding plan.
#[derive(Clone, Copy, serde::Deserialize)]
#[serde(try_from = "u8")]
pub enum QuotaUnit {
   Hour,
   Week,
}

impl TryFrom<u8> for QuotaUnit {
   type Error = String;

   fn try_from(code: u8) -> Result<Self, Self::Error> {
      match code {
         3 => Ok(Self::Hour),
         6 => Ok(Self::Week),
         other => Err(format!("unknown quota unit {other}")),
      }
   }
}

pub struct GlmClient {
   egresses: Egresses,
   cfg: RelayConfig,
}

impl GlmClient {
   pub fn new(cfg: RelayConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "glm", None)?;
      Ok(Self { egresses, cfg })
   }

   pub const fn egresses(&self) -> &Egresses {
      &self.egresses
   }

   fn base_url(&self) -> &str {
      self.cfg.base_url_or(BASE_URL)
   }

   pub async fn models(&self, key: &str) -> Result<Vec<Model>, SendError> {
      let resp = self
         .egresses
         .send(|http| zcode(http.get(format!("{}/v1/models", self.base_url())), key).send())
         .await?;
      Ok(json::<Listing>(resp, Classify::STRICT).await?.data)
   }

   pub async fn quota(&self, key: &str) -> Result<QuotaReport, SendError> {
      let resp = self
         .egresses
         .send(|http| http.get(QUOTA_URL).header("authorization", key).send())
         .await?;
      Ok(json::<QuotaEnvelope>(resp, Classify::STRICT).await?.data)
   }

   pub async fn post(
      &self,
      key: &str,
      path: &str,
      body: &Bytes,
      session: &str,
   ) -> Result<reqwest::Response, SendError> {
      let resp = self
         .egresses
         .send(|http| {
            zcode(http.post(format!("{}{path}", self.base_url())), key)
               .header(CONTENT_TYPE, "application/json")
               .header("x-request-id", Uuid::new_v4().to_string())
               .header("x-zcode-trace-id", Uuid::new_v4().to_string())
               .header("x-session-id", session)
               .body(body.clone())
               .send()
         })
         .await?;
      classify(resp, RULES).await
   }
}
