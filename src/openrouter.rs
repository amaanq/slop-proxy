//! Verbatim relay to `OpenRouter`'s Anthropic-compatible /v1/messages.

use axum::body::Bytes;

use crate::config::RelayConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

const BASE_URL: &str = "https://openrouter.ai/api";

pub struct OpenRouterClient {
   egresses: Egresses,
   cfg: RelayConfig,
}

impl OpenRouterClient {
   pub fn new(cfg: RelayConfig) -> misstep::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "openrouter", None)?;
      Ok(Self { egresses, cfg })
   }

   pub const fn egresses(&self) -> &Egresses {
      &self.egresses
   }

   pub async fn post(
      &self,
      key: &str,
      path: &str,
      body: &Bytes,
   ) -> Result<reqwest::Response, SendError> {
      let resp = self
         .egresses
         .send(|http| {
            http
               .post(format!("{}{path}", self.cfg.base_url_or(BASE_URL)))
               .bearer_auth(key)
               .header("content-type", "application/json")
               .body(body.clone())
               .send()
         })
         .await?;
      classify(resp, Classify::STRICT).await
   }
}
