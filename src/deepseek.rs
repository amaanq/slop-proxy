//! `DeepSeek` publishes an Anthropic-compatible endpoint under `/anthropic`, so
//! a request is the one the caller already sent and the reply needs no
//! translation. Its `/anthropic/v1/models` 404s though, so the catalog comes
//! from the OpenAI-shaped list at the root instead.

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;

use crate::config::RelayConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, IdList, SendError, classify, json};

/// The site root, not the messages surface. `DeepSeek` serves its catalog from
/// the root and its Anthropic dialect from `/anthropic` under it.
const BASE_URL: &str = "https://api.deepseek.com";

pub struct DeepSeekClient {
   egresses: Egresses,
   cfg: RelayConfig,
}

impl DeepSeekClient {
   pub fn new(cfg: RelayConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "deepseek", None)?;
      Ok(Self { egresses, cfg })
   }

   pub const fn egresses(&self) -> &Egresses {
      &self.egresses
   }

   fn root(&self) -> &str {
      self.cfg.base_url_or(BASE_URL)
   }

   pub async fn models(&self, key: &str) -> Result<Vec<String>, SendError> {
      let resp = self
         .egresses
         .send(|http| {
            http
               .get(format!("{}/models", self.root()))
               .bearer_auth(key)
               .send()
         })
         .await?;
      let listing: IdList = json(resp, Classify::STRICT).await?;
      Ok(listing.data.into_iter().map(|entry| entry.id).collect())
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
               .post(format!("{}/anthropic{path}", self.root()))
               .header("x-api-key", key)
               .header("anthropic-version", "2023-06-01")
               .header(CONTENT_TYPE, "application/json")
               .body(body.clone())
               .send()
         })
         .await?;
      classify(resp, Classify::STRICT).await
   }
}
