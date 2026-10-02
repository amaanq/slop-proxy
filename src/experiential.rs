//! Verbatim relay to the Experiential gateway over /v1/messages only.

use axum::body::Bytes;

use crate::config::RelayConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

const BASE_URL: &str = "https://api.experientiallabs.ai";

pub struct ExperientialClient {
   egresses: Egresses,
   cfg: RelayConfig,
}

impl ExperientialClient {
   pub fn new(cfg: RelayConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "experiential", None)?;
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
