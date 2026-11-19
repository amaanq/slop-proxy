//! Verbatim relay to a self-hosted radiance server's /v1/messages.

use axum::body::Bytes;

use crate::config::RelayConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, IdList, SendError, classify, json};

const BASE_URL: &str = "http://[::1]:8080";

pub struct RadianceClient {
   egresses: Egresses,
   cfg: RelayConfig,
}

impl RadianceClient {
   pub fn new(cfg: RelayConfig) -> misstep::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "radiance", None)?;
      Ok(Self { egresses, cfg })
   }

   pub const fn egresses(&self) -> &Egresses {
      &self.egresses
   }

   pub async fn models(&self, key: &str) -> Result<Vec<String>, SendError> {
      let resp = self
         .egresses
         .send(|http| {
            http
               .get(format!("{}/v1/models", self.cfg.base_url_or(BASE_URL)))
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
