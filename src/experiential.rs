//! Verbatim relay to the Experiential gateway over /v1/messages only.

use axum::body::Bytes;

use crate::config::ExperientialConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

pub struct ExperientialClient {
   egresses: Egresses,
   cfg: ExperientialConfig,
}

impl ExperientialClient {
   pub fn new(cfg: ExperientialConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress.urls()?, "experiential", None)?;
      Ok(Self { egresses, cfg })
   }

   pub async fn post(
      &self,
      key: &str,
      path: &str,
      body: &Bytes,
   ) -> Result<reqwest::Response, SendError> {
      let resp = self
         .egresses
         .send(|http| async move {
            http
               .post(format!("{}{path}", self.cfg.base_url.trim_end_matches('/')))
               .bearer_auth(key)
               .header("content-type", "application/json")
               .body(body.clone())
               .send()
               .await
               .map_err(|err| SendError::Network(err.to_string()))
         })
         .await?;
      classify(resp, Classify::STRICT).await
   }
}
