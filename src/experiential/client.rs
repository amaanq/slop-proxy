//! Verbatim relay to the Experiential gateway over /v1/messages only.

use std::time::Duration;

use axum::body::Bytes;

use crate::config::ExperientialConfig;
use crate::upstream::{Classify, SendError, classify};

pub struct ExperientialClient {
   http: reqwest::Client,
   cfg: ExperientialConfig,
}

impl ExperientialClient {
   pub fn new(cfg: ExperientialConfig) -> Self {
      let http = reqwest::Client::builder()
         .connect_timeout(Duration::from_secs(30))
         .tcp_keepalive(Duration::from_secs(30))
         .build()
         .expect("building http client");
      Self { http, cfg }
   }

   pub async fn post(
      &self,
      key: &str,
      path: &str,
      body: &Bytes,
   ) -> Result<reqwest::Response, SendError> {
      let resp = self
         .http
         .post(format!("{}{path}", self.cfg.base_url.trim_end_matches('/')))
         .bearer_auth(key)
         .header("content-type", "application/json")
         .body(body.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      classify(resp, Classify::STRICT).await
   }
}
