//! Z.ai publishes an Anthropic-compatible endpoint, so a GLM request is the
//! one the caller already sent and the reply needs no translation.

use std::time::Duration;

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;

use crate::config::GlmConfig;
use crate::upstream::{Classify, SendError, classify};

pub struct GlmClient {
   http: reqwest::Client,
   cfg: GlmConfig,
}

impl GlmClient {
   pub fn new(cfg: GlmConfig) -> Self {
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
         .header("x-api-key", key)
         .header("anthropic-version", "2023-06-01")
         .header("content-type", "application/json")
         .header(CONTENT_TYPE, "application/json")
         .body(body.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      match classify(resp, Classify::STRICT).await {
         Err(SendError::RateLimited { body: text, .. })
            if text.contains("Insufficient balance") =>
         {
            Err(SendError::Auth(text))
         },
         other => other,
      }
   }
}
