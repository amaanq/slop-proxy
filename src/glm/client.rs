//! Z.ai publishes an Anthropic-compatible endpoint, so a GLM request is the
//! one the caller already sent and the reply needs no translation.

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;

use crate::anthropic::Model;
use crate::config::GlmConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

pub struct GlmClient {
   egresses: Egresses,
   cfg: GlmConfig,
}

impl GlmClient {
   pub fn new(cfg: GlmConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress.urls()?, "glm", None)?;
      Ok(Self { egresses, cfg })
   }

   pub async fn models(&self, key: &str) -> Result<Vec<Model>, SendError> {
      #[derive(serde::Deserialize)]
      struct Listing {
         data: Vec<Model>,
      }
      let resp = self
         .egresses
         .send(|http| async move {
            http
               .get(format!(
                  "{}/v1/models",
                  self.cfg.base_url.trim_end_matches('/')
               ))
               .header("x-api-key", key)
               .header("anthropic-version", "2023-06-01")
               .send()
               .await
               .map_err(|err| SendError::Network(err.to_string()))
         })
         .await?;
      let status = resp.status().as_u16();
      let body = resp.text().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("reading models response: {err}"),
      })?;
      serde_json::from_str::<Listing>(&body)
         .map(|listing| listing.data)
         .map_err(|err| SendError::Upstream {
            status,
            body: format!("parsing models response: {err}"),
         })
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
               .header("x-api-key", key)
               .header("anthropic-version", "2023-06-01")
               .header(CONTENT_TYPE, "application/json")
               .body(body.clone())
               .send()
               .await
               .map_err(|err| SendError::Network(err.to_string()))
         })
         .await?;
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
