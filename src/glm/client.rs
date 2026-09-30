//! Z.ai publishes an Anthropic-compatible endpoint, so a GLM request is the
//! one the caller already sent and the reply needs no translation.

use axum::body::Bytes;
use reqwest::RequestBuilder;
use reqwest::header::CONTENT_TYPE;
use uuid::Uuid;

use crate::anthropic::Model;
use crate::config::GlmConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

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

pub struct GlmClient {
   egresses: Egresses,
   cfg: GlmConfig,
}

impl GlmClient {
   pub fn new(cfg: GlmConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress.urls()?, "glm", None)?;
      Ok(Self { egresses, cfg })
   }

   fn base_url(&self) -> &str {
      self.cfg.base_url.trim_end_matches('/')
   }

   pub async fn models(&self, key: &str) -> Result<Vec<Model>, SendError> {
      #[derive(serde::Deserialize)]
      struct Listing {
         data: Vec<Model>,
      }
      let resp = self
         .egresses
         .send(|http| async move {
            zcode(http.get(format!("{}/v1/models", self.base_url())), key)
               .send()
               .await
               .map_err(|err| SendError::Network(err.to_string()))
         })
         .await?;
      let resp = classify(resp, Classify::STRICT).await?;
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
      session: &str,
   ) -> Result<reqwest::Response, SendError> {
      let resp = self
         .egresses
         .send(|http| async move {
            zcode(http.post(format!("{}{path}", self.base_url())), key)
               .header(CONTENT_TYPE, "application/json")
               .header("x-request-id", Uuid::new_v4().to_string())
               .header("x-zcode-trace-id", Uuid::new_v4().to_string())
               .header("x-session-id", session)
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
