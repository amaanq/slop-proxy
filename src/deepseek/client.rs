//! `DeepSeek` publishes an Anthropic-compatible endpoint under `/anthropic`, so
//! a request is the one the caller already sent and the reply needs no
//! translation. Its `/anthropic/v1/models` 404s though, so the catalog comes
//! from the OpenAI-shaped list at the root instead.

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;

use crate::config::DeepSeekConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, SendError, classify};

pub struct DeepSeekClient {
   egresses: Egresses,
   cfg: DeepSeekConfig,
}

impl DeepSeekClient {
   pub fn new(cfg: DeepSeekConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress.urls()?, "deepseek", None)?;
      Ok(Self { egresses, cfg })
   }

   fn root(&self) -> &str {
      self.cfg.base_url.trim_end_matches('/')
   }

   pub async fn models(&self, key: &str) -> Result<Vec<String>, SendError> {
      #[derive(serde::Deserialize)]
      struct Entry {
         id: String,
      }
      #[derive(serde::Deserialize)]
      struct Listing {
         data: Vec<Entry>,
      }
      let resp = self
         .egresses
         .send(|http| async move {
            http
               .get(format!("{}/models", self.root()))
               .bearer_auth(key)
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
         .map(|listing| listing.data.into_iter().map(|entry| entry.id).collect())
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
               .post(format!("{}/anthropic{path}", self.root()))
               .header("x-api-key", key)
               .header("anthropic-version", "2023-06-01")
               .header(CONTENT_TYPE, "application/json")
               .body(body.clone())
               .send()
               .await
               .map_err(|err| SendError::Network(err.to_string()))
         })
         .await?;
      classify(resp, Classify::STRICT).await
   }
}
