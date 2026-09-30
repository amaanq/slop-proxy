use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::anthropic::Model;
use crate::glm::client::GlmClient;
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;

/// Session-sticky pool over Z.ai keys.
pub type GlmPool = Pool<GlmClient>;

impl GlmPool {
   pub async fn models(&self) -> Vec<Model> {
      for slot in self.slots.list().await {
         let Ok(key) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.models(&key).await {
            Ok(models) => return models,
            Err(err) => tracing::debug!("models for {}: {err}", slot.display),
         }
      }
      Vec::new()
   }
}

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
}

impl Backend for GlmClient {
   const PROVIDER: Provider = Provider::Glm;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
   type Request = Relay;
   type Response = reqwest::Response;

   fn reason(body: String) -> String {
      ChatError::reason(body)
   }

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      Self::post(self, token, req.path, &req.body, route.session_key).await
   }
}
