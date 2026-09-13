use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::deepseek::client::DeepSeekClient;
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;

/// Session-sticky pool over `DeepSeek` keys.
pub type DeepSeekPool = Pool<DeepSeekClient>;

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
}

impl DeepSeekPool {
   pub async fn models(&self) -> Vec<String> {
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

impl Backend for DeepSeekClient {
   const PROVIDER: Provider = Provider::DeepSeek;
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
      _route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      Self::post(self, token, req.path, &req.body).await
   }
}
