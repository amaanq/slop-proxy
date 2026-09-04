use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::provider::Provider;
use crate::upstream::SendError;
use crate::zen::client::ZenClient;

/// Zen over whatever credentials are stored, and over none at all when the
/// table is empty. The free models are served without a key today, so an
/// empty pool is a working pool rather than an error.
pub type ZenPool = Pool<ZenClient>;

impl Backend for ZenClient {
   const PROVIDER: Provider = Provider::Zen;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
   const ANONYMOUS: bool = true;
   type Request = Bytes;
   type Response = reqwest::Response;

   fn reason(body: String) -> String {
      crate::translate::chat::ChatError::reason(body)
   }

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      _route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      Self::post(self, Some(token), req).await
   }

   async fn send_anonymous(&self, req: &Self::Request) -> Result<Self::Response, SendError> {
      Self::post(self, None, req).await
   }
}

impl Pool<ZenClient> {
   pub async fn models(&self) -> Vec<String> {
      self.backend.models().await.unwrap_or_default()
   }
}
