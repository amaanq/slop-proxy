use super::{Backend, Pool, Relay, Route, Slot};
use crate::deepseek::DeepSeekClient;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over `DeepSeek` keys.
pub type DeepSeekPool = Pool<DeepSeekClient>;

impl DeepSeekPool {
   pub async fn models(&self) -> Vec<String> {
      self
         .first_answer(async |backend, key, _| backend.models(key).await)
         .await
         .unwrap_or_default()
   }
}

impl Backend for DeepSeekClient {
   const PROVIDER: Provider = Provider::DeepSeek;
   type Request = Relay;
   type Response = reqwest::Response;

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
