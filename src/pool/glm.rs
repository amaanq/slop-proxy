use crate::anthropic::Model;
use crate::glm::GlmClient;
use crate::pool::{Backend, Pool, Relay, Route, Slot};
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Z.ai keys.
pub type GlmPool = Pool<GlmClient>;

impl GlmPool {
   pub async fn models(&self) -> Vec<Model> {
      self
         .first_answer(async |backend, key, _| backend.models(key).await)
         .await
         .unwrap_or_default()
   }
}

impl Backend for GlmClient {
   const PROVIDER: Provider = Provider::Glm;
   type Request = Relay;
   type Response = reqwest::Response;

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
