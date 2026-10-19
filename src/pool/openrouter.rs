use crate::openrouter::OpenRouterClient;
use crate::pool::{Backend, Pool, Relay, Route, Slot};
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over `OpenRouter` keys.
pub type OpenRouterPool = Pool<OpenRouterClient>;

impl Backend for OpenRouterClient {
   const PROVIDER: Provider = Provider::OpenRouter;
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
