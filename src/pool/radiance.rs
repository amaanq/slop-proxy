use crate::radiance::RadianceClient;
use crate::pool::{Backend, Pool, Relay, Route, Slot};
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over radiance server keys.
pub type RadiancePool = Pool<RadianceClient>;

impl RadiancePool {
   pub async fn models(&self) -> Option<Vec<String>> {
      self
         .first_answer(async |backend, key, _| backend.models(key).await)
         .await
   }
}

impl Backend for RadianceClient {
   const PROVIDER: Provider = Provider::Radiance;
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
