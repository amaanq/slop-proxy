use super::{Backend, Pool, Relay, Route, Slot};
use crate::experiential::ExperientialClient;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Experiential gateway keys. Each key belongs to
/// an org with its own free daily bucket, so rotation multiplies the quota.
pub type ExperientialPool = Pool<ExperientialClient>;

impl Backend for ExperientialClient {
   const PROVIDER: Provider = Provider::Experiential;
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
