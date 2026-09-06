use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::experiential::client::ExperientialClient;
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;

/// Session-sticky pool over Experiential gateway keys. Each key belongs to
/// an org with its own free daily bucket, so rotation multiplies the quota.
pub type ExperientialPool = Pool<ExperientialClient>;

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
}

impl Backend for ExperientialClient {
   const PROVIDER: Provider = Provider::Experiential;
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
