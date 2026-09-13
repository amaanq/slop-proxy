use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;
use crate::zen::client::ZenClient;

/// Zen over whatever credentials are stored, and over none at all when the
/// table is empty. The free models are served without a key today, so an
/// empty pool is a working pool rather than an error.
pub type ZenPool = Pool<ZenClient>;

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
   /// Zen hashes the session id's last four characters to pick the upstream
   /// that serves a model, so a retry has to move the tail or it lands on the
   /// same dead one. Zero keeps the stable id, and its prompt cache with it.
   pub attempt: usize,
}

impl Backend for ZenClient {
   const PROVIDER: Provider = Provider::Zen;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
   const ANONYMOUS: bool = true;
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
      let session = session(route, req.attempt);
      Self::post(self, Some(token), &session, req.path, &req.body).await
   }

   async fn send_anonymous(
      &self,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      let session = session(route, req.attempt);
      Self::post(self, None, &session, req.path, &req.body).await
   }
}

const ALPHABET: &[u8; 62] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn session(route: Route<'_>, attempt: usize) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   if route.session_key.is_empty() {
      hasher.update(route.user.as_bytes());
      hasher.update(b"\0");
      hasher.update(route.model.as_bytes());
   } else {
      hasher.update(route.session_key.as_bytes());
   }
   if attempt > 0 {
      hasher.update(b"\0retry\0");
      hasher.update(attempt.to_le_bytes());
   }
   let digest = hasher.finalize();
   let body = digest
      .iter()
      .take(26)
      .map(|byte| char::from(ALPHABET[usize::from(*byte) % ALPHABET.len()]))
      .collect::<String>();
   format!("ses_{body}")
}

impl Pool<ZenClient> {
   pub async fn models(&self) -> Vec<String> {
      self.backend.models().await.unwrap_or_default()
   }
}
