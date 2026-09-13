use std::sync::LazyLock;

use axum::body::Bytes;
use rand::{Rng as _, thread_rng};

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::clock::unix_now_ms;
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
      let session = session(route);
      Self::post(self, Some(token), &session, req.path, &req.body).await
   }

   async fn send_anonymous(
      &self,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      let session = session(route);
      Self::post(self, None, &session, req.path, &req.body).await
   }
}

const ALPHABET: &[u8; 62] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ID_TIME_MASK: i64 = 0xffff_ffff_ffff;
static SESSION_PREFIX: LazyLock<String> = LazyLock::new(session_prefix);

fn session(route: Route<'_>) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   if route.session_key.is_empty() {
      hasher.update(route.user.as_bytes());
      hasher.update(b"\0");
      hasher.update(route.model.as_bytes());
   } else {
      hasher.update(route.session_key.as_bytes());
   }
   let digest = hasher.finalize();
   let suffix = digest
      .iter()
      .take(14)
      .map(|byte| char::from(ALPHABET[usize::from(*byte) % ALPHABET.len()]))
      .collect::<String>();
   format!("ses_{}{suffix}", SESSION_PREFIX.as_str())
}

fn session_prefix() -> String {
   let counter = thread_rng().gen_range(1..=0xfff_i64);
   let timestamp = !(unix_now_ms().saturating_mul(0x1000) + counter) & ID_TIME_MASK;
   format!("{timestamp:012x}")
}

impl Pool<ZenClient> {
   pub async fn models(&self) -> Vec<String> {
      self.backend.models().await.unwrap_or_default()
   }
}
