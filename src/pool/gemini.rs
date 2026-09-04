use axum::body::Bytes;

use crate::translate::chat::ChatRequest;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::gemini::client::{GeminiClient, GeminiProtocol, GeminiResponse};
use crate::provider::Provider;
use crate::upstream::SendError;

/// What to send upstream. A caller already speaking the native dialect skips
/// the translation entirely, but shares the retry and cooldown policy.
#[derive(Clone)]
pub enum Call {
   OpenAi(Box<ChatRequest>),
   Native {
      model: String,
      action: String,
      query: Option<String>,
      body: Bytes,
   },
}

/// Google answers a 429 with no retry-after and refills a token bucket rather
/// than clearing a window, so the wait is measured: exhausting one key and
/// probing it back took 20s, 27s and 37s.
const RATE_LIMIT_COOLDOWN_SECS: i64 = 40;

/// Google caches per key, so a conversation that moves keys re-sends its whole
/// prefix as fresh input, which both re-bills it and spends more of the budget
/// that caused the failover. Cache hit rate is 84.6% here against 96.9% on
/// codex. Matching the cooldown means one wait covers exactly one backoff.
const STICKY_WAIT_SECS: i64 = RATE_LIMIT_COOLDOWN_SECS;

/// Session-sticky pool over Gemini accounts.
pub type GeminiPool = Pool<GeminiClient>;

impl Backend for GeminiClient {
   const PROVIDER: Provider = Provider::Gemini;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: RATE_LIMIT_COOLDOWN_SECS,
   };
   // A static key cannot be refreshed into a working one, so a
   // rejected key sits out rather than retrying in place.
   const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
   const STICKY_WAIT_SECS: i64 = STICKY_WAIT_SECS;
   type Request = Call;
   type Response = GeminiResponse;

   fn reason(body: String) -> String {
      crate::translate::chat::ChatError::reason(body)
   }

   fn soft_limit(&self) -> f64 {
      self.soft_utilization_limit()
   }

   fn retry_budget(&self) -> std::time::Duration {
      Self::retry_budget_duration(self)
   }

   async fn send(
      &self,
      token: &str,
      slot: &Slot,
      _route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      match req {
         Call::OpenAi(body) => Self::post(self, token, slot.http_referer.as_deref(), body).await,
         Call::Native {
            model,
            action,
            query,
            body,
         } => Self::send_native(
            self,
            token,
            slot.http_referer.as_deref(),
            model,
            action,
            query.as_deref(),
            body,
         )
         .await
         .map(|response| GeminiResponse {
            response,
            protocol: GeminiProtocol::Native,
         }),
      }
   }
}

impl Pool<GeminiClient> {
   /// The first account that answers. Every key sees the same catalog, so
   /// there is nothing to merge across accounts.
   pub async fn models(&self) -> Vec<String> {
      for slot in self.slots.list().await {
         let Ok(key) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self
            .backend
            .models(&key, slot.http_referer.as_deref())
            .await
         {
            Ok(ids) => return ids,
            Err(e) => tracing::debug!("models for {}: {e}", slot.display),
         }
      }
      Vec::new()
   }
}
