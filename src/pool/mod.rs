pub mod anthropic;
pub mod codex;
pub mod gemini;
pub mod glm;
pub mod pools;
pub mod slots;
pub mod zen;

use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;
use std::cmp::Reverse;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

pub use pools::Pools;
#[cfg(test)]
pub use slots::test_slots;
pub use slots::{AccountSnapshot, AccountUsage, ModelWindow, Slot, UsageWindow, window_seconds};
pub use slots::{Slots, rendezvous_score};

#[derive(Debug, Error)]
pub enum PoolError {
   #[error("no usable {0} accounts; run `slop-proxy login`")]
   NoAccounts(Provider),
   #[error("all upstream accounts are cooling down")]
   AllCoolingDown { retry_after: i64 },
   #[error("the {provider} backend rejected {model}: {body}")]
   BadRequest {
      provider: Provider,
      model: String,
      body: String,
   },
   #[error("upstream failure: {0}")]
   Upstream(String),
}

impl From<SendError> for PoolError {
   fn from(e: SendError) -> Self {
      Self::Upstream(e.to_string())
   }
}

pub struct Cooldown {
   pub max: i64,
   pub base: i64,
}

pub enum AuthPolicy {
   RefreshOnce,
   CoolKey(i64),
}

#[derive(Clone, Copy)]
pub struct Route<'a> {
   pub session_key: &'a str,
   pub model: &'a str,
   pub prefer_trusted: bool,
}

pub trait Backend: Send + Sync + 'static {
   const PROVIDER: Provider;
   const RATE_LIMIT: Cooldown;
   const ON_AUTH: AuthPolicy;
   const ATTEMPTS: usize = 3;
   /// Accounts come in two tiers and a token may prefer one, codex only.
   const TIERED: bool = false;
   /// A session waits this long for its own account's cooldown rather than losing the prompt cache.
   const STICKY_WAIT_SECS: i64 = 0;
   /// The backend serves without an account, zen's free tier.
   const ANONYMOUS: bool = false;
   type Request: Clone + Send + Sync;
   type Response: Send;
   fn soft_limit(&self) -> f64 {
      1.0
   }
   fn retry_budget(&self) -> Duration {
      Duration::ZERO
   }
   async fn send(
      &self,
      token: &str,
      slot: &Slot,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError>;
   async fn send_anonymous(&self, req: &Self::Request) -> Result<Self::Response, SendError> {
      let _ = req;
      Err(SendError::Network("no accounts".into()))
   }
   /// A 400 that describes this account rather than the request.
   fn retryable_bad_request(&self, body: &str) -> bool {
      let _ = body;
      false
   }
   /// Each dialect buries its one useful sentence at a different depth.
   fn reason(body: String) -> String {
      body
   }
   fn usage_from(&self, resp: &Self::Response) -> Option<AccountUsage> {
      let _ = resp;
      None
   }
}

pub struct Pool<B: Backend> {
   slots: Slots,
   backend: B,
}

impl<B: Backend> Pool<B> {
   pub async fn load(db: Db, backend: B) -> eyre::Result<Self> {
      Ok(Self {
         slots: Slots::load(db, B::PROVIDER).await?,
         backend,
      })
   }

   pub const fn backend(&self) -> &B {
      &self.backend
   }

   pub async fn len(&self) -> usize {
      self.slots.len().await
   }

   pub async fn reload(&self) -> eyre::Result<()> {
      self.slots.reload().await
   }

   pub async fn snapshot(&self) -> Vec<AccountSnapshot> {
      self.slots.snapshot().await
   }

   /// Candidates are tried with capacity ahead of preference, so an account
   /// with room left beats a preferred one that is nearly spent, and only
   /// then does the token's trusted preference break the tie. Within a group
   /// a session sticks to one account, since a prompt cache lives on the
   /// account that built it and scattering re-bills the whole prefix.
   pub(crate) async fn ranked(&self, route: Route<'_>) -> Vec<Arc<Slot>> {
      let mut scored = Vec::new();
      for slot in self.slots.list().await {
         let band = self.slots.band(&slot, self.backend.soft_limit()).await;
         scored.push((
            band,
            B::TIERED && slot.trusted != route.prefer_trusted,
            Reverse(rendezvous_score(route.session_key, slot.id)),
            slot,
         ));
      }
      scored.sort_by_key(|(band, mismatch, score, _)| (*band, *mismatch, *score));
      scored.into_iter().map(|(_, _, _, slot)| slot).collect()
   }

   async fn served(&self, slot: &Slot, resp: B::Response) -> B::Response {
      self.slots.mark_ok(slot).await;
      if let Some(usage) = self.backend.usage_from(&resp) {
         self.slots.note_usage(slot, usage).await;
      }
      resp
   }

   /// Google refills a token bucket in 20-40s and the whole pool empties at
   /// once, so a sweep repeats until the budget is spent. The wait is
   /// jittered to stop a queue waking together and draining the refill.
   pub async fn execute(
      &self,
      route: Route<'_>,
      req: B::Request,
   ) -> Result<(Option<i64>, B::Response), PoolError> {
      let budget = self.backend.retry_budget();
      let deadline = std::time::Instant::now() + budget;
      loop {
         let err = match self.sweep(route, &req).await {
            Err(e @ PoolError::AllCoolingDown { .. }) if !budget.is_zero() => e,
            other => return other,
         };
         let left = deadline.saturating_duration_since(std::time::Instant::now());
         if left.is_zero() {
            return Err(err);
         }
         let wait = Duration::from_secs(self.slots.min_cooldown().await.max(1) as u64)
            .min(left)
            .saturating_add(Duration::from_millis(rand::random::<u64>() % 1500));
         tracing::info!(
             provider = %B::PROVIDER,
             wait_ms = wait.as_millis() as u64,
             "pool is empty, holding the request rather than returning a 429"
         );
         tokio::time::sleep(wait).await;
      }
   }

   async fn sweep(
      &self,
      route: Route<'_>,
      req: &B::Request,
   ) -> Result<(Option<i64>, B::Response), PoolError> {
      let ranked = self.ranked(route).await;
      if ranked.is_empty() {
         if B::ANONYMOUS {
            return match self.backend.send_anonymous(req).await {
               Ok(r) => Ok((None, r)),
               Err(SendError::BadRequest(body)) => Err(PoolError::BadRequest {
                  provider: B::PROVIDER,
                  model: route.model.into(),
                  body: B::reason(body),
               }),
               Err(SendError::RateLimited { retry_after, .. }) => Err(PoolError::AllCoolingDown {
                  retry_after: retry_after.unwrap_or(30),
               }),
               Err(e) => Err(PoolError::Upstream(e.to_string())),
            };
         }
         return Err(PoolError::NoAccounts(B::PROVIDER));
      }
      if B::STICKY_WAIT_SECS > 0
         && let Some(preferred) = ranked.first()
      {
         let left = self.slots.cooldown_left(preferred).await;
         if (1..=B::STICKY_WAIT_SECS).contains(&left) {
            tracing::debug!(
                account = %preferred.display,
                left,
                "waiting for the sticky gemini key rather than losing its cache"
            );
            tokio::time::sleep(Duration::from_secs(left as u64 + 1)).await;
         }
      }
      let mut last_err = Option::<SendError>::None;
      let mut attempts = 0;
      for slot in ranked {
         if attempts >= B::ATTEMPTS {
            break;
         }
         if !self.slots.try_claim(&slot).await {
            continue;
         }
         attempts += 1;
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.send(&token, &slot, route, req).await {
            Ok(resp) => return Ok((Some(slot.id), self.served(&slot, resp).await)),
            Err(SendError::Auth(text)) => match B::ON_AUTH {
               AuthPolicy::CoolKey(secs) => {
                  self.slots.cool(&slot, secs, "key rejected").await;
                  last_err = Some(SendError::Auth(text));
               },
               AuthPolicy::RefreshOnce => {
                  tracing::warn!("account {} got 401, forcing refresh", slot.display);
                  if let Ok(fresh) = self.slots.fresh_token(&slot, true).await {
                     match self.backend.send(&fresh, &slot, route, req).await {
                        Ok(resp) => {
                           return Ok((Some(slot.id), self.served(&slot, resp).await));
                        },
                        Err(e) => {
                           self.slots.cool(&slot, 60, "post-refresh failure").await;
                           last_err = Some(e);
                        },
                     }
                  } else {
                     last_err = Some(SendError::Auth(text));
                  }
               },
            },
            Err(SendError::RateLimited { retry_after, body }) => {
               self
                  .slots
                  .cool_rate_limited(&slot, retry_after, B::RATE_LIMIT.max, B::RATE_LIMIT.base)
                  .await;
               last_err = Some(SendError::RateLimited { retry_after, body });
            },
            Err(SendError::BadRequest(body)) if self.backend.retryable_bad_request(&body) => {
               tracing::warn!(
                   account = %slot.display,
                   "account cannot serve this model, trying another: {body}"
               );
               last_err = Some(SendError::BadRequest(body));
            },
            Err(SendError::BadRequest(body)) => {
               return Err(PoolError::BadRequest {
                  provider: B::PROVIDER,
                  model: route.model.into(),
                  body: B::reason(body),
               });
            },
            Err(e) => {
               self.slots.cool_failure(&slot).await;
               last_err = Some(e);
            },
         }
      }
      match last_err {
         Some(SendError::BadRequest(body)) => Err(PoolError::BadRequest {
            provider: B::PROVIDER,
            model: route.model.into(),
            body: B::reason(body),
         }),
         Some(SendError::RateLimited { .. }) | None => Err(PoolError::AllCoolingDown {
            retry_after: self.slots.min_cooldown().await.max(30),
         }),
         Some(e) => Err(PoolError::Upstream(e.to_string())),
      }
   }
}

#[cfg(test)]
mod retry_tests {
   use std::sync::atomic::{AtomicUsize, Ordering};

   use super::*;

   struct Flaky {
      calls: AtomicUsize,
      frees_after: usize,
      budget: Duration,
   }

   impl Backend for Flaky {
      const PROVIDER: Provider = Provider::Gemini;
      const RATE_LIMIT: Cooldown = Cooldown { max: 1, base: 1 };
      const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(60);
      type Request = ();
      type Response = usize;

      fn retry_budget(&self) -> Duration {
         self.budget
      }

      async fn send(
         &self,
         _token: &str,
         _slot: &Slot,
         _route: Route<'_>,
         _req: &Self::Request,
      ) -> Result<Self::Response, SendError> {
         let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
         if n <= self.frees_after {
            return Err(SendError::RateLimited {
               retry_after: None,
               body: "quota".into(),
            });
         }
         Ok(n)
      }
   }

   fn pool(frees_after: usize, budget: Duration) -> Pool<Flaky> {
      let db_path = std::env::temp_dir().join(format!("slop-retry-{}.db", uuid::Uuid::new_v4()));
      let db = Db::open(&db_path).unwrap();
      Pool {
         slots: test_slots(db, Provider::Gemini, &[(1, false)]),
         backend: Flaky {
            calls: AtomicUsize::new(0),
            frees_after,
            budget,
         },
      }
   }

   fn route() -> Route<'static> {
      Route {
         session_key: "s",
         model: "m",
         prefer_trusted: false,
      }
   }

   #[tokio::test]
   async fn a_rate_limited_pool_is_waited_out_rather_than_handed_back() {
      let pool = pool(1, Duration::from_secs(10));
      let (_, calls) = pool.execute(route(), ()).await.unwrap();
      assert_eq!(calls, 2, "the second sweep should have been served");
   }

   #[tokio::test]
   async fn no_budget_keeps_the_old_behaviour() {
      let pool = pool(usize::MAX, Duration::ZERO);
      assert!(matches!(
         pool.execute(route(), ()).await,
         Err(PoolError::AllCoolingDown { .. })
      ));
      assert_eq!(pool.backend.calls.load(Ordering::SeqCst), 1);
   }
}

#[cfg(test)]
mod reason_tests {
   use super::*;
   use crate::anthropic::client::AnthropicClient;
   use crate::codex::client::CodexClient;
   use crate::gemini::client::GeminiClient;

   #[test]
   fn each_envelope_gives_up_its_one_sentence() {
      assert_eq!(
         CodexClient::reason(r#"{"detail":"no such model"}"#.into()),
         "no such model"
      );
      assert_eq!(
         GeminiClient::reason(
            r#"{"error":{"code":400,"message":"contents is not specified"}}"#.into()
         ),
         "contents is not specified"
      );
      assert_eq!(
            AnthropicClient::reason(
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens is too large"}}"#.into()
            ),
            "max_tokens is too large"
        );
   }
}
