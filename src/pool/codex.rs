use std::collections::BTreeMap;
use std::sync::Arc;

use axum::body::Bytes;
use futures_util::{StreamExt as _, stream};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clock::unix_now;
use crate::codex::client::CodexClient;
use crate::codex::models::{ModelInfo, ModelsResponse, ServiceTier};
use crate::codex::turn_state::{self, TurnState};
use crate::codex::types::ErrorEnvelope;
use crate::codex::websocket::Connection;
use crate::pool::{
   AccountUsage, AuthPolicy, Backend, Cooldown, Pool, PoolError, Route, Served, Slot, UsageWindow,
   window_seconds,
};
use crate::provider::Provider;
use crate::upstream::SendError;

#[derive(Default, Deserialize)]
struct RateLimitEvent {
   metered_limit_name: Option<String>,
   limit_name: Option<String>,
   #[serde(default)]
   rate_limits: RateLimitsIn,
}

#[derive(Default, Deserialize)]
struct RateLimitsIn {
   primary: Option<WindowIn>,
   secondary: Option<WindowIn>,
}

#[derive(Deserialize)]
struct WindowIn {
   window_minutes: Option<i64>,
   used_percent: Option<f64>,
   reset_at: Option<i64>,
}

#[derive(Default, Serialize)]
struct RateLimitsOut {
   primary: Option<WindowOut>,
   secondary: Option<WindowOut>,
}

#[derive(Serialize)]
struct WindowOut {
   used_percent: f64,
   window_minutes: i64,
   reset_at: Option<i64>,
}

/// Session-sticky pool over codex accounts, owning the backend client.
pub type CodexPool = Pool<CodexClient>;

/// Floor for an exhausted account when the backend names no reset.
const EXHAUSTED_COOLDOWN: i64 = 15 * 60;

/// A catalog only moves when a model ships or an account's access changes.
const CATALOG_TTL: i64 = 300;

#[derive(Clone)]
pub enum Call {
   Http { body: Bytes, headers: HeaderMap },
   Search { body: Bytes, headers: HeaderMap },
   Backend { path: String, body: Option<Bytes> },
   WebSocket(HeaderMap),
}

pub enum Reply {
   Http(reqwest::Response),
   WebSocket(Box<Connection>),
}

impl Backend for CodexClient {
   const PROVIDER: Provider = Provider::OpenAi;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 6 * 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
   const TIERED: bool = true;
   const SESSION_AFFINITY: bool = true;
   type Request = Call;
   type Response = Reply;

   fn reason(body: String) -> String {
      ErrorEnvelope::reason(body)
   }

   fn soft_limit(&self) -> f64 {
      self.soft_utilization_limit()
   }

   async fn send(
      &self,
      token: &str,
      slot: &Slot,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      let session = session_uuid(route.session_key);
      match *req {
         Call::Http {
            ref body,
            ref headers,
         } => self
            .post(
               token,
               &slot.provider_account_id,
               body,
               &session,
               route.model,
               headers,
            )
            .await
            .map(Reply::Http),
         Call::Search {
            ref body,
            ref headers,
         } => self
            .search(
               token,
               &slot.provider_account_id,
               body,
               &session,
               route.model,
               headers,
            )
            .await
            .map(Reply::Http),
         Call::Backend { ref path, ref body } => self
            .backend(token, &slot.provider_account_id, path, body.as_ref())
            .await
            .map(Reply::Http),
         Call::WebSocket(ref headers) => {
            connect_pinned(self, slot, token, &session, route.model, headers)
               .await
               .map(|connection| Reply::WebSocket(Box::new(connection)))
         },
      }
   }

   fn retryable_bad_request(&self, body: &str) -> bool {
      // A preview model can be enabled per account, so "not supported"
      // describes this key rather than the request, and the next
      // account may well serve it.
      body.contains("is not supported")
   }

   fn usage_from(&self, resp: &Self::Response) -> Option<AccountUsage> {
      usage_from_headers(match *resp {
         Reply::Http(ref response) => response.headers(),
         Reply::WebSocket(ref connection) => &connection.headers,
      })
   }

   fn is_handshake(&self, resp: &Self::Response) -> bool {
      matches!(resp, Reply::WebSocket(_))
   }
}

/// Dials with the account's cleanest turn-state in place of the caller's, and
/// falls back to the caller's once the backend refuses it, so a stale pin
/// costs a handshake rather than a cooled account.
async fn connect_pinned(
   client: &CodexClient,
   slot: &Slot,
   token: &str,
   session: &str,
   model: &str,
   headers: &HeaderMap,
) -> Result<Connection, SendError> {
   let pinned = match TurnState::from_headers(headers) {
      Some(presented) if client.pins_turn_state() => slot.preferred_turn_state(&presented).await,
      _ => None,
   };
   let Some(pinned) = pinned else {
      return dial(client, slot, token, session, model, headers).await;
   };
   let Ok(value) = HeaderValue::from_str(&pinned.token) else {
      return dial(client, slot, token, session, model, headers).await;
   };
   let mut swapped = headers.clone();
   swapped.insert(turn_state::HEADER, value);
   match dial(client, slot, token, session, model, &swapped).await {
      Ok(connection) => Ok(connection),
      Err(err) => {
         tracing::warn!(
            account = %slot.display,
            blocks = pinned.blocks,
            error = %err,
            "backend refused the pinned codex turn-state, dialling with the caller's"
         );
         slot.refuse_turn_state().await;
         dial(client, slot, token, session, model, headers).await
      },
   }
}

/// One handshake, boxed, so the relay loop's future does not carry it inline.
async fn dial(
   client: &CodexClient,
   slot: &Slot,
   token: &str,
   session: &str,
   model: &str,
   headers: &HeaderMap,
) -> Result<Connection, SendError> {
   Box::pin(client.connect_websocket(token, &slot.provider_account_id, session, model, headers))
      .await
}

impl Pool<CodexClient> {
   pub async fn websocket_failed(&self, account_id: Option<i64>, why: &str) {
      if let Some(slot) = self.slot(account_id).await {
         self.slots.cool_failure(&slot, why).await;
      }
   }

   /// The 60s refusal cooldown walks back into the same wall, and `ranked`
   /// keeps the account first off a quota figure the backend stopped honouring.
   pub async fn websocket_exhausted(&self, account_id: Option<i64>) {
      let Some(slot) = self.slot(account_id).await else {
         return;
      };
      let now = unix_now();
      let secs = self
         .slots
         .limit_windows(&slot, None)
         .await
         .iter()
         .filter_map(|window| window.resets_at)
         .filter(|reset| *reset > now)
         .min()
         .map_or(EXHAUSTED_COOLDOWN, |reset| reset - now)
         .clamp(EXHAUSTED_COOLDOWN, CodexClient::RATE_LIMIT.max);
      self.slots.cool(&slot, secs, "usage limit reached").await;
   }

   pub async fn websocket_completed(&self, account_id: Option<i64>) {
      if let Some(slot) = self.slot(account_id).await {
         self.slots.mark_ok(&slot).await;
      }
   }

   /// Keeps the cleanest turn-state token an account's handshakes have returned.
   pub async fn note_turn_state(&self, account_id: Option<i64>, observed: TurnState) {
      let Some(slot) = self.slot(account_id).await else {
         return;
      };
      self.slots.note_turn_state(&slot, observed).await;
   }

   pub async fn rewrite_rate_limits(
      &self,
      account_id: Option<i64>,
      user: &str,
      pinned_account: Option<i64>,
      event: &mut Value,
   ) {
      let reading = RateLimitEvent::deserialize(&*event).unwrap_or_default();
      let limit = reading
         .metered_limit_name
         .as_deref()
         .or(reading.limit_name.as_deref())
         .map(str::trim)
         .filter(|name| !name.is_empty())
         .unwrap_or("codex")
         .to_ascii_lowercase()
         .replace('-', "_");
      let named_limit = (limit != "codex").then_some(limit.as_str());
      let windows = [reading.rate_limits.primary, reading.rate_limits.secondary]
         .into_iter()
         .filter_map(|window| {
            let window = window?;
            let minutes = window.window_minutes?;
            if minutes <= 0 || minutes.checked_mul(60).is_none() {
               return None;
            }
            let percent = window.used_percent?;
            (percent.is_finite() && percent >= 0.0_f64).then(|| UsageWindow {
               name: window_name(minutes),
               utilization: percent / 100.0,
               resets_at: window.reset_at,
            })
         })
         .collect();
      if let Some(slot) = self.slot(account_id).await {
         self
            .slots
            .note_limit_windows(&slot, named_limit, windows)
            .await;
      }
      let mut pooled = self.pool_windows(user, pinned_account, named_limit).await;
      pooled.sort_by_key(|window| window_seconds(&window.name).unwrap_or(i64::MAX));
      let mut limits = RateLimitsOut::default();
      for (tier, window) in [&mut limits.primary, &mut limits.secondary]
         .into_iter()
         .zip(pooled)
      {
         let Some(seconds) = window_seconds(&window.name) else {
            continue;
         };
         *tier = Some(WindowOut {
            used_percent: window.utilization * 100.0_f64,
            window_minutes: seconds / 60,
            reset_at: window.resets_at,
         });
      }
      event["rate_limits"] = serde_json::to_value(limits).unwrap_or_default();
      if let Some(event) = event.as_object_mut() {
         event.remove("credits");
         event.remove("plan_type");
      }
   }

   pub async fn post(
      &self,
      route: Route<'_>,
      body: Bytes,
      headers: HeaderMap,
   ) -> Result<Served<reqwest::Response>, PoolError> {
      self
         .dispatch(route, Call::Http { body, headers }, |reply| match reply {
            Reply::Http(response) => Ok(response),
            Reply::WebSocket(_) => Err(PoolError::Upstream("unexpected WebSocket reply".into())),
         })
         .await
   }

   pub async fn search(
      &self,
      route: Route<'_>,
      body: Bytes,
      headers: HeaderMap,
   ) -> Result<Served<reqwest::Response>, PoolError> {
      self
         .dispatch(route, Call::Search { body, headers }, |reply| match reply {
            Reply::Http(response) => Ok(response),
            Reply::WebSocket(_) => Err(PoolError::Upstream("unexpected WebSocket reply".into())),
         })
         .await
   }

   pub async fn backend(
      &self,
      route: Route<'_>,
      path: String,
      body: Option<Bytes>,
   ) -> Result<Served<reqwest::Response>, PoolError> {
      self
         .dispatch(route, Call::Backend { path, body }, |reply| match reply {
            Reply::Http(response) => Ok(response),
            Reply::WebSocket(_) => Err(PoolError::Upstream("unexpected WebSocket reply".into())),
         })
         .await
   }

   pub async fn websocket(
      &self,
      route: Route<'_>,
      headers: HeaderMap,
   ) -> Result<Served<Connection>, PoolError> {
      self
         .dispatch(route, Call::WebSocket(headers), |reply| match reply {
            Reply::WebSocket(connection) => Ok(*connection),
            Reply::Http(_) => Err(PoolError::Upstream("unexpected HTTP reply".into())),
         })
         .await
   }

   async fn dispatch<T>(
      &self,
      route: Route<'_>,
      call: Call,
      unwrap: fn(Reply) -> Result<T, PoolError>,
   ) -> Result<Served<T>, PoolError> {
      if route.explicit_tier().is_some() {
         self.catalogs(route.user, route.pinned_account).await?;
      }
      let served = self.execute(route, call).await?;
      Ok(Served {
         account_id: served.account_id,
         response: unwrap(served.response)?,
         attempts: served.attempts,
      })
   }

   /// Reads quota for every account from the usage endpoint, so idle
   /// accounts report current figures instead of whatever they last saw on
   /// a served response.
   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
         if self.slots.is_disabled(&slot).await {
            continue;
         }
         if let Err(error) = self.account_catalog(&slot).await {
            tracing::debug!(account = %slot.display, %error, "reading codex catalog failed");
         }
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.usage(&token, &slot.provider_account_id).await {
            Ok(usage) => {
               let windows = usage
                  .rate_limit
                  .windows()
                  .filter(|window| window.limit_window_seconds > 0)
                  .map(|window| UsageWindow {
                     name: window_name(window.limit_window_seconds / 60),
                     utilization: window.used_percent / 100.0,
                     resets_at: window.reset_at,
                  })
                  .collect::<Vec<_>>();
               if windows.is_empty() {
                  continue;
               }
               let healthy = !usage.rate_limit.limit_reached
                  && windows.iter().all(|window| window.utilization < 1.0_f64);
               let now = unix_now();
               self
                  .slots
                  .clear_cooldown_if(&slot, |until| healthy && until - now > EXHAUSTED_COOLDOWN)
                  .await;
               self
                  .slots
                  .note_usage(
                     &slot,
                     AccountUsage {
                        windows,
                        locked: usage.rate_limit.limit_reached,
                        ..AccountUsage::default()
                     },
                  )
                  .await;
            },
            Err(err) => tracing::debug!("usage for {}: {err}", slot.display),
         }
      }
   }

   /// Accounts worth asking for the models listing, best first. Trusted
   /// first, since gated models are absent from an untrusted account's
   /// catalog. Cooldowns are ignored, a listing spends no quota and a fleet
   /// that is entirely cooling after a restart must still serve one. A
   /// disabled account is not: it is idle, so `ranked` bands it on no quota
   /// at all and sorts it ahead of the working fleet.
   async fn listing_slots(&self, user: &str, pinned_account: Option<i64>) -> Vec<Arc<Slot>> {
      let ranked = self
         .ranked(Route {
            session_key: "",
            model: "",
            service_tier: None,
            user,
            pinned_account,
            prefer_trusted: true,
            reserved_only: false,
         })
         .await;
      let mut usable = Vec::with_capacity(ranked.len());
      for slot in ranked {
         if !self.slots.is_disabled(&slot).await {
            usable.push(slot);
         }
      }
      usable
   }

   pub async fn list_models(&self) -> Result<Vec<ModelInfo>, PoolError> {
      Ok(self.catalog("", None).await?.models)
   }

   pub async fn catalog(
      &self,
      user: &str,
      pinned_account: Option<i64>,
   ) -> Result<ModelsResponse, PoolError> {
      let catalogs = self.catalogs(user, pinned_account).await?;
      let mut entries = catalogs.into_iter();
      let mut combined = entries
         .next()
         .expect("catalogs returns at least one catalog")
         .as_ref()
         .clone();
      for entry in entries {
         combined.merge(&entry);
      }
      combined.add_service_tier(
         "gpt-5.6-sol",
         ServiceTier {
            id: "ultrafast".into(),
            name: "Ultrafast".into(),
            description: "The fastest available responses for latency-sensitive work.".into(),
            rest: BTreeMap::default(),
         },
      );
      Ok(combined)
   }

   async fn catalogs(
      &self,
      user: &str,
      pinned_account: Option<i64>,
   ) -> Result<Vec<Arc<ModelsResponse>>, PoolError> {
      let slots = self.listing_slots(user, pinned_account).await;
      let mut requests = stream::iter(slots)
         .map(|slot| async move {
            let result = self.account_catalog(&slot).await;
            if self.slots.is_disabled(&slot).await {
               return Err(PoolError::NoAccounts(Provider::OpenAi));
            }
            match result {
               Ok(catalog) => Ok(catalog),
               Err(error) => {
                  tracing::warn!(account = %slot.display, %error, "reading codex catalog failed");
                  self.slots.catalog(&slot).await.ok_or(error)
               },
            }
         })
         .buffered(8);
      let mut catalogs = Vec::new();
      let mut last = None;
      while let Some(result) = requests.next().await {
         match result {
            Ok(catalog) => catalogs.push(catalog),
            Err(error) => last = Some(error),
         }
      }
      if catalogs.is_empty() {
         return Err(last.unwrap_or(PoolError::NoAccounts(Provider::OpenAi)));
      }
      Ok(catalogs)
   }

   async fn account_catalog(&self, slot: &Slot) -> Result<Arc<ModelsResponse>, PoolError> {
      let _refresh = slot.catalog_refresh.lock().await;
      if !self.slots.catalog_older_than(slot, CATALOG_TTL).await
         && let Some(cached) = self.slots.catalog(slot).await
      {
         return Ok(cached);
      }
      let access = self
         .slots
         .fresh_token(slot, false)
         .await
         .map_err(|()| PoolError::Upstream("refreshing catalog credentials failed".into()))?;
      let catalog = Arc::new(
         self
            .backend
            .catalog(&access, &slot.provider_account_id)
            .await?,
      );
      self.slots.note_catalog(slot, Arc::clone(&catalog)).await;
      Ok(catalog)
   }

   pub async fn websocket_serves(
      &self,
      account_id: Option<i64>,
      route: Route<'_>,
   ) -> Result<bool, PoolError> {
      let Some(tier) = route.explicit_tier() else {
         return Ok(true);
      };
      self.catalogs(route.user, route.pinned_account).await?;
      let Some(slot) = self.slot(account_id).await else {
         return Ok(false);
      };
      Ok(slot.serves(route.user)
         && !self.slots.is_disabled(&slot).await
         && self.slots.serves_tier(&slot, route.model, tier).await)
   }
}

/// The codex backend reports quota on every successful response rather than
/// from a queryable endpoint, so consumption is only known once an account
/// has served traffic.
fn usage_from_headers(headers: &HeaderMap) -> Option<AccountUsage> {
   let get = |name: &str| headers.get(name)?.to_str().ok()?.parse::<i64>().ok();
   let mut windows = Vec::new();
   for tier in ["primary", "secondary"] {
      let minutes = get(&format!("x-codex-{tier}-window-minutes")).unwrap_or(0);
      let Some(percent) = get(&format!("x-codex-{tier}-used-percent")) else {
         continue;
      };
      if minutes <= 0 {
         continue;
      }
      windows.push(UsageWindow {
         name: window_name(minutes),
         utilization: percent as f64 / 100.0,
         resets_at: get(&format!("x-codex-{tier}-reset-at")),
      });
   }
   (!windows.is_empty()).then_some(AccountUsage {
      windows,
      ..AccountUsage::default()
   })
}

/// Codex sends one session id per conversation, not per request. Deriving it
/// from the same key means upstream sees a continuing thread rather than a
/// stranger every turn.
fn session_uuid(session_key: &str) -> String {
   if session_key.is_empty() {
      return uuid::Uuid::new_v4().to_string();
   }
   let digest = hmac_sha256::Hash::hash(session_key.as_bytes());
   let mut bytes = [0_u8; 16];
   bytes.copy_from_slice(&digest[..16]);
   uuid::Builder::from_random_bytes(bytes)
      .into_uuid()
      .to_string()
}

fn window_name(minutes: i64) -> String {
   if minutes % 1440 == 0 {
      format!("{}d", minutes / 1440)
   } else if minutes % 60 == 0 {
      format!("{}h", minutes / 60)
   } else {
      format!("{minutes}m")
   }
}
