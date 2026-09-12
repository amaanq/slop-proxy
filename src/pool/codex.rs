use std::collections::BTreeMap;
use std::sync::Arc;

use axum::body::Bytes;
use futures_util::{StreamExt as _, stream};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

use super::{
   AccountUsage, AuthPolicy, Backend, Cooldown, Pool, PoolError, Route, Slot, UsageWindow,
   window_seconds,
};
use crate::clock::unix_now;
use crate::codex::client::CodexClient;
use crate::codex::models::{ModelInfo, ModelsResponse, ServiceTier};
use crate::codex::types::ErrorEnvelope;
use crate::codex::websocket::Connection;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over codex accounts, owning the backend client.
pub type CodexPool = Pool<CodexClient>;

/// Floor for an exhausted account when the backend names no reset.
const EXHAUSTED_COOLDOWN: i64 = 15 * 60;

/// A catalog only moves when a model ships or an account's access changes.
const CATALOG_TTL: i64 = 300;

#[derive(Clone)]
pub enum Call {
   Http { body: Bytes, headers: HeaderMap },
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
   /// A capacity refusal cools the account for 60s and the same model
   /// refuses again after the wait, so a bound session moves on instead.
   const BOUND_WAIT_SECS: i64 = 0;
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
         Call::WebSocket(ref headers) => self
            .connect_websocket(
               token,
               &slot.provider_account_id,
               &session,
               route.model,
               headers,
            )
            .await
            .map(|connection| Reply::WebSocket(Box::new(connection))),
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

impl Pool<CodexClient> {
   pub async fn websocket_failed(&self, account_id: Option<i64>) {
      if let Some(id) = account_id
         && let Some(slot) = self.slots.by_id(id).await
      {
         self.slots.cool_failure(&slot).await;
      }
   }

   /// The 60s refusal cooldown walks back into the same wall, and `ranked`
   /// keeps the account first off a quota figure the backend stopped honouring.
   pub async fn websocket_exhausted(&self, account_id: Option<i64>) {
      let Some(id) = account_id else { return };
      let Some(slot) = self.slots.by_id(id).await else {
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
      if let Some(id) = account_id
         && let Some(slot) = self.slots.by_id(id).await
      {
         self.slots.mark_ok(&slot).await;
      }
   }

   pub async fn rewrite_rate_limits(
      &self,
      account_id: Option<i64>,
      user: &str,
      pinned_account: Option<i64>,
      event: &mut Value,
   ) {
      let limit = event
         .get("metered_limit_name")
         .and_then(Value::as_str)
         .or_else(|| event.get("limit_name").and_then(Value::as_str))
         .map(str::trim)
         .filter(|name| !name.is_empty())
         .unwrap_or("codex")
         .to_ascii_lowercase()
         .replace('-', "_");
      let named_limit = (limit != "codex").then_some(limit.as_str());
      let windows = ["primary", "secondary"]
         .into_iter()
         .filter_map(|tier| {
            let window = event.get("rate_limits")?.get(tier)?;
            let minutes = window.get("window_minutes")?.as_i64()?;
            if minutes <= 0 || minutes.checked_mul(60).is_none() {
               return None;
            }
            let percent = window.get("used_percent")?.as_f64()?;
            (percent.is_finite() && percent >= 0.0_f64).then(|| UsageWindow {
               name: window_name(minutes),
               utilization: percent / 100.0,
               resets_at: window.get("reset_at").and_then(Value::as_i64),
            })
         })
         .collect();
      if let Some(id) = account_id
         && let Some(slot) = self.slots.by_id(id).await
      {
         self
            .slots
            .note_limit_windows(&slot, named_limit, windows)
            .await;
      }
      let mut pooled = self.pool_windows(user, pinned_account, named_limit).await;
      pooled.sort_by_key(|window| window_seconds(&window.name).unwrap_or(i64::MAX));
      let mut limits = json!({ "primary": null, "secondary": null });
      for (tier, window) in ["primary", "secondary"].into_iter().zip(pooled) {
         let Some(seconds) = window_seconds(&window.name) else {
            continue;
         };
         limits[tier] = json!({
            "used_percent": window.utilization * 100.0_f64,
            "window_minutes": seconds / 60,
            "reset_at": window.resets_at,
         });
      }
      event["rate_limits"] = limits;
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
   ) -> Result<(Option<i64>, reqwest::Response), PoolError> {
      if route.explicit_tier().is_some() {
         self.catalogs(route.user, route.pinned_account).await?;
      }
      let (account, reply) = self.execute(route, Call::Http { body, headers }).await?;
      match reply {
         Reply::Http(response) => Ok((account, response)),
         Reply::WebSocket(_) => Err(PoolError::Upstream("unexpected WebSocket reply".into())),
      }
   }

   pub async fn websocket(
      &self,
      route: Route<'_>,
      headers: HeaderMap,
   ) -> Result<(Option<i64>, Connection), PoolError> {
      if route.explicit_tier().is_some() {
         self.catalogs(route.user, route.pinned_account).await?;
      }
      let (account, reply) = self.execute(route, Call::WebSocket(headers)).await?;
      match reply {
         Reply::WebSocket(connection) => Ok((account, *connection)),
         Reply::Http(_) => Err(PoolError::Upstream("unexpected HTTP reply".into())),
      }
   }

   pub const fn client(&self) -> &CodexClient {
      self.backend()
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
               match self
                  .slots
                  .clear_cooldown_if(&slot, |until| healthy && until - now > EXHAUSTED_COOLDOWN)
                  .await
               {
                  Ok(true) => tracing::info!(
                     account = %slot.display,
                     "cleared a cooldown the account's own quota no longer justifies"
                  ),
                  Ok(false) => {},
                  Err(err) => tracing::warn!(
                     account = %slot.display,
                     error = %err,
                     "failed to clear an obsolete codex cooldown"
                  ),
               }
               self
                  .slots
                  .note_usage(
                     &slot,
                     AccountUsage {
                        windows,
                        model_windows: Vec::new(),
                        locked: usage.rate_limit.limit_reached,
                        observed_at: 0,
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

   pub async fn any_active_credentials(&self) -> Option<(String, String)> {
      for slot in self.listing_slots("", None).await {
         let Ok(access) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         return Some((access, slot.provider_account_id.clone()));
      }
      None
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
      let Some(id) = account_id else {
         return Ok(false);
      };
      let Some(slot) = self.slots.by_id(id).await else {
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
      model_windows: Vec::new(),
      locked: false,
      observed_at: 0,
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
