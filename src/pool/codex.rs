use axum::body::Bytes;
use reqwest::header::HeaderMap;

use super::{
   AccountUsage, AuthPolicy, Backend, Cooldown, Pool, PoolError, Route, Slot, UsageWindow,
};
use crate::codex::client::CodexClient;
use crate::codex::models::ModelInfo;
use crate::codex::types::ErrorEnvelope;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over codex accounts, owning the backend client.
pub type CodexPool = Pool<CodexClient>;

impl Backend for CodexClient {
   const PROVIDER: Provider = Provider::OpenAi;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 6 * 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
   const TIERED: bool = true;
   type Request = Bytes;
   type Response = reqwest::Response;

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
      Self::post(
         self,
         token,
         &slot.provider_account_id,
         req,
         &session_uuid(route.session_key),
      )
      .await
   }

   fn retryable_bad_request(&self, body: &str) -> bool {
      // A preview model can be enabled per account, so "not supported"
      // describes this key rather than the request, and the next
      // account may well serve it.
      body.contains("is not supported")
   }

   fn usage_from(&self, resp: &Self::Response) -> Option<AccountUsage> {
      usage_from_headers(resp.headers())
   }
}

impl Pool<CodexClient> {
   pub const fn client(&self) -> &CodexClient {
      self.backend()
   }

   /// Reads quota for every account from the usage endpoint, so idle
   /// accounts report current figures instead of whatever they last saw on
   /// a served response.
   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
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

   /// Fresh (`access_token`, `account_id`) for the models listing. Trusted
   /// first, since gated models are absent from an untrusted account's
   /// catalog.
   pub async fn any_active_credentials(&self) -> Option<(String, String)> {
      for slot in self
         .ranked(Route {
            session_key: "",
            model: "",
            user: "",
            pinned_account: None,
            prefer_trusted: true,
         })
         .await
      {
         if !self.slots.try_claim(&slot).await {
            continue;
         }
         let Ok(access) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         return Some((access, slot.provider_account_id.clone()));
      }
      None
   }

   pub async fn list_models(&self) -> Result<Vec<ModelInfo>, PoolError> {
      let (access, account_id) = self
         .any_active_credentials()
         .await
         .ok_or(PoolError::NoAccounts(Provider::OpenAi))?;
      Ok(self.backend.list_models(&access, &account_id).await?)
   }

   /// The catalog body untouched, for relaying to a codex client verbatim.
   pub async fn models_raw(&self) -> Result<String, PoolError> {
      let (access, account_id) = self
         .any_active_credentials()
         .await
         .ok_or(PoolError::NoAccounts(Provider::OpenAi))?;
      let (status, body) = self.backend.models_raw(&access, &account_id).await?;
      if !status.is_success() {
         return Err(PoolError::Upstream(format!(
            "{status}: {}",
            body.chars().take(400).collect::<String>()
         )));
      }
      Ok(body)
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
