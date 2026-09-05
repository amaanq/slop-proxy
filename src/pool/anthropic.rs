use axum::body::Bytes;

use super::{
   AccountUsage, AuthPolicy, Backend, Cooldown, ModelWindow, Pool, PoolError, Route, Slot,
   UsageWindow,
};
use crate::anthropic::client::{AnthropicClient, RelayHeaders};
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;

/// Session-sticky pool over Anthropic Max accounts, owning the relay client.
pub type AnthropicPool = Pool<AnthropicClient>;

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
   pub hdrs: RelayHeaders,
}

impl Backend for AnthropicClient {
   const PROVIDER: Provider = Provider::Anthropic;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 7 * 24 * 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
   type Request = Relay;
   type Response = reqwest::Response;

   fn reason(body: String) -> String {
      ChatError::reason(body)
   }

   fn soft_limit(&self) -> f64 {
      self.soft_utilization_limit()
   }

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      _route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      Self::post(self, token, req.path, &req.body, &req.hdrs).await
   }
}

impl Pool<AnthropicClient> {
   /// The catalog body untouched, for relaying to an Anthropic client.
   pub async fn models_raw(&self) -> Result<String, PoolError> {
      for slot in self.slots.list().await {
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.models_raw(&token).await {
            Ok((status, body)) if status.is_success() => return Ok(body),
            Ok((status, _)) => tracing::debug!("models for {}: {status}", slot.display),
            Err(err) => tracing::debug!("models for {}: {err}", slot.display),
         }
      }
      Err(PoolError::NoAccounts(Provider::Anthropic))
   }

   /// Reads each account's rolling-window consumption from the provider.
   /// This needs no inference request, so idle accounts report real numbers
   /// and a locked account is known before it rejects traffic.
   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.usage(&token).await {
            Ok(usage) => {
               match self
                  .slots
                  .clear_cooldown_if(&slot, |until| usage.cooldown_is_obsolete(until))
                  .await
               {
                  Ok(true) => tracing::info!(
                     account = %slot.display,
                     "cleared cooldown for an inactive Anthropic quota window"
                  ),
                  Ok(false) => {},
                  Err(err) => tracing::warn!(
                     account = %slot.display,
                     error = %err,
                     "failed to clear obsolete Anthropic cooldown"
                  ),
               }
               let windows = usage
                  .windows()
                  .map(|(name, window)| UsageWindow {
                     name: name.to_owned(),
                     // The endpoint reports percentages while the
                     // response headers report fractions.
                     utilization: window.utilization / 100.0,
                     resets_at: window.resets_at_unix(),
                  })
                  .collect();
               self
                  .slots
                  .note_usage(
                     &slot,
                     AccountUsage {
                        windows,
                        model_windows: usage
                           .model_windows()
                           .map(|(model, window, limit)| ModelWindow {
                              model,
                              window: window.to_owned(),
                              utilization: limit.percent / 100.0,
                              is_active: limit.is_active,
                              resets_at: limit.resets_at_unix(),
                           })
                           .collect(),
                        locked: usage.locked(),
                        observed_at: 0,
                     },
                  )
                  .await;
            },
            Err(err) => tracing::debug!("usage for {}: {err}", slot.display),
         }
      }
   }
}

#[cfg(test)]
mod tests {
   use axum::routing::get;
   use std::collections::HashSet;
   use std::env;
   use std::sync::Arc;
   use tokio::net::TcpListener;

   use super::*;
   use crate::clock;
   use crate::config::AnthropicConfig;
   use crate::db::Db;
   use crate::db::accounts::{AccountStatus, NewAccount};
   use crate::oauth::TokenSet;
   use crate::provider::AuthMode;
   use uuid::Uuid;

   fn test_pool(ids: &[(i64, bool)]) -> AnthropicPool {
      let db_path = env::temp_dir().join(format!("slop-rdv-{}.db", Uuid::new_v4()));
      let db = Db::open(&db_path).unwrap();
      AnthropicPool {
         slots: super::super::test_slots(db, Provider::Anthropic, ids),
         backend: AnthropicClient::new(AnthropicConfig::default()),
      }
   }

   async fn ranked(pool: &AnthropicPool, session_key: &str) -> Vec<Arc<Slot>> {
      pool
         .ranked(Route {
            session_key,
            model: "",
            prefer_trusted: false,
         })
         .await
   }

   #[tokio::test]
   async fn rendezvous_is_sticky_and_spreads() {
      let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]);

      let first = ranked(&pool, "session-a").await[0].id;
      for _ in 0_usize..10_usize {
         assert_eq!(ranked(&pool, "session-a").await[0].id, first);
      }

      let mut winners = HashSet::new();
      for idx in 0_usize..64_usize {
         winners.insert(ranked(&pool, &format!("session-{idx}")).await[0].id);
      }
      assert!(winners.len() > 1, "all sessions landed on one account");
   }

   #[tokio::test]
   async fn quota_that_resets_soonest_is_spent_first() {
      let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]);
      let now = clock::unix_now();
      let hours = |hrs: f64| now + (hrs * 3600.0_f64) as i64;
      for (id, used, resets_in) in [
         (1_i64, 0.64_f64, 47.2_f64),
         (2_i64, 0.90_f64, 37.2_f64),
         (3_i64, 0.04_f64, 131.0_f64),
         (4_i64, 0.69_f64, 8.2_f64),
      ] {
         let slot = pool
            .slots
            .list()
            .await
            .into_iter()
            .find(|slot| slot.id == id)
            .unwrap();
         pool
            .slots
            .note_usage(
               &slot,
               AccountUsage {
                  windows: vec![UsageWindow {
                     name: "7d".into(),
                     utilization: used,
                     resets_at: Some(hours(resets_in)),
                  }],
                  model_windows: Vec::new(),
                  locked: false,
                  observed_at: 0,
               },
            )
            .await;
      }
      let order = ranked(&pool, "session-a").await;
      assert_eq!(order[0].id, 4, "the account resetting in 8h should lead");
      assert_eq!(
         order[3].id, 2,
         "the account that runs out before it resets should sort last"
      );
   }

   #[tokio::test]
   async fn a_spent_window_sinks_below_its_peers() {
      let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]);
      let key = "session-strain";
      let head = Arc::clone(&ranked(&pool, key).await[0]);

      pool
         .slots
         .note_usage(
            &head,
            AccountUsage {
               model_windows: Vec::new(),
               windows: vec![UsageWindow {
                  name: "5h".into(),
                  utilization: 0.97,
                  resets_at: None,
               }],
               locked: false,
               observed_at: 0,
            },
         )
         .await;
      let after = ranked(&pool, key).await;
      assert_eq!(after[3].id, head.id, "spent account should sort last");

      // A locked account sinks the same way regardless of its fraction.
      let fresh = Arc::clone(&after[0]);
      pool
         .slots
         .note_usage(
            &fresh,
            AccountUsage {
               model_windows: Vec::new(),
               windows: vec![UsageWindow {
                  name: "5h".into(),
                  utilization: 0.01,
                  resets_at: None,
               }],
               locked: true,
               observed_at: 0,
            },
         )
         .await;
      let reranked = ranked(&pool, key).await;
      assert_ne!(
         reranked[0].id, fresh.id,
         "locked account still ranked first"
      );
   }

   #[tokio::test]
   async fn polling_preserves_model_quota_activity_and_resets() {
      let reset = 1_800_000_000_i64;
      let usage = serde_json::json!({
         "five_hour": {"utilization": 26.0_f64},
         "seven_day": {"utilization": 99.0_f64},
         "limits": [
            {"group": "weekly", "is_active": false, "percent": 99_i32},
            {"group": "weekly", "is_active": true, "percent": 25_i32,
             "resets_at": jiff::Timestamp::from_second(reset).unwrap().to_string(),
             "scope": {"model": {"display_name": "Fable"}}},
            {"group": "weekly", "is_active": false, "percent": 100_i32,
             "resets_at": "invalid", "scope": {"model": {"display_name": "Dormant"}}},
            {"group": "weekly", "percent": 12_i32,
             "scope": {"model": {"display_name": "Unknown"}}}
         ]
      });
      let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
      let addr = listener.local_addr().unwrap();
      let app = axum::Router::new().route(
         "/api/oauth/usage",
         get(move || {
            let usage = usage.clone();
            async move { axum::Json(usage) }
         }),
      );
      let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
      let db = Db::open(&env::temp_dir().join(format!("slop-model-quota-{}.db", Uuid::new_v4())))
         .unwrap();
      db.upsert_account(NewAccount {
         provider: Provider::Anthropic,
         id: "quota-test",
         email: None,
         label: None,
         plan: None,
         tokens: &TokenSet {
            access_token: "test-token".into(),
            refresh_token: "test-refresh".into(),
            id_token: None,
            expires_at: Some(clock::unix_now() + 3600),
         },
         auth_mode: AuthMode::OAuth,
      })
      .await
      .unwrap();
      let pool = AnthropicPool::load(
         db,
         AnthropicClient::new(AnthropicConfig {
            base_url: format!("http://{addr}"),
            ..AnthropicConfig::default()
         }),
      )
      .await
      .unwrap();
      pool.poll_usage().await;
      server.abort();

      let snapshots = pool.slots.snapshot().await;
      let reported = snapshots[0].usage.as_ref().unwrap();
      assert_eq!(reported.windows.len(), 1);
      assert_eq!(reported.windows[0].name, "5h");
      assert!((reported.peak() - 0.26_f64).abs() < f64::EPSILON);
      let models: Vec<_> = reported
         .model_windows
         .iter()
         .map(|window| {
            (
               window.model.as_str(),
               window.window.as_str(),
               window.utilization,
               window.is_active,
               window.resets_at,
            )
         })
         .collect();
      assert_eq!(
         models,
         vec![
            ("fable", "7d", 0.25_f64, Some(true), Some(reset)),
            ("dormant", "7d", 1.0_f64, Some(false), None),
            ("unknown", "7d", 0.12_f64, None, None),
         ]
      );
   }

   #[tokio::test]
   async fn polling_clears_only_the_obsolete_window_and_persists_it() {
      let now = clock::unix_now();
      let weekly_reset = now + 86400;
      let session_reset = now + 3600;
      let usage = serde_json::json!({
         "five_hour": {"utilization": 26.0_f64},
         "limits": [
            {"group": "session", "is_active": true, "percent": 26_i32},
            {"group": "weekly", "is_active": false, "percent": 6_i32,
             "resets_at": jiff::Timestamp::from_second(weekly_reset).unwrap().to_string()}
         ]
      });
      let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
      let addr = listener.local_addr().unwrap();
      let app = axum::Router::new().route(
         "/api/oauth/usage",
         get(move || {
            let usage = usage.clone();
            async move { axum::Json(usage) }
         }),
      );
      let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
      let db =
         Db::open(&env::temp_dir().join(format!("slop-inactive-quota-{}.db", Uuid::new_v4())))
            .unwrap();
      for (label, status, until) in [
         ("obsolete", AccountStatus::Cooldown, weekly_reset - 1),
         ("session", AccountStatus::Cooldown, session_reset),
         ("disabled", AccountStatus::Disabled, weekly_reset - 1),
      ] {
         let id = db
            .upsert_account(NewAccount {
               provider: Provider::Anthropic,
               id: label,
               email: None,
               label: Some(label),
               plan: None,
               tokens: &TokenSet {
                  access_token: "test-token".into(),
                  refresh_token: "test-refresh".into(),
                  id_token: None,
                  expires_at: Some(now + 3600),
               },
               auth_mode: AuthMode::OAuth,
            })
            .await
            .unwrap();
         db.set_account_status(id, status, Some(until), None)
            .await
            .unwrap();
      }
      for label in ["session", "disabled"] {
         let account = db.find_account(label).await.unwrap().unwrap();
         assert!(
            !db.clear_account_cooldown(account.id, weekly_reset - 1)
               .await
               .unwrap()
         );
      }
      let config = AnthropicConfig {
         base_url: format!("http://{addr}"),
         ..AnthropicConfig::default()
      };
      let pool = AnthropicPool::load(db.clone(), AnthropicClient::new(config.clone()))
         .await
         .unwrap();
      pool.poll_usage().await;
      server.abort();

      let reloaded = AnthropicPool::load(db.clone(), AnthropicClient::new(config))
         .await
         .unwrap();
      for checked in [&pool, &reloaded] {
         for slot in checked.slots.list().await {
            assert_eq!(
               checked.slots.try_claim(&slot).await,
               slot.display == "obsolete"
            );
         }
      }
      for (label, status, until) in [
         ("obsolete", AccountStatus::Active, None),
         ("session", AccountStatus::Cooldown, Some(session_reset)),
         ("disabled", AccountStatus::Disabled, Some(weekly_reset - 1)),
      ] {
         let account = db.find_account(label).await.unwrap().unwrap();
         assert_eq!(account.status, status);
         assert_eq!(account.cooldown_until, until);
      }
   }
}
