use axum::body::Bytes;

use super::{
    AccountUsage, AuthPolicy, Backend, Cooldown, ModelWindow, Pool, PoolError, Route, Slot,
    UsageWindow,
};
use crate::anthropic::client::{AnthropicClient, RelayHeaders};
use crate::provider::Provider;
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
        crate::translate::chat::ChatError::reason(body)
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
    /// Averaged across accounts, so a caller's figures do not jump when
    /// routing moves it.
    pub async fn pool_windows(&self) -> Vec<UsageWindow> {
        let mut by_name: std::collections::BTreeMap<String, (f64, usize, Option<i64>)> =
            std::collections::BTreeMap::default();
        for account in self.slots.snapshot().await {
            let Some(usage) = account.usage else { continue };
            for w in usage.windows {
                let slot = by_name.entry(w.name.clone()).or_insert((0.0, 0, None));
                slot.0 += w.utilization;
                slot.1 += 1;
                slot.2 = match (slot.2, w.resets_at) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
            }
        }
        by_name
            .into_iter()
            .map(|(name, (sum, count, resets_at))| UsageWindow {
                name,
                utilization: sum / count.max(1) as f64,
                resets_at,
            })
            .collect()
    }

    /// The catalog body untouched, for relaying to an Anthropic client.
    pub async fn models_raw(&self) -> Result<String, PoolError> {
        for slot in self.slots.list().await {
            let Ok(token) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.backend.models_raw(&token).await {
                Ok((status, body)) if status.is_success() => return Ok(body),
                Ok((status, _)) => tracing::debug!("models for {}: {status}", slot.display),
                Err(e) => tracing::debug!("models for {}: {e}", slot.display),
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
                    let windows = usage
                        .windows()
                        .map(|(name, w)| UsageWindow {
                            name: name.to_owned(),
                            // The endpoint reports percentages while the
                            // response headers report fractions.
                            utilization: w.utilization / 100.0,
                            resets_at: w.resets_at_unix(),
                        })
                        .collect();
                    self.slots
                        .note_usage(
                            &slot,
                            AccountUsage {
                                windows,
                                model_windows: usage
                                    .model_windows()
                                    .map(|(model, window, utilization)| ModelWindow {
                                        model,
                                        window: window.to_owned(),
                                        utilization,
                                    })
                                    .collect(),
                                locked: usage.locked(),
                                observed_at: 0,
                            },
                        )
                        .await;
                }
                Err(e) => tracing::debug!("usage for {}: {e}", slot.display),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::config::AnthropicConfig;
    use crate::db::Db;

    fn test_pool(ids: &[(i64, bool)]) -> AnthropicPool {
        let db_path = std::env::temp_dir().join(format!("slop-rdv-{}.db", uuid::Uuid::new_v4()));
        let db = Db::open(&db_path).unwrap();
        AnthropicPool {
            slots: super::super::test_slots(db, Provider::Anthropic, ids),
            backend: AnthropicClient::new(AnthropicConfig::default()),
        }
    }

    async fn ranked(pool: &AnthropicPool, session_key: &str) -> Vec<std::sync::Arc<Slot>> {
        pool.ranked(Route {
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
        for _ in 0..10 {
            assert_eq!(ranked(&pool, "session-a").await[0].id, first);
        }

        let mut winners = HashSet::new();
        for i in 0..64 {
            winners.insert(ranked(&pool, &format!("session-{i}")).await[0].id);
        }
        assert!(winners.len() > 1, "all sessions landed on one account");
    }

    #[tokio::test]
    async fn quota_that_resets_soonest_is_spent_first() {
        let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]);
        let now = crate::clock::unix_now();
        let hours = |h: f64| now + (h * 3600.0) as i64;
        for (id, used, resets_in) in [
            (1, 0.64, 47.2),
            (2, 0.90, 37.2),
            (3, 0.04, 131.0),
            (4, 0.69, 8.2),
        ] {
            let slot = pool
                .slots
                .list()
                .await
                .into_iter()
                .find(|s| s.id == id)
                .unwrap();
            pool.slots
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
        let head = std::sync::Arc::clone(&ranked(&pool, key).await[0]);

        pool.slots
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
        let fresh = std::sync::Arc::clone(&after[0]);
        pool.slots
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
        assert_ne!(reranked[0].id, fresh.id, "locked account still ranked first");
    }
}
