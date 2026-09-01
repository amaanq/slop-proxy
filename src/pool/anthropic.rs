use std::sync::Arc;

use serde_json::Value;

use super::{AccountUsage, ModelWindow, PoolError, Slot, Slots, UsageWindow};
use crate::anthropic::client::{AnthropicClient, RelayHeaders};
use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Anthropic Max accounts, owning the relay client.
pub struct AnthropicPool {
    slots: Slots,
    client: AnthropicClient,
    soft_limit: f64,
}

impl AnthropicPool {
    pub async fn load(db: Db, client: AnthropicClient) -> eyre::Result<Self> {
        Ok(Self {
            soft_limit: client.soft_utilization_limit(),
            slots: Slots::load(db, Provider::Anthropic).await?,
            client,
        })
    }

    pub async fn len(&self) -> usize {
        self.slots.len().await
    }

    pub async fn is_empty(&self) -> bool {
        self.slots.is_empty().await
    }

    pub async fn reload(&self) -> eyre::Result<()> {
        self.slots.reload().await
    }

    pub async fn snapshot(&self) -> Vec<super::AccountSnapshot> {
        self.slots.snapshot().await
    }

    /// Reads each account's rolling-window consumption from the provider.
    /// This needs no inference request, so idle accounts report real numbers
    /// and a locked account is known before it rejects traffic.
    pub async fn poll_usage(&self) {
        for slot in self.slots.list().await {
            let Ok(token) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.client.usage(&token).await {
                Ok(usage) => {
                    let windows = usage
                        .windows()
                        .map(|(name, w)| UsageWindow {
                            name: name.to_string(),
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
                                        window: window.to_string(),
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

    /// Rendezvous-hashed order: a session sticks to its highest-scoring
    /// account so upstream prompt caches keep hitting.
    async fn ranked(&self, session_key: &str, prefer_trusted: bool) -> Vec<Arc<Slot>> {
        let mut scored = Vec::new();
        for slot in self.slots.list().await {
            let strained = self.slots.is_strained(&slot, self.soft_limit).await;
            let score = super::rendezvous_score(session_key, slot.id);
            scored.push((score, strained, slot));
        }
        // Strain outranks the trust preference: an account with capacity left
        // beats a preferred one that is nearly spent, since the preference is
        // about reserving scarce capacity rather than guaranteeing it.
        scored.sort_by_key(|(score, strained, slot)| {
            (
                *strained,
                std::cmp::Reverse(slot.trusted == prefer_trusted),
                std::cmp::Reverse(*score),
            )
        });
        scored.into_iter().map(|(_, _, s)| s).collect()
    }

    pub async fn execute(
        &self,
        path: &str,
        body: &Value,
        hdrs: &RelayHeaders,
        session_key: &str,
        prefer_trusted: bool,
    ) -> Result<(i64, reqwest::Response), PoolError> {
        let ranked = self.ranked(session_key, prefer_trusted).await;
        if ranked.is_empty() {
            return Err(PoolError::NoAccounts(Provider::Anthropic));
        }
        let mut last_err = Option::<SendError>::None;
        let mut attempts = 0;

        for slot in ranked {
            if attempts >= 3 {
                break;
            }
            if !self.slots.try_claim(&slot).await {
                continue;
            }
            attempts += 1;
            let Ok(token) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };

            match self.client.send(&token, path, body, hdrs).await {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    return Ok((slot.id, resp));
                }
                Err(SendError::Auth(body_text)) => {
                    tracing::warn!("account {} got 401, forcing refresh", slot.display);
                    if let Ok(token) = self.slots.fresh_token(&slot, true).await {
                        match self.client.send(&token, path, body, hdrs).await {
                            Ok(resp) => {
                                self.slots.mark_ok(&slot).await;
                                return Ok((slot.id, resp));
                            }
                            Err(e) => {
                                self.slots.cool(&slot, 60, "post-refresh failure").await;
                                last_err = Some(e);
                            }
                        }
                    } else {
                        last_err = Some(SendError::Auth(body_text));
                    }
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    self.slots
                        .cool_rate_limited(&slot, retry_after, 7 * 24 * 3600)
                        .await;
                    last_err = Some(SendError::RateLimited { retry_after, body });
                }
                Err(e) => {
                    self.slots.cool_failure(&slot).await;
                    last_err = Some(e);
                }
            }
        }

        match last_err {
            Some(SendError::RateLimited { .. }) | None => Err(PoolError::AllCoolingDown {
                retry_after: self.slots.min_cooldown().await.max(30),
            }),
            Some(e) => Err(PoolError::Upstream(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::config::AnthropicConfig;

    async fn test_pool(ids: &[(i64, bool)]) -> AnthropicPool {
        let db_path = std::env::temp_dir().join(format!("slop-rdv-{}.db", uuid::Uuid::new_v4()));
        let db = Db::open(&db_path).await.unwrap();
        AnthropicPool {
            slots: super::super::test_slots(db, Provider::Anthropic, ids),
            client: AnthropicClient::new(AnthropicConfig::default()),
            soft_limit: 0.9,
        }
    }

    #[tokio::test]
    async fn rendezvous_is_sticky_and_spreads() {
        let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]).await;

        let first = pool.ranked("session-a", false).await[0].id;
        for _ in 0..10 {
            assert_eq!(pool.ranked("session-a", false).await[0].id, first);
        }

        let mut winners = HashSet::new();
        for i in 0..64 {
            winners.insert(pool.ranked(&format!("session-{i}"), false).await[0].id);
        }
        assert!(winners.len() > 1, "all sessions landed on one account");

        let order = pool.ranked("session-a", false).await;
        assert_eq!(order.len(), 4);
        let ids = order.iter().map(|s| s.id).collect::<HashSet<i64>>();
        assert_eq!(ids.len(), 4);
    }

    #[tokio::test]
    async fn a_spent_window_sinks_below_its_peers() {
        let pool = test_pool(&[(1, false), (2, false), (3, false), (4, false)]).await;
        let key = "session-strain";
        let head = pool.ranked(key, false).await[0].clone();

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
        let after = pool.ranked(key, false).await;
        assert_ne!(after[0].id, head.id, "spent account still ranked first");
        assert_eq!(after[3].id, head.id, "spent account should sort last");

        // A locked account sinks the same way regardless of its fraction.
        let fresh = after[0].clone();
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
        let after = pool.ranked(key, false).await;
        assert_ne!(after[0].id, fresh.id, "locked account still ranked first");
    }

    #[tokio::test]
    async fn capacity_outranks_the_trust_preference() {
        let pool = test_pool(&[(1, true), (2, true), (3, false), (4, false)]).await;
        // Both trusted accounts are nearly spent while the untrusted ones are idle.
        for slot in pool.slots.list().await {
            if slot.trusted {
                pool.slots
                    .note_usage(
                        &slot,
                        AccountUsage {
                            model_windows: Vec::new(),
                            windows: vec![UsageWindow {
                                name: "5h".into(),
                                utilization: 0.95,
                                resets_at: None,
                            }],
                            locked: false,
                            observed_at: 0,
                        },
                    )
                    .await;
            }
        }
        let order = pool.ranked("session-a", true).await;
        assert!(
            !order[0].trusted,
            "a spent trusted account was preferred over an idle untrusted one"
        );
        assert!(
            order[2].trusted && order[3].trusted,
            "spent accounts should sort last"
        );
    }

    #[tokio::test]
    async fn trusted_preference_partitions_both_ways() {
        let pool = test_pool(&[(1, false), (2, true), (3, false), (4, true)]).await;

        for i in 0..16 {
            let order = pool.ranked(&format!("session-{i}"), true).await;
            let heads = [order[0].id, order[1].id];
            assert!(
                heads.contains(&2) && heads.contains(&4),
                "untrusted ranked first"
            );

            let plain = pool.ranked(&format!("session-{i}"), false).await;
            let plain_heads = [plain[0].id, plain[1].id];
            assert!(
                plain_heads.contains(&1) && plain_heads.contains(&3),
                "trusted accounts served an ordinary token before untrusted ones"
            );
        }

        let plain = pool.ranked("session-x", false).await;
        let preferred = pool.ranked("session-x", true).await;
        let plain_trusted = plain
            .iter()
            .filter(|s| s.trusted)
            .map(|s| s.id)
            .collect::<Vec<_>>();
        let preferred_trusted = preferred[..2].iter().map(|s| s.id).collect::<Vec<_>>();
        assert_eq!(
            plain_trusted, preferred_trusted,
            "stickiness broken within trusted group"
        );
    }
}
