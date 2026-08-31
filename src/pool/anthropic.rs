use std::sync::Arc;

use serde_json::Value;

use super::{PoolError, Slot, Slots};
use crate::anthropic::client::{AnthropicClient, RelayHeaders};
use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Anthropic Max accounts, owning the relay client.
pub struct AnthropicPool {
    slots: Slots,
    client: AnthropicClient,
}

impl AnthropicPool {
    pub async fn load(db: Db, client: AnthropicClient) -> eyre::Result<Self> {
        Ok(Self {
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

    /// Rendezvous-hashed order: a session sticks to its highest-scoring
    /// account so upstream prompt caches keep hitting, and only moves while
    /// that account is cooling down.
    async fn ranked(&self, session_key: &str, prefer_trusted: bool) -> Vec<Arc<Slot>> {
        let mut scored = self
            .slots
            .list()
            .await
            .iter()
            .map(|s| {
                let mut h = hmac_sha256::Hash::new();
                h.update(session_key.as_bytes());
                h.update(s.id.to_le_bytes());
                let d = h.finalize();
                (u64::from_le_bytes(d[..8].try_into().unwrap()), s.clone())
            })
            .collect::<Vec<(u64, Arc<Slot>)>>();
        scored.sort_by_key(|(score, slot)| {
            (
                std::cmp::Reverse(prefer_trusted && slot.trusted),
                std::cmp::Reverse(*score),
            )
        });
        scored.into_iter().map(|(_, s)| s).collect()
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
                    self.slots
                        .note_utilization(&slot, utilization(resp.headers()))
                        .await;
                    return Ok((slot.id, resp));
                }
                Err(SendError::Auth(body_text)) => {
                    tracing::warn!("account {} got 401, forcing refresh", slot.display);
                    if let Ok(token) = self.slots.fresh_token(&slot, true).await {
                        match self.client.send(&token, path, body, hdrs).await {
                            Ok(resp) => {
                                self.slots.mark_ok(&slot).await;
                                self.slots
                                    .note_utilization(&slot, utilization(resp.headers()))
                                    .await;
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

/// The unified-limit windows Anthropic reports utilization for, e.g.
/// `anthropic-ratelimit-unified-5h-utilization: 0.4`.
fn utilization(headers: &reqwest::header::HeaderMap) -> Vec<(String, f64)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let window = name
                .as_str()
                .strip_prefix("anthropic-ratelimit-unified-")?
                .strip_suffix("-utilization")?;
            let ratio = value.to_str().ok()?.parse::<f64>().ok()?;
            Some((window.to_string(), ratio))
        })
        .collect()
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
        assert!(
            winners.len() > 1,
            "all sessions landed on one account"
        );

        let order = pool.ranked("session-a", false).await;
        assert_eq!(order.len(), 4);
        let ids = order.iter().map(|s| s.id).collect::<HashSet<i64>>();
        assert_eq!(ids.len(), 4);
    }

    #[tokio::test]
    async fn prefer_trusted_ranks_trusted_first() {
        let pool = test_pool(&[(1, false), (2, true), (3, false), (4, true)]).await;

        for i in 0..16 {
            let order = pool.ranked(&format!("session-{i}"), true).await;
            let heads = [order[0].id, order[1].id];
            assert!(heads.contains(&2) && heads.contains(&4), "untrusted ranked first");
        }

        let plain = pool.ranked("session-x", false).await;
        let preferred = pool.ranked("session-x", true).await;
        let plain_trusted = plain
            .iter()
            .filter(|s| s.trusted)
            .map(|s| s.id)
            .collect::<Vec<_>>();
        let preferred_trusted = preferred[..2].iter().map(|s| s.id).collect::<Vec<_>>();
        assert_eq!(plain_trusted, preferred_trusted, "stickiness broken within trusted group");
    }
}
