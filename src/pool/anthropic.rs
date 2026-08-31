use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};

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
    pub async fn load(db: Db, client: AnthropicClient) -> anyhow::Result<Self> {
        Ok(Self {
            slots: Slots::load(db, Provider::Anthropic).await?,
            client,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Rendezvous-hashed order: a session sticks to its highest-scoring
    /// account so upstream prompt caches keep hitting, and only moves while
    /// that account is cooling down.
    fn ranked(&self, session_key: &str) -> Vec<Arc<Slot>> {
        let mut scored = self
            .slots
            .all()
            .iter()
            .map(|s| {
                let mut h = Sha256::new();
                h.update(session_key.as_bytes());
                h.update(s.id.to_le_bytes());
                let d = h.finalize();
                (u64::from_le_bytes(d[..8].try_into().unwrap()), s.clone())
            })
            .collect::<Vec<(u64, Arc<Slot>)>>();
        scored.sort_by_key(|s| std::cmp::Reverse(s.0));
        scored.into_iter().map(|(_, s)| s).collect()
    }

    pub async fn execute(
        &self,
        path: &str,
        body: &Value,
        hdrs: &RelayHeaders,
        session_key: &str,
    ) -> Result<(i64, reqwest::Response), PoolError> {
        if self.slots.is_empty() {
            return Err(PoolError::NoAccounts(Provider::Anthropic));
        }
        let mut last_err = Option::<SendError>::None;
        let mut attempts = 0;

        for slot in self.ranked(session_key) {
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

    async fn test_pool(ids: &[i64]) -> AnthropicPool {
        let db_path = std::env::temp_dir().join(format!("slop-rdv-{}.db", uuid::Uuid::new_v4()));
        let db = Db::open(&db_path).await.unwrap();
        AnthropicPool {
            slots: super::super::test_slots(db, Provider::Anthropic, ids),
            client: AnthropicClient::new(AnthropicConfig::default()),
        }
    }

    #[tokio::test]
    async fn rendezvous_is_sticky_and_spreads() {
        let pool = test_pool(&[1, 2, 3, 4]).await;

        let first = pool.ranked("session-a")[0].id;
        for _ in 0..10 {
            assert_eq!(pool.ranked("session-a")[0].id, first);
        }

        let winners = (0..64)
            .map(|i| pool.ranked(&format!("session-{i}"))[0].id)
            .collect::<HashSet<i64>>();
        assert!(winners.len() > 1, "all sessions landed on one account");

        let order = pool.ranked("session-a");
        assert_eq!(order.len(), 4);
        let ids = order.iter().map(|s| s.id).collect::<HashSet<i64>>();
        assert_eq!(ids.len(), 4);
    }
}
