use axum::body::Bytes;

use super::{PoolError, Slots};
use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;
use crate::zen::client::ZenClient;

/// Zen over whatever credentials are stored, and over none at all when the
/// table is empty. The free models are served without a key today, so an
/// empty pool is a working pool rather than an error.
pub struct ZenPool {
    slots: Slots,
    client: ZenClient,
}

impl ZenPool {
    pub async fn load(db: Db, client: ZenClient) -> eyre::Result<Self> {
        Ok(Self {
            slots: Slots::load(db, Provider::Zen).await?,
            client,
        })
    }

    pub async fn len(&self) -> usize {
        self.slots.len().await
    }

    pub async fn reload(&self) -> eyre::Result<()> {
        self.slots.reload().await
    }

    pub async fn snapshot(&self) -> Vec<super::AccountSnapshot> {
        self.slots.snapshot().await
    }

    pub async fn models(&self) -> Vec<String> {
        self.client.models().await.unwrap_or_default()
    }

    pub async fn execute(
        &self,
        req: &Bytes,
        session_key: &str,
    ) -> Result<(Option<i64>, reqwest::Response), PoolError> {
        let mut ranked = self.slots.list().await;
        ranked.sort_by_key(|slot| std::cmp::Reverse(super::rendezvous_score(session_key, slot.id)));

        if ranked.is_empty() {
            return match self.client.send(None, req).await {
                Ok(resp) => Ok((None, resp)),
                Err(SendError::BadRequest(body)) => Err(PoolError::BadRequest(body)),
                Err(SendError::RateLimited { retry_after, .. }) => Err(PoolError::AllCoolingDown {
                    retry_after: retry_after.unwrap_or(30),
                }),
                Err(e) => Err(PoolError::Upstream(e.to_string())),
            };
        }

        let mut last_err = Option::<SendError>::None;
        for slot in ranked.into_iter().take(3) {
            if !self.slots.try_claim(&slot).await {
                continue;
            }
            let Ok(key) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.client.send(Some(&key), req).await {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    return Ok((Some(slot.id), resp));
                }
                Err(SendError::Auth(text)) => {
                    self.slots.cool(&slot, 15 * 60, "key rejected").await;
                    last_err = Some(SendError::Auth(text));
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    self.slots
                        .cool_rate_limited(&slot, retry_after, 3600, 60)
                        .await;
                    last_err = Some(SendError::RateLimited { retry_after, body });
                }
                Err(SendError::BadRequest(text)) => return Err(PoolError::BadRequest(text)),
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
