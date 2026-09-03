use axum::body::Bytes;

use super::{PoolError, Slots};
use crate::config::GlmConfig;
use crate::db::Db;
use crate::glm::client::GlmClient;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Z.ai keys.
pub struct GlmPool {
    slots: Slots,
    client: GlmClient,
}

impl GlmPool {
    pub async fn load(db: Db, cfg: GlmConfig) -> eyre::Result<Self> {
        Ok(Self {
            slots: Slots::load(db, Provider::Glm).await?,
            client: GlmClient::new(cfg),
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

    pub async fn execute(
        &self,
        path: &str,
        body: &Bytes,
        session_key: &str,
    ) -> Result<(i64, reqwest::Response), PoolError> {
        let mut ranked = self.slots.list().await;
        ranked.sort_by_key(|slot| std::cmp::Reverse(super::rendezvous_score(session_key, slot.id)));
        if ranked.is_empty() {
            return Err(PoolError::NoAccounts(Provider::Glm));
        }

        let mut last_err = Option::<SendError>::None;
        for slot in ranked.into_iter().take(3) {
            if !self.slots.try_claim(&slot).await {
                continue;
            }
            let Ok(key) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.client.send(&key, path, body).await {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    return Ok((slot.id, resp));
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
