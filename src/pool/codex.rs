use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use super::{PoolError, Slot, Slots};
use crate::codex::client::CodexClient;
use crate::codex::models::ModelInfo;
use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Round-robin pool over codex accounts, owning the backend client.
pub struct CodexPool {
    slots: Slots,
    cursor: AtomicUsize,
    client: CodexClient,
}

impl CodexPool {
    pub async fn load(db: Db, client: CodexClient) -> eyre::Result<Self> {
        Ok(Self {
            slots: Slots::load(db, Provider::Codex).await?,
            cursor: AtomicUsize::new(0),
            client,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn client(&self) -> &CodexClient {
        &self.client
    }

    pub async fn snapshot(&self) -> Vec<super::AccountSnapshot> {
        self.slots.snapshot().await
    }

    /// Fresh (access_token, account_id) for any active account, for
    /// non-completion calls like the models listing.
    pub async fn any_active_credentials(&self) -> Option<(String, String)> {
        let slot = self.next_available().await?;
        let access = self.slots.fresh_token(&slot, false).await.ok()?;
        Some((access, slot.provider_account_id.clone()))
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        let (access, account_id) = self
            .any_active_credentials()
            .await
            .ok_or_else(|| "no usable account; run `slop-proxy login`".to_string())?;
        self.client.list_models(&access, &account_id).await
    }

    pub async fn execute(&self, req: &Value) -> Result<(i64, reqwest::Response), PoolError> {
        if self.slots.is_empty() {
            return Err(PoolError::NoAccounts(Provider::Codex));
        }
        let attempts = self.slots.len().min(3);
        let mut last_err = Option::<SendError>::None;

        for _ in 0..attempts {
            let Some(slot) = self.next_available().await else {
                break;
            };
            let Ok(creds) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };

            match self.client.send(&creds, &slot.provider_account_id, req).await {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    return Ok((slot.id, resp));
                }
                Err(SendError::Auth(body)) => {
                    tracing::warn!("account {} got 401, forcing refresh", slot.display);
                    if let Ok(creds) = self.slots.fresh_token(&slot, true).await {
                        match self.client.send(&creds, &slot.provider_account_id, req).await {
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
                        last_err = Some(SendError::Auth(body));
                    }
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    self.slots
                        .cool_rate_limited(&slot, retry_after, 6 * 3600)
                        .await;
                    last_err = Some(SendError::RateLimited { retry_after, body });
                }
                Err(SendError::BadRequest(body)) => {
                    return Err(PoolError::BadRequest(body));
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

    async fn next_available(&self) -> Option<Arc<Slot>> {
        let n = self.slots.len();
        for _ in 0..n {
            let i = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
            let slot = &self.slots.all()[i];
            if self.slots.try_claim(slot).await {
                return Some(slot.clone());
            }
        }
        None
    }
}
