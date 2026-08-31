use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use thiserror::Error;
use tokio::sync::Mutex;

use serde_json::Value;

use crate::codex::client::{CodexClient, SendError};
use crate::db::Db;
use crate::oauth::refresh::{refresh, RefreshError};

pub struct AccountPool {
    slots: Vec<Arc<Slot>>,
    cursor: AtomicUsize,
    db: Db,
}

pub struct Slot {
    pub id: i64,
    pub chatgpt_account_id: String,
    pub display: String,
    state: Mutex<SlotState>,
}

struct SlotState {
    access_token: String,
    refresh_token: String,
    expires_at: Option<i64>,
    status: Status,
    consecutive_fails: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Active,
    Cooldown { until: i64 },
    Disabled,
}

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("no usable codex accounts; run `slop-proxy login`")]
    NoAccounts,
    #[error("all codex accounts are cooling down")]
    AllCoolingDown { retry_after: i64 },
    #[error("upstream rejected the request: {0}")]
    BadRequest(String),
    #[error("upstream failure: {0}")]
    Upstream(String),
}

impl AccountPool {
    pub async fn load(db: Db) -> anyhow::Result<Self> {
        let now = chrono::Utc::now().timestamp();
        let slots = db
            .list_accounts()
            .await?
            .into_iter()
            .map(|a| {
                let status = match a.status.as_str() {
                    "disabled" => Status::Disabled,
                    "cooldown" if a.cooldown_until.unwrap_or(0) > now => Status::Cooldown {
                        until: a.cooldown_until.unwrap_or(0),
                    },
                    _ => Status::Active,
                };
                Arc::new(Slot {
                    id: a.id,
                    chatgpt_account_id: a.chatgpt_account_id,
                    display: a
                        .label
                        .or(a.email)
                        .unwrap_or_else(|| format!("account#{}", a.id)),
                    state: Mutex::new(SlotState {
                        access_token: a.access_token,
                        refresh_token: a.refresh_token,
                        expires_at: a.access_expires_at,
                        status,
                        consecutive_fails: 0,
                    }),
                })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            slots,
            cursor: AtomicUsize::new(0),
            db,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Fresh (access_token, chatgpt_account_id) for any active account, for
    /// non-completion calls like the models listing.
    pub async fn any_active_credentials(&self) -> Option<(String, String)> {
        let slot = self.next_available().await?;
        let access = self.fresh_token(&slot, false).await.ok()?;
        Some((access, slot.chatgpt_account_id.clone()))
    }

    pub async fn execute(
        &self,
        client: &CodexClient,
        req: &Value,
    ) -> Result<(i64, reqwest::Response), PoolError> {
        if self.slots.is_empty() {
            return Err(PoolError::NoAccounts);
        }
        let attempts = self.slots.len().min(3);
        let mut last_err: Option<SendError> = None;

        for _ in 0..attempts {
            let Some(slot) = self.next_available().await else {
                break;
            };
            let creds = match self.fresh_token(&slot, false).await {
                Ok(c) => c,
                Err(()) => continue,
            };

            match client.send(&creds, &slot.chatgpt_account_id, req).await {
                Ok(resp) => {
                    slot.state.lock().await.consecutive_fails = 0;
                    return Ok((slot.id, resp));
                }
                Err(SendError::Auth(body)) => {
                    tracing::warn!("account {} got 401, forcing refresh", slot.display);
                    if let Ok(creds) = self.fresh_token(&slot, true).await {
                        match client.send(&creds, &slot.chatgpt_account_id, req).await {
                            Ok(resp) => {
                                slot.state.lock().await.consecutive_fails = 0;
                                return Ok((slot.id, resp));
                            }
                            Err(e) => {
                                self.cool(&slot, 60, "post-refresh failure").await;
                                last_err = Some(e);
                            }
                        }
                    } else {
                        last_err = Some(SendError::Auth(body));
                    }
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    let fails = slot.state.lock().await.consecutive_fails;
                    let secs = retry_after
                        .unwrap_or(60i64.saturating_mul(1 << fails.min(6)))
                        .clamp(30, 6 * 3600);
                    self.cool(&slot, secs, "rate limited").await;
                    last_err = Some(SendError::RateLimited { retry_after, body });
                }
                Err(SendError::BadRequest(body)) => {
                    return Err(PoolError::BadRequest(body));
                }
                Err(e) => {
                    let fails = slot.state.lock().await.consecutive_fails;
                    let secs = 15i64.saturating_mul(1 << fails.min(6)).min(900);
                    self.cool(&slot, secs, "upstream failure").await;
                    last_err = Some(e);
                }
            }
        }

        match last_err {
            Some(SendError::RateLimited { .. }) | None => Err(PoolError::AllCoolingDown {
                retry_after: self.min_cooldown().await.max(30),
            }),
            Some(e) => Err(PoolError::Upstream(e.to_string())),
        }
    }

    async fn next_available(&self) -> Option<Arc<Slot>> {
        let now = chrono::Utc::now().timestamp();
        let n = self.slots.len();
        for _ in 0..n {
            let i = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
            let slot = &self.slots[i];
            let mut st = slot.state.lock().await;
            match st.status {
                Status::Disabled => continue,
                Status::Cooldown { until } if until > now => continue,
                _ => {
                    st.status = Status::Active;
                    return Some(slot.clone());
                }
            }
        }
        None
    }

    /// Refresh serialization matters: refresh tokens rotate, so two tasks
    /// refreshing the same account concurrently would invalidate each other.
    /// The slot's state mutex is held across the refresh call for that reason.
    async fn fresh_token(&self, slot: &Arc<Slot>, force: bool) -> Result<String, ()> {
        let now = chrono::Utc::now().timestamp();
        let mut st = slot.state.lock().await;
        if !force && st.expires_at.map(|e| e - now > 60).unwrap_or(false) {
            return Ok(st.access_token.clone());
        }
        tracing::info!("refreshing access token for {}", slot.display);
        match refresh(&st.refresh_token).await {
            Ok(tokens) => {
                if let Err(e) = self.db.update_account_tokens(slot.id, &tokens).await {
                    tracing::error!("persisting refreshed tokens for {}: {e}", slot.display);
                    return Err(());
                }
                st.access_token = tokens.access_token.clone();
                st.refresh_token = tokens.refresh_token;
                st.expires_at = tokens.expires_at;
                Ok(tokens.access_token)
            }
            Err(RefreshError::Terminal(msg)) => {
                tracing::error!(
                    "account {} refresh token is dead ({msg}); disabling",
                    slot.display
                );
                st.status = Status::Disabled;
                drop(st);
                let _ = self
                    .db
                    .set_account_status(slot.id, "disabled", None, Some(&msg))
                    .await;
                Err(())
            }
            Err(RefreshError::Transient(msg)) => {
                tracing::warn!("account {} refresh failed transiently: {msg}", slot.display);
                drop(st);
                self.cool(slot, 30, "refresh failure").await;
                Err(())
            }
        }
    }

    async fn cool(&self, slot: &Arc<Slot>, secs: i64, why: &str) {
        let until = chrono::Utc::now().timestamp() + secs;
        {
            let mut st = slot.state.lock().await;
            st.status = Status::Cooldown { until };
            st.consecutive_fails += 1;
        }
        tracing::warn!("account {} cooling down {secs}s ({why})", slot.display);
        let _ = self
            .db
            .set_account_status(slot.id, "cooldown", Some(until), None)
            .await;
    }

    async fn min_cooldown(&self) -> i64 {
        let now = chrono::Utc::now().timestamp();
        let mut min = i64::MAX;
        for slot in &self.slots {
            let st = slot.state.lock().await;
            if let Status::Cooldown { until } = st.status {
                min = min.min(until - now);
            }
        }
        if min == i64::MAX {
            30
        } else {
            min.max(1)
        }
    }
}
