use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use super::{AccountUsage, PoolError, Slot, Slots, UsageWindow};
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

    pub async fn len(&self) -> usize {
        self.slots.len().await
    }

    pub async fn is_empty(&self) -> bool {
        self.slots.is_empty().await
    }

    pub async fn reload(&self) -> eyre::Result<()> {
        self.slots.reload().await
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
        let slot = self.next_available(false).await?;
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

    pub async fn execute(
        &self,
        req: &Value,
        prefer_trusted: bool,
    ) -> Result<(i64, reqwest::Response), PoolError> {
        let attempts = self.slots.len().await.min(3);
        if attempts == 0 {
            return Err(PoolError::NoAccounts(Provider::Codex));
        }
        let mut last_err = Option::<SendError>::None;

        for _ in 0..attempts {
            let Some(slot) = self.next_available(prefer_trusted).await else {
                break;
            };
            let Ok(creds) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };

            match self.client.send(&creds, &slot.provider_account_id, req).await {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    if let Some(usage) = usage_from_headers(resp.headers()) {
                        self.slots.note_usage(&slot, usage).await;
                    }
                    return Ok((slot.id, resp));
                }
                Err(SendError::Auth(body)) => {
                    tracing::warn!("account {} got 401, forcing refresh", slot.display);
                    if let Ok(creds) = self.slots.fresh_token(&slot, true).await {
                        match self.client.send(&creds, &slot.provider_account_id, req).await {
                            Ok(resp) => {
                                self.slots.mark_ok(&slot).await;
                                if let Some(usage) = usage_from_headers(resp.headers()) {
                                    self.slots.note_usage(&slot, usage).await;
                                }
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

    /// Accounts matching the token's trusted preference are tried first, so
    /// ordinary traffic spends the untrusted accounts and only spills onto
    /// the scarce trusted ones when nothing else is available.
    async fn next_available(&self, prefer_trusted: bool) -> Option<Arc<Slot>> {
        let (preferred, rest): (Vec<_>, Vec<_>) = self
            .slots
            .list()
            .await
            .into_iter()
            .partition(|s| s.trusted == prefer_trusted);
        if let Some(slot) = self.claim_round_robin(&preferred).await {
            return Some(slot);
        }
        self.claim_round_robin(&rest).await
    }

    async fn claim_round_robin(&self, slots: &[Arc<Slot>]) -> Option<Arc<Slot>> {
        let n = slots.len();
        if n == 0 {
            return None;
        }
        for _ in 0..n {
            let i = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
            let slot = &slots[i];
            if self.slots.try_claim(slot).await {
                return Some(slot.clone());
            }
        }
        None
    }
}

/// The codex backend reports quota on every successful response rather than
/// from a queryable endpoint, so consumption is only known once an account
/// has served traffic.
fn usage_from_headers(headers: &reqwest::header::HeaderMap) -> Option<AccountUsage> {
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
        locked: false,
        observed_at: 0,
    })
}

fn window_name(minutes: i64) -> String {
    match minutes {
        m if m % 1440 == 0 => format!("{}d", m / 1440),
        m if m % 60 == 0 => format!("{}h", m / 60),
        m => format!("{m}m"),
    }
}
