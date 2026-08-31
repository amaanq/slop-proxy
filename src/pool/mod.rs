pub mod anthropic;
pub mod codex;

use std::sync::Arc;

use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

use crate::db::Db;
use crate::oauth::refresh::RefreshError;
use crate::provider::Provider;

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("no usable {0} accounts; run `slop-proxy login`")]
    NoAccounts(Provider),
    #[error("all upstream accounts are cooling down")]
    AllCoolingDown { retry_after: i64 },
    #[error("upstream rejected the request: {0}")]
    BadRequest(String),
    #[error("upstream failure: {0}")]
    Upstream(String),
}

pub(crate) struct Slot {
    pub id: i64,
    pub provider_account_id: String,
    pub display: String,
    pub trusted: bool,
    pub plan: Option<String>,
    state: Mutex<SlotState>,
}

struct SlotState {
    access_token: String,
    refresh_token: String,
    expires_at: Option<i64>,
    status: Status,
    consecutive_fails: u32,
    usage: Option<AccountUsage>,
}

/// Provider-reported consumption of an account's rolling limit windows.
#[derive(Debug, Default, Clone)]
pub struct AccountUsage {
    pub windows: Vec<UsageWindow>,
    /// The provider has stopped serving this account until a window resets,
    /// which is a harder signal than a high fraction.
    pub locked: bool,
    /// When this sample was taken. Codex only reports quota on a served
    /// response, so an idle account's figures go stale and a dashboard needs
    /// to know that rather than trusting them.
    pub observed_at: i64,
}

#[derive(Debug, Clone)]
pub struct UsageWindow {
    pub name: String,
    /// Fraction consumed, 0.0 to 1.0.
    pub utilization: f64,
    /// Unix seconds at which the window rolls over, when reported.
    pub resets_at: Option<i64>,
}

impl AccountUsage {
    pub fn peak(&self) -> f64 {
        self.windows
            .iter()
            .map(|w| w.utilization)
            .fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Active,
    Cooldown { until: i64 },
    Disabled,
}

/// Point-in-time view of one account for the metrics endpoint.
pub struct AccountSnapshot {
    pub provider: Provider,
    pub display: String,
    pub plan: Option<String>,
    pub trusted: bool,
    pub status: u8,
    pub cooldown_seconds: i64,
    pub consecutive_fails: u32,
    pub usage: Option<AccountUsage>,
}

/// The per-account state machine shared by both backend pools: cooldown
/// bookkeeping and serialized token refresh. Selection strategy stays with
/// the owning pool.
pub(crate) struct Slots {
    provider: Provider,
    slots: RwLock<Vec<Arc<Slot>>>,
    db: Db,
}

impl Slots {
    pub async fn load(db: Db, provider: Provider) -> eyre::Result<Self> {
        let slots = Self {
            provider,
            slots: RwLock::new(Vec::new()),
            db,
        };
        slots.reload().await?;
        Ok(slots)
    }

    /// Syncs slots with the accounts table so logins land without a restart.
    /// Existing slots keep their in-memory cooldown and token state unless the
    /// db row's refresh token changed, which means someone re-logged-in and
    /// the in-memory grant is stale.
    pub async fn reload(&self) -> eyre::Result<()> {
        let accounts: Vec<_> = self
            .db
            .list_accounts()
            .await?
            .into_iter()
            .filter(|a| a.provider == self.provider)
            .collect();

        let mut slots = self.slots.write().await;
        let mut next = Vec::with_capacity(accounts.len());
        let mut added = 0;
        for a in accounts {
            let existing = match slots.iter().find(|s| s.id == a.id) {
                Some(s) if s.state.lock().await.refresh_token == a.refresh_token => Some(s),
                _ => None,
            };
            match existing {
                Some(s) => next.push(s.clone()),
                None => {
                    added += 1;
                    next.push(Arc::new(slot_from_account(a)));
                }
            }
        }
        let removed = slots
            .iter()
            .filter(|s| !next.iter().any(|n| n.id == s.id))
            .count();
        if added > 0 || removed > 0 {
            tracing::info!(
                "reloaded {} accounts: {added} added or replaced, {removed} removed",
                self.provider
            );
        }
        *slots = next;
        Ok(())
    }

    pub async fn len(&self) -> usize {
        self.slots.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.slots.read().await.is_empty()
    }

    pub async fn list(&self) -> Vec<Arc<Slot>> {
        self.slots.read().await.clone()
    }

    /// Claims the slot for a request if it is not disabled or cooling down.
    pub async fn try_claim(&self, slot: &Arc<Slot>) -> bool {
        let now = crate::clock::unix_now();
        let mut st = slot.state.lock().await;
        match st.status {
            Status::Disabled => false,
            Status::Cooldown { until } if until > now => false,
            _ => {
                st.status = Status::Active;
                true
            }
        }
    }

    pub async fn mark_ok(&self, slot: &Arc<Slot>) {
        slot.state.lock().await.consecutive_fails = 0;
    }

    pub async fn note_usage(&self, slot: &Arc<Slot>, mut usage: AccountUsage) {
        usage.observed_at = crate::clock::unix_now();
        slot.state.lock().await.usage = Some(usage);
    }

    /// Fraction of the busiest window consumed, and whether the provider has
    /// locked the account out entirely.
    pub async fn usage_of(&self, slot: &Arc<Slot>) -> Option<AccountUsage> {
        slot.state.lock().await.usage.clone()
    }

    pub async fn snapshot(&self) -> Vec<AccountSnapshot> {
        let now = crate::clock::unix_now();
        let slots = self.list().await;
        let mut out = Vec::with_capacity(slots.len());
        for slot in &slots {
            let st = slot.state.lock().await;
            let (status, cooldown_seconds) = match st.status {
                Status::Active => (0, 0),
                Status::Cooldown { until } if until > now => (1, until - now),
                Status::Cooldown { .. } => (0, 0),
                Status::Disabled => (2, 0),
            };
            out.push(AccountSnapshot {
                provider: self.provider,
                display: slot.display.clone(),
                plan: slot.plan.clone(),
                trusted: slot.trusted,
                status,
                cooldown_seconds,
                consecutive_fails: st.consecutive_fails,
                usage: st.usage.clone(),
            });
        }
        out
    }

    /// Refresh serialization matters: refresh tokens rotate, so two tasks
    /// refreshing the same account concurrently would invalidate each other.
    /// The slot's state mutex is held across the refresh call for that reason.
    pub async fn fresh_token(&self, slot: &Arc<Slot>, force: bool) -> Result<String, ()> {
        let now = crate::clock::unix_now();
        let mut st = slot.state.lock().await;
        if !force && st.expires_at.map(|e| e - now > 60).unwrap_or(false) {
            return Ok(st.access_token.clone());
        }
        tracing::info!(
            "refreshing {} access token for {}",
            self.provider,
            slot.display
        );
        let refreshed = match self.provider {
            Provider::Codex => crate::oauth::refresh::refresh(&st.refresh_token).await,
            Provider::Anthropic => crate::oauth::anthropic::refresh(&st.refresh_token).await,
        };
        match refreshed {
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

    pub async fn cool(&self, slot: &Arc<Slot>, secs: i64, why: &str) {
        let until = crate::clock::unix_now() + secs;
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

    /// Backoff for a 429: the reported retry-after when present, else
    /// exponential, clamped to the given ceiling.
    pub async fn cool_rate_limited(&self, slot: &Arc<Slot>, retry_after: Option<i64>, max: i64) {
        let fails = slot.state.lock().await.consecutive_fails;
        let secs = retry_after
            .unwrap_or(60i64.saturating_mul(1 << fails.min(6)))
            .clamp(30, max);
        self.cool(slot, secs, "rate limited").await;
    }

    pub async fn cool_failure(&self, slot: &Arc<Slot>) {
        let fails = slot.state.lock().await.consecutive_fails;
        let secs = 15i64.saturating_mul(1 << fails.min(6)).min(900);
        self.cool(slot, secs, "upstream failure").await;
    }

    pub async fn min_cooldown(&self) -> i64 {
        let now = crate::clock::unix_now();
        let mut min = i64::MAX;
        for slot in &self.list().await {
            let st = slot.state.lock().await;
            if let Status::Cooldown { until } = st.status {
                min = min.min(until - now);
            }
        }
        if min == i64::MAX { 30 } else { min.max(1) }
    }
}

fn slot_from_account(a: crate::db::accounts::Account) -> Slot {
    let now = crate::clock::unix_now();
    let status = match a.status.as_str() {
        "disabled" => Status::Disabled,
        "cooldown" if a.cooldown_until.unwrap_or(0) > now => Status::Cooldown {
            until: a.cooldown_until.unwrap_or(0),
        },
        _ => Status::Active,
    };
    Slot {
        id: a.id,
        provider_account_id: a.provider_account_id,
        trusted: a.trusted,
        plan: a.plan_type,
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
            usage: None,
        }),
    }
}

#[cfg(test)]
pub(crate) fn test_slots(db: Db, provider: Provider, ids: &[(i64, bool)]) -> Slots {
    Slots {
        provider,
        slots: RwLock::new(ids
            .iter()
            .map(|&(id, trusted)| {
                Arc::new(Slot {
                    id,
                    provider_account_id: format!("acct-{id}"),
                    display: format!("a{id}"),
                    trusted,
                    plan: None,
                    state: Mutex::new(SlotState {
                        access_token: "at".into(),
                        refresh_token: "rt".into(),
                        expires_at: None,
                        status: Status::Active,
                        consecutive_fails: 0,
                        usage: None,
                    }),
                })
            })
            .collect()),
        db,
    }
}
