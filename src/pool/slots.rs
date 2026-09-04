use crate::db::Db;
use crate::db::accounts::AccountStatus;
use crate::oauth::refresh::RefreshError;
use crate::provider::{AuthMode, Provider};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

pub struct Slot {
    pub id: i64,
    pub provider_account_id: String,
    pub display: String,
    pub trusted: bool,
    pub auth_mode: AuthMode,
    pub plan: Option<String>,
    pub http_referer: Option<String>,
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
    /// Sub-limits for individual models. Held apart from `windows` because
    /// each is measured against its own allowance, so mixing them in would
    /// make `peak` report strain the account does not have.
    pub model_windows: Vec<ModelWindow>,
    /// The provider has stopped serving this account until a window resets,
    /// which is a harder signal than a high fraction.
    pub locked: bool,
    /// When this sample was taken. Codex only reports quota on a served
    /// response, so an idle account's figures go stale and a dashboard needs
    /// to know that rather than trusting them.
    pub observed_at: i64,
}

#[derive(Debug, Clone)]
pub struct ModelWindow {
    pub model: String,
    pub window: String,
    pub utilization: f64,
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

    /// How far ahead of a level burn the account is, as a ratio: headroom
    /// divided by the headroom it would have if it had spent evenly since the
    /// window opened. Above 1 it has capacity to spare, below 1 it runs out
    /// before the window resets.
    ///
    /// Dividing by the time left is what makes quota that is about to reset
    /// worth more than quota that is not, since the unspent part is lost.
    ///
    /// Measured on the longest window only.
    fn slack(&self, now: i64) -> Option<f64> {
        let w = self
            .windows
            .iter()
            .filter(|w| w.resets_at.is_some() && window_seconds(&w.name).is_some())
            .max_by_key(|w| window_seconds(&w.name))?;
        let resets_in = (w.resets_at? - now).max(MIN_RESET_SECS) as f64;
        let span = window_seconds(&w.name)? as f64;
        Some((1.0 - w.utilization).max(0.0) * span / resets_in)
    }

    /// Coarse on purpose. Sessions stay pinned to one account while it holds
    /// its band, so a prompt cache is only given up when the account's
    /// standing actually changes rather than on every drift in the numbers.
    pub fn band(&self, soft_limit: f64, now: i64) -> Band {
        if self.locked || self.peak() >= soft_limit {
            return Band::Spent;
        }
        match self.slack(now) {
            Some(s) if s >= 2.0 => Band::Ample,
            Some(s) if s >= 1.0 => Band::Steady,
            Some(_) => Band::Behind,
            None => Band::Steady,
        }
    }
}

/// A window that has just reset reports a tiny time remaining, which would
/// divide the headroom into a near-infinite score.
const MIN_RESET_SECS: i64 = 300;

pub fn window_seconds(name: &str) -> Option<i64> {
    let (value, unit) = name.split_at(name.len().checked_sub(1)?);
    let value: i64 = value.parse().ok()?;
    match unit {
        "m" => Some(value * 60),
        "h" => Some(value * 3600),
        "d" => Some(value * 86400),
        _ => None,
    }
}

/// Routing order for an account, best first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    Ample,
    Steady,
    Behind,
    Spent,
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
pub struct Slots {
    provider: Provider,
    inner: RwLock<Vec<Arc<Slot>>>,
    db: Db,
}

impl Slots {
    pub async fn load(db: Db, provider: Provider) -> eyre::Result<Self> {
        let slots = Self {
            provider,
            inner: RwLock::new(Vec::new()),
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

        let mut slots = self.inner.write().await;
        let mut next = Vec::with_capacity(accounts.len());
        let mut added = 0;
        for a in accounts {
            let existing = match slots.iter().find(|s| s.id == a.id) {
                Some(s) if s.state.lock().await.refresh_token == a.refresh_token => Some(s),
                _ => None,
            };
            match existing {
                // Swapping an unchanged slot would strand an in-flight
                // cooldown write on the orphaned Arc.
                Some(s) if slot_matches(s, &a) => next.push(Arc::clone(s)),
                Some(s) => next.push(Arc::new(reslot(&a, s).await)),
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
        self.inner.read().await.len()
    }


    pub async fn list(&self) -> Vec<Arc<Slot>> {
        self.inner.read().await.clone()
    }

    /// Claims the slot for a request if it is not disabled or cooling down.
    pub async fn try_claim(&self, slot: &Slot) -> bool {
        let now = crate::clock::unix_now();
        let mut st = slot.state.lock().await;
        match st.status {
            Status::Disabled => false,
            Status::Cooldown { until } if until > now => false,
            Status::Active | Status::Cooldown { .. } => {
                st.status = Status::Active;
                true
            }
        }
    }

    pub async fn mark_ok(&self, slot: &Slot) {
        slot.state.lock().await.consecutive_fails = 0;
    }

    pub async fn note_usage(&self, slot: &Slot, mut usage: AccountUsage) {
        usage.observed_at = crate::clock::unix_now();
        slot.state.lock().await.usage = Some(usage);
    }

    /// Where the account sits relative to a level burn of its windows.
    /// Accounts with no usage report yet are assumed healthy so a fresh
    /// account is not held back before it has served anything.
    pub async fn band(&self, slot: &Slot, soft_limit: f64) -> Band {
        let now = crate::clock::unix_now();
        slot.state
            .lock()
            .await
            .usage
            .as_ref()
            .map_or(Band::Steady, |u| u.band(soft_limit, now))
    }

    pub async fn snapshot(&self) -> Vec<AccountSnapshot> {
        let now = crate::clock::unix_now();
        let slots = self.list().await;
        let mut out = Vec::with_capacity(slots.len());
        for slot in &slots {
            let st = slot.state.lock().await;
            let (status, cooldown_seconds) = match st.status {
                Status::Cooldown { until } if until > now => (1, until - now),
                Status::Active | Status::Cooldown { .. } => (0, 0),
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
    pub async fn fresh_token(&self, slot: &Slot, force: bool) -> Result<String, ()> {
        let now = crate::clock::unix_now();
        let mut st = slot.state.lock().await;
        if !slot.auth_mode.refreshable() {
            return Ok(st.access_token.clone());
        }
        if !force && st.expires_at.is_some_and(|e| e - now > 60) {
            return Ok(st.access_token.clone());
        }
        tracing::info!(
            "refreshing {} access token for {}",
            self.provider,
            slot.display
        );
        let refreshed = match self.provider {
            Provider::OpenAi => crate::oauth::refresh::refresh(&st.refresh_token).await,
            Provider::Anthropic => crate::oauth::anthropic::refresh(&st.refresh_token).await,
            Provider::Gemini => Err(RefreshError::Terminal(
                "google oauth grants are not implemented, add the account with an api key".into(),
            )),
            Provider::Glm => Err(RefreshError::Terminal(
                "z.ai issues static keys, there is nothing to exchange".into(),
            )),
            Provider::Zen => Err(RefreshError::Terminal(
                "zen issues static keys, there is nothing to exchange".into(),
            )),
        };
        match refreshed {
            Ok(tokens) => {
                if let Err(e) = self.db.update_account_tokens(slot.id, &tokens).await {
                    tracing::error!("persisting refreshed tokens for {}: {e}", slot.display);
                    return Err(());
                }
                st.access_token.clone_from(&tokens.access_token);
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
                    .set_account_status(slot.id, AccountStatus::Disabled, None, Some(&msg))
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

    pub async fn cool(&self, slot: &Slot, secs: i64, why: &str) {
        let until = crate::clock::unix_now() + secs;
        {
            let mut st = slot.state.lock().await;
            st.status = Status::Cooldown { until };
            st.consecutive_fails += 1;
        }
        tracing::warn!("account {} cooling down {secs}s ({why})", slot.display);
        let _ = self
            .db
            .set_account_status(slot.id, AccountStatus::Cooldown, Some(until), None)
            .await;
    }

    /// Backoff for a 429: the reported retry-after when present, else
    /// exponential, clamped to the given ceiling.
    /// `base` is the first backoff to use when the provider names no
    /// retry-after, and it doubles from there.
    pub async fn cool_rate_limited(
        &self,
        slot: &Slot,
        retry_after: Option<i64>,
        max: i64,
        base: i64,
    ) {
        let fails = slot.state.lock().await.consecutive_fails;
        let secs = retry_after
            .unwrap_or_else(|| base.saturating_mul(1 << fails.min(6)))
            .clamp(base.min(30), max);
        self.cool(slot, secs, "rate limited").await;
    }

    pub async fn cool_failure(&self, slot: &Slot) {
        let fails = slot.state.lock().await.consecutive_fails;
        let secs = 15_i64.saturating_mul(1 << fails.min(6)).min(900);
        self.cool(slot, secs, "upstream failure").await;
    }

    /// Seconds until this one slot is claimable, 0 when it already is.
    pub async fn cooldown_left(&self, slot: &Slot) -> i64 {
        let now = crate::clock::unix_now();
        match slot.state.lock().await.status {
            Status::Cooldown { until } if until > now => until - now,
            Status::Active | Status::Disabled | Status::Cooldown { .. } => 0,
        }
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

/// Rendezvous score for a session against a slot. Ordering by it keeps a
/// conversation on one account, which is what lets an upstream prompt cache
/// keep hitting instead of paying for the whole prefix again.
pub fn rendezvous_score(session: &str, id: i64) -> u64 {
    let mut h = hmac_sha256::Hash::new();
    h.update(session.as_bytes());
    h.update(id.to_le_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().unwrap())
}

fn display_for(a: &crate::db::accounts::Account) -> String {
    a.label
        .clone()
        .or_else(|| a.email.clone())
        .unwrap_or_else(|| format!("account#{}", a.id))
}

fn slot_matches(s: &Slot, a: &crate::db::accounts::Account) -> bool {
    (s.trusted, &s.plan, &s.display, &s.http_referer)
        == (a.trusted, &a.plan_type, &display_for(a), &a.http_referer)
}

/// A fresh slot carrying the previous one's cooldown, tokens and quota sample.
async fn reslot(a: &crate::db::accounts::Account, prev: &Slot) -> Slot {
    let state = prev.state.lock().await;
    Slot {
        id: a.id,
        provider_account_id: a.provider_account_id.clone(),
        trusted: a.trusted,
        auth_mode: a.auth_mode,
        plan: a.plan_type.clone(),
        http_referer: a.http_referer.clone(),
        display: display_for(a),
        state: Mutex::new(SlotState {
            access_token: state.access_token.clone(),
            refresh_token: state.refresh_token.clone(),
            expires_at: state.expires_at,
            status: state.status.clone(),
            consecutive_fails: state.consecutive_fails,
            usage: state.usage.clone(),
        }),
    }
}

fn slot_from_account(a: crate::db::accounts::Account) -> Slot {
    let now = crate::clock::unix_now();
    let status = match a.status {
        AccountStatus::Disabled => Status::Disabled,
        AccountStatus::Cooldown if a.cooldown_until.unwrap_or(0) > now => Status::Cooldown {
            until: a.cooldown_until.unwrap_or(0),
        },
        AccountStatus::Active | AccountStatus::Cooldown => Status::Active,
    };
    Slot {
        id: a.id,
        provider_account_id: a.provider_account_id,
        trusted: a.trusted,
        auth_mode: a.auth_mode,
        plan: a.plan_type,
        http_referer: a.http_referer,
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
pub fn test_slots(db: Db, provider: Provider, ids: &[(i64, bool)]) -> Slots {
    Slots {
        provider,
        inner: RwLock::new(
            ids.iter()
                .map(|&(id, trusted)| {
                    Arc::new(Slot {
                        id,
                        provider_account_id: format!("acct-{id}"),
                        display: format!("a{id}"),
                        trusted,
                        auth_mode: match provider {
                            Provider::OpenAi | Provider::Anthropic => AuthMode::OAuth,
                            Provider::Gemini | Provider::Glm | Provider::Zen => AuthMode::ApiKey,
                        },
                        plan: None,
                        http_referer: None,
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
                .collect(),
        ),
        db,
    }
}

#[cfg(test)]
mod band_tests {
    use super::*;

    fn usage(windows: &[(&str, f64, i64)], now: i64) -> AccountUsage {
        AccountUsage {
            windows: windows
                .iter()
                .map(|(name, utilization, resets_in)| UsageWindow {
                    name: (*name).to_owned(),
                    utilization: *utilization,
                    resets_at: Some(now + resets_in),
                })
                .collect(),
            model_windows: Vec::new(),
            locked: false,
            observed_at: 0,
        }
    }

    #[test]
    fn expiring_headroom_outranks_larger_headroom_with_time_to_spare() {
        let now = 1_000_000;
        let h = 3600;
        let expiring = usage(&[("7d", 0.69, 8 * h)], now);
        let roomy = usage(&[("7d", 0.04, 131 * h)], now);
        assert!(expiring.band(0.9, now) < roomy.band(0.9, now));
    }

    #[test]
    fn an_account_burning_faster_than_its_window_falls_behind() {
        let now = 1_000_000;
        let h = 3600;
        assert_eq!(
            usage(&[("7d", 0.80, 37 * h)], now).band(0.9, now),
            Band::Behind
        );
    }

    /// Ranking follows the weekly window, not whichever is tighter.
    #[test]
    fn the_longest_window_decides() {
        let now = 1_000_000;
        let h = 3600;
        let u = usage(&[("7d", 0.10, 100 * h), ("5h", 0.85, 4 * h)], now);
        assert_eq!(u.band(0.9, now), Band::Steady);
    }

    #[test]
    fn a_window_about_to_reset_does_not_score_infinitely() {
        let now = 1_000_000;
        let u = usage(&[("7d", 0.99, 1)], now);
        assert!(u.slack(now).unwrap().is_finite());
    }

    #[test]
    fn usage_without_a_reset_time_keeps_the_old_behaviour() {
        let now = 1_000_000;
        let mut u = usage(&[("7d", 0.5, 3600)], now);
        u.windows[0].resets_at = None;
        assert_eq!(u.band(0.9, now), Band::Steady);
    }
}

#[cfg(test)]
mod idle_window_tests {
    use super::*;

    /// But a short window past the soft limit is still benched, since the next
    /// request would be refused outright.
    #[test]
    fn a_full_short_window_is_still_spent() {
        let now = 1_000_000;
        let h = 3600;
        let u = AccountUsage {
            windows: vec![
                UsageWindow {
                    name: "5h".into(),
                    utilization: 0.95,
                    resets_at: Some(now + 4 * h),
                },
                UsageWindow {
                    name: "7d".into(),
                    utilization: 0.10,
                    resets_at: Some(now + 100 * h),
                },
            ],
            model_windows: Vec::new(),
            locked: false,
            observed_at: 0,
        };
        assert_eq!(u.band(0.9, now), Band::Spent);
    }
}
