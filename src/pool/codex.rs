use axum::body::Bytes;

use super::{AccountUsage, AuthPolicy, Backend, Cooldown, Pool, Route, Slot, UsageWindow};
use crate::codex::client::CodexClient;
use crate::codex::models::ModelInfo;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over codex accounts, owning the backend client.
pub type CodexPool = Pool<CodexClient>;

impl Backend for CodexClient {
    const PROVIDER: Provider = Provider::OpenAi;
    const RATE_LIMIT: Cooldown = Cooldown {
        max: 6 * 3600,
        base: 60,
    };
    const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
    const TIERED: bool = true;
    type Request = Bytes;
    type Response = reqwest::Response;

    fn reason(body: String) -> String {
        crate::codex::types::ErrorEnvelope::reason(body)
    }

    fn soft_limit(&self) -> f64 {
        self.soft_utilization_limit()
    }

    async fn send(
        &self,
        token: &str,
        slot: &Slot,
        route: Route<'_>,
        req: &Self::Request,
    ) -> Result<Self::Response, SendError> {
        CodexClient::send(
            self,
            token,
            &slot.provider_account_id,
            req,
            &session_uuid(route.session_key),
        )
        .await
    }

    fn retryable_bad_request(&self, body: &str) -> bool {
        // A preview model can be enabled per account, so "not supported"
        // describes this key rather than the request, and the next
        // account may well serve it.
        body.contains("is not supported")
    }

    fn usage_from(&self, resp: &Self::Response) -> Option<AccountUsage> {
        usage_from_headers(resp.headers())
    }
}

impl Pool<CodexClient> {
    pub fn client(&self) -> &CodexClient {
        self.backend()
    }

    /// Averaged across accounts, so a client's figures do not jump when
    /// routing moves it.
    pub async fn pool_windows(&self) -> Vec<UsageWindow> {
        let mut by_name: std::collections::BTreeMap<String, (f64, usize, Option<i64>)> =
            Default::default();
        for account in self.slots.snapshot().await {
            let Some(usage) = account.usage else { continue };
            for w in usage.windows {
                let slot = by_name.entry(w.name.clone()).or_insert((0.0, 0, None));
                slot.0 += w.utilization;
                slot.1 += 1;
                slot.2 = match (slot.2, w.resets_at) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
            }
        }
        by_name
            .into_iter()
            .map(|(name, (sum, count, resets_at))| UsageWindow {
                name,
                utilization: sum / count.max(1) as f64,
                resets_at,
            })
            .collect()
    }

    /// Reads quota for every account from the usage endpoint, so idle
    /// accounts report current figures instead of whatever they last saw on
    /// a served response.
    pub async fn poll_usage(&self) {
        for slot in self.slots.list().await {
            let Ok(token) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.backend.usage(&token, &slot.provider_account_id).await {
                Ok(usage) => {
                    let windows = usage
                        .rate_limit
                        .windows()
                        .filter(|w| w.limit_window_seconds > 0)
                        .map(|w| UsageWindow {
                            name: window_name(w.limit_window_seconds / 60),
                            utilization: w.used_percent / 100.0,
                            resets_at: w.reset_at,
                        })
                        .collect::<Vec<_>>();
                    if windows.is_empty() {
                        continue;
                    }
                    self.slots
                        .note_usage(
                            &slot,
                            AccountUsage {
                                windows,
                                model_windows: Vec::new(),
                                locked: usage.rate_limit.limit_reached,
                                observed_at: 0,
                            },
                        )
                        .await;
                }
                Err(e) => tracing::debug!("usage for {}: {e}", slot.display),
            }
        }
    }

    /// Fresh (access_token, account_id) for the models listing. Trusted
    /// first, since gated models are absent from an untrusted account's
    /// catalog.
    pub async fn any_active_credentials(&self) -> Option<(String, String)> {
        for slot in self
            .ranked(Route {
                session_key: "",
                model: "",
                prefer_trusted: true,
            })
            .await
        {
            if !self.slots.try_claim(&slot).await {
                continue;
            }
            let Ok(access) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            return Some((access, slot.provider_account_id.clone()));
        }
        None
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        let (access, account_id) = self
            .any_active_credentials()
            .await
            .ok_or_else(|| "no usable account; run `slop-proxy login`".to_string())?;
        self.backend.list_models(&access, &account_id).await
    }

    /// The catalog body untouched, for relaying to a codex client verbatim.
    pub async fn models_raw(&self) -> Result<String, String> {
        let (access, account_id) = self
            .any_active_credentials()
            .await
            .ok_or_else(|| "no usable account; run `slop-proxy login`".to_string())?;
        let (status, body) = self.backend.models_raw(&access, &account_id).await?;
        if !status.is_success() {
            return Err(format!(
                "{status}: {}",
                body.chars().take(400).collect::<String>()
            ));
        }
        Ok(body)
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
        model_windows: Vec::new(),
        locked: false,
        observed_at: 0,
    })
}

/// Codex sends one session id per conversation, not per request. Deriving it
/// from the same key means upstream sees a continuing thread rather than a
/// stranger every turn.
fn session_uuid(session_key: &str) -> String {
    if session_key.is_empty() {
        return uuid::Uuid::new_v4().to_string();
    }
    let digest = hmac_sha256::Hash::hash(session_key.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Builder::from_random_bytes(bytes)
        .into_uuid()
        .to_string()
}

fn window_name(minutes: i64) -> String {
    match minutes {
        m if m % 1440 == 0 => format!("{}d", m / 1440),
        m if m % 60 == 0 => format!("{}h", m / 60),
        m => format!("{m}m"),
    }
}
