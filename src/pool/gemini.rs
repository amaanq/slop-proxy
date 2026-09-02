use std::sync::Arc;

use serde_json::Value;

use super::{PoolError, Slot, Slots};
use crate::db::Db;
use crate::gemini::client::{GeminiClient, GeminiResponse};
use crate::provider::Provider;
use crate::upstream::SendError;

/// What to send upstream. A caller already speaking the native dialect skips
/// the translation entirely, but shares the retry and cooldown policy.
#[derive(Clone, Copy)]
pub enum Call<'a> {
    OpenAi(&'a Value),
    Native {
        model: &'a str,
        action: &'a str,
        query: Option<&'a str>,
        body: &'a Value,
    },
}

/// Google answers a 429 with no retry-after and refills a token bucket rather
/// than clearing a window, so the wait is measured: exhausting one key and
/// probing it back took 20s, 27s and 37s.
const RATE_LIMIT_COOLDOWN_SECS: i64 = 40;

/// Google caches per key, so a conversation that moves keys re-sends its whole
/// prefix as fresh input, which both re-bills it and spends more of the budget
/// that caused the failover. Cache hit rate is 84.6% here against 96.9% on
/// codex. Matching the cooldown means one wait covers exactly one backoff.
const STICKY_WAIT_SECS: i64 = RATE_LIMIT_COOLDOWN_SECS;

/// Session-sticky pool over Gemini accounts.
pub struct GeminiPool {
    slots: Slots,
    client: GeminiClient,
    soft_limit: f64,
}

impl GeminiPool {
    pub async fn load(db: Db, client: GeminiClient) -> eyre::Result<Self> {
        Ok(Self {
            soft_limit: client.soft_utilization_limit(),
            slots: Slots::load(db, Provider::Gemini).await?,
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

    async fn ranked(&self, session_key: &str) -> Vec<Arc<Slot>> {
        let mut scored = Vec::new();
        for slot in self.slots.list().await {
            let band = self.slots.band(&slot, self.soft_limit).await;
            let score = super::rendezvous_score(session_key, slot.id);
            scored.push((score, band, slot));
        }
        scored.sort_by_key(|(score, band, _)| (*band, std::cmp::Reverse(*score)));
        scored.into_iter().map(|(_, _, s)| s).collect()
    }

    /// The first account that answers. Every key sees the same catalog, so
    /// there is nothing to merge across accounts.
    pub async fn models(&self) -> Vec<String> {
        for slot in self.slots.list().await {
            let Ok(key) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };
            match self.client.models(&key, slot.http_referer.as_deref()).await {
                Ok(ids) => return ids,
                Err(e) => tracing::debug!("models for {}: {e}", slot.display),
            }
        }
        Vec::new()
    }

    pub async fn execute(
        &self,
        call: Call<'_>,
        session_key: &str,
    ) -> Result<(i64, GeminiResponse), PoolError> {
        let ranked = self.ranked(session_key).await;
        if ranked.is_empty() {
            return Err(PoolError::NoAccounts(Provider::Gemini));
        }
        let mut last_err = Option::<SendError>::None;
        let mut attempts = 0;

        if let Some(preferred) = ranked.first() {
            let left = self.slots.cooldown_left(preferred).await;
            if (1..=STICKY_WAIT_SECS).contains(&left) {
                tracing::debug!(
                    account = %preferred.display,
                    left,
                    "waiting for the sticky gemini key rather than losing its cache"
                );
                tokio::time::sleep(std::time::Duration::from_secs(left as u64 + 1)).await;
            }
        }

        for slot in ranked {
            if attempts >= 3 {
                break;
            }
            if !self.slots.try_claim(&slot).await {
                continue;
            }
            attempts += 1;
            let Ok(key) = self.slots.fresh_token(&slot, false).await else {
                continue;
            };

            let referer = slot.http_referer.as_deref();
            let sent = match call {
                Call::OpenAi(body) => self.client.send(&key, referer, body).await,
                Call::Native {
                    model,
                    action,
                    query,
                    body,
                } => self
                    .client
                    .send_native(&key, referer, model, action, query, body)
                    .await
                    .map(|response| GeminiResponse {
                        response,
                        protocol: crate::gemini::client::GeminiProtocol::Native,
                    }),
            };
            match sent {
                Ok(resp) => {
                    self.slots.mark_ok(&slot).await;
                    return Ok((slot.id, resp));
                }
                // A static key cannot be refreshed into a working one, so a
                // rejected key sits out rather than retrying in place.
                Err(SendError::Auth(text)) => {
                    self.slots.cool(&slot, 15 * 60, "key rejected").await;
                    last_err = Some(SendError::Auth(text));
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    self.slots
                        .cool_rate_limited(&slot, retry_after, 3600, RATE_LIMIT_COOLDOWN_SECS)
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
