pub mod anthropic;
pub mod auth;
pub mod error;
pub mod metrics;
pub mod openai;
pub mod relay;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use sha2::{Digest, Sha256};

use crate::anthropic::client::AnthropicClient;
use crate::codex::client::CodexClient;
use crate::codex::models::ModelInfo;
use crate::config::Config;
use crate::db::usage::UsageRecord;
use crate::db::Db;
use crate::pool::anthropic::AnthropicPool;
use crate::pool::codex::CodexPool;
use crate::translate::UsageCapture;

#[derive(Clone)]
pub struct AppState {
    pub db: Db,
    pub codex: Arc<CodexPool>,
    pub anthropic: Arc<AnthropicPool>,
    pub cfg: Arc<Config>,
    pub models: Arc<ModelCache>,
}

pub struct ModelCache {
    inner: Mutex<Option<(Instant, Vec<ModelInfo>)>>,
    ttl: Duration,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            ttl: Duration::from_secs(300),
        }
    }

    pub fn get(&self) -> Option<Vec<ModelInfo>> {
        let g = self.inner.lock().unwrap();
        g.as_ref()
            .filter(|(t, _)| t.elapsed() < self.ttl)
            .map(|(_, m)| m.clone())
    }

    pub fn put(&self, models: Vec<ModelInfo>) {
        *self.inner.lock().unwrap() = Some((Instant::now(), models));
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn serve(db: Db, cfg: Config, bind: &str) -> Result<()> {
    let codex = CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone())).await?;
    if codex.is_empty() {
        tracing::warn!("no codex accounts in the database; run `slop-proxy login`");
    } else {
        tracing::info!("loaded {} codex account(s)", codex.len());
    }
    let anthropic =
        AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone())).await?;
    if anthropic.is_empty() {
        tracing::warn!(
            "no anthropic accounts in the database; run `slop-proxy login --provider anthropic`"
        );
    } else {
        tracing::info!("loaded {} anthropic account(s)", anthropic.len());
    }

    let state = AppState {
        db,
        codex: Arc::new(codex),
        anthropic: Arc::new(anthropic),
        cfg: Arc::new(cfg),
        models: Arc::new(ModelCache::new()),
    };
    if let Some(mbind) = state.cfg.metrics_bind.clone() {
        let mstate = state.clone();
        let listener = tokio::net::TcpListener::bind(&mbind)
            .await
            .with_context(|| format!("binding metrics listener {mbind}"))?;
        tracing::info!("metrics on http://{mbind}/metrics");
        tokio::spawn(async move {
            let app = Router::new()
                .route("/metrics", get(metrics::metrics))
                .with_state(mstate);
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("metrics listener failed: {e}");
            }
        });
    }
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding {bind}"))?;
    tracing::info!("listening on http://{bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::models))
        .route("/v1/responses", post(openai::responses_passthrough))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_token,
        ))
        .with_state(state)
}

/// Logs the request on drop, so client disconnects mid-stream still get a row.
pub struct LogGuard {
    db: Db,
    capture: UsageCapture,
    record: UsageRecord,
    start: Instant,
}

impl LogGuard {
    pub fn new(db: Db, capture: UsageCapture, record: UsageRecord) -> Self {
        Self {
            db,
            capture,
            record,
            start: Instant::now(),
        }
    }
}

impl Drop for LogGuard {
    fn drop(&mut self) {
        let mut record = self.record.clone();
        let cap = self.capture.snapshot();
        record.input_tokens = cap.input_tokens;
        record.output_tokens = cap.output_tokens;
        record.cache_read_tokens = cap.cache_read_tokens;
        record.reasoning_tokens = cap.reasoning_tokens;
        record.error_kind = cap.error_kind;
        if !cap.completed && record.error_kind.is_none() {
            record.error_kind = Some("client_disconnect".into());
        }
        record.duration_ms = Some(self.start.elapsed().as_millis() as i64);
        let db = self.db.clone();
        tokio::spawn(async move {
            if let Err(e) = db.log_usage(&record).await {
                tracing::error!("writing usage log: {e}");
            }
        });
    }
}

pub fn log_error(db: &Db, mut record: UsageRecord, status: i64, kind: &str) {
    record.status = status;
    record.error_kind = Some(kind.to_string());
    log_usage(db, record);
}

pub fn log_usage(db: &Db, record: UsageRecord) {
    let db = db.clone();
    tokio::spawn(async move {
        if let Err(e) = db.log_usage(&record).await {
            tracing::error!("writing usage log: {e}");
        }
    });
}

/// Stable per-conversation cache key so upstream prompt caching can kick in.
pub fn cache_key(user: &str, req: &crate::codex::types::ResponsesRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(user.as_bytes());
    hasher.update(&req.instructions);
    if let Some(first) = req.input.first() {
        hasher.update(serde_json::to_string(first).unwrap_or_default());
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Builder::from_random_bytes(bytes)
        .into_uuid()
        .to_string()
}
