pub mod anthropic;
pub mod auth;
pub mod clientcfg;
pub mod error;
pub mod metrics;
pub mod openai;
pub mod relay;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::middleware;
use axum::routing::{get, post};
use eyre::{Result, WrapErr};

use crate::anthropic::client::AnthropicClient;
use crate::codex::client::CodexClient;
use crate::codex::models::ModelInfo;
use crate::config::Config;
use crate::db::Db;
use crate::db::usage::UsageRecord;
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

impl AppState {
    /// The catalog exactly as the codex backend sent it.
    pub async fn catalog_raw(&self) -> Option<String> {
        if let Some(cached) = self.models.get() {
            return Some(cached);
        }
        match self.codex.models_raw().await {
            Ok(body) => {
                self.models.put(body.clone());
                Some(body)
            }
            Err(e) => {
                tracing::warn!("fetching models from codex backend: {e}");
                None
            }
        }
    }

    pub async fn catalog(&self) -> Option<Vec<ModelInfo>> {
        let raw = self.catalog_raw().await?;
        match serde_json::from_str::<crate::codex::models::ModelsResponse>(&raw) {
            Ok(parsed) => Some(parsed.models),
            Err(e) => {
                tracing::warn!("parsing models response: {e}");
                None
            }
        }
    }
}

pub struct ModelCache {
    inner: Mutex<Option<(Instant, String)>>,
    ttl: Duration,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            ttl: Duration::from_secs(300),
        }
    }

    pub fn get(&self) -> Option<String> {
        let g = self.inner.lock().unwrap();
        g.as_ref()
            .filter(|(t, _)| t.elapsed() < self.ttl)
            .map(|(_, m)| m.clone())
    }

    pub fn put(&self, catalog: String) {
        *self.inner.lock().unwrap() = Some((Instant::now(), catalog));
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn serve(db: Db, cfg: Config, bind: &str) -> Result<()> {
    let codex = CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone())).await?;
    if codex.is_empty().await {
        tracing::warn!("no codex accounts in the database; run `slop-proxy login`");
    } else {
        tracing::info!("loaded {} codex account(s)", codex.len().await);
    }
    let anthropic =
        AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone())).await?;
    if anthropic.is_empty().await {
        tracing::warn!(
            "no anthropic accounts in the database; run `slop-proxy login --provider anthropic`"
        );
    } else {
        tracing::info!("loaded {} anthropic account(s)", anthropic.len().await);
    }

    let state = AppState {
        db,
        codex: Arc::new(codex),
        anthropic: Arc::new(anthropic),
        cfg: Arc::new(cfg),
        models: Arc::new(ModelCache::new()),
    };
    let reload_state = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        tick.tick().await;
        loop {
            tick.tick().await;
            if let Err(e) = reload_state.codex.reload().await {
                tracing::warn!("reloading codex accounts: {e}");
            }
            if let Err(e) = reload_state.anthropic.reload().await {
                tracing::warn!("reloading anthropic accounts: {e}");
            }
            reload_state.codex.poll_usage().await;
            reload_state.anthropic.poll_usage().await;
        }
    });

    if let Some(mbind) = state.cfg.metrics_bind.clone() {
        let mstate = state.clone();
        let listener = tokio::net::TcpListener::bind(&mbind)
            .await
            .wrap_err_with(|| format!("binding metrics listener {mbind}"))?;
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
        .wrap_err_with(|| format!("binding {bind}"))?;
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
        .route("/config/codex/auth.json", get(clientcfg::codex_auth))
        .route("/config/codex/config.toml", get(clientcfg::codex_config))
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
        record.cache_write_tokens = cap.cache_write_tokens;
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

/// A request rejected before dispatch has already spent the caller's
/// admission, so without a row it burns their limit invisibly.
pub fn log_rejected(db: &Db, auth: &auth::AuthInfo, dialect: &'static str, model: &str) {
    log_error(
        db,
        UsageRecord {
            meter_id: Some(auth.meter_id),
            token_id: Some(auth.token_id),
            user: auth.user.clone(),
            dialect,
            requested_model: model.to_string(),
            ..Default::default()
        },
        400,
        "invalid_request",
    );
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
    let mut hasher = hmac_sha256::Hash::new();
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
