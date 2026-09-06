pub mod anthropic;
pub mod auth;
pub mod clientcfg;
pub mod decompress;
pub mod error;
pub mod facts;
mod gemini;
pub mod metrics;
pub mod openai;
pub mod pipeline;
pub mod relay;
#[cfg(test)]
mod tests;

use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use eyre::Result;
use eyre::WrapErr as _;
use tokio::net::TcpListener;
use tokio::{signal, time};

use crate::codex::models::{ModelInfo, ModelsResponse};
use crate::codex::types::ResponsesRequest;
use crate::config::Config;
use crate::db::Db;
use crate::db::usage::UsageRecord;
use crate::pool::Pools;
use crate::pricing::{Prices, Tokens};
use crate::translate::UsageCapture;

#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
   pub db: Db,
   pub cfg: Config,
   pub prices: Prices,
   pub models: ModelCache,
   pub pools: Pools,
}

impl Deref for AppState {
   type Target = Inner;
   fn deref(&self) -> &Inner {
      &self.0
   }
}

impl AppState {
   /// The catalog exactly as the codex backend sent it.
   pub async fn catalog_raw(&self) -> Option<String> {
      if let Some(cached) = self.models.get() {
         return Some(cached);
      }
      match self.pools.codex.models_raw().await {
         Ok(body) => {
            self.models.put(body.clone());
            Some(body)
         },
         Err(err) => {
            tracing::warn!("fetching models from codex backend: {err}");
            None
         },
      }
   }

   pub async fn catalog(&self) -> Option<Vec<ModelInfo>> {
      let raw = self.catalog_raw().await?;
      match serde_json::from_str::<ModelsResponse>(&raw) {
         Ok(parsed) => Some(parsed.models),
         Err(err) => {
            tracing::warn!("parsing models response: {err}");
            None
         },
      }
   }
}

pub struct ModelCache {
   inner: Mutex<Option<(Instant, String)>>,
   ttl: Duration,
}

impl ModelCache {
   pub const fn new() -> Self {
      Self {
         inner: Mutex::new(None),
         ttl: Duration::from_mins(5),
      }
   }

   pub fn get(&self) -> Option<String> {
      let guard = self.inner.lock().unwrap();
      guard
         .as_ref()
         .filter(|&&(ref tick, _)| tick.elapsed() < self.ttl)
         .map(|&(_, ref model)| model.clone())
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
   let pools = Pools::load(&db, &cfg).await?;
   let prices = Prices::new(&cfg.db_path, cfg.pricing.url.clone());
   prices.load().await;
   let state = AppState(Arc::new(Inner {
      db,
      cfg,
      prices,
      models: ModelCache::new(),
      pools,
   }));
   price_history(&state).await;
   let price_state = state.clone();
   tokio::spawn(async move {
      let mut tick = time::interval(Duration::from_hours(12));
      tick.tick().await;
      loop {
         tick.tick().await;
         if price_state.prices.refresh().await.is_ok() {
            price_history(&price_state).await;
         }
      }
   });
   let reload_state = state.clone();
   tokio::spawn(async move {
      let mut tick = time::interval(Duration::from_mins(1));
      tick.tick().await;
      loop {
         tick.tick().await;
         reload_state.pools.reload().await;
         reload_state.pools.poll_usage().await;
      }
   });

   if let Some(mbind) = state.cfg.metrics_bind.clone() {
      let mstate = state.clone();
      let listener = TcpListener::bind(&mbind)
         .await
         .wrap_err_with(|| format!("binding metrics listener {mbind}"))?;
      tracing::info!("metrics on http://{mbind}/metrics");
      tokio::spawn(async move {
         let app = Router::new()
            .route("/metrics", get(metrics::metrics))
            .with_state(mstate);
         if let Err(err) = axum::serve(listener, app).await {
            tracing::error!("metrics listener failed: {err}");
         }
      });
   }
   let app = router(state.clone());

   let listener = TcpListener::bind(bind)
      .await
      .wrap_err_with(|| format!("binding {bind}"))?;
   tracing::info!("listening on http://{bind}");
   axum::serve(listener, app)
      .with_graceful_shutdown(async {
         let _ = signal::ctrl_c().await;
      })
      .await?;
   state.db.flush().await
}

/// Costs the rows that were logged before their model had a price, so a table
/// that arrives late still bills the history behind it.
async fn price_history(state: &AppState) {
   let rows = match state.db.unpriced_usage().await {
      Ok(rows) => rows,
      Err(err) => {
         tracing::warn!("reading unpriced usage: {err}");
         return;
      },
   };
   let table = state.prices.table();
   let priced: Vec<_> = rows
      .iter()
      .map(|row| {
         (
            row.id,
            state.prices.cost(&row.model, row.tokens),
            table.list_cost(&row.model, row.tokens),
         )
      })
      .filter(|&(_, ref cost, ref list_cost)| *cost > 0.0_f64 || *list_cost > 0.0_f64)
      .collect();
   if priced.is_empty() {
      return;
   }
   match state.db.price_usage(&priced).await {
      Ok(()) => tracing::info!("priced {} earlier request(s)", priced.len()),
      Err(err) => tracing::warn!("pricing earlier requests: {err}"),
   }
}

pub fn router(state: AppState) -> Router {
   Router::new()
      .route("/v1/messages", post(anthropic::messages))
      .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
      .route("/v1/chat/completions", post(openai::chat_completions))
      .route("/v1/models", get(openai::models))
      .route("/v1beta/models/{spec}", post(gemini::native))
      .route("/config/codex/auth.json", get(clientcfg::codex_auth))
      .route("/config/codex/config.toml", get(clientcfg::codex_config))
      .route(
         "/v1/responses",
         post(openai::responses_passthrough).get(openai::responses_upgrade_required),
      )
      .layer(middleware::from_fn(decompress::zstd_requests))
      .layer(DefaultBodyLimit::max(decompress::MAX_BODY))
      .layer(middleware::from_fn_with_state(
         state.clone(),
         auth::require_token,
      ))
      .with_state(state)
}

/// Reasoning tokens are a subset of the output the provider already billed,
/// so they are deliberately absent here.
const fn billable(record: &UsageRecord) -> Tokens {
   Tokens {
      input: record.input_tokens,
      output: record.output_tokens,
      cache_read: record.cache_read_tokens,
      cache_write: record.cache_write_tokens,
   }
}

/// Logs the request on drop, so client disconnects mid-stream still get a row.
pub struct LogGuard {
   state: AppState,
   capture: UsageCapture,
   record: UsageRecord,
   start: Instant,
}

impl LogGuard {
   /// `started` is the handler's own clock. Timing from here instead would
   /// begin after the upstream response headers, hiding the wait that time
   /// to first byte exists to measure.
   pub const fn new(
      state: AppState,
      capture: UsageCapture,
      record: UsageRecord,
      started: Instant,
   ) -> Self {
      Self {
         state,
         capture,
         record,
         start: started,
      }
   }
}

impl Drop for LogGuard {
   fn drop(&mut self) {
      let mut record = self.record.clone();
      let cap = self.capture.snapshot();
      pipeline::apply_snapshot(&mut record, &cap);
      record.error_kind = cap.error_kind;
      record.response_bytes = cap.response_bytes;
      let finished = cap.completed || cap.stop_reason.is_some();
      record.stop_reason = cap.stop_reason.unwrap_or_default();
      record.tools_called = cap.tools_called.join(",");
      record.ttft_ms = cap
         .first_byte_at
         .map(|tick| tick.saturating_duration_since(self.start).as_millis() as i64);
      if !finished && record.error_kind.is_none() {
         record.error_kind = Some(if cap.upstream_eof {
            "upstream_eof".into()
         } else if record.status >= 400 {
            "upstream_rejected".into()
         } else {
            "client_disconnect".into()
         });
      }
      if !finished {
         tracing::warn!(
             user = %record.user,
             model = %record.upstream_model,
             kind = record.error_kind.as_deref().unwrap_or("?"),
             last_event = cap.last_event.as_deref().unwrap_or("none"),
             upstream_head = cap.upstream_head.as_deref().unwrap_or(""),
             after_ms = self.start.elapsed().as_millis() as i64,
             "stream ended without usage"
         );
      }
      record.duration_ms = Some(self.start.elapsed().as_millis() as i64);
      price(&self.state.prices, &mut record);
      write_usage(&self.state.db, record);
   }
}

/// A request rejected before dispatch has already spent the caller's
/// admission, so without a row it burns their limit invisibly.
pub fn log_rejected(state: &AppState, auth: &auth::AuthInfo, dialect: &'static str, model: &str) {
   log_error(
      state,
      UsageRecord {
         meter_id: Some(auth.meter_id),
         token_id: Some(auth.token_id),
         user: auth.user.clone(),
         dialect,
         requested_model: model.to_owned(),
         ..Default::default()
      },
      400,
      "invalid_request",
   );
}

/// A rejected request served no tokens, so it is written without consulting
/// the price table.
pub fn log_error(state: &AppState, mut record: UsageRecord, status: i64, kind: &str) {
   record.status = status;
   record.error_kind = Some(kind.to_owned());
   write_usage(&state.db, record);
}

/// Both write paths price a row
fn price(prices: &Prices, record: &mut UsageRecord) {
   let billable = billable(record);
   record.cost_usd = prices.cost(&record.upstream_model, billable);
   record.list_cost_usd = prices.table().list_cost(&record.upstream_model, billable);
}

pub fn log_usage(state: &AppState, mut record: UsageRecord) {
   price(&state.prices, &mut record);
   write_usage(&state.db, record);
}

fn write_usage(db: &Db, record: UsageRecord) {
   if let Err(err) = db.enqueue_usage(record) {
      tracing::error!("writing usage log failed {err}");
   }
}

/// Stable per-conversation cache key so upstream prompt caching can kick in.
pub fn cache_key(user: &str, req: &ResponsesRequest) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   hasher.update(user.as_bytes());
   hasher.update(&req.instructions);
   if let Some(first) = req.input.first() {
      hasher.update(serde_json::to_string(first).unwrap_or_default());
   }
   let digest = hasher.finalize();
   let mut bytes = [0_u8; 16];
   bytes.copy_from_slice(&digest[..16]);
   uuid::Builder::from_random_bytes(bytes)
      .into_uuid()
      .to_string()
}

#[cfg(test)]
mod end_reason_tests {
   use crate::translate::UsageCapture;

   #[test]
   fn the_two_ways_a_stream_dies_are_distinguishable() {
      let cut_by_client = UsageCapture::default();
      cut_by_client.note_event("response.output_text.delta");
      let snap = cut_by_client.snapshot();
      assert!(!snap.completed && !snap.upstream_eof);
      assert_eq!(
         snap.last_event.as_deref(),
         Some("response.output_text.delta")
      );

      let cut_by_upstream = UsageCapture::default();
      cut_by_upstream.note_upstream_eof();
      let upstream_snap = cut_by_upstream.snapshot();
      assert!(!upstream_snap.completed && upstream_snap.upstream_eof);
   }
}
