pub mod anthropic;
pub mod auth;
mod cache;
pub mod chat;
pub mod clientcfg;
pub mod copilot;
pub mod decompress;
mod desktop;
pub mod error;
pub mod facts;
pub mod handoff;
mod gemini;
pub mod metrics;
pub mod openai;
pub mod pipeline;
mod profile;
pub mod relay;
#[cfg(test)]
mod tests;

use std::mem;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use misstep::Result;
use tokio::signal::unix::{self, SignalKind};
use tokio::sync::watch;
use tokio::{signal, time};

use crate::codex::models::ModelsResponse;
use crate::codex::types::ResponsesRequest;
use crate::config::Config;
use crate::db::Db;
use crate::db::usage::UsageRecord;
use crate::server::handoff::{Inherited, Predecessor, Socket};
use crate::pool::Pools;
use crate::pricing::{Prices, Tokens};
use crate::translate::UsageCapture;

#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
   pub db: Db,
   pub cfg: Config,
   pub prices: Prices,
   pub pools: Pools,
}

impl Deref for AppState {
   type Target = Inner;
   fn deref(&self) -> &Inner {
      &self.0
   }
}

impl AppState {
   pub async fn catalog(&self, user: &str, pinned_account: Option<i64>) -> Option<ModelsResponse> {
      match self.pools.codex.catalog(user, pinned_account).await {
         Ok(catalog) => Some(catalog),
         Err(err) => {
            tracing::warn!("fetching models from codex backend: {err}");
            None
         },
      }
   }
}

/// How long a draining process keeps serving its open streams after a
/// successor or a stop has taken over the sockets.
const DRAIN_LIMIT: Duration = Duration::from_mins(15);

pub async fn serve(db: Db, cfg: Config, bind: &str) -> Result<()> {
   // Bound before the slow startup below, so connections queue in the
   // kernel meanwhile rather than being refused.
   let mut inherited = Inherited::take()?;
   let listener = inherited.listen(Socket::Api, bind).await?;
   let metrics_listener = match cfg.metrics_bind.as_deref() {
      Some(addr) => Some(inherited.listen(Socket::Metrics, addr).await?),
      None => None,
   };

   let pools = Pools::load(&db, &cfg).await?;
   pools.refresh_catalogs().await;
   let prices = Prices::new(&cfg.db_path, cfg.pricing.url.clone());
   prices.load().await;
   let state = AppState(Arc::new(Inner {
      db,
      cfg,
      prices,
      pools,
   }));
   let (start_drain, drain) = watch::channel(false);
   let mut tasks = Vec::new();

   let price_state = state.clone();
   tasks.push(tokio::spawn(async move {
      price_history(&price_state).await;
      let mut tick = time::interval(Duration::from_hours(12));
      tick.tick().await;
      loop {
         tick.tick().await;
         if price_state.prices.refresh().await.is_ok() {
            price_history(&price_state).await;
         }
      }
   }));
   let catalog_state = state.clone();
   tasks.push(tokio::spawn(async move {
      let mut tick = time::interval(Duration::from_mins(5));
      tick.tick().await;
      loop {
         tick.tick().await;
         catalog_state.pools.refresh_catalogs().await;
      }
   }));
   let reload_state = state.clone();
   tasks.push(tokio::spawn(async move {
      let mut tick = time::interval(Duration::from_mins(1));
      tick.tick().await;
      loop {
         tick.tick().await;
         reload_state.pools.reload().await;
         reload_state.pools.refresh_egresses(&reload_state.cfg);
         reload_state.pools.poll_usage().await;
      }
   }));

   let mut sockets = vec![(Socket::Api, &listener)];
   sockets.extend(metrics_listener.as_ref().map(|metrics| (Socket::Metrics, metrics)));
   handoff::accept(&sockets, drain.clone())?;
   if let Some(metrics_socket) = metrics_listener {
      tracing::info!("metrics on http://{}/metrics", metrics_socket.local_addr()?);
      tasks.extend(metrics::spawn(state.clone(), metrics_socket, drain.clone()));
   }
   tracing::info!("listening on http://{}", listener.local_addr()?);
   handoff::notify_ready();
   if let Some(predecessor) = Predecessor::from_env() {
      predecessor.relieve();
   }
   tokio::spawn(async move {
      wait_for_stop().await;
      tracing::info!("draining open connections");
      let _ = start_drain.send(true);
   });

   let mut stopped = drain.clone();
   let server = axum::serve(listener, router(state.clone()))
      .with_graceful_shutdown(async move {
         let _ = stopped.wait_for(|&draining| draining).await;
      })
      .into_future();
   tokio::pin!(server);
   let mut limit = drain;
   tokio::select! {
      outcome = &mut server => outcome?,
      () = async {
         let _ = limit.wait_for(|&draining| draining).await;
         for task in &tasks {
            task.abort();
         }
         time::sleep(DRAIN_LIMIT).await;
      } => tracing::warn!("dropping connections still open after {}s", DRAIN_LIMIT.as_secs()),
   }
   state.db.flush().await
}

async fn wait_for_stop() {
   let Ok(mut term) = unix::signal(SignalKind::terminate()) else {
      let _ = signal::ctrl_c().await;
      return;
   };
   tokio::select! {
      _ = term.recv() => {},
      _ = signal::ctrl_c() => {},
   }
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
   let priced = rows
      .iter()
      .filter_map(|row| {
         let cost = state.prices.cost(&row.model, row.tokens);
         let list_cost = table.list_cost(&row.model, row.tokens);
         let billable = cost > 0.0_f64 || list_cost > 0.0_f64;
         let changed =
            cost.to_bits() != row.cost.to_bits() || list_cost.to_bits() != row.list_cost.to_bits();
         (billable && changed).then_some((row.id, cost, list_cost))
      })
      .collect::<Vec<_>>();
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
      .route("/v1/alpha/search", post(openai::search))
      .route("/v1/cache/{session}", get(cache::status))
      .route("/v1beta/models", get(gemini::models))
      .route("/v1beta/models/{spec}", post(gemini::native))
      .route("/config/codex/auth.json", get(clientcfg::codex_auth))
      .route("/config/codex/config.toml", get(clientcfg::codex_config))
      .route("/backend-api/aura/site_status", get(openai::backend_get))
      .merge(desktop::routes())
      .merge(profile::routes())
      .route(
         "/v1/responses",
         post(openai::responses_passthrough).get(openai::websocket::responses),
      )
      .layer(middleware::from_fn(decompress::zstd_requests))
      .layer(DefaultBodyLimit::max(decompress::MAX_BODY))
      .layer(middleware::from_fn_with_state(
         state.clone(),
         auth::require_token,
      ))
      .route("/oauth/token", post(clientcfg::refresh))
      .with_state(state)
}

/// Logs the request on drop, so client disconnects mid-stream still get a row.
pub struct LogGuard {
   state: AppState,
   capture: UsageCapture,
   record: UsageRecord,
   started: Instant,
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
         started,
      }
   }
}

impl Drop for LogGuard {
   fn drop(&mut self) {
      let mut record = mem::take(&mut self.record);
      let cap = self.capture.snapshot();
      pipeline::apply_snapshot(&mut record, &cap, self.started);
      let finished = cap.completed || cap.stop_reason.is_some();
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
             after_ms = self.started.elapsed().as_millis() as i64,
             idle_ms = cap.last_byte_at.map_or(-1, |last| last.elapsed().as_millis() as i64),
             egress = cap.egress.map_or(-1, |index| index as i64),
             "stream ended without usage"
         );
      }
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
         meter_id: auth.meter_id,
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

fn price(prices: &Prices, record: &mut UsageRecord) {
   // Reasoning tokens are a subset of the output the provider already billed,
   // so they are deliberately absent here.
   let billable = Tokens {
      input: record.input_tokens,
      output: record.output_tokens,
      cache_read: record.cache_read_tokens,
      cache_write: record.cache_write_tokens,
   };
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
      hasher.update(serde_json::to_string(first).expect("a Value always serializes"));
   }
   let digest = hasher.finalize();
   let mut bytes = [0_u8; 16];
   bytes.copy_from_slice(&digest[..16]);
   uuid::Builder::from_random_bytes(bytes)
      .into_uuid()
      .to_string()
}
