//! The steps every handler repeats between parsing its dialect and
//! answering in it.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::time::Instant;

use axum::body::HttpBody as _;
use axum::body::{Body, Bytes};
use axum::http::response::Builder;
use axum::response::IntoResponse as _;
use axum::response::Response;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt as _;
use futures_util::stream;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response, pool_error_status};
use super::facts::RequestFacts;
use super::{AppState, LogGuard, log_error};
use crate::codex::sse::EventStream;
use crate::codex::types::ResponsesEvent;
use crate::db::usage::UsageRecord;
use crate::pool::PoolError;
use crate::provider::Provider;
use crate::translate::{CapturedUsage, UsageCapture};

pub fn record(
   auth: &AuthInfo,
   dialect: &'static str,
   provider: Provider,
   requested_model: String,
   upstream_model: String,
   facts: RequestFacts,
) -> UsageRecord {
   UsageRecord {
      meter_id: Some(auth.meter_id),
      token_id: Some(auth.token_id),
      user: auth.user.clone(),
      provider: Some(provider),
      dialect,
      requested_model,
      upstream_model,
      status: 200,
      turn_index: facts.turn_index,
      tools_declared: facts.tools_declared,
      thinking_budget: facts.thinking_budget,
      image_count: facts.image_count,
      request_bytes: facts.request_bytes,
      ..Default::default()
   }
}

pub fn dispatch_failed(
   state: &AppState,
   record: UsageRecord,
   dialect: Dialect,
   err: PoolError,
) -> Response {
   log_error(state, record, pool_error_status(&err), "pool");
   pool_error_response(dialect, &state.cfg.models, err)
}

pub async fn read_body(
   state: &AppState,
   record: &UsageRecord,
   dialect: Dialect,
   resp: reqwest::Response,
) -> Result<Bytes, Response> {
   resp.bytes().await.map_err(|err| {
      log_error(state, record.clone(), 502, "upstream_read");
      error_response(dialect, 502, "api_error", &err.to_string())
   })
}

pub fn apply_snapshot(record: &mut UsageRecord, snap: &CapturedUsage, started: Instant) {
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   record.ttft_ms = snap
      .first_byte_at
      .map(|tick| tick.saturating_duration_since(started).as_millis() as i64);
   record.input_tokens = snap.input_tokens;
   record.output_tokens = snap.output_tokens;
   record.cache_read_tokens = snap.cache_read_tokens;
   record.cache_write_tokens = snap.cache_write_tokens;
   record.reasoning_tokens = snap.reasoning_tokens;
   if snap.error_kind.is_some() {
      record.error_kind.clone_from(&snap.error_kind);
   }
   if let Some(reason) = snap.stop_reason.as_ref() {
      record.stop_reason.clone_from(reason);
   }
   record.tools_called = snap.tools_called.join(",");
   record.response_bytes = snap.response_bytes;
}

pub fn logged_json<T>(state: &AppState, mut record: UsageRecord, value: T) -> Response
where
   T: serde::Serialize,
{
   let response = axum::Json(value).into_response();
   record.status = i64::from(response.status().as_u16());
   record.response_bytes = response.body().size_hint().exact().unwrap_or(0) as i64;
   super::log_usage(state, record);
   response
}

/// Upstream bytes to the client, `each` seeing every chunk on the way (and
/// free to rewrite it), `tail` appended once upstream closes.
pub fn relayed<F, T>(
   builder: Builder,
   resp: reqwest::Response,
   guard: LogGuard,
   capture: UsageCapture,
   dialect: Dialect,
   mut each: F,
   tail: T,
) -> Response
where
   F: FnMut(Bytes) -> Bytes + Send + 'static,
   T: FnOnce() -> Bytes + Send + 'static,
{
   let eof = capture.clone();
   let stream = resp
      .bytes_stream()
      .map(move |item| {
         let _ = &guard;
         match item {
            Ok(bytes) => {
               let bytes = each(bytes);
               capture.note_bytes(bytes.len());
               Ok(bytes)
            },
            Err(err) => {
               capture.fail("upstream_stream_error");
               Err(err)
            },
         }
      })
      .chain(stream::once(async move {
         eof.note_upstream_eof();
         let bytes = tail();
         eof.note_bytes(bytes.len());
         Ok(bytes)
      }));
   builder
      .body(Body::from_stream(stream))
      .unwrap_or_else(|err| error_response(dialect, 502, "api_error", &err.to_string()))
}

/// Responses events rendered as another dialect's SSE. `step` gets `None`
/// once upstream closes, for whatever the dialect ends with.
pub fn translated<S>(upstream: EventStream, guard: LogGuard, step: S) -> Response
where
   S: FnMut(Option<ResponsesEvent>) -> Vec<Event> + Send + 'static,
{
   struct State<F> {
      upstream: EventStream,
      step: F,
      queue: VecDeque<Event>,
      finished: bool,
      _guard: LogGuard,
   }
   let state = State {
      upstream,
      step,
      queue: VecDeque::new(),
      finished: false,
      _guard: guard,
   };
   let stream = stream::unfold(state, |mut state| async move {
      loop {
         if let Some(event) = state.queue.pop_front() {
            return Some((Ok::<_, Infallible>(event), state));
         }
         if state.finished {
            return None;
         }
         let next = state.upstream.next().await;
         state.finished = next.is_none();
         state.queue.extend((state.step)(next));
      }
   });
   Sse::new(stream)
      .keep_alive(KeepAlive::default())
      .into_response()
}
