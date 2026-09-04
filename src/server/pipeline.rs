//! The steps every handler repeats between parsing its dialect and
//! answering in it.

use std::collections::VecDeque;
use std::convert::Infallible;

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
   resp.bytes().await.map_err(|e| {
      log_error(state, record.clone(), 502, "upstream_read");
      error_response(dialect, 502, "api_error", &e.to_string())
   })
}

pub const fn apply_snapshot(record: &mut UsageRecord, snap: &CapturedUsage) {
   record.input_tokens = snap.input_tokens;
   record.output_tokens = snap.output_tokens;
   record.cache_read_tokens = snap.cache_read_tokens;
   record.cache_write_tokens = snap.cache_write_tokens;
   record.reasoning_tokens = snap.reasoning_tokens;
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
            Ok(bytes) => Ok(each(bytes)),
            Err(e) => {
               capture.fail("upstream_stream_error");
               Err(e)
            },
         }
      })
      .chain(stream::once(async move {
         eof.note_upstream_eof();
         Ok(tail())
      }));
   builder
      .body(Body::from_stream(stream))
      .unwrap_or_else(|e| error_response(dialect, 502, "api_error", &e.to_string()))
}

/// Responses events rendered as another dialect's SSE. `step` gets `None`
/// once upstream closes, for whatever the dialect ends with.
pub fn translated<S>(upstream: EventStream, guard: LogGuard, step: S) -> Response
where
   S: FnMut(Option<ResponsesEvent>) -> Vec<Event> + Send + 'static,
{
   struct St<F> {
      upstream: EventStream,
      step: F,
      queue: VecDeque<Event>,
      finished: bool,
      _guard: LogGuard,
   }
   let st = St {
      upstream,
      step,
      queue: VecDeque::new(),
      finished: false,
      _guard: guard,
   };
   let stream = stream::unfold(st, |mut st| async move {
      loop {
         if let Some(ev) = st.queue.pop_front() {
            return Some((Ok::<_, Infallible>(ev), st));
         }
         if st.finished {
            return None;
         }
         let next = st.upstream.next().await;
         st.finished = next.is_none();
         st.queue.extend((st.step)(next));
      }
   });
   Sse::new(stream)
      .keep_alive(KeepAlive::default())
      .into_response()
}
