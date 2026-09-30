//! The steps every handler repeats between parsing its dialect and
//! answering in it.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::time::{Duration, Instant};

use axum::body::HttpBody as _;
use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use axum::http::response::Builder;
use axum::response::IntoResponse as _;
use axum::response::Response;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt as _;
use futures_util::stream;
use serde::Serialize;
use tokio::time::timeout;

use crate::codex::sse::EventStream;
use crate::codex::types::{ResponsesEvent, ResponsesRequest};
use crate::db::usage::UsageRecord;
use crate::egress::egress_of;
use crate::pool::{PoolError, Route, Served};
use crate::provider::Provider;
use crate::server::auth::AuthInfo;
use crate::server::error::{Dialect, error_response, pool_error_response};
use crate::server::facts::RequestFacts;
use crate::server::relay::forwarded_response;
use crate::server::{AppState, LogGuard, log_error, log_rejected, log_usage};
use crate::translate::{Aggregated, CapturedUsage, StopKind, UsageCapture, aggregate};

pub fn record(
   auth: &AuthInfo,
   dialect: &'static str,
   provider: Provider,
   requested_model: String,
   upstream_model: String,
   facts: RequestFacts,
) -> UsageRecord {
   UsageRecord {
      meter_id: auth.meter_id,
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

pub fn admit(
   state: &AppState,
   auth: &AuthInfo,
   dialect: Dialect,
   endpoint: &'static str,
   requested: &str,
   model: &str,
) -> Result<Provider, Box<Response>> {
   if state.cfg.models.blocked(model) {
      log_rejected(state, auth, endpoint, requested);
      return Err(Box::new(error_response(
         dialect,
         StatusCode::FORBIDDEN,
         "permission_error",
         &format!("{model} is blocked on this proxy"),
      )));
   }
   let provider = state.cfg.models.route(model);
   if !auth.limits.may_use(provider) {
      log_rejected(state, auth, endpoint, requested);
      // Names the provider the token lacks, so a scoped key does not read as
      // the model being broken for everyone.
      return Err(Box::new(error_response(
         dialect,
         StatusCode::FORBIDDEN,
         "permission_error",
         &format!(
            "this token is not scoped to the {} backend",
            provider.as_str()
         ),
      )));
   }
   Ok(provider)
}

pub fn dispatch_failed(
   state: &AppState,
   mut record: UsageRecord,
   dialect: Dialect,
   err: PoolError,
) -> Response {
   record.attempts = i64::from(err.attempts());
   let kind = match err {
      PoolError::NoAccounts(_) => "pool_no_accounts",
      PoolError::AllCoolingDown { .. } => "pool_cooling_down",
      PoolError::BadRequest { .. } => "pool_bad_request",
      PoolError::Upstream(_) => "pool_upstream",
   };
   let response = pool_error_response(dialect, &state.cfg.models, err);
   log_error(state, record, i64::from(response.status().as_u16()), kind);
   response
}

pub async fn read_body(
   state: &AppState,
   record: &UsageRecord,
   dialect: Dialect,
   resp: reqwest::Response,
) -> Result<Bytes, Response> {
   resp.bytes().await.map_err(|err| {
      log_error(state, record.clone(), 502, "upstream_read");
      error_response(
         dialect,
         StatusCode::BAD_GATEWAY,
         "api_error",
         &err.to_string(),
      )
   })
}

pub fn respond(builder: Builder, dialect: Dialect, body: Body) -> Response {
   builder.body(body).unwrap_or_else(|err| {
      error_response(
         dialect,
         StatusCode::BAD_GATEWAY,
         "api_error",
         &err.to_string(),
      )
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
   T: Serialize,
{
   let response = axum::Json(value).into_response();
   record.status = i64::from(response.status().as_u16());
   record.response_bytes = response.body().size_hint().exact().unwrap_or(0) as i64;
   log_usage(state, record);
   response
}

/// Reads usage out of a reply relayed in the dialect it arrived in.
pub trait Scan: Send + 'static {
   const DIALECT: Dialect;
   const REJECTED: &'static str;

   fn chunk(&mut self, bytes: Bytes) -> Bytes;

   fn body(&mut self, bytes: Bytes) -> Result<Bytes, String>;

   fn tail(&mut self) -> Bytes {
      Bytes::new()
   }

   fn head(&mut self) -> Option<Bytes> {
      None
   }
}

pub async fn forward<S, F>(
   state: AppState,
   mut record: UsageRecord,
   served: Result<Served<reqwest::Response>, PoolError>,
   headers: Vec<(String, String)>,
   started: Instant,
   streaming: bool,
   scan: F,
) -> Response
where
   S: Scan,
   F: FnOnce(UsageCapture) -> S + Send,
{
   let served = match served {
      Ok(served) => served,
      Err(err) => return dispatch_failed(&state, record, S::DIALECT, err),
   };
   let resp = served.response;
   record.account_id = served.account_id;
   record.attempts = i64::from(served.attempts);
   record.status = i64::from(resp.status().as_u16());
   let mut builder = forwarded_response(&resp);
   for (name, value) in headers {
      builder = builder.header(name, value);
   }
   let capture = UsageCapture::default();
   if let Some(index) = egress_of(&resp) {
      capture.note_egress(index);
   }
   let mut scan = scan(capture.clone());

   // A non-2xx carries no SSE frames, so the usage scanner would log a
   // phantom `client_disconnect` and drop the body.
   if !resp.status().is_success() {
      let mut bytes = resp.bytes().await.unwrap_or_default();
      if let Some(head) = scan.head() {
         bytes = Bytes::from([head, bytes].concat());
      }
      tracing::warn!(
          user = %record.user,
          model = %record.requested_model,
          dialect = record.dialect,
          status = record.status,
          body = %String::from_utf8_lossy(&bytes).chars().take(2000).collect::<String>(),
          "{} rejected the request",
          record.provider.map_or("upstream", Provider::as_str)
      );
      record.error_kind = Some(S::REJECTED.into());
      record.response_bytes = bytes.len() as i64;
      record.duration_ms = Some(started.elapsed().as_millis() as i64);
      log_usage(&state, record);
      return respond(builder, S::DIALECT, Body::from(bytes));
   }
   if streaming {
      let head = stream::iter(scan.head().map(Ok::<Bytes, reqwest::Error>));
      let guard = LogGuard::new(state, capture.clone(), record, started);
      return relayed(
         builder,
         head.chain(resp.bytes_stream()),
         guard,
         capture,
         scan,
      );
   }

   let bytes = match read_body(&state, &record, S::DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   let bytes = match scan.body(bytes) {
      Ok(bytes) => bytes,
      Err(error) => {
         log_error(&state, record, 502, "upstream_decode");
         return error_response(S::DIALECT, StatusCode::BAD_GATEWAY, "api_error", &error);
      },
   };
   apply_snapshot(&mut record, &capture.snapshot(), started);
   record.response_bytes = bytes.len() as i64;
   log_usage(&state, record);
   respond(builder, S::DIALECT, Body::from(bytes))
}

/// Upstream bytes to the client, the scan seeing every chunk on the way (and
/// free to rewrite it) and its tail appended once upstream closes.
fn relayed<B, E, S>(
   builder: Builder,
   body: B,
   guard: LogGuard,
   capture: UsageCapture,
   mut scan: S,
) -> Response
where
   B: stream::Stream<Item = Result<Bytes, E>> + Send + 'static,
   E: Into<axum::BoxError> + Send + 'static,
   S: Scan,
{
   let stream = body
      .map(Some)
      .chain(stream::once(async { None }))
      .map(move |item| {
         let _ = &guard;
         match item {
            Some(Ok(bytes)) => {
               let bytes = scan.chunk(bytes);
               capture.note_bytes(bytes.len());
               Ok(bytes)
            },
            Some(Err(err)) => {
               capture.fail("upstream_stream_error");
               Err(err)
            },
            None => {
               capture.note_upstream_eof();
               let bytes = scan.tail();
               capture.note_bytes(bytes.len());
               Ok(bytes)
            },
         }
      });
   respond(builder, S::DIALECT, Body::from_stream(kept_alive(stream)))
}

/// Cloudflare 524s an origin silent for 100s. Upstream chunks split anywhere,
/// so a beat only goes out after one ended on a blank line.
fn kept_alive<S, E>(stream: S) -> impl stream::Stream<Item = Result<Bytes, E>> + Send
where
   S: stream::Stream<Item = Result<Bytes, E>> + Send + 'static,
{
   const EVERY: Duration = Duration::from_secs(15);
   stream::unfold(
      (Box::pin(stream), true),
      |(mut upstream, boundary)| async move {
         loop {
            match timeout(EVERY, upstream.next()).await {
               Ok(item) => {
                  let ended = item.as_ref().is_some_and(|item| {
                     item.as_ref().is_ok_and(|bytes: &Bytes| {
                        bytes.ends_with(b"\n\n") || bytes.ends_with(b"\r\n\r\n")
                     })
                  });
                  return item.map(|item| (item, (upstream, ended)));
               },
               Err(_) if boundary => {
                  let beat = Bytes::from_static(b": heartbeat\n\n");
                  return Some((Ok(beat), (upstream, true)));
               },
               Err(_) => {},
            }
         }
      },
   )
}

pub struct Reply<F, R> {
   pub dialect: Dialect,
   pub stream: Option<F>,
   pub render: R,
}

pub async fn serve_translated<F, S, R, T>(
   state: AppState,
   auth: &AuthInfo,
   mut record: UsageRecord,
   provider: Provider,
   upstream_req: &ResponsesRequest,
   started: Instant,
   reply: Reply<F, R>,
) -> Response
where
   F: FnOnce(UsageCapture) -> S + Send,
   S: FnMut(Option<ResponsesEvent>) -> Vec<Event> + Send + 'static,
   R: FnOnce(&Aggregated) -> T + Send,
   T: Serialize,
{
   record.effort = upstream_req
      .reasoning
      .as_ref()
      .map(|reasoning| reasoning.effort.clone())
      .unwrap_or_default();
   record.session_key = upstream_req.prompt_cache_key.clone().unwrap_or_default();
   let route = Route {
      service_tier: upstream_req.service_tier.as_deref(),
      ..auth.route(&record.session_key, &upstream_req.model)
   };
   let dispatched = match state
      .pools
      .responses(&state.cfg.models, provider, route, upstream_req)
      .await
   {
      Ok(dispatched) => dispatched,
      Err(err) => return dispatch_failed(&state, record, reply.dialect, err),
   };
   record.account_id = dispatched.account_id;
   record.attempts = i64::from(dispatched.attempts);

   let capture = UsageCapture::default();
   let events = dispatched
      .upstream
      .events(&upstream_req.model, capture.clone());
   if let Some(stream) = reply.stream {
      let step = stream(capture.clone());
      return translated(events, LogGuard::new(state, capture, record, started), step);
   }
   let agg = aggregate(events, &capture).await;
   let snap = capture.snapshot();
   apply_snapshot(&mut record, &snap, started);
   if agg.stop == StopKind::Error {
      let msg = agg
         .error_message
         .unwrap_or_else(|| "upstream failure".into());
      record.status = 502;
      log_usage(&state, record);
      return error_response(reply.dialect, StatusCode::BAD_GATEWAY, "api_error", &msg);
   }
   record.error_kind = snap.error_kind;
   logged_json(&state, record, (reply.render)(&agg))
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
