use std::convert::Infallible;

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use axum::{Extension, Json};
use eventsource_stream::Eventsource as _;
use futures_util::StreamExt as _;
use serde_json::value::RawValue;

use super::auth::AuthInfo;
use super::error::{Dialect, translation_error};
use super::pipeline::{self, apply_snapshot, dispatch_failed, translated};
use super::{AppState, LogGuard, cache_key, log_error, log_rejected, log_usage};
use crate::clock::unix_now;
use crate::codex::types::{OutputItem, ResponsesEvent, ResponsesRequest, Usage};
use crate::db::usage::UsageRecord;
use crate::pool::pools::{Dispatched, Upstream};
use crate::pool::{Route, UsageWindow, window_seconds};
use crate::provider::Provider;
use crate::translate::chat::ChatRequest;
use crate::translate::openai_req;
use crate::translate::openai_stream::{OpenAiStream, render_aggregated};
use crate::translate::{StopKind, UsageCapture, aggregate, model_map};

const DIALECT: Dialect = Dialect::OpenAi;

#[derive(serde::Deserialize)]
pub struct ModelsQuery {
   client_version: Option<String>,
}

pub async fn chat_completions(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let started = Instant::now();
   let req = match serde_json::from_slice::<ChatRequest>(&body) {
      Ok(req) => req,
      Err(err) => {
         log_rejected(&state, &auth, "chat", "unknown");
         return translation_error(DIALECT, &format!("invalid request: {err}"));
      },
   };
   let facts = super::facts::RequestFacts::from_chat(&req, &headers);
   let provider = state.cfg.models.route(&req.model);
   if !auth.may_use(provider) {
      log_rejected(&state, &auth, "chat", &req.model);
      return super::error::out_of_scope(DIALECT, provider);
   }
   match provider {
      Provider::Anthropic => {
         log_rejected(&state, &auth, "chat", &req.model);
         return translation_error(
            DIALECT,
            "this model is relayed to Anthropic and only available on /v1/messages",
         );
      },
      Provider::Gemini => {
         let model = req.model.clone();
         return super::gemini::chat_completions(state, auth, req, model, facts).await;
      },
      Provider::Glm => {
         log_rejected(&state, &auth, "chat", &req.model);
         return translation_error(DIALECT, "this model is served over /v1/messages");
      },
      Provider::Zen | Provider::OpenAi => {},
   }
   let mut upstream_req = match openai_req::to_responses(&req, &state.cfg) {
      Ok(upstream) => upstream,
      Err(err) => {
         log_rejected(&state, &auth, "chat", &req.model);
         return translation_error(DIALECT, &err.to_string());
      },
   };
   upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));

   let mut record = pipeline::record(
      &auth,
      "chat",
      provider,
      req.model.clone(),
      upstream_req.model.clone(),
      facts,
   );
   record.effort = upstream_req
      .reasoning
      .as_ref()
      .map(|reasoning| reasoning.effort.clone())
      .unwrap_or_default();

   let session_key = upstream_req.prompt_cache_key.clone().unwrap_or_default();
   let route = Route {
      session_key: &session_key,
      model: &req.model,
      prefer_trusted: auth.limits.prefer_trusted,
   };
   let Dispatched {
      account_id,
      upstream,
   } = match state.pools.responses(provider, route, &upstream_req).await {
      Ok(dispatched) => dispatched,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   record.account_id = account_id;
   record.session_key = session_key;

   let capture = UsageCapture::default();
   let events = upstream.events(&upstream_req.model, capture.clone());

   if req.stream.unwrap_or(false) {
      let mut translator =
         OpenAiStream::new(req.model.clone(), req.include_usage(), capture.clone());
      let guard = LogGuard::new(state.clone(), capture, record, started);
      translated(events, guard, move |event| {
         let (chunks, done) = match event {
            Some(event) => (translator.handle(event), false),
            None => (translator.finalize(), true),
         };
         let mut out: Vec<Event> = chunks
            .into_iter()
            .map(|chunk| Event::default().data(chunk))
            .collect();
         if done {
            out.push(Event::default().data("[DONE]"));
         }
         out
      })
   } else {
      let agg = aggregate(events, &capture).await;
      let snap = capture.snapshot();
      apply_snapshot(&mut record, &snap);
      if agg.stop == StopKind::Error {
         let msg = agg
            .error_message
            .unwrap_or_else(|| "upstream failure".into());
         log_error(&state, record, 502, "upstream_failed");
         return super::error::error_response(DIALECT, 502, "api_error", &msg);
      }
      record.error_kind = snap.error_kind;
      log_usage(&state, record);
      Json(render_aggregated(&agg, &req.model)).into_response()
   }
}

/// Codex opens a WebSocket to this path before falling back to HTTP, and only
/// 426 short-circuits that.
pub async fn responses_upgrade_required() -> Response {
   (
      StatusCode::UPGRADE_REQUIRED,
      "this proxy serves the responses API over HTTP only",
   )
      .into_response()
}

/// Codex asks with a `client_version` query and reads its context window out
/// of the reply, so it gets the backend payload untouched.
pub async fn models(
   State(state): State<AppState>,
   headers: HeaderMap,
   Query(query): Query<ModelsQuery>,
) -> Response {
   #[derive(serde::Serialize)]
   struct ModelEntry {
      id: String,
      object: &'static str,
      created: i64,
      owned_by: &'static str,
      #[serde(skip_serializing_if = "Option::is_none")]
      context_window: Option<i64>,
   }

   #[derive(serde::Serialize)]
   struct ModelList {
      object: &'static str,
      data: Vec<ModelEntry>,
   }

   // Both harnesses ask the same path for a catalog, and each only
   // understands its own. `anthropic-version` is required on every Anthropic
   // API call, so its presence identifies the caller.
   if headers.contains_key("anthropic-version") {
      return match state.pools.anthropic.models_raw().await {
         Ok(body) => ([("content-type", "application/json")], body).into_response(),
         Err(err) => super::error::error_response(
            super::error::Dialect::Anthropic,
            503,
            "api_error",
            &format!("reading the model catalog: {err}"),
         ),
      };
   }
   if query.client_version.is_some() {
      return state.catalog_raw().await.map_or_else(
         || {
            super::error::error_response(
               DIALECT,
               503,
               "api_error",
               "no usable codex account to read the model catalog from",
            )
         },
         |body| ([("content-type", "application/json")], body).into_response(),
      );
   }

   let created = unix_now();

   let live = state.catalog().await;

   let mut data = if let Some(models) = live {
      models
         .iter()
         .filter(|model| model.listed())
         .map(|model| ModelEntry {
            id: model.slug.clone(),
            object: "model",
            created,
            owned_by: "openai",
            context_window: model.context_window,
         })
         .collect::<Vec<ModelEntry>>()
   } else {
      let mut ids = state.cfg.models.known.clone();
      let default = &state.cfg.models.default;
      if !default.is_empty() && !ids.contains(default) {
         ids.push(default.clone());
      }
      ids.into_iter()
         .map(|id| ModelEntry {
            id,
            object: "model",
            created,
            owned_by: "slop-proxy",
            context_window: None,
         })
         .collect::<Vec<ModelEntry>>()
   };

   data.extend(state.pools.zen.models().await.into_iter().filter_map(|id| {
      state
         .cfg
         .models
         .route(&id)
         .eq(&Provider::Zen)
         .then_some(ModelEntry {
            id,
            object: "model",
            created,
            owned_by: "opencode",
            context_window: None,
         })
   }));

   data.extend(
      state
         .pools
         .gemini
         .models()
         .await
         .into_iter()
         .filter_map(|id| {
            state
               .cfg
               .models
               .route(&id)
               .eq(&Provider::Gemini)
               .then_some(ModelEntry {
                  id,
                  object: "model",
                  created,
                  owned_by: "google",
                  context_window: None,
               })
         }),
   );

   Json(ModelList {
      object: "list",
      data,
   })
   .into_response()
}

/// The fields the proxy rewrites on a `/v1/responses` request. Everything
/// else stays in `rest` and goes upstream as the caller wrote it.
#[derive(serde::Deserialize, serde::Serialize)]
struct PassthroughRequest {
   model: Option<String>,
   /// Codex sets this to `priority` for `/fast`.
   #[serde(skip_serializing_if = "Option::is_none")]
   service_tier: Option<String>,
   store: Option<bool>,
   stream: Option<bool>,
   #[serde(skip_serializing_if = "Option::is_none")]
   instructions: Option<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   reasoning: Option<ReasoningPatch>,
   /// Codex sends its own, stable per conversation.
   #[serde(skip_serializing_if = "Option::is_none")]
   prompt_cache_key: Option<String>,
   #[serde(flatten)]
   rest: serde_json::Map<String, serde_json::Value>,
}

#[derive(Default, serde::Deserialize, serde::Serialize)]
struct ReasoningPatch {
   #[serde(skip_serializing_if = "Option::is_none")]
   effort: Option<String>,
   #[serde(flatten)]
   rest: serde_json::Map<String, serde_json::Value>,
}

pub async fn responses_passthrough(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let started = Instant::now();
   let mut req = match serde_json::from_slice::<PassthroughRequest>(&body) {
      Ok(req) => req,
      Err(err) => return translation_error(DIALECT, &format!("invalid request: {err}")),
   };

   let requested_model = req
      .model
      .unwrap_or_else(|| state.cfg.models.default.clone());
   let resolved = model_map::resolve(&state.cfg.models, &requested_model);
   // Scope is decided by where the model resolves, not by the endpoint. This
   // surface is the Responses API, which zen speaks as well as codex does.
   let provider = state.cfg.models.route(&resolved.model);
   if !matches!(
      provider,
      Provider::OpenAi | Provider::Zen | Provider::Gemini
   ) {
      return translation_error(DIALECT, "this model is not served over the responses api");
   }
   if !auth.may_use(provider) {
      return super::error::out_of_scope(DIALECT, provider);
   }
   req.model = Some(resolved.model.clone());
   if let Some(effort) = model_map::suffix_effort(&requested_model) {
      req.reasoning.get_or_insert_default().effort =
         Some(model_map::clamp_effort(&resolved.model, &effort));
   }

   let client_streams = req.stream.unwrap_or(false);
   req.store = Some(false);
   req.stream = Some(true);
   if req.instructions.is_none() && provider == Provider::OpenAi {
      req.instructions = Some(state.cfg.codex.instructions());
   }
   let encoded = match serde_json::to_vec(&req) {
      Ok(value) => Bytes::from(value),
      Err(err) => return translation_error(DIALECT, &format!("serializing request: {err}")),
   };
   let session_key = req
      .prompt_cache_key
      .clone()
      .unwrap_or_else(|| auth.user.clone());

   let typed = match serde_json::from_slice::<ResponsesRequest>(&encoded) {
      Ok(typed) => Some(typed),
      Err(error) => {
         tracing::warn!(%error, "responses body did not type for the bridge");
         None
      },
   };
   let facts = typed
      .as_ref()
      .map(|typed| super::facts::RequestFacts::from_responses(typed, &headers))
      .unwrap_or_default();
   let mut record = pipeline::record(
      &auth,
      "responses",
      provider,
      requested_model,
      resolved.model,
      facts,
   );
   record.effort = req
      .reasoning
      .as_ref()
      .and_then(|reasoning| reasoning.effort.clone())
      .unwrap_or_default();
   record.service_tier = req.service_tier.clone().unwrap_or_default();
   record.session_key = session_key.clone();
   let route = Route {
      session_key: &session_key,
      model: &record.requested_model,
      prefer_trusted: auth.limits.prefer_trusted,
   };
   let Dispatched {
      account_id,
      upstream,
   } = match state
      .pools
      .responses_raw(provider, route, encoded, typed.as_ref())
      .await
   {
      Ok(dispatched) => dispatched,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   record.account_id = account_id;
   let capture = UsageCapture::default();
   let resp = match upstream {
      Upstream::Responses(resp) => resp,
      bridged @ Upstream::Bridged { .. } => {
         let model = record.upstream_model.clone();
         return bridged_responses(state, record, bridged, model, client_streams, started).await;
      },
   };

   if client_streams {
      let limits = rate_limit_headers(&state.pools.codex.pool_windows().await);
      return relay_stream(
         resp,
         LogGuard::new(state.clone(), capture.clone(), record, started),
         capture,
         limits,
      );
   }

   {
      // Upstream only streams; recover the final response object from the
      // terminal event for non-streaming clients.
      let mut raw_events = resp.bytes_stream().eventsource();
      let mut final_response = Option::<Box<RawValue>>::None;
      while let Some(event) = raw_events.next().await {
         let Ok(event) = event else { break };
         let Ok(TerminalEvent {
            kind,
            response: Some(response),
         }) = serde_json::from_str::<TerminalEvent>(&event.data)
         else {
            continue;
         };
         if !TerminalEvent::is_terminal(&kind) {
            continue;
         }
         if let Ok(env) = serde_json::from_str::<UsageEnvelope>(&event.data)
            && let Some(usage) = env.response.usage
         {
            capture.record(&usage);
         }
         final_response = Some(response);
      }
      let snap = capture.snapshot();
      apply_snapshot(&mut record, &snap);
      log_usage(&state, record);
      final_response.map_or_else(
         || {
            super::error::error_response(
               DIALECT,
               502,
               "api_error",
               "upstream stream ended unexpectedly",
            )
         },
         |value| {
            (
               [("content-type", "application/json")],
               value.get().to_owned(),
            )
               .into_response()
         },
      )
   }
}

/// The bridge's frames are relayed rather than the events they parse into.
async fn bridged_responses(
   state: AppState,
   mut record: UsageRecord,
   upstream: Upstream,
   model: String,
   client_streams: bool,
   started: Instant,
) -> Response {
   let capture = UsageCapture::default();
   let mut events = upstream.events(&model, capture.clone());

   if client_streams {
      let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
      let stream = events.map(move |event| {
         let _ = &guard;
         if let ResponsesEvent::Completed { ref response }
         | ResponsesEvent::Incomplete { ref response }
         | ResponsesEvent::Failed { ref response } = event
         {
            if let Some(usage) = response.usage.as_ref() {
               capture.record(usage);
            }
            capture.note_stop_reason(response.status.as_deref().unwrap_or("completed"));
         }
         let data = serde_json::to_string(&event).unwrap_or_default();
         capture.note_bytes(data.len());
         Ok::<_, Infallible>(Event::default().event(event.kind()).data(data))
      });
      return Sse::new(stream)
         .keep_alive(KeepAlive::default())
         .into_response();
   }

   let mut output = Vec::new();
   while let Some(event) = events.next().await {
      match event {
         ResponsesEvent::Completed { response }
         | ResponsesEvent::Incomplete { response }
         | ResponsesEvent::Failed { response } => {
            if let Some(usage) = response.usage {
               capture.record(&usage);
            }
            capture.note_stop_reason(response.status.as_deref().unwrap_or("completed"));
         },
         ResponsesEvent::OutputItemDone { item, .. } => output.push(item),
         ResponsesEvent::Created { .. }
         | ResponsesEvent::InProgress
         | ResponsesEvent::OutputItemAdded { .. }
         | ResponsesEvent::ContentPartAdded { .. }
         | ResponsesEvent::ContentPartDone
         | ResponsesEvent::OutputTextDelta { .. }
         | ResponsesEvent::OutputTextDone { .. }
         | ResponsesEvent::ReasoningSummaryPartAdded
         | ResponsesEvent::ReasoningSummaryPartDone
         | ResponsesEvent::ReasoningSummaryTextDelta { .. }
         | ResponsesEvent::ReasoningSummaryTextDone
         | ResponsesEvent::ReasoningTextDelta { .. }
         | ResponsesEvent::ReasoningTextDone
         | ResponsesEvent::FunctionCallArgumentsDelta { .. }
         | ResponsesEvent::FunctionCallArgumentsDone { .. }
         | ResponsesEvent::CustomToolCallInputDone { .. }
         | ResponsesEvent::Other => {},
      }
   }
   let snap = capture.snapshot();
   apply_snapshot(&mut record, &snap);
   log_usage(&state, record);
   Json(NonStreamResponse {
      id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
      object: "response",
      status: "completed",
      model,
      output,
   })
   .into_response()
}

#[derive(serde::Serialize)]
struct NonStreamResponse {
   id: String,
   object: &'static str,
   status: &'static str,
   model: String,
   output: Vec<OutputItem>,
}

/// The stream's terminal events; `response` stays raw because it is handed
/// back to the client verbatim, which rules out a tagged enum since serde
/// buffers those and `RawValue` cannot be read back out of the buffer.
#[derive(serde::Deserialize)]
struct TerminalEvent {
   #[serde(rename = "type")]
   kind: String,
   response: Option<Box<RawValue>>,
}

impl TerminalEvent {
   fn is_terminal(kind: &str) -> bool {
      matches!(
         kind,
         "response.completed" | "response.incomplete" | "response.failed"
      )
   }
}

#[derive(serde::Deserialize)]
struct UsageEnvelope {
   response: ResponseUsage,
}

#[derive(serde::Deserialize)]
struct ResponseUsage {
   #[serde(default)]
   usage: Option<Usage>,
   #[serde(default)]
   status: Option<String>,
   #[serde(default)]
   incomplete_details: Option<IncompleteDetails>,
}

#[derive(serde::Deserialize)]
struct IncompleteDetails {
   reason: Option<String>,
}

/// `response.output_item.done` names the tool it emitted. Its `arguments`
/// field is the caller's own content and is never read.
#[derive(serde::Deserialize)]
struct OutputItemDone {
   item: ToolName,
}

#[derive(serde::Deserialize)]
struct ToolName {
   #[serde(default)]
   name: Option<String>,
}

/// What codex reads for `/status`.
fn rate_limit_headers(windows: &[UsageWindow]) -> Vec<(String, String)> {
   let mut sorted: Vec<_> = windows.iter().collect();
   sorted.sort_by_key(|window| window_seconds(&window.name).unwrap_or(i64::MAX));
   let mut out = Vec::new();
   for (tier, window) in ["primary", "secondary"].iter().zip(sorted) {
      let Some(minutes) = window_seconds(&window.name).map(|secs| secs / 60) else {
         continue;
      };
      out.push((
         format!("x-codex-{tier}-window-minutes"),
         minutes.to_string(),
      ));
      out.push((
         format!("x-codex-{tier}-used-percent"),
         ((window.utilization * 100.0).round() as i64).to_string(),
      ));
      if let Some(resets_at) = window.resets_at {
         out.push((format!("x-codex-{tier}-reset-at"), resets_at.to_string()));
      }
   }
   out
}

fn relay_stream(
   resp: reqwest::Response,
   guard: LogGuard,
   capture: UsageCapture,
   limits: Vec<(String, String)>,
) -> Response {
   struct Held<S> {
      inner: S,
      capture: UsageCapture,
      _guard: LogGuard,
   }
   impl<S: futures_util::Stream + Unpin> futures_util::Stream for Held<S> {
      type Item = S::Item;
      fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
         let polled = Pin::new(&mut self.inner).poll_next(cx);
         // Never reached if the caller went away first.
         if matches!(polled, Poll::Ready(None)) {
            self.capture.note_upstream_eof();
         }
         polled
      }
   }
   let eof_capture = capture.clone();
   let stream = resp.bytes_stream().eventsource().filter_map(move |event| {
      let capture = capture.clone();
      async move {
         match event {
            Ok(event) => {
               if !event.event.is_empty() {
                  capture.note_event(&event.event);
               }
               capture.note_bytes(event.data.len());
               if (event.event == "response.completed"
                  || event.data.contains("\"response.completed\""))
                  && let Ok(env) = serde_json::from_str::<UsageEnvelope>(&event.data)
               {
                  if let Some(usage) = env.response.usage {
                     capture.record(&usage);
                  }
                  let reason = env
                     .response
                     .incomplete_details
                     .and_then(|detail| detail.reason)
                     .or(env.response.status);
                  if let Some(reason) = reason {
                     capture.note_stop_reason(&reason);
                  }
               }
               if event.event == "response.output_item.done"
                  && let Ok(done) = serde_json::from_str::<OutputItemDone>(&event.data)
                  && let Some(name) = done.item.name
               {
                  capture.note_tool_call(&name);
               }
               let mut out = Event::default().data(event.data);
               if !event.event.is_empty() && event.event != "message" {
                  out = out.event(event.event);
               }
               Some(Ok::<_, Infallible>(out))
            },
            Err(err) => {
               tracing::warn!("passthrough SSE error: {err}");
               capture.fail("upstream_sse_error");
               None
            },
         }
      }
   });
   let held = Held {
      inner: Box::pin(stream),
      capture: eof_capture,
      _guard: guard,
   };
   let mut response = Sse::new(held)
      .keep_alive(KeepAlive::default())
      .into_response();
   for (name, value) in limits {
      if let (Ok(name), Ok(value)) = (HeaderName::try_from(name), HeaderValue::try_from(value)) {
         response.headers_mut().insert(name, value);
      }
   }
   response
}

#[cfg(test)]
mod passthrough_tests {
   use super::*;

   #[test]
   fn the_service_tier_is_read_and_forwarded_unchanged() {
      let body = serde_json::json!({
          "model": "gpt-5.6-sol",
          "service_tier": "priority",
          "input": [],
      });
      let req: PassthroughRequest = serde_json::from_value(body).unwrap();
      assert_eq!(req.service_tier.as_deref(), Some("priority"));
      let out = serde_json::to_value(&req).unwrap();
      assert_eq!(out["service_tier"], "priority");
   }

   /// The typed request would serialise an item it does not know as
   /// `{"type":"Other"}` and drop a custom tool's grammar.
   #[test]
   fn what_the_proxy_has_no_opinion_on_goes_upstream_as_written() {
      let body = serde_json::json!({
          "model": "gpt-5.6-sol",
          "input": [{"type": "apply_patch_call", "call_id": "c1", "status": "completed"}],
          "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark"}}],
          "reasoning": {"summary": "auto"},
      });
      let req: PassthroughRequest = serde_json::from_value(body.clone()).unwrap();
      let out = serde_json::to_value(&req).unwrap();
      assert_eq!(out["input"], body["input"]);
      assert_eq!(out["tools"], body["tools"]);
      assert_eq!(out["reasoning"], body["reasoning"]);
   }
}

#[cfg(test)]
mod terminal_event_tests {
   use super::TerminalEvent;

   /// Parsed from the wire, not from a Value, because the bug this pins only
   /// shows on the streaming deserializer.
   #[test]
   fn the_final_response_survives_the_terminal_frame() {
      let frame = r#"{"type":"response.completed","sequence_number":9,"response":{"id":"resp_1","usage":{"input_tokens":1}}}"#;
      let event: TerminalEvent = serde_json::from_str(frame).unwrap();
      assert!(TerminalEvent::is_terminal(&event.kind));
      assert_eq!(
         event.response.unwrap().get(),
         r#"{"id":"resp_1","usage":{"input_tokens":1}}"#
      );
   }
}

#[cfg(test)]
mod rate_limit_header_tests {
   use super::*;
   use crate::pool::UsageWindow;

   fn window(name: &str, utilization: f64, resets_at: Option<i64>) -> UsageWindow {
      UsageWindow {
         name: name.into(),
         utilization,
         resets_at,
      }
   }

   #[test]
   fn the_shorter_window_is_primary() {
      let out = rate_limit_headers(&[
         window("7d", 0.42, Some(1000)),
         window("5h", 0.11, Some(500)),
      ]);
      let get = |key: &str| {
         out.iter()
            .find(|&&(ref name, _)| name == key)
            .map(|&(_, ref value)| value.as_str())
            .unwrap()
      };
      assert_eq!(get("x-codex-primary-window-minutes"), "300");
      assert_eq!(get("x-codex-primary-used-percent"), "11");
      assert_eq!(get("x-codex-secondary-window-minutes"), "10080");
      assert_eq!(get("x-codex-secondary-used-percent"), "42");
   }

   #[test]
   fn a_window_without_a_reset_still_reports() {
      let out = rate_limit_headers(&[window("7d", 0.8, None)]);
      assert!(
         out.iter()
            .any(|&(ref name, ref value)| name == "x-codex-primary-used-percent" && value == "80")
      );
      assert!(
         !out
            .iter()
            .any(|&(ref name, _)| name == "x-codex-primary-reset-at")
      );
   }
}
