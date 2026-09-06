use std::collections::{BTreeMap, HashSet};
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
use serde_json::Value;
use serde_json::value::RawValue;

use super::auth::AuthInfo;
use super::error::{Dialect, translation_error};
use super::pipeline::{self, apply_snapshot, dispatch_failed, translated};
use super::{AppState, LogGuard, cache_key, log_error, log_rejected, log_usage};
use crate::clock::unix_now;
use crate::codex::models::with_zen_entries;
use crate::codex::types::{OutputItem, ResponsesEvent, ResponsesRequest};
use crate::db::usage::UsageRecord;
use crate::pool::pools::{Dispatched, Upstream};
use crate::pool::{PoolError, Route, UsageWindow, window_seconds};
use crate::provider::Provider;
use crate::translate::chat::ChatRequest;
use crate::translate::openai_req;
use crate::translate::openai_stream::{OpenAiStream, render_aggregated};
use crate::translate::{StopKind, UsageCapture, aggregate, model_map, usable_cap};

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
   let mut req = match serde_json::from_slice::<ChatRequest>(&body) {
      Ok(req) => req,
      Err(err) => {
         log_rejected(&state, &auth, "chat", "unknown");
         return translation_error(DIALECT, &format!("invalid request: {err}"));
      },
   };
   let facts = super::facts::RequestFacts::from_chat(&req, &headers);
   let resolved = model_map::resolve(&state.cfg.models, &req.model);
   let provider = state.cfg.models.route(&resolved.model);
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
         req.model = resolved.model;
         req.reasoning_effort = req.reasoning_effort.or(resolved.effort);
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
      model: &upstream_req.model,
      user: &auth.user,
      pinned_account: auth.limits.pinned_account,
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
      apply_snapshot(&mut record, &snap, started);
      if agg.stop == StopKind::Error {
         let msg = agg
            .error_message
            .unwrap_or_else(|| "upstream failure".into());
         record.status = 502;
         log_usage(&state, record);
         return super::error::error_response(DIALECT, 502, "api_error", &msg);
      }
      record.error_kind = snap.error_kind;
      pipeline::logged_json(&state, record, render_aggregated(&agg, &req.model))
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
      let zen: Vec<String> = state
         .pools
         .zen
         .models()
         .await
         .into_iter()
         .filter(|id| state.cfg.models.route(id) == Provider::Zen)
         .collect();
      return state.catalog_raw().await.map_or_else(
         || {
            super::error::error_response(
               DIALECT,
               503,
               "api_error",
               "no usable codex account to read the model catalog from",
            )
         },
         |body| {
            let body = with_zen_entries(&body, &state.cfg.models.default, &zen).unwrap_or(body);
            ([("content-type", "application/json")], body).into_response()
         },
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

#[derive(Default)]
struct ZenFixups {
   hoisted: usize,
   rewritten: usize,
   dropped: usize,
   unpaired: usize,
}

#[derive(serde::Serialize)]
struct AssistantText {
   #[serde(rename = "type")]
   kind: &'static str,
   role: &'static str,
   content: [TextPart; 1],
}

#[derive(serde::Serialize)]
struct TextPart {
   #[serde(rename = "type")]
   kind: &'static str,
   text: String,
}

/// Zen 400s any `max_output_tokens` under 16, and no other provider wants it either.
fn drop_unusable_max_output_tokens(rest: &mut serde_json::Map<String, Value>, user: &str) {
   let Some(value) = rest.get("max_output_tokens") else {
      return;
   };
   let Some(cap) = value.as_u64() else {
      return;
   };
   if usable_cap(Some(cap)).is_none() {
      tracing::debug!(cap, user = %user, "dropped a max_output_tokens below the upstream floor");
      rest.remove("max_output_tokens");
   }
}

/// Zen 400s these codex-only items as `input[N] did not match any supported type`.
fn zen_input_fixups(rest: &mut serde_json::Map<String, Value>) -> ZenFixups {
   let mut fixes = ZenFixups::default();
   let mut hoisted = Vec::new();
   if let Some(&mut Value::Array(ref mut items)) = rest.get_mut("input") {
      let before = items.len();
      for item in items.iter_mut() {
         let kind = item
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_default();
         match kind.as_str() {
            "additional_tools" => {
               if let Some(&mut Value::Array(ref mut tools)) = item.get_mut("tools") {
                  hoisted.append(tools);
               }
               *item = Value::Null;
            },
            "agent_message" => {
               let text = item
                  .get("content")
                  .and_then(Value::as_array)
                  .map(|parts| {
                     parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<String>()
                  })
                  .unwrap_or_default();
               *item = if text.is_empty() {
                  Value::Null
               } else {
                  fixes.rewritten += 1;
                  serde_json::to_value(AssistantText {
                     kind: "message",
                     role: "assistant",
                     content: [TextPart {
                        kind: "output_text",
                        text,
                     }],
                  })
                  .unwrap_or(Value::Null)
               };
            },
            "local_shell_call" | "context_compaction" => *item = Value::Null,
            _ => {},
         }
      }
      items.retain(|item| !item.is_null());
      fixes.dropped = before - items.len();
   }
   if !hoisted.is_empty() {
      fixes.hoisted = hoisted.len();
      match rest.get_mut("tools") {
         Some(&mut Value::Array(ref mut tools)) => tools.append(&mut hoisted),
         _ => {
            rest.insert("tools".into(), Value::Array(hoisted));
         },
      }
   }
   fixes.unpaired = drop_unpaired_tool_items(rest);
   fixes
}

/// An unpaired `*_call` or `*_call_output` is a 400 on zen.
fn drop_unpaired_tool_items(rest: &mut serde_json::Map<String, Value>) -> usize {
   let Some(&mut Value::Array(ref mut items)) = rest.get_mut("input") else {
      return 0;
   };
   let side = |item: &Value| -> Option<(bool, String)> {
      let kind = item.get("type")?.as_str()?;
      let call_id = item.get("call_id")?.as_str()?.to_owned();
      if kind.ends_with("_call_output") {
         Some((false, call_id))
      } else if kind.ends_with("_call") {
         Some((true, call_id))
      } else {
         None
      }
   };
   let mut calls = HashSet::new();
   let mut outputs = HashSet::new();
   for item in items.iter() {
      match side(item) {
         Some((true, call_id)) => calls.insert(call_id),
         Some((false, call_id)) => outputs.insert(call_id),
         None => false,
      };
   }
   let before = items.len();
   items.retain(|item| match side(item) {
      Some((true, ref call_id)) => outputs.contains(call_id),
      Some((false, ref call_id)) => calls.contains(call_id),
      None => true,
   });
   before - items.len()
}

fn rejected_index(body: &str) -> Option<usize> {
   let tail = body.split_once("\"param\":\"input[")?.1;
   tail.split(']').next()?.parse().ok()
}

fn note_rejected_item(body: &str, rest: &serde_json::Map<String, Value>) {
   let Some(index) = rejected_index(body) else {
      return;
   };
   let Some(item) = rest
      .get("input")
      .and_then(Value::as_array)
      .and_then(|items| items.get(index))
   else {
      return;
   };
   let kind = item.get("type").and_then(Value::as_str).unwrap_or("?");
   let keys: Vec<&str> = item
      .as_object()
      .map(|fields| fields.keys().map(String::as_str).collect())
      .unwrap_or_default();
   tracing::warn!(index, kind, ?keys, "zen rejected an input item");
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
   if req
      .reasoning
      .as_ref()
      .is_none_or(|reasoning| reasoning.effort.is_none())
      && let Some(effort) = resolved.effort
   {
      req.reasoning.get_or_insert_default().effort =
         Some(model_map::clamp_effort(&resolved.model, &effort));
   }
   drop_unusable_max_output_tokens(&mut req.rest, &auth.user);

   if provider == Provider::Zen {
      let fixes = zen_input_fixups(&mut req.rest);
      if fixes.hoisted + fixes.rewritten + fixes.dropped + fixes.unpaired > 0 {
         tracing::warn!(
            fixes.hoisted,
            fixes.rewritten,
            fixes.dropped,
            fixes.unpaired,
            user = %auth.user,
            "reshaped codex-only input items for zen"
         );
      }
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
      model: &record.upstream_model,
      user: &auth.user,
      pinned_account: auth.limits.pinned_account,
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
      Err(err) => {
         if provider == Provider::Zen
            && let PoolError::BadRequest {
               body: ref error_body,
               ..
            } = err
         {
            note_rejected_item(error_body, &req.rest);
         }
         return dispatch_failed(&state, record, DIALECT, err);
      },
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

   raw_response(state, record, resp, capture, started).await
}

async fn raw_response(
   state: AppState,
   mut record: UsageRecord,
   resp: reqwest::Response,
   capture: UsageCapture,
   started: Instant,
) -> Response {
   // Upstream only streams; recover the final response object from the
   // terminal event for non-streaming clients.
   let mut raw_events = resp.bytes_stream().eventsource();
   let mut final_response = Option::<Box<RawValue>>::None;
   while let Some(event) = raw_events.next().await {
      let Ok(event) = event else { break };
      if let Ok(parsed) = serde_json::from_str::<ResponsesEvent>(&event.data) {
         capture.observe(&parsed);
      }
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
      final_response = Some(response);
   }
   let snap = capture.snapshot();
   apply_snapshot(&mut record, &snap, started);
   record.response_bytes = final_response
      .as_ref()
      .map_or(0, |value| value.get().len() as i64);
   let Some(value) = final_response else {
      log_error(&state, record, 502, "upstream_eof");
      return super::error::error_response(
         DIALECT,
         502,
         "api_error",
         "upstream stream ended unexpectedly",
      );
   };
   log_usage(&state, record);
   (
      [("content-type", "application/json")],
      value.get().to_owned(),
   )
      .into_response()
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
         capture.observe(&event);
         let data = serde_json::to_string(&event).unwrap_or_default();
         capture.note_bytes(data.len());
         Ok::<_, Infallible>(Event::default().event(event.kind()).data(data))
      });
      return Sse::new(stream)
         .keep_alive(KeepAlive::default())
         .into_response();
   }

   let mut output = BTreeMap::new();
   let mut terminal = None;
   while let Some(event) = events.next().await {
      capture.observe(&event);
      if let Some((kind, response)) = event.terminal() {
         let mut response = response.clone();
         response.status = Some(kind.as_str().into());
         terminal = Some(response);
      }
      if let ResponsesEvent::OutputItemDone { output_index, item } = event {
         output.insert(output_index, item);
      }
   }
   let snap = capture.snapshot();
   apply_snapshot(&mut record, &snap, started);
   let Some(mut response) = terminal else {
      log_error(&state, record, 502, "upstream_eof");
      return super::error::error_response(
         DIALECT,
         502,
         "api_error",
         "upstream stream ended unexpectedly",
      );
   };
   response
      .id
      .get_or_insert_with(|| format!("resp_{}", uuid::Uuid::new_v4().simple()));
   pipeline::logged_json(
      &state,
      record,
      NonStreamResponse {
         response,
         object: "response",
         model,
         output: output.into_values().collect(),
      },
   )
}

#[derive(serde::Serialize)]
struct NonStreamResponse {
   #[serde(flatten)]
   response: crate::codex::types::ResponseObj,
   object: &'static str,
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
               if let Ok(parsed) = serde_json::from_str::<ResponsesEvent>(&event.data) {
                  capture.observe(&parsed);
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
mod tests;
