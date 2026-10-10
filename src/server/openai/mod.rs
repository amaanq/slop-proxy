use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use axum::{Extension, Json};
use eventsource_stream::Eventsource as _;
use futures_util::{StreamExt as _, stream};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use crate::anthropic::{Catalog, Model};
use crate::clock::{rfc3339, unix_now};
use crate::codex::models::with_zen_entries;
use crate::codex::types::{
   ContentPart, InputItem, OutputItem, ResponseObj, ResponsesEvent, ResponsesRequest,
};
use crate::config::ZenDialect;
use crate::db::usage::UsageRecord;
use crate::egress::egress_of;
use crate::pool::pools::{Catalogs, Dispatched, Upstream};
use crate::pool::{Route, UsageWindow, window_seconds};
use crate::provider::Provider;
use crate::server::auth::AuthInfo;
use crate::server::error::{Dialect, error_response, pool_error_response, translation_error};
use crate::server::facts::RequestFacts;
use crate::server::pipeline::{self, Reply, apply_snapshot, dispatch_failed, translated};
use crate::server::{
   AppState, LogGuard, cache_key, copilot, gemini, log_error, log_rejected, log_usage,
};
use crate::translate::chat::ChatRequest;
use crate::translate::openai_req;
use crate::translate::openai_stream::{OpenAiStream, render_aggregated};
use crate::translate::{Aggregated, UsageCapture, model_map, usable_cap};
use crate::zen::ZenModel;

pub mod websocket;

const DIALECT: Dialect = Dialect::OpenAi;

#[derive(serde::Deserialize)]
pub struct ModelsQuery {
   client_version: Option<String>,
}

/// `context_length` and `input` are the keys a discovering client reads, so a
/// harness needs no hand-written model list of its own.
#[derive(serde::Serialize)]
pub struct ModelEntry {
   pub id: String,
   pub object: &'static str,
   pub created: i64,
   pub owned_by: &'static str,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub context_length: Option<i64>,
   #[serde(skip_serializing_if = "Vec::is_empty")]
   pub input: Vec<String>,
}

impl ModelEntry {
   fn new(id: String, owned_by: &'static str, context_length: Option<i64>) -> Self {
      Self {
         id,
         object: "model",
         created: unix_now(),
         owned_by,
         context_length,
         input: Vec::new(),
      }
   }
}

#[derive(serde::Serialize)]
pub struct ModelList {
   pub object: &'static str,
   pub data: Vec<ModelEntry>,
}

/// The Gemini pool's chat-capable catalog, shared by `/v1/models` and the
/// `/v1beta` surface a native-dialect caller discovers from.
pub fn gemini_entries(state: &AppState, catalogs: &Catalogs) -> Vec<ModelEntry> {
   catalogs
      .gemini
      .iter()
      .filter(|model| state.cfg.models.route(&model.id) == Provider::Gemini)
      .map(|model| ModelEntry::new(model.id.clone(), "google", model.context_window))
      .collect()
}

fn zen_responses_models<'cat>(
   state: &AppState,
   catalogs: &'cat Catalogs,
) -> impl Iterator<Item = &'cat ZenModel> {
   catalogs.zen.iter().filter(|model| {
      state.cfg.models.route(&model.id) == Provider::Zen
         && state.cfg.models.zen_dialect(&model.id) != ZenDialect::Messages
   })
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
   let facts = RequestFacts::from_chat(&req, &headers);
   let resolved = model_map::resolve(&state.cfg.models, &req.model);
   let provider = match pipeline::admit(&state, &auth, DIALECT, "chat", &req.model, &resolved.model)
   {
      Ok(provider) => provider,
      Err(response) => return *response,
   };
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
         return gemini::chat_completions(state, auth, req, model, facts).await;
      },
      Provider::Copilot => {
         let model = req.model.clone();
         req.model = resolved.model;
         req.reasoning_effort = req.reasoning_effort.or(resolved.effort);
         return copilot::chat_completions(state, auth, req, model, facts).await;
      },
      Provider::Zen if state.cfg.models.zen_dialect(&resolved.model) != ZenDialect::Messages => {},
      Provider::OpenAi => {},
      Provider::Glm | Provider::DeepSeek | Provider::Experiential | Provider::Zen => {
         log_rejected(&state, &auth, "chat", &req.model);
         return translation_error(DIALECT, "this model is served over /v1/messages");
      },
   }
   let mut upstream = match openai_req::to_responses(&req, &state.cfg) {
      Ok(upstream) => upstream,
      Err(err) => {
         log_rejected(&state, &auth, "chat", &req.model);
         return translation_error(DIALECT, &err.to_string());
      },
   };
   upstream.prompt_cache_key = Some(cache_key(&auth.user, &upstream));

   let record = pipeline::record(
      &auth,
      "chat",
      provider,
      req.model.clone(),
      upstream.model.clone(),
      facts,
   );
   let model = req.model.clone();
   let include_usage = req.include_usage();
   let stream = move |capture: UsageCapture| {
      let mut translator = OpenAiStream::new(model, include_usage, capture);
      move |event: Option<ResponsesEvent>| {
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
      }
   };
   let reply = Reply {
      dialect: DIALECT,
      stream: req.stream.unwrap_or(false).then_some(stream),
      render: |agg: &Aggregated| render_aggregated(agg, &req.model),
   };
   pipeline::serve_translated(state, &auth, record, provider, &upstream, started, reply).await
}

/// Codex opens a WebSocket to this path before falling back to HTTP, and only
/// 426 short-circuits that.
pub fn responses_upgrade_required() -> Response {
   (
      StatusCode::UPGRADE_REQUIRED,
      "WebSocket unavailable for this request, use HTTP POST",
   )
      .into_response()
}

fn messages_catalog(state: &AppState) -> Catalog {
   let catalogs = state.pools.catalogs();
   let mut data = catalogs.anthropic.to_vec();
   let now = rfc3339(unix_now());
   let synthetic = |id: String| Model {
      display_name: id.clone(),
      id,
      created_at: now.clone(),
      kind: "model".to_owned(),
   };

   data.extend(
      catalogs
         .glm
         .iter()
         .filter(|model| state.cfg.models.route(&model.id) == Provider::Glm)
         .cloned(),
   );
   data.extend(
      catalogs
         .deepseek
         .iter()
         .filter(|id| state.cfg.models.route(id) == Provider::DeepSeek)
         .map(|id| synthetic(id.clone())),
   );
   data.extend(
      catalogs
         .zen
         .iter()
         .filter(|model| state.cfg.models.zen_dialect(&model.id) == ZenDialect::Messages)
         .map(|model| synthetic(model.id.clone())),
   );

   Catalog {
      first_id: data.first().map(|model| model.id.clone()),
      last_id: data.last().map(|model| model.id.clone()),
      has_more: false,
      data,
   }
}

/// Codex asks with a `client_version` query and reads its context window out
/// of the reply, so its catalog preserves the backend's model metadata.
pub async fn models(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   Query(query): Query<ModelsQuery>,
) -> Response {
   // Both harnesses ask the same path for a catalog, and each only
   // understands its own. `anthropic-version` is required on every Anthropic
   // API call, so its presence identifies the caller.
   if headers.contains_key("anthropic-version") {
      return Json(messages_catalog(&state)).into_response();
   }

   let catalogs = state.pools.catalogs();
   if query.client_version.is_some() {
      return state
         .catalog(&auth.user, auth.limits.pinned_account)
         .await
         .map_or_else(
            || {
               error_response(
                  DIALECT,
                  StatusCode::SERVICE_UNAVAILABLE,
                  "api_error",
                  "no usable codex account to read the model catalog from",
               )
            },
            |catalog| {
               let body = serde_json::to_string(&catalog).expect("catalog serializes");
               let body = with_zen_entries(
                  &body,
                  &state.cfg.models.default,
                  zen_responses_models(&state, &catalogs),
               )
               .unwrap_or(body);
               ([("content-type", "application/json")], body).into_response()
            },
         );
   }

   let live = state.catalog(&auth.user, auth.limits.pinned_account).await;
   let mut data = if let Some(models) = live {
      models
         .models
         .iter()
         .filter(|model| model.listed())
         .map(|model| ModelEntry {
            input: model.input_modalities.clone(),
            ..ModelEntry::new(model.slug.clone(), "openai", model.context_window)
         })
         .collect::<Vec<ModelEntry>>()
   } else {
      let mut ids = state.cfg.models.known.clone();
      let default = &state.cfg.models.default;
      if !default.is_empty() && !ids.contains(default) {
         ids.push(default.clone());
      }
      ids.into_iter()
         .map(|id| ModelEntry::new(id, "slop-proxy", None))
         .collect::<Vec<ModelEntry>>()
   };

   // This catalog is what an openai-dialect client discovers from, so a zen
   // model the responses surface refuses must not appear in it.
   data.extend(
      zen_responses_models(&state, &catalogs)
         .map(|model| ModelEntry::new(model.id.clone(), "opencode", model.context_window)),
   );
   data.extend(gemini_entries(&state, &catalogs));
   // The Copilot catalog narrowed to whatever `copilot_patterns` claims, so
   // `/v1/models` only advertises what this proxy will actually serve.
   data.extend(
      catalogs
         .copilot
         .iter()
         .filter(|id| state.cfg.models.route(id) == Provider::Copilot)
         .map(|id| ModelEntry::new(id.clone(), "github", None)),
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
   #[serde(skip_serializing_if = "Option::is_none")]
   store: Option<bool>,
   #[serde(skip_serializing_if = "Option::is_none")]
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

/// Zen 400s any `max_output_tokens` under 16, and no other provider wants it either.
fn drop_unusable_max_output_tokens(rest: &mut Map<String, Value>, user: &str) {
   if let Some(cap) = rest.get("max_output_tokens").and_then(Value::as_u64)
      && usable_cap(Some(cap)).is_none()
   {
      tracing::debug!(cap, user = %user, "dropped a max_output_tokens below the upstream floor");
      rest.remove("max_output_tokens");
   }
}

const ENCRYPTED_PAYLOAD_NOTE: &str = "[this payload was encrypted by OpenAI before it reached the proxy and cannot be read here. The parent session has to send its requests through the proxy as well.]";

fn input_items(rest: &mut Map<String, Value>) -> Option<&mut Vec<Value>> {
   rest.get_mut("input").and_then(Value::as_array_mut)
}

fn for_each_tool_list(rest: &mut Map<String, Value>, mut each: impl FnMut(&mut [Value])) {
   if let Some(&mut Value::Array(ref mut tools)) = rest.get_mut("tools") {
      each(tools);
   }
   for item in input_items(rest).into_iter().flatten() {
      if item.get("type").and_then(Value::as_str) == Some("additional_tools")
         && let Some(&mut Value::Array(ref mut tools)) = item.get_mut("tools")
      {
         each(tools);
      }
   }
}

/// The backend returns any argument whose schema says `encrypted: true` as a Fernet
/// token only its own backend can read, which is how a spawned agent on
/// another provider ends up with the header of a task and no payload.
fn strip_encrypted_argument_flags(rest: &mut Map<String, Value>) -> usize {
   let mut stripped = 0;
   for_each_tool_list(rest, |tools| stripped += strip_encrypted_from_tools(tools));
   stripped
}

/// The backend 400s any edit to a tool under this namespace as "reserved for
/// use by this model and must match the configured schema".
const RESERVED_NAMESPACE: &str = "collaboration";

fn strip_encrypted_from_tools(tools: &mut [Value]) -> usize {
   let mut stripped = 0;
   for tool in tools.iter_mut() {
      if tool.get("name").and_then(Value::as_str) == Some(RESERVED_NAMESPACE) {
         continue;
      }
      if let Some(&mut Value::Array(ref mut nested)) = tool.get_mut("tools") {
         stripped += strip_encrypted_from_tools(nested);
         continue;
      }
      let Some(&mut Value::Object(ref mut properties)) = tool.pointer_mut("/parameters/properties")
      else {
         continue;
      };
      for schema in properties.values_mut() {
         if let Some(schema) = schema.as_object_mut()
            && schema.remove("encrypted").is_some()
         {
            stripped += 1;
         }
      }
   }
   stripped
}

/// Namespace the collaboration tools are declared under on the way up. The
/// backend leaves a schema under any other name alone, so the `encrypted`
/// flag can come off and the spawn message arrives readable.
const PROXY_NAMESPACE: &str = "slop_collab";

fn rename_tools(tools: &mut [Value]) -> usize {
   let mut renamed = 0;
   for tool in tools.iter_mut() {
      let reserved = tool.get("name").and_then(Value::as_str) == Some(RESERVED_NAMESPACE)
         && tool.get("tools").is_some();
      if reserved && let Some(tool) = tool.as_object_mut() {
         tool.insert("name".into(), Value::String(PROXY_NAMESPACE.into()));
         renamed += 1;
      }
   }
   renamed
}

fn rename_reserved_namespace(rest: &mut Map<String, Value>) -> usize {
   let mut renamed = 0;
   for_each_tool_list(rest, |tools| renamed += rename_tools(tools));
   let mention = format!("functions.{RESERVED_NAMESPACE}.");
   let replacement = format!("functions.{PROXY_NAMESPACE}.");
   for item in input_items(rest).into_iter().flatten() {
      let reserved_call = item.get("namespace").and_then(Value::as_str) == Some(RESERVED_NAMESPACE);
      if reserved_call && let Some(item) = item.as_object_mut() {
         item.insert("namespace".into(), Value::String(PROXY_NAMESPACE.into()));
         renamed += 1;
         continue;
      }
      if item.get("type").and_then(Value::as_str) == Some("message")
         && item.get("role").and_then(Value::as_str) == Some("developer")
         && let Some(&mut Value::Array(ref mut parts)) = item.get_mut("content")
      {
         for part in parts.iter_mut() {
            let rewritten = part
               .get("text")
               .and_then(Value::as_str)
               .filter(|text| text.contains(&mention))
               .map(|text| text.replace(&mention, &replacement));
            if let Some(text) = rewritten
               && let Some(part) = part.as_object_mut()
            {
               part.insert("text".into(), Value::String(text));
            }
         }
      }
   }
   renamed
}

/// The router on the client resolves a call by namespace and name, so a
/// frame naming the proxy's namespace has to reach it under the original.
fn restore_reserved_namespace(data: String) -> String {
   let marker = format!("\"namespace\":\"{PROXY_NAMESPACE}\"");
   if data.contains(&marker) {
      data.replace(&marker, &format!("\"namespace\":\"{RESERVED_NAMESPACE}\""))
   } else {
      data
   }
}

fn is_fernet_token(text: &str) -> bool {
   text.starts_with("gAAAA")
}

/// Codex wraps a plaintext task as `encrypted_content` whenever the backend
/// omits `encrypted_function_args`, and the backend 400s that with
/// `invalid_encrypted_content`.
fn unwrap_plaintext_agent_payloads(rest: &mut Map<String, Value>) -> usize {
   let mut unwrapped = 0;
   for item in input_items(rest).into_iter().flatten() {
      if item.get("type").and_then(Value::as_str) != Some("agent_message") {
         continue;
      }
      let Some(&mut Value::Array(ref mut parts)) = item.get_mut("content") else {
         continue;
      };
      for part in parts.iter_mut() {
         let Some(text) = part.get("encrypted_content").and_then(Value::as_str) else {
            continue;
         };
         if is_fernet_token(text) {
            continue;
         }
         *part = serde_json::to_value(ContentPart::InputText {
            text: text.to_owned(),
         })
         .unwrap_or(Value::Null);
         unwrapped += 1;
      }
   }
   unwrapped
}

fn drop_composite_reasoning(rest: &mut Map<String, Value>) -> usize {
   let valid = |id: &str| {
      id.bytes()
         .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
   };
   let Some(items) = input_items(rest) else {
      return 0;
   };
   let before = items.len();
   items.retain(|item| {
      item.get("type").and_then(Value::as_str) != Some("reasoning")
         || item.get("id").and_then(Value::as_str).is_none_or(valid)
   });
   before - items.len()
}

/// Zen 400s these codex-only items as `input[N] did not match any supported type`.
fn zen_input_fixups(rest: &mut Map<String, Value>, user: &str) {
   let mut rewritten = 0;
   let mut dropped = 0;
   let mut hoisted = Vec::new();
   if let Some(items) = input_items(rest) {
      let before = items.len();
      for item in items.iter_mut() {
         let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
         match kind {
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
                        .filter_map(|part| {
                           part.get("text").and_then(Value::as_str).or_else(|| {
                              part
                                 .get("encrypted_content")
                                 .map(|_| ENCRYPTED_PAYLOAD_NOTE)
                           })
                        })
                        .collect::<String>()
                  })
                  .unwrap_or_default();
               *item = if text.is_empty() {
                  Value::Null
               } else {
                  rewritten += 1;
                  serde_json::to_value(InputItem::Message {
                     role: "assistant".into(),
                     content: vec![ContentPart::OutputText { text }],
                  })
                  .unwrap_or(Value::Null)
               };
            },
            "reasoning" if item.get("encrypted_content").is_some() => *item = Value::Null,
            "local_shell_call" | "context_compaction" => *item = Value::Null,
            _ => {},
         }
      }
      items.retain(|item| !item.is_null());
      dropped = before - items.len();
   }
   let hoisted_count = hoisted.len();
   if !hoisted.is_empty() {
      match rest.get_mut("tools") {
         Some(&mut Value::Array(ref mut tools)) => tools.append(&mut hoisted),
         _ => {
            rest.insert("tools".into(), Value::Array(hoisted));
         },
      }
   }
   let (repaired, malformed) = repair_tool_arguments(rest);
   let unpaired = drop_unpaired_tool_items(rest);
   if hoisted_count + rewritten + dropped + unpaired + repaired + malformed > 0 {
      tracing::warn!(
         hoisted = hoisted_count,
         rewritten,
         dropped,
         unpaired,
         repaired,
         malformed,
         user = %user,
         "reshaped codex-only input items for zen"
      );
   }
}

/// Zen 400s `param: "arguments"` on a call whose arguments are not JSON, and
/// an empty string is not.
fn repair_tool_arguments(rest: &mut Map<String, Value>) -> (usize, usize) {
   let Some(items) = input_items(rest) else {
      return (0, 0);
   };
   let mut repaired = 0;
   let mut malformed = 0;
   for item in items.iter_mut() {
      let call = item
         .get("type")
         .and_then(Value::as_str)
         .is_some_and(|kind| kind.ends_with("_call"));
      if !call {
         continue;
      }
      let Some(arguments) = item.get("arguments").and_then(Value::as_str) else {
         continue;
      };
      if arguments.trim().is_empty() {
         if let Some(item) = item.as_object_mut() {
            item.insert("arguments".into(), Value::String("{}".into()));
            repaired += 1;
         }
      } else if serde_json::from_str::<Value>(arguments).is_err() {
         *item = Value::Null;
         malformed += 1;
      }
   }
   items.retain(|item| !item.is_null());
   (repaired, malformed)
}

/// An unpaired `*_call` or `*_call_output` is a 400 on zen.
fn drop_unpaired_tool_items(rest: &mut Map<String, Value>) -> usize {
   fn side(item: &Value) -> Option<(bool, &str)> {
      let kind = item.get("type")?.as_str()?;
      let call_id = item.get("call_id")?.as_str()?;
      if kind.ends_with("_call_output") {
         Some((false, call_id))
      } else if kind.ends_with("_call") {
         Some((true, call_id))
      } else {
         None
      }
   }

   let Some(items) = input_items(rest) else {
      return 0;
   };
   let mut calls = HashSet::new();
   let mut outputs = HashSet::new();
   for item in items.iter() {
      if let Some((is_call, call_id)) = side(item) {
         if is_call {
            calls.insert(call_id.to_owned());
         } else {
            outputs.insert(call_id.to_owned());
         }
      }
   }
   let before = items.len();
   items.retain(|item| match side(item) {
      Some((true, call_id)) => outputs.contains(call_id),
      Some((false, call_id)) => calls.contains(call_id),
      None => true,
   });
   before - items.len()
}

fn prepare_request(
   state: &AppState,
   auth: &AuthInfo,
   mut req: PassthroughRequest,
) -> Result<(PassthroughRequest, String, Provider), Box<Response>> {
   let requested_model = req
      .model
      .unwrap_or_else(|| state.cfg.models.default.clone());
   let resolved = model_map::resolve(&state.cfg.models, &requested_model);
   // Scope is decided by where the model resolves, not by the endpoint. This
   // surface is the Responses API, which zen speaks as well as codex does.
   let provider = pipeline::admit(
      state,
      auth,
      DIALECT,
      "responses",
      &requested_model,
      &resolved.model,
   )?;
   let responses_native = match provider {
      Provider::OpenAi | Provider::Gemini => true,
      Provider::Zen => state.cfg.models.zen_dialect(&resolved.model) != ZenDialect::Messages,
      Provider::Anthropic
      | Provider::Glm
      | Provider::DeepSeek
      | Provider::Experiential
      | Provider::Copilot => false,
   };
   if !responses_native {
      return Err(Box::new(translation_error(
         DIALECT,
         "this model is not served over the responses api",
      )));
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
   // The parent that writes an inter-agent payload runs on OpenAI, so gating
   // this by provider silences it exactly where it has to fire and the child
   // on another backend receives ciphertext it cannot read. Reserved functions
   // are skipped per tool in strip_encrypted_from_tools instead.
   let renamed = if provider == Provider::OpenAi {
      req.instructions
         .get_or_insert_with(|| state.cfg.codex.instructions());
      let dropped_reasoning = drop_composite_reasoning(&mut req.rest);
      if dropped_reasoning > 0 {
         let user = &auth.user;
         tracing::debug!(dropped_reasoning, %user, "dropped reasoning from another backend");
      }
      rename_reserved_namespace(&mut req.rest)
   } else {
      0
   };
   let flags = strip_encrypted_argument_flags(&mut req.rest);
   let payloads = unwrap_plaintext_agent_payloads(&mut req.rest);
   if renamed + flags + payloads > 0 {
      tracing::debug!(renamed, flags, payloads, user = %auth.user, "kept inter-agent payloads readable");
   }
   if provider == Provider::Zen {
      zen_input_fixups(&mut req.rest, &auth.user);
   }
   req.store = Some(false);
   Ok((req, requested_model, provider))
}

#[derive(Deserialize)]
struct SearchPeek {
   id: Option<String>,
   model: String,
}

/// Codex's web search tool posts here instead of `/responses`, and
/// only the codex backend serves it, whatever model the turn runs on.
pub async fn search(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let peek = match serde_json::from_slice::<SearchPeek>(&body) {
      Ok(peek) => peek,
      Err(err) => return translation_error(DIALECT, &format!("invalid request: {err}")),
   };
   let session_key = peek.id.unwrap_or_else(|| auth.user.clone());
   let route = auth.route(&session_key, &peek.model);
   let served = match state.pools.codex.search(route, body, headers).await {
      Ok(served) => served,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   relay_json(served.response).await
}

/// Reads from the codex backend whose answer is the same for every account.
pub async fn backend_get(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   uri: Uri,
) -> Response {
   let Some(path) = uri.path_and_query().and_then(|path| path.as_str().strip_prefix("/backend-api"))
   else {
      return translation_error(DIALECT, "not a backend-api path");
   };
   let route = auth.route(&auth.user, "");
   let served = match state
      .pools
      .codex
      .backend(route, path.to_owned(), None)
      .await
   {
      Ok(served) => served,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   relay_json(served.response).await
}

async fn relay_json(response: reqwest::Response) -> Response {
   let status = response.status();
   match response.bytes().await {
      Ok(bytes) => (status, [(CONTENT_TYPE, "application/json")], bytes).into_response(),
      Err(err) => error_response(
         DIALECT,
         StatusCode::BAD_GATEWAY,
         "upstream_error",
         &format!("reading upstream response: {err}"),
      ),
   }
}

fn passthrough_record(
   auth: &AuthInfo,
   provider: Provider,
   requested_model: String,
   req: &PassthroughRequest,
   facts: RequestFacts,
   session_key: &str,
) -> UsageRecord {
   let mut record = pipeline::record(
      auth,
      "responses",
      provider,
      requested_model,
      req.model.clone().unwrap_or_default(),
      facts,
   );
   record.effort = req
      .reasoning
      .as_ref()
      .and_then(|reasoning| reasoning.effort.clone())
      .unwrap_or_default();
   record.service_tier = req.service_tier.clone().unwrap_or_default();
   record.session_key = req
      .prompt_cache_key
      .clone()
      .unwrap_or_else(|| session_key.to_owned());
   record
}

pub async fn responses_passthrough(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let started = Instant::now();
   let req = match serde_json::from_slice::<PassthroughRequest>(&body) {
      Ok(req) => req,
      Err(err) => return translation_error(DIALECT, &format!("invalid request: {err}")),
   };
   let (mut req, requested_model, provider) = match prepare_request(&state, &auth, req) {
      Ok(prepared) => prepared,
      Err(response) => return *response,
   };
   let client_streams = req.stream.unwrap_or(false);
   req.stream = Some(true);
   let encoded = match serde_json::to_vec(&req) {
      Ok(value) => Bytes::from(value),
      Err(err) => return translation_error(DIALECT, &format!("serializing request: {err}")),
   };

   let typed = match serde_json::from_slice::<ResponsesRequest>(&encoded) {
      Ok(typed) => Some(typed),
      Err(error) => {
         tracing::warn!(%error, "responses body did not type for the bridge");
         None
      },
   };
   let facts = typed
      .as_ref()
      .map(|typed| RequestFacts::from_responses(typed, &headers))
      .unwrap_or_default();
   let mut record = passthrough_record(&auth, provider, requested_model, &req, facts, &auth.user);
   let route = Route {
      service_tier: req.service_tier.as_deref(),
      ..auth.route(&record.session_key, &record.upstream_model)
   };
   let Dispatched {
      account_id,
      upstream,
      attempts,
   } = match state
      .pools
      .responses_raw(
         &state.cfg.models,
         provider,
         route,
         encoded,
         typed.as_ref(),
         &headers,
      )
      .await
   {
      Ok(dispatched) => dispatched,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   record.account_id = account_id;
   record.attempts = i64::from(attempts);
   let capture = UsageCapture::default();
   let resp = match upstream {
      Upstream::Responses(resp) => {
         if let Some(index) = egress_of(&resp) {
            capture.note_egress(index);
         }
         resp
      },
      bridged @ Upstream::Bridged { .. } => {
         let model = record.upstream_model.clone();
         return bridged_responses(state, record, bridged, model, client_streams, started).await;
      },
   };

   if client_streams {
      let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
      let mut response = relay_stream(resp, guard, capture);
      if provider == Provider::OpenAi {
         let windows = state
            .pools
            .codex
            .pool_windows(&auth.user, auth.limits.pinned_account, None)
            .await;
         rate_limit_headers(&windows, response.headers_mut());
      }
      return response;
   }

   raw_response(state, record, resp, capture, started).await
}

fn upstream_eof(state: &AppState, record: UsageRecord) -> Response {
   log_error(state, record, 502, "upstream_eof");
   error_response(
      DIALECT,
      StatusCode::BAD_GATEWAY,
      "api_error",
      "upstream stream ended unexpectedly",
   )
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
      if let Ok(TerminalEvent {
         kind,
         response: Some(response),
      }) = serde_json::from_str::<TerminalEvent>(&event.data)
         && matches!(
            kind.as_str(),
            "response.completed" | "response.incomplete" | "response.failed"
         )
      {
         final_response = Some(response);
      }
   }
   let snap = capture.snapshot();
   apply_snapshot(&mut record, &snap, started);
   record.response_bytes = final_response
      .as_ref()
      .map_or(0, |value| value.get().len() as i64);
   let Some(value) = final_response else {
      return upstream_eof(&state, record);
   };
   log_usage(&state, record);
   (
      [("content-type", "application/json")],
      restore_reserved_namespace(value.get().to_owned()),
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
      let guard = LogGuard::new(state, capture.clone(), record, started);
      return translated(events, guard, move |event| {
         let Some(event) = event else {
            return Vec::new();
         };
         capture.observe(&event);
         let data = serde_json::to_string(&event).unwrap_or_default();
         capture.note_bytes(data.len());
         vec![Event::default().event(event.kind()).data(data)]
      });
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
      return upstream_eof(&state, record);
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
   response: ResponseObj,
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

/// What codex reads for `/status`.
fn rate_limit_headers(windows: &[UsageWindow], headers: &mut HeaderMap) {
   let mut sorted: Vec<_> = windows.iter().collect();
   sorted.sort_by_key(|window| window_seconds(&window.name).unwrap_or(i64::MAX));
   for (tier, window) in ["primary", "secondary"].iter().zip(sorted) {
      let Some(minutes) = window_seconds(&window.name).map(|secs| secs / 60) else {
         continue;
      };
      let mut insert = |field: &str, value: i64| {
         if let Ok(name) = HeaderName::try_from(format!("x-codex-{tier}-{field}")) {
            headers.insert(name, HeaderValue::from(value));
         }
      };
      insert("window-minutes", minutes);
      insert("used-percent", (window.utilization * 100.0).round() as i64);
      if let Some(resets_at) = window.resets_at {
         insert("reset-at", resets_at);
      }
   }
}

fn relay_stream(resp: reqwest::Response, guard: LogGuard, capture: UsageCapture) -> Response {
   let eof = capture.clone();
   // Never reached if the caller went away first.
   let ended = stream::once(async move { eof.note_upstream_eof() }).filter_map(|()| async { None });
   let stream = resp
      .bytes_stream()
      .eventsource()
      .filter_map(move |event| {
         let _ = &guard;
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
                  if event.data.starts_with(r#"{"type":"error""#)
                     || event.data.starts_with(r#"{"type":"response.failed""#)
                  {
                     let head: String = event.data.chars().take(600).collect();
                     tracing::warn!(frame = %head, "upstream failed inside a 200");
                  }
                  let mut out = Event::default().data(restore_reserved_namespace(event.data));
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
      })
      .chain(ended);
   Sse::new(stream)
      .keep_alive(KeepAlive::default())
      .into_response()
}

#[cfg(test)]
mod tests;
