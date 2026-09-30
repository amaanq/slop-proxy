use std::mem;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::response::Builder;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::value::RawValue;
use tokio::time::timeout;

use crate::anthropic::RelayHeaders;
use crate::config::ModelsConfig;
use crate::pool::anthropic::Relay as AnthropicRelay;
use crate::pool::{PoolError, Relay, Route, Served, UsageWindow};
use crate::provider::Provider;
use crate::server::auth::AuthInfo;
use crate::server::error::{Dialect, error_response, pool_error_response};
use crate::server::facts::RequestFacts;
use crate::server::pipeline::{self, Scan, respond};
use crate::server::{AppState, log_error};
use crate::translate::UsageCapture;
use crate::translate::anthropic_req::AnthropicRequest;
use crate::translate::model_map::resolve;

const DIALECT: Dialect = Dialect::Anthropic;

/// The few request fields the proxy itself needs; the body is forwarded
/// verbatim regardless.
pub struct Peek {
   pub model: String,
   pub upstream_model: String,
   pub effort: String,
   user_id: Option<String>,
   system: Option<Box<RawValue>>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct PeekBody {
   model: Option<String>,
   effort: Option<String>,
   thinking: Option<ThinkingPeek>,
   metadata: Option<MetadataPeek>,
   system: Option<Box<RawValue>>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct ThinkingPeek {
   #[serde(rename = "type")]
   kind: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct MetadataPeek {
   user_id: Option<String>,
}

impl Peek {
   pub fn from_slice(body: &[u8], cfg: &ModelsConfig) -> Self {
      let peek: PeekBody = serde_json::from_slice(body).unwrap_or_default();
      let model = peek.model.unwrap_or_default();
      let resolved = resolve(cfg, &model);
      Self {
         model,
         upstream_model: resolved.model,
         // Claude Code sends `effort` only when it is choosing the level
         // itself; with adaptive thinking on, `thinking.type` is what
         // carries the same signal.
         effort: peek
            .effort
            .or(resolved.effort)
            .or_else(|| peek.thinking?.kind)
            .unwrap_or_default(),
         user_id: peek.metadata.and_then(|meta| meta.user_id),
         system: peek.system,
      }
   }

   /// Claude Code's `metadata.user_id` is stable for a session, which is
   /// exactly the granularity upstream prompt caching wants.
   fn session_key(&self, auth: &AuthInfo) -> String {
      if let Some(uid) = self.user_id.as_ref() {
         return uid.clone();
      }
      let mut hasher = hmac_sha256::Hash::new();
      hasher.update(auth.user.as_bytes());
      if let Some(system) = self.system.as_ref() {
         hasher.update(system.get());
      }
      let digest = hasher.finalize();
      format!(
         "sys-{:016x}",
         u64::from_le_bytes(digest[..8].try_into().unwrap())
      )
   }
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(default)]
struct RelayUsage {
   input_tokens: i64,
   output_tokens: i64,
   cache_read_input_tokens: i64,
   /// Priced above fresh input, so dropping it undercounts the users who
   /// start new sessions most.
   cache_creation_input_tokens: i64,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RelayEvent {
   MessageStart {
      message: MessageEnvelope,
   },
   MessageDelta {
      usage: Option<RelayUsage>,
      delta: Option<StopDelta>,
   },
   ContentBlockStart {
      content_block: ContentBlock,
   },
   MessageStop,
   Error,
   #[serde(other)]
   Other,
}

#[derive(Deserialize, Default)]
struct MessageEnvelope {
   #[serde(default)]
   usage: RelayUsage,
}

#[derive(Deserialize)]
struct StopDelta {
   stop_reason: Option<String>,
}

/// Only the tool's name is read. Its `input` is the caller's shell command or
/// source text, and never reaches the log.
#[derive(Deserialize)]
struct ContentBlock {
   name: Option<String>,
}

pub fn header_str<'map>(headers: &'map HeaderMap, name: &str) -> Option<&'map str> {
   headers.get(name)?.to_str().ok()
}

fn relay_headers(headers: &HeaderMap) -> RelayHeaders {
   let get = |name: &str| header_str(headers, name).map(String::from);
   RelayHeaders {
      version: get("anthropic-version"),
      beta: get("anthropic-beta"),
      user_agent: get("user-agent"),
   }
}

/// The beta and the user agent Claude Code sends on every call. A request
/// missing either is some other client wearing an Anthropic API shape.
fn is_claude_code(headers: &HeaderMap) -> bool {
   let has =
      |name: &str, want: &str| header_str(headers, name).is_some_and(|value| value.contains(want));
   has("anthropic-beta", "claude-code-") && has("user-agent", "claude-cli/")
}

/// Logs what the caller actually sent, because the payload alone cannot tell
/// a refused harness apart from a Claude Code request missing its headers.
fn refuses_non_claude_code(
   state: &AppState,
   auth: &AuthInfo,
   headers: &HeaderMap,
) -> Option<Response> {
   if !state.cfg.anthropic.require_claude_code
      || auth.limits.reserved_only
      || is_claude_code(headers)
   {
      return None;
   }
   let show = |name: &str| header_str(headers, name).unwrap_or("<absent>");
   tracing::warn!(
      "refusing non-claude-code request from {}: user-agent={:?} anthropic-beta={:?}",
      auth.user,
      show("user-agent"),
      show("anthropic-beta"),
   );
   Some(error_response(
      DIALECT,
      StatusCode::FORBIDDEN,
      "permission_error",
      "this proxy serves Anthropic subscriptions, which only cover Claude Code",
   ))
}

/// The body goes upstream untouched, so the parse here only feeds the log.
fn anthropic_facts(body: &[u8], headers: &HeaderMap) -> RequestFacts {
   serde_json::from_slice::<AnthropicRequest>(body).map_or_else(
      |_| RequestFacts::empty(headers),
      |req| RequestFacts::from_anthropic(&req, headers),
   )
}

fn normalized_body(body: &Bytes, peek: &Peek, provider: Provider) -> Bytes {
   if peek.model == peek.upstream_model && provider != Provider::Glm {
      return body.clone();
   }
   let Ok(mut value) = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(body)
   else {
      return body.clone();
   };
   value.insert(
      "model".into(),
      serde_json::Value::String(peek.upstream_model.clone()),
   );
   if provider == Provider::Glm {
      mark_cache_breakpoint(&mut value);
   }
   Bytes::from(serde_json::to_vec(&value).expect("request serializes"))
}

/// `ZCode` marks its last message as a cache breakpoint, so the coding plan
/// bills a cached prefix the way it does for its own client.
fn mark_cache_breakpoint(value: &mut serde_json::Map<String, serde_json::Value>) {
   let Some(last) = value
      .get_mut("messages")
      .and_then(serde_json::Value::as_array_mut)
      .and_then(|messages| messages.last_mut())
   else {
      return;
   };
   if let Some(content) = last.get_mut("content")
      && let serde_json::Value::String(ref mut text) = *content
   {
      let blocks = serde_json::json!([{ "type": "text", "text": mem::take(text) }]);
      *content = blocks;
   }
   let Some(blocks) = last
      .get_mut("content")
      .and_then(serde_json::Value::as_array_mut)
   else {
      return;
   };
   if blocks
      .iter()
      .any(|block| block.get("cache_control").is_some())
   {
      return;
   }
   if let Some(block) = blocks.last_mut().and_then(serde_json::Value::as_object_mut) {
      block.insert(
         "cache_control".into(),
         serde_json::json!({ "type": "ephemeral" }),
      );
   }
}

pub async fn messages(
   state: AppState,
   auth: AuthInfo,
   headers: HeaderMap,
   body: Bytes,
   peek: Peek,
   provider: Provider,
) -> Response {
   let started = Instant::now();
   let facts = anthropic_facts(&body, &headers);
   let mut record = pipeline::record(
      &auth,
      "messages",
      provider,
      peek.model.clone(),
      peek.upstream_model.clone(),
      facts,
   );
   record.effort = peek.effort.clone();
   record.session_key = peek.session_key(&auth);
   if provider == Provider::Anthropic
      && let Some(refused) = refuses_non_claude_code(&state, &auth, &headers)
   {
      log_error(&state, record, 403, "not_claude_code");
      return refused;
   }
   let route = auth.route(&record.session_key, &peek.upstream_model);
   let body = normalized_body(&body, &peek, provider);
   let (served, first) = dispatch(&state, route, provider, body, &headers, &peek).await;
   let limits = if provider == Provider::Anthropic && served.is_ok() {
      pool_rate_limit_headers(
         &state
            .pools
            .anthropic
            .pool_windows(&auth.user, auth.limits.pinned_account, None)
            .await,
      )
   } else {
      Vec::new()
   };
   let streaming = served
      .as_ref()
      .is_ok_and(|served| is_event_stream(&served.response));
   pipeline::forward(
      state,
      record,
      served,
      limits,
      started,
      streaming,
      |capture| SseScan::new(capture, first),
   )
   .await
}

/// Hands the body to whichever pool serves the model. Zen is the one backend
/// that must see its opening frame before the response counts as started, so
/// that frame comes back alongside the result for the caller to re-emit.
async fn dispatch(
   state: &AppState,
   route: Route<'_>,
   provider: Provider,
   body: Bytes,
   headers: &HeaderMap,
   peek: &Peek,
) -> (Result<Served<reqwest::Response>, PoolError>, Option<Bytes>) {
   let mut first = None;
   let pools = &state.pools;
   let relay = move |path| Relay { path, body };
   let result = match provider {
      Provider::Anthropic => {
         let Relay { path, body: bytes } = relay("/v1/messages");
         let hdrs = relay_headers(headers);
         let request = AnthropicRelay {
            path,
            body: bytes,
            hdrs,
         };
         pools.anthropic.execute(route, request).await
      },
      Provider::Glm => pools.glm.execute(route, relay("/v1/messages")).await,
      Provider::DeepSeek => pools.deepseek.execute(route, relay("/v1/messages")).await,
      Provider::Experiential => {
         pools
            .experiential
            .execute(route, relay("/v1/messages"))
            .await
      },
      Provider::Zen => {
         let relay = relay("/messages");
         let opened = timeout(ZEN_FIRST_FRAME, async {
            let served = pools.zen.execute(route, relay).await?;
            let mut resp = served.response;
            let opening = if is_event_stream(&resp) {
               resp
                  .chunk()
                  .await
                  .map_err(|err| PoolError::Upstream(err.to_string()))?
            } else {
               None
            };
            Ok(Served {
               account_id: served.account_id,
               response: (resp, opening),
               attempts: served.attempts,
            })
         })
         .await;
         match opened {
            Ok(Ok(served))
               if served.response.1.is_none() && is_event_stream(&served.response.0) =>
            {
               tracing::warn!(
                  model = %peek.upstream_model,
                  "zen answered 200 and closed the stream without a byte"
               );
               Err(PoolError::Upstream(
                  "zen answered 200 and closed the stream without a byte".into(),
               ))
            },
            Ok(Ok(served)) => {
               first = served.response.1;
               Ok(Served {
                  account_id: served.account_id,
                  response: served.response.0,
                  attempts: served.attempts,
               })
            },
            Ok(Err(err)) => Err(err),
            Err(_) => {
               tracing::warn!(
                  model = %peek.upstream_model,
                  "zen sent no frame before the deadline"
               );
               Err(PoolError::Upstream(
                  "zen sent no frame before the deadline".into(),
               ))
            },
         }
      },
      Provider::OpenAi | Provider::Gemini | Provider::Copilot => Err(PoolError::BadRequest {
         provider,
         model: peek.upstream_model.clone(),
         body: "not served over the messages api".into(),
      }),
   };
   (result, first)
}

const ZEN_FIRST_FRAME: Duration = Duration::from_secs(12);

fn is_event_stream(resp: &reqwest::Response) -> bool {
   header_str(resp.headers(), "content-type")
      .is_some_and(|content_type| content_type.contains("text/event-stream"))
}

pub async fn count_tokens(
   state: AppState,
   auth: AuthInfo,
   headers: HeaderMap,
   body: Bytes,
   peek: Peek,
) -> Response {
   if let Some(refused) = refuses_non_claude_code(&state, &auth, &headers) {
      return refused;
   }

   let key = peek.session_key(&auth);
   let resp = match state
      .pools
      .anthropic
      .execute(
         auth.route(&key, &peek.upstream_model),
         AnthropicRelay {
            path: "/v1/messages/count_tokens",
            body: normalized_body(&body, &peek, Provider::Anthropic),
            hdrs: relay_headers(&headers),
         },
      )
      .await
   {
      Ok(served) => served.response,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   let builder = forwarded_response(&resp);
   match resp.bytes().await {
      Ok(bytes) => respond(builder, DIALECT, Body::from(bytes)),
      Err(err) => error_response(
         DIALECT,
         StatusCode::BAD_GATEWAY,
         "api_error",
         &err.to_string(),
      ),
   }
}

pub fn forwarded_response(resp: &reqwest::Response) -> Builder {
   let mut builder = Response::builder().status(resp.status().as_u16());
   for (name, value) in resp.headers() {
      let key = name.as_str();
      if is_rate_limit_header(key) {
         continue;
      }
      if key == "content-type"
         || key == "request-id"
         || key == "retry-after"
         || key.starts_with("anthropic-")
      {
         builder = builder.header(name, value);
      }
   }
   builder
}

/// These describe whichever account served the turn, and a client shows them
/// as the caller's own quota.
fn is_rate_limit_header(name: &str) -> bool {
   name.starts_with("anthropic-ratelimit-")
}

/// Claude Code warns on the utilization in these, so they carry the pool's
/// figures rather than one account's.
pub fn pool_rate_limit_headers(windows: &[UsageWindow]) -> Vec<(String, String)> {
   let mut out = Vec::new();
   let mut soonest: Option<i64> = None;
   for window in windows {
      let prefix = format!("anthropic-ratelimit-unified-{}", window.name);
      out.push((format!("{prefix}-status"), "allowed".into()));
      out.push((
         format!("{prefix}-utilization"),
         format!("{:.2}", window.utilization),
      ));
      if let Some(resets_at) = window.resets_at {
         out.push((format!("{prefix}-reset"), resets_at.to_string()));
         soonest = Some(soonest.map_or(resets_at, |prev: i64| prev.min(resets_at)));
      }
   }
   if !out.is_empty() {
      out.push((
         "anthropic-ratelimit-unified-status".into(),
         "allowed".into(),
      ));
      if let Some(reset) = soonest {
         out.push((
            "anthropic-ratelimit-unified-reset".into(),
            reset.to_string(),
         ));
      }
   }
   out
}

/// Taps the relayed SSE bytes for usage numbers without altering them. Only
/// the four bookkeeping event types get their JSON parsed; content deltas
/// pass through unparsed.
struct SseScan {
   buf: String,
   interesting: bool,
   capture: UsageCapture,
   first: Option<Bytes>,
}

impl SseScan {
   const fn new(capture: UsageCapture, first: Option<Bytes>) -> Self {
      Self {
         buf: String::new(),
         interesting: false,
         capture,
         first,
      }
   }

   fn feed(&mut self, chunk: &[u8]) {
      self.buf.push_str(&String::from_utf8_lossy(chunk));
      let mut consumed = 0;
      while let Some(newline) = self.buf.get(consumed..).and_then(|text| text.find('\n')) {
         let line = self
            .buf
            .get(consumed..consumed + newline)
            .map_or("", |text| text.trim());
         if let Some(event) = line.strip_prefix("event:") {
            self.capture.note_event(event.trim());
            self.interesting = matches!(
               event.trim(),
               "message_start" | "message_delta" | "message_stop" | "content_block_start" | "error"
            );
         } else if self.interesting
            && let Some(data) = line.strip_prefix("data:")
            && let Ok(event) = serde_json::from_str::<RelayEvent>(data.trim_start())
         {
            apply_event(&self.capture, event);
         }
         consumed += newline + 1;
      }
      self.buf.drain(..consumed);
   }
}

impl Scan for SseScan {
   const DIALECT: Dialect = DIALECT;
   const REJECTED: &'static str = "upstream_error";

   fn chunk(&mut self, bytes: Bytes) -> Bytes {
      self.feed(&bytes);
      bytes
   }

   fn body(&mut self, bytes: Bytes) -> Result<Bytes, String> {
      if let Ok(message) = serde_json::from_slice::<MessageEnvelope>(&bytes) {
         apply_event(&self.capture, RelayEvent::MessageStart { message });
      }
      Ok(bytes)
   }

   fn head(&mut self) -> Option<Bytes> {
      self.first.take()
   }
}

fn apply_event(capture: &UsageCapture, event: RelayEvent) {
   let mut guard = capture.0.lock().unwrap();
   match event {
      RelayEvent::MessageStart { message } => {
         guard.input_tokens = message.usage.input_tokens;
         guard.output_tokens = message.usage.output_tokens;
         guard.cache_read_tokens = message.usage.cache_read_input_tokens;
         guard.cache_write_tokens = message.usage.cache_creation_input_tokens;
      },
      RelayEvent::MessageDelta { usage, delta } => {
         if let Some(usage) = usage {
            guard.output_tokens = usage.output_tokens;
            // Zen reports a streamed turn's input as zero in message_start
            // and only settles it here, where anthropic sends nothing.
            if usage.input_tokens > 0 {
               guard.input_tokens = usage.input_tokens;
               guard.cache_read_tokens = usage.cache_read_input_tokens;
               guard.cache_write_tokens = usage.cache_creation_input_tokens;
            }
         }
         if let Some(reason) = delta.and_then(|stop| stop.stop_reason) {
            guard.stop_reason = Some(reason);
         }
      },
      RelayEvent::ContentBlockStart { content_block } => {
         if let Some(name) = content_block.name
            && !guard.tools_called.contains(&name)
         {
            guard.tools_called.push(name);
         }
      },
      RelayEvent::MessageStop => guard.completed = true,
      RelayEvent::Error => {
         if guard.error_kind.is_none() {
            guard.error_kind = Some("upstream_error".into());
         }
      },
      RelayEvent::Other => {},
   }
}

#[cfg(test)]
mod tests {
   use axum::http::{HeaderMap, HeaderName};

   fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
      let mut map = HeaderMap::new();
      for &(key, value) in pairs {
         map.insert(HeaderName::from_static(key), value.parse().unwrap());
      }
      map
   }

   /// Captured from Claude Code 2.1.252.
   #[test]
   fn a_real_claude_code_request_passes() {
      assert!(super::is_claude_code(&headers(&[
         (
            "anthropic-beta",
            "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27"
         ),
         ("user-agent", "claude-cli/2.1.252 (external, cli)"),
      ])));
      assert!(super::is_claude_code(&headers(&[
         ("anthropic-beta", "claude-code-20250219"),
         ("user-agent", "claude-cli/2.1.252 (external, sdk-cli)"),
      ])));
   }

   #[test]
   fn another_harness_is_refused() {
      assert!(!super::is_claude_code(&headers(&[(
         "user-agent",
         "claude-cli/2.1.252 (external, cli)"
      )])));
      assert!(!super::is_claude_code(&headers(&[(
         "anthropic-beta",
         "claude-code-20250219"
      )])));
      assert!(!super::is_claude_code(&headers(&[
         ("anthropic-beta", "oauth-2025-04-20"),
         ("user-agent", "python-httpx/0.27"),
      ])));
   }

   #[test]
   fn only_glm_bodies_are_rewritten() {
      use crate::config::ModelsConfig;
      use crate::provider::Provider;

      let body = super::Bytes::from_static(
         br#"{"model":"glm-5","messages":[{"role":"user","content":"hi"}]}"#,
      );
      let peek = super::Peek::from_slice(&body, &ModelsConfig::default());
      let anthropic = super::normalized_body(&body, &peek, Provider::Anthropic);
      assert_eq!(anthropic, body);
      let glm = super::normalized_body(&body, &peek, Provider::Glm);
      let parsed: serde_json::Value = serde_json::from_slice(&glm).unwrap();
      assert_eq!(
         parsed["messages"][0]["content"][0]["cache_control"]["type"],
         "ephemeral"
      );
   }
}

#[cfg(test)]
mod pool_header_tests {
   use super::*;

   #[test]
   fn one_accounts_quota_never_reaches_the_caller() {
      assert!(is_rate_limit_header(
         "anthropic-ratelimit-unified-7d-utilization"
      ));
      assert!(is_rate_limit_header("anthropic-ratelimit-unified-status"));
      assert!(!is_rate_limit_header("anthropic-version"));
      assert!(!is_rate_limit_header("anthropic-beta"));
   }

   #[test]
   fn the_pool_headers_name_each_window() {
      let out = pool_rate_limit_headers(&[
         UsageWindow {
            name: "5h".into(),
            utilization: 0.5,
            resets_at: Some(100),
         },
         UsageWindow {
            name: "7d".into(),
            utilization: 0.25,
            resets_at: Some(900),
         },
      ]);
      let get = |key: &str| {
         out.iter()
            .find(|&&(ref name, _)| name == key)
            .map(|&(_, ref value)| value.as_str())
      };
      assert_eq!(
         get("anthropic-ratelimit-unified-5h-utilization"),
         Some("0.50")
      );
      assert_eq!(
         get("anthropic-ratelimit-unified-7d-utilization"),
         Some("0.25")
      );
      // The summary reset is the soonest of them, not the last one seen.
      assert_eq!(get("anthropic-ratelimit-unified-reset"), Some("100"));
   }

   #[test]
   fn no_windows_emits_nothing_rather_than_a_false_allowed() {
      assert!(pool_rate_limit_headers(&[]).is_empty());
   }
}
