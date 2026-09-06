use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::http::response::Builder;
use axum::response::Response;
use serde::Deserialize;
use serde_json::value::RawValue;
use std::io::{Result, Write};
use std::time::Instant;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response};
use super::pipeline::{dispatch_failed, read_body, relayed};
use super::{AppState, LogGuard, log_error, log_usage};
use crate::anthropic::client::RelayHeaders;
use crate::config::ModelsConfig;
use crate::db::usage::UsageRecord;
use crate::pool::anthropic::Relay as AnthropicRelay;
use crate::pool::experiential::Relay as ExperientialRelay;
use crate::pool::glm::Relay as GlmRelay;
use crate::pool::{PoolError, Route, UsageWindow};
use crate::provider::Provider;
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
      struct HashWriter(hmac_sha256::Hash);
      impl Write for HashWriter {
         fn write(&mut self, buf: &[u8]) -> Result<usize> {
            self.0.update(buf);
            Ok(buf.len())
         }
         fn flush(&mut self) -> Result<()> {
            Ok(())
         }
      }

      if let Some(uid) = self.user_id.as_ref() {
         return uid.clone();
      }
      let mut hasher = HashWriter(hmac_sha256::Hash::new());
      hasher.0.update(auth.user.as_bytes());
      if let Some(system) = self.system.as_ref() {
         let _ = serde_json::to_writer(&mut hasher, system);
      }
      let digest = hasher.0.finalize();
      format!(
         "sys-{:016x}",
         u64::from_le_bytes(digest[..8].try_into().unwrap())
      )
   }
}

#[derive(Deserialize, Default, Clone, Copy)]
struct RelayUsage {
   #[serde(default)]
   input_tokens: i64,
   #[serde(default)]
   output_tokens: i64,
   #[serde(default)]
   cache_read_input_tokens: i64,
   /// Priced above fresh input, so dropping it undercounts the users who
   /// start new sessions most.
   #[serde(default)]
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

fn relay_headers(headers: &HeaderMap) -> RelayHeaders {
   let get = |name: &str| {
      headers
         .get(name)
         .and_then(|value| value.to_str().ok())
         .map(String::from)
   };
   RelayHeaders {
      version: get("anthropic-version"),
      beta: get("anthropic-beta"),
      user_agent: get("user-agent"),
   }
}

/// The beta and the user agent Claude Code sends on every call. A request
/// missing either is some other client wearing an Anthropic API shape.
fn is_claude_code(headers: &HeaderMap) -> bool {
   let has = |name: &str, want: &str| {
      headers
         .get(name)
         .and_then(|value| value.to_str().ok())
         .is_some_and(|value| value.contains(want))
   };
   has("anthropic-beta", "claude-code-") && has("user-agent", "claude-cli/")
}

/// Logs what the caller actually sent, because the payload alone cannot tell
/// a refused harness apart from a Claude Code request missing its headers.
fn not_claude_code(user: &str, headers: &HeaderMap) -> Response {
   let show = |name: &str| {
      headers
         .get(name)
         .and_then(|value| value.to_str().ok())
         .unwrap_or("<absent>")
   };
   tracing::warn!(
      "refusing non-claude-code request from {user}: user-agent={:?} anthropic-beta={:?}",
      show("user-agent"),
      show("anthropic-beta"),
   );
   error_response(
      DIALECT,
      403,
      "permission_error",
      "this proxy serves Anthropic subscriptions, which only cover Claude Code",
   )
}

/// The body goes upstream untouched, so the parse here only feeds the log.
fn anthropic_facts(body: &[u8], headers: &HeaderMap) -> super::facts::RequestFacts {
   serde_json::from_slice::<AnthropicRequest>(body).map_or_else(
      |_| super::facts::RequestFacts::empty(headers),
      |req| super::facts::RequestFacts::from_anthropic(&req, headers),
   )
}

fn normalized_body(body: &Bytes, peek: &Peek) -> Bytes {
   if peek.model == peek.upstream_model {
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
   Bytes::from(serde_json::to_vec(&value).expect("request serializes"))
}

/// Z.ai answers the messages API directly, so this is the anthropic relay
/// without the subscription guard, which covers Claude Code and not a paid
/// third-party key.
/// The Experiential gateway answers the messages API directly, same shape
/// as the z.ai relay.
pub async fn messages(
   state: AppState,
   auth: AuthInfo,
   headers: HeaderMap,
   body: Bytes,
   peek: Peek,
   provider: Provider,
) -> Response {
   let started = Instant::now();
   let key = peek.session_key(&auth);
   let facts = anthropic_facts(&body, &headers);
   let mut record = super::pipeline::record(
      &auth,
      "messages",
      provider,
      peek.model.clone(),
      peek.upstream_model.clone(),
      facts,
   );
   record.effort = peek.effort.clone();
   record.session_key = key.clone();
   if provider == Provider::Anthropic
      && state.cfg.anthropic.require_claude_code
      && !is_claude_code(&headers)
   {
      log_error(&state, record, 403, "not_claude_code");
      return not_claude_code(&auth.user, &headers);
   }
   let route = Route {
      session_key: &key,
      model: &peek.upstream_model,
      user: &auth.user,
      pinned_account: auth.limits.pinned_account,
      prefer_trusted: false,
   };
   let body = normalized_body(&body, &peek);
   let result = match provider {
      Provider::Anthropic => {
         state
            .pools
            .anthropic
            .execute(
               route,
               AnthropicRelay {
                  path: "/v1/messages",
                  body,
                  hdrs: relay_headers(&headers),
               },
            )
            .await
      },
      Provider::Glm => {
         state
            .pools
            .glm
            .execute(
               route,
               GlmRelay {
                  path: "/v1/messages",
                  body,
               },
            )
            .await
      },
      Provider::Experiential => {
         state
            .pools
            .experiential
            .execute(
               route,
               ExperientialRelay {
                  path: "/v1/messages",
                  body,
               },
            )
            .await
      },
      Provider::OpenAi | Provider::Gemini | Provider::Zen => Err(PoolError::BadRequest {
         provider,
         model: peek.upstream_model.clone(),
         body: "not served over the messages api".into(),
      }),
   };
   let (account_id, resp) = match result {
      Ok(result) => result,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   record.account_id = account_id;
   record.status = i64::from(resp.status().as_u16());
   let mut builder = forwarded_response(&resp);
   if provider == Provider::Anthropic {
      for (name, value) in pool_rate_limit_headers(&state.pools.anthropic.pool_windows().await) {
         builder = builder.header(name, value);
      }
   }
   relay_response(state, record, resp, builder, started).await
}

async fn relay_response(
   state: AppState,
   mut record: UsageRecord,
   resp: reqwest::Response,
   builder: Builder,
   started: Instant,
) -> Response {
   let streaming = resp
      .headers()
      .get("content-type")
      .and_then(|value| value.to_str().ok())
      .is_some_and(|content_type| content_type.contains("text/event-stream"));

   if streaming {
      let capture = UsageCapture::default();
      let mut scan = SseScan::new(capture.clone());
      return relayed(
         builder,
         resp,
         LogGuard::new(state, capture.clone(), record, started),
         capture,
         DIALECT,
         move |bytes| {
            scan.feed(&bytes);
            bytes
         },
         Bytes::new,
      );
   }

   let ok = resp.status().is_success();
   let bytes = match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   if ok {
      if let Ok(msg) = serde_json::from_slice::<MessageEnvelope>(&bytes) {
         record.input_tokens = msg.usage.input_tokens;
         record.output_tokens = msg.usage.output_tokens;
         record.cache_read_tokens = msg.usage.cache_read_input_tokens;
         record.cache_write_tokens = msg.usage.cache_creation_input_tokens;
      }
   } else {
      record.error_kind = Some("upstream_error".into());
   }
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   record.response_bytes = bytes.len() as i64;
   log_usage(&state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}

pub async fn count_tokens(
   state: AppState,
   auth: AuthInfo,
   headers: HeaderMap,
   body: Bytes,
   peek: Peek,
) -> Response {
   if state.cfg.anthropic.require_claude_code && !is_claude_code(&headers) {
      return not_claude_code(&auth.user, &headers);
   }

   let hdrs = relay_headers(&headers);
   let key = peek.session_key(&auth);
   let resp = match state
      .pools
      .anthropic
      .execute(
         Route {
            session_key: &key,
            model: &peek.upstream_model,
            user: &auth.user,
            pinned_account: auth.limits.pinned_account,
            prefer_trusted: false,
         },
         AnthropicRelay {
            path: "/v1/messages/count_tokens",
            body: normalized_body(&body, &peek),
            hdrs: hdrs.clone(),
         },
      )
      .await
   {
      Ok((_, resp)) => resp,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   let builder = forwarded_response(&resp);
   match resp.bytes().await {
      Ok(bytes) => builder
         .body(Body::from(bytes))
         .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string())),
      Err(err) => error_response(DIALECT, 502, "api_error", &err.to_string()),
   }
}

pub(super) fn forwarded_response(resp: &reqwest::Response) -> Builder {
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
pub(super) fn pool_rate_limit_headers(windows: &[UsageWindow]) -> Vec<(String, String)> {
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
}

impl SseScan {
   const fn new(capture: UsageCapture) -> Self {
      Self {
         buf: String::new(),
         interesting: false,
         capture,
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
