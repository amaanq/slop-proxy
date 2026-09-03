use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::response::Response;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::value::RawValue;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response, pool_error_status};
use super::{AppState, LogGuard, log_error, log_usage};
use crate::anthropic::client::RelayHeaders;
use crate::db::usage::UsageRecord;
use crate::pool::Route;
use crate::pool::anthropic::Relay as AnthropicRelay;
use crate::pool::glm::Relay as GlmRelay;
use crate::provider::Provider;
use crate::translate::UsageCapture;

const DIALECT: Dialect = Dialect::Anthropic;

/// The few request fields the proxy itself needs; the body is forwarded
/// verbatim regardless.
pub struct Peek {
    pub model: String,
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
    pub fn from_slice(body: &[u8]) -> Self {
        let peek: PeekBody = serde_json::from_slice(body).unwrap_or_default();
        Self {
            model: peek.model.unwrap_or_default(),
            // Claude Code sends `effort` only when it is choosing the level
            // itself; with adaptive thinking on, `thinking.type` is what
            // carries the same signal.
            effort: peek
                .effort
                .or_else(|| peek.thinking.and_then(|t| t.kind))
                .unwrap_or_default(),
            user_id: peek.metadata.and_then(|m| m.user_id),
            system: peek.system,
        }
    }

    /// Claude Code's metadata.user_id is stable for a session, which is
    /// exactly the granularity upstream prompt caching wants.
    fn session_key(&self, auth: &AuthInfo) -> String {
        if let Some(uid) = &self.user_id {
            return uid.clone();
        }
        struct HashWriter(hmac_sha256::Hash);
        impl std::io::Write for HashWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.update(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let mut h = HashWriter(hmac_sha256::Hash::new());
        h.0.update(auth.user.as_bytes());
        if let Some(system) = &self.system {
            let _ = serde_json::to_writer(&mut h, system);
        }
        let d = h.0.finalize();
        format!(
            "sys-{:016x}",
            u64::from_le_bytes(d[..8].try_into().unwrap())
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
            .and_then(|v| v.to_str().ok())
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
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains(want))
    };
    has("anthropic-beta", "claude-code-") && has("user-agent", "claude-cli/")
}

/// Logs what the caller actually sent, because the payload alone cannot tell
/// a refused harness apart from a Claude Code request missing its headers.
fn not_claude_code(user: &str, headers: &HeaderMap) -> Response {
    let show = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
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
    serde_json::from_slice::<crate::translate::anthropic_req::AnthropicRequest>(body)
        .map(|r| super::facts::RequestFacts::from_anthropic(&r, headers))
        .unwrap_or_else(|_| super::facts::RequestFacts::empty(headers))
}

pub async fn messages(
    state: AppState,
    auth: AuthInfo,
    headers: HeaderMap,
    body: Bytes,
    peek: Peek,
) -> Response {
    let started = std::time::Instant::now();
    let key = peek.session_key(&auth);
    let facts = anthropic_facts(&body, &headers);
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        provider: Some(Provider::Anthropic),
        dialect: "messages",
        requested_model: peek.model.clone(),
        upstream_model: peek.model.clone(),
        effort: peek.effort.clone(),
        status: 200,
        session_key: key.clone(),
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };

    if state.cfg.anthropic.require_claude_code && !is_claude_code(&headers) {
        log_error(&state, record, 403, "not_claude_code");
        return not_claude_code(&auth.user, &headers);
    }

    let hdrs = relay_headers(&headers);
    let (account_id, resp) = match state
        .pools
        .anthropic
        .execute(
            Route {
                session_key: &key,
                model: &peek.model,
                prefer_trusted: false,
            },
            AnthropicRelay {
                path: "/v1/messages",
                body: body.clone(),
                hdrs: hdrs.clone(),
            },
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, &state.cfg.models, e);
        }
    };
    let mut record = record;
    record.account_id = account_id;
    record.status = resp.status().as_u16() as i64;

    let streaming = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));
    let mut builder = forwarded_response(&resp);
    for (name, value) in pool_rate_limit_headers(&state.pools.anthropic.pool_windows().await) {
        builder = builder.header(name, value);
    }

    if streaming {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
        let mut scan = SseScan::new(capture.clone());
        // `chain` only runs if the caller stayed to drain the body.
        let stream = resp
            .bytes_stream()
            .map(move |item| {
                let _ = &guard;
                match &item {
                    Ok(bytes) => scan.feed(bytes),
                    Err(_) => scan.capture().fail("upstream_stream_error"),
                }
                item
            })
            .chain(futures_util::stream::once(async move {
                capture.note_upstream_eof();
                Ok(axum::body::Bytes::new())
            }));
        builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
    } else {
        let ok = resp.status().is_success();
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                log_error(&state, record, 502, "upstream_read");
                return error_response(DIALECT, 502, "api_error", &e.to_string());
            }
        };
        if ok {
            if let Ok(m) = serde_json::from_slice::<MessageEnvelope>(&bytes) {
                record.input_tokens = m.usage.input_tokens;
                record.output_tokens = m.usage.output_tokens;
                record.cache_read_tokens = m.usage.cache_read_input_tokens;
                record.cache_write_tokens = m.usage.cache_creation_input_tokens;
            }
        } else {
            record.error_kind = Some("upstream_error".into());
        }
        log_usage(&state, record);
        builder
            .body(Body::from(bytes))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
    }
}

/// Z.ai answers the messages API directly, so this is the anthropic relay
/// without the subscription guard, which covers Claude Code and not a paid
/// third-party key.
pub async fn glm(
    state: AppState,
    auth: AuthInfo,
    headers: HeaderMap,
    body: Bytes,
    peek: Peek,
) -> Response {
    let started = std::time::Instant::now();
    let key = peek.session_key(&auth);
    let facts = anthropic_facts(&body, &headers);
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        provider: Some(Provider::Glm),
        dialect: "messages",
        requested_model: peek.model.clone(),
        upstream_model: peek.model.clone(),
        effort: peek.effort.clone(),
        status: 200,
        session_key: key.clone(),
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };

    let (account_id, resp) = match state
        .pools
        .glm
        .execute(
            Route {
                session_key: &key,
                model: &peek.model,
                prefer_trusted: false,
            },
            GlmRelay {
                path: "/v1/messages",
                body: body.clone(),
            },
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, &state.cfg.models, e);
        }
    };
    let mut record = record;
    record.account_id = account_id;
    record.status = resp.status().as_u16() as i64;

    let streaming = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));
    let builder = forwarded_response(&resp);

    if streaming {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
        let mut scan = SseScan::new(capture.clone());
        let stream = resp
            .bytes_stream()
            .map(move |item| {
                let _ = &guard;
                match &item {
                    Ok(bytes) => scan.feed(bytes),
                    Err(_) => scan.capture().fail("upstream_stream_error"),
                }
                item
            })
            .chain(futures_util::stream::once(async move {
                capture.note_upstream_eof();
                Ok(axum::body::Bytes::new())
            }));
        return builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()));
    }

    let ok = resp.status().is_success();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            log_error(&state, record, 502, "upstream_read");
            return error_response(DIALECT, 502, "api_error", &e.to_string());
        }
    };
    if ok {
        if let Ok(m) = serde_json::from_slice::<MessageEnvelope>(&bytes) {
            record.input_tokens = m.usage.input_tokens;
            record.output_tokens = m.usage.output_tokens;
            record.cache_read_tokens = m.usage.cache_read_input_tokens;
            record.cache_write_tokens = m.usage.cache_creation_input_tokens;
        }
    } else {
        record.error_kind = Some("upstream_error".into());
    }
    log_usage(&state, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
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
                model: &peek.model,
                prefer_trusted: false,
            },
            AnthropicRelay {
                path: "/v1/messages/count_tokens",
                body: body.clone(),
                hdrs: hdrs.clone(),
            },
        )
        .await
    {
        Ok((_, r)) => r,
        Err(e) => return pool_error_response(DIALECT, &state.cfg.models, e),
    };
    let builder = forwarded_response(&resp);
    match resp.bytes().await {
        Ok(bytes) => builder
            .body(Body::from(bytes))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string())),
        Err(e) => error_response(DIALECT, 502, "api_error", &e.to_string()),
    }
}

pub(super) fn forwarded_response(resp: &reqwest::Response) -> axum::http::response::Builder {
    let mut builder = Response::builder().status(resp.status().as_u16());
    for (name, value) in resp.headers() {
        let n = name.as_str();
        if is_rate_limit_header(n) {
            continue;
        }
        if n == "content-type"
            || n == "request-id"
            || n == "retry-after"
            || n.starts_with("anthropic-")
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
pub(super) fn pool_rate_limit_headers(
    windows: &[crate::pool::UsageWindow],
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut soonest: Option<i64> = None;
    for w in windows {
        let prefix = format!("anthropic-ratelimit-unified-{}", w.name);
        out.push((format!("{prefix}-status"), "allowed".into()));
        out.push((
            format!("{prefix}-utilization"),
            format!("{:.2}", w.utilization),
        ));
        if let Some(resets_at) = w.resets_at {
            out.push((format!("{prefix}-reset"), resets_at.to_string()));
            soonest = Some(soonest.map_or(resets_at, |s: i64| s.min(resets_at)));
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
    fn capture(&self) -> &UsageCapture {
        &self.capture
    }

    fn new(capture: UsageCapture) -> Self {
        Self {
            buf: String::new(),
            interesting: false,
            capture,
        }
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.capture.note_bytes(chunk.len());
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        let mut consumed = 0;
        while let Some(nl) = self.buf[consumed..].find('\n') {
            let line = self.buf[consumed..consumed + nl].trim();
            if let Some(event) = line.strip_prefix("event:") {
                self.capture.note_event(event.trim());
                self.interesting = matches!(
                    event.trim(),
                    "message_start"
                        | "message_delta"
                        | "message_stop"
                        | "content_block_start"
                        | "error"
                );
            } else if self.interesting
                && let Some(data) = line.strip_prefix("data:")
                && let Ok(ev) = serde_json::from_str::<RelayEvent>(data.trim_start())
            {
                apply_event(&self.capture, ev);
            }
            consumed += nl + 1;
        }
        self.buf.drain(..consumed);
    }
}

fn apply_event(capture: &UsageCapture, ev: RelayEvent) {
    let mut c = capture.0.lock().unwrap();
    match ev {
        RelayEvent::MessageStart { message } => {
            c.input_tokens = message.usage.input_tokens;
            c.output_tokens = message.usage.output_tokens;
            c.cache_read_tokens = message.usage.cache_read_input_tokens;
            c.cache_write_tokens = message.usage.cache_creation_input_tokens;
        }
        RelayEvent::MessageDelta { usage, delta } => {
            if let Some(usage) = usage {
                c.output_tokens = usage.output_tokens;
            }
            if let Some(reason) = delta.and_then(|d| d.stop_reason) {
                c.stop_reason = Some(reason);
            }
        }
        RelayEvent::ContentBlockStart { content_block } => {
            if let Some(name) = content_block.name
                && !c.tools_called.contains(&name)
            {
                c.tools_called.push(name);
            }
        }
        RelayEvent::MessageStop => c.completed = true,
        RelayEvent::Error => {
            if c.error_kind.is_none() {
                c.error_kind = Some("upstream_error".into());
            }
        }
        RelayEvent::Other => {}
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(axum::http::HeaderName::from_static(k), v.parse().unwrap());
        }
        h
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
        assert!(!super::is_claude_code(&headers(&[])));
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
    use crate::pool::UsageWindow;

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
        let get = |k: &str| out.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
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
