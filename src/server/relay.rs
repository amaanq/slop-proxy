use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response, pool_error_status};
use super::{AppState, LogGuard, log_error, log_usage};
use crate::anthropic::client::RelayHeaders;
use crate::db::usage::UsageRecord;
use crate::translate::UsageCapture;

const DIALECT: Dialect = Dialect::Anthropic;

/// The few request fields the proxy itself needs; the body is forwarded
/// verbatim regardless.
pub struct Peek {
    pub model: String,
    pub effort: String,
    user_id: Option<String>,
}

impl Peek {
    pub fn from_body(body: &Value) -> Self {
        let get = |ptr: &str| body.pointer(ptr).and_then(Value::as_str).map(String::from);
        Self {
            model: get("/model").unwrap_or_default(),
            // Claude Code sends `effort` only when it is choosing the level
            // itself; with adaptive thinking on, `thinking.type` is what
            // carries the same signal.
            effort: get("/effort")
                .or_else(|| get("/thinking/type"))
                .unwrap_or_default(),
            user_id: get("/metadata/user_id"),
        }
    }

    /// Claude Code's metadata.user_id is stable for a session, which is
    /// exactly the granularity upstream prompt caching wants.
    fn session_key(&self, body: &Value, auth: &AuthInfo) -> String {
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
        if let Some(system) = body.get("system") {
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

pub async fn messages(
    state: AppState,
    auth: AuthInfo,
    headers: HeaderMap,
    body: Value,
    peek: Peek,
) -> Response {
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "messages",
        requested_model: peek.model.clone(),
        upstream_model: peek.model.clone(),
        effort: peek.effort.clone(),
        status: 200,
        ..Default::default()
    };

    if state.cfg.anthropic.require_claude_code && !is_claude_code(&headers) {
        log_error(&state.db, record, 403, "not_claude_code");
        return not_claude_code(&auth.user, &headers);
    }

    let hdrs = relay_headers(&headers);
    let key = peek.session_key(&body, &auth);
    let (account_id, resp) = match state
        .anthropic
        .execute("/v1/messages", &body, &hdrs, &key)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state.db, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, e);
        }
    };
    let mut record = record;
    record.account_id = Some(account_id);
    record.status = resp.status().as_u16() as i64;

    let streaming = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));
    let builder = forwarded_response(&resp);

    if streaming {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(state.db.clone(), state.prices.clone(), capture.clone(), record);
        let mut scan = SseScan::new(capture);
        let stream = resp.bytes_stream().map(move |item| {
            let _ = &guard;
            if let Ok(bytes) = &item {
                scan.feed(bytes);
            }
            item
        });
        builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
    } else {
        let ok = resp.status().is_success();
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                log_error(&state.db, record, 502, "upstream_read");
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
        log_usage(&state.db, &state.prices, record);
        builder
            .body(Body::from(bytes))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
    }
}

pub async fn count_tokens(
    state: AppState,
    auth: AuthInfo,
    headers: HeaderMap,
    body: Value,
    peek: Peek,
) -> Response {
    if state.cfg.anthropic.require_claude_code && !is_claude_code(&headers) {
        return not_claude_code(&auth.user, &headers);
    }

    let hdrs = relay_headers(&headers);
    let key = peek.session_key(&body, &auth);
    let resp = match state
        .anthropic
        .execute(
            "/v1/messages/count_tokens",
            &body,
            &hdrs,
            &key,
        )
        .await
    {
        Ok((_, r)) => r,
        Err(e) => return pool_error_response(DIALECT, e),
    };
    let builder = forwarded_response(&resp);
    match resp.bytes().await {
        Ok(bytes) => builder
            .body(Body::from(bytes))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string())),
        Err(e) => error_response(DIALECT, 502, "api_error", &e.to_string()),
    }
}

fn forwarded_response(resp: &reqwest::Response) -> axum::http::response::Builder {
    let mut builder = Response::builder().status(resp.status().as_u16());
    for (name, value) in resp.headers() {
        let n = name.as_str();
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

/// Taps the relayed SSE bytes for usage numbers without altering them. Only
/// the four bookkeeping event types get their JSON parsed; content deltas
/// pass through unparsed.
struct SseScan {
    buf: String,
    interesting: bool,
    capture: UsageCapture,
}

impl SseScan {
    fn new(capture: UsageCapture) -> Self {
        Self {
            buf: String::new(),
            interesting: false,
            capture,
        }
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        let mut consumed = 0;
        while let Some(nl) = self.buf[consumed..].find('\n') {
            let line = self.buf[consumed..consumed + nl].trim();
            if let Some(event) = line.strip_prefix("event:") {
                self.interesting = matches!(
                    event.trim(),
                    "message_start" | "message_delta" | "message_stop" | "error"
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
        RelayEvent::MessageDelta { usage: Some(usage) } => {
            c.output_tokens = usage.output_tokens;
        }
        RelayEvent::MessageDelta { usage: None } => {}
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
            h.insert(
                axum::http::HeaderName::from_static(k),
                v.parse().unwrap(),
            );
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
