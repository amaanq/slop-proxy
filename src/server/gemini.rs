use axum::body::{Body, Bytes};
use axum::response::Response;
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::Value;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response, pool_error_status};
use super::relay::forwarded_response;
use super::{AppState, LogGuard, log_error};
use crate::codex::types::{TokenDetails, Usage};
use crate::db::usage::UsageRecord;
use crate::gemini::client::GeminiProtocol;
use crate::gemini::native::NativeStream;
use crate::translate::UsageCapture;

const DIALECT: Dialect = Dialect::OpenAi;

/// The chat-completions usage block, which names the same quantities the
/// Responses API reports under different keys.
#[derive(Debug, Default, Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: i64,
    #[serde(default)]
    completion_tokens: i64,
    #[serde(default)]
    total_tokens: i64,
    #[serde(default)]
    prompt_tokens_details: TokenDetails,
    #[serde(default)]
    completion_tokens_details: TokenDetails,
}

impl From<ChatUsage> for Usage {
    /// Google leaves thinking out of `completion_tokens` and reports it only
    /// in `total_tokens`.
    fn from(c: ChatUsage) -> Self {
        let billed_output = (c.total_tokens - c.prompt_tokens).max(c.completion_tokens);
        let mut output_details = c.completion_tokens_details;
        if output_details.reasoning_tokens == 0 {
            output_details.reasoning_tokens = billed_output - c.completion_tokens;
        }
        Usage {
            input_tokens: c.prompt_tokens,
            output_tokens: billed_output,
            input_tokens_details: c.prompt_tokens_details,
            output_tokens_details: output_details,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatEnvelope {
    usage: Option<ChatUsage>,
}

/// Google's OpenAI-compatible surface speaks the dialect the caller already
/// sent, so the body is relayed rather than translated and only usage is read
/// back out.
pub async fn chat_completions(
    state: AppState,
    auth: AuthInfo,
    mut body: Value,
    model: String,
    facts: super::facts::RequestFacts,
) -> Response {
    let started = std::time::Instant::now();
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    // Without this the terminal chunk carries no usage and the request bills
    // as zero tokens.
    if streaming && let Some(obj) = body.as_object_mut() {
        obj.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
    }

    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "chat",
        requested_model: model.clone(),
        upstream_model: model.clone(),
        status: 200,
        session_key: session_key(&auth.user, &body),
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };

    let session_key = record.session_key.clone();
    let (account_id, upstream) = match state
        .gemini
        .execute(crate::pool::gemini::Call::OpenAi(&body), &session_key)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state.db, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, e);
        }
    };
    let protocol = upstream.protocol;
    let resp = upstream.response;
    let mut record = record;
    record.account_id = Some(account_id);
    record.status = resp.status().as_u16() as i64;

    let builder = forwarded_response(&resp);
    let ok = resp.status().is_success();
    if streaming && ok && protocol == GeminiProtocol::Native {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(
            state.db.clone(),
            state.prices.clone(),
            capture.clone(),
            record,
            started,
        );
        let mut native = NativeStream::new(&model);
        let mut scan = ChatUsageScan::new(capture);
        let stream = resp
            .bytes_stream()
            .map(move |item| {
                let _ = &guard;
                match item {
                    Ok(bytes) => native
                        .feed(&bytes)
                        .into_iter()
                        .map(|bytes| {
                            scan.feed(&bytes);
                            Ok::<Bytes, reqwest::Error>(Bytes::from(bytes))
                        })
                        .collect::<Vec<_>>(),
                    Err(error) => vec![Err(error)],
                }
            })
            .flat_map(stream::iter);
        return builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()));
    }
    if streaming && protocol == GeminiProtocol::OpenAi {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(
            state.db.clone(),
            state.prices.clone(),
            capture.clone(),
            record,
            started,
        );
        let mut scan = ChatUsageScan::new(capture);
        let stream = resp.bytes_stream().map(move |item| {
            let _ = &guard;
            if let Ok(bytes) = &item {
                scan.feed(bytes);
            }
            item
        });
        return builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()));
    }

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            log_error(&state.db, record, 502, "upstream_read");
            return error_response(DIALECT, 502, "api_error", &e.to_string());
        }
    };
    let bytes = if ok && protocol == GeminiProtocol::Native {
        match crate::gemini::native::response(&bytes, &model) {
            Ok(body) => Bytes::from(body),
            Err(error) => {
                log_error(&state.db, record, 502, "upstream_decode");
                return error_response(DIALECT, 502, "api_error", &error);
            }
        }
    } else {
        bytes
    };
    if ok
        && let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
        && let Some(u) = env.usage
    {
        let capture = UsageCapture::default();
        capture.record(&u.into());
        let snap = capture.snapshot();
        record.input_tokens = snap.input_tokens;
        record.output_tokens = snap.output_tokens;
        record.cache_read_tokens = snap.cache_read_tokens;
        record.reasoning_tokens = snap.reasoning_tokens;
    }
    super::log_usage(&state.db, &state.prices, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
}

/// Pins a conversation to one account.
fn session_key(user: &str, body: &Value) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(user.as_bytes());
    if let Some(first) = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|m| m.first())
    {
        h.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
    }
    data_encoding::HEXLOWER.encode(&h.finalize())
}

/// Reads usage out of the `data:` frames of a chat stream. Only the terminal
/// frame carries it, so every frame is tried and the last one wins.
struct ChatUsageScan {
    capture: UsageCapture,
    buf: Vec<u8>,
}

impl ChatUsageScan {
    fn new(capture: UsageCapture) -> Self {
        Self {
            capture,
            buf: Vec::new(),
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if data == b"[DONE]" {
                continue;
            }
            if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(data)
                && let Some(u) = env.usage
            {
                self.capture.record(&u.into());
            }
        }
    }
}

/// The native surface Gemini CLI speaks. Nothing is translated in either
/// direction, so the reply is byte-identical to Google's and only usage is
/// read out of it on the way past.
pub async fn native(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::Extension(auth): axum::Extension<AuthInfo>,
    axum::extract::Path(spec): axum::extract::Path<String>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> Response {
    let Some((model, action)) = spec.split_once(':') else {
        return error_response(
            DIALECT,
            404,
            "invalid_request_error",
            "expected /v1beta/models/{model}:{generateContent|streamGenerateContent}",
        );
    };
    if !matches!(action, "generateContent" | "streamGenerateContent") {
        return error_response(
            DIALECT,
            404,
            "invalid_request_error",
            "unsupported action on the native surface",
        );
    }
    if state.cfg.models.route(model) != crate::provider::Provider::Gemini {
        return error_response(
            DIALECT,
            400,
            "invalid_request_error",
            "this model is not served by the gemini backend",
        );
    }

    let started = std::time::Instant::now();
    if !auth.may_use(crate::provider::Provider::Gemini) {
        return super::error::out_of_scope(DIALECT, crate::provider::Provider::Gemini);
    }
    let streaming = action == "streamGenerateContent";
    let key = native_session_key(&auth.user, &body);
    let facts = super::facts::RequestFacts::extract(&body, &headers);
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "native",
        requested_model: model.to_string(),
        upstream_model: model.to_string(),
        status: 200,
        session_key: key.clone(),
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };
    let call = crate::pool::gemini::Call::Native {
        model,
        action,
        query: query.as_deref(),
        body: &body,
    };
    let (account_id, upstream) = match state.gemini.execute(call, &key).await {
        Ok(r) => r,
        Err(e) => {
            log_error(&state.db, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, e);
        }
    };
    let resp = upstream.response;
    let mut record = record;
    record.account_id = Some(account_id);
    record.status = resp.status().as_u16() as i64;
    let ok = resp.status().is_success();
    let builder = forwarded_response(&resp);

    if streaming {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(
            state.db.clone(),
            state.prices.clone(),
            capture.clone(),
            record,
            started,
        );
        let mut scan = NativeUsageScan::new(capture);
        let stream = resp.bytes_stream().map(move |item| {
            let _ = &guard;
            if let Ok(bytes) = &item {
                scan.feed(bytes);
            }
            item
        });
        return builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()));
    }

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            log_error(&state.db, record, 502, "upstream_read");
            return error_response(DIALECT, 502, "api_error", &e.to_string());
        }
    };
    if ok && let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
        if let Some(reason) = finish_reason(&v) {
            record.stop_reason = reason.to_ascii_lowercase();
        }
        if let Some(u) = v.get("usageMetadata") {
            apply_native_usage(&mut record, u);
        }
    }
    super::log_usage(&state.db, &state.prices, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
}

fn finish_reason(chunk: &Value) -> Option<&str> {
    chunk
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
}

fn apply_native_usage(record: &mut UsageRecord, usage: &Value) {
    let Ok(chat) = serde_json::from_value::<ChatUsage>(crate::gemini::native::usage_value(usage))
    else {
        return;
    };
    let capture = UsageCapture::default();
    capture.record(&chat.into());
    let snap = capture.snapshot();
    record.input_tokens = snap.input_tokens;
    record.output_tokens = snap.output_tokens;
    record.cache_read_tokens = snap.cache_read_tokens;
    record.reasoning_tokens = snap.reasoning_tokens;
}

/// The native request nests its first turn under `contents`, where the chat
/// dialect uses `messages`.
fn native_session_key(user: &str, body: &Value) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(user.as_bytes());
    if let Some(first) = body
        .get("contents")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    {
        h.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
    }
    data_encoding::HEXLOWER.encode(&h.finalize())
}

/// Reads `usageMetadata` out of a native SSE stream. Only the terminal chunk
/// carries totals, so every frame is tried and the last one wins.
struct NativeUsageScan {
    capture: UsageCapture,
    buf: Vec<u8>,
}

impl NativeUsageScan {
    fn new(capture: UsageCapture) -> Self {
        Self {
            capture,
            buf: Vec::new(),
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            let Ok(v) = serde_json::from_slice::<Value>(data) else {
                continue;
            };
            if let Some(reason) = finish_reason(&v) {
                self.capture.note_stop_reason(&reason.to_ascii_lowercase());
            }
            if let Some(u) = v.get("usageMetadata")
                && let Ok(chat) =
                    serde_json::from_value::<ChatUsage>(crate::gemini::native::usage_value(u))
            {
                self.capture.record(&chat.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terminal_frame_supplies_usage() {
        let capture = UsageCapture::default();
        let mut scan = ChatUsageScan::new(capture.clone());
        scan.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        scan.feed(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":7,\
              \"total_tokens\":107,\"prompt_tokens_details\":{\"cached_tokens\":40}}}\n\n",
        );
        scan.feed(b"data: [DONE]\n\n");
        let snap = capture.snapshot();
        // Cached tokens come out of prompt_tokens so the two bill separately.
        assert_eq!(snap.input_tokens, 60);
        assert_eq!(snap.cache_read_tokens, 40);
        assert_eq!(snap.output_tokens, 7);
    }

    #[test]
    fn thinking_left_out_of_completion_tokens_is_recovered() {
        let capture = UsageCapture::default();
        let mut scan = ChatUsageScan::new(capture.clone());
        scan.feed(
            b"data: {\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":10,\
              \"total_tokens\":309}}\n",
        );
        let snap = capture.snapshot();
        assert_eq!(snap.input_tokens, 13);
        assert_eq!(snap.output_tokens, 296);
        assert_eq!(snap.reasoning_tokens, 286);
    }

    #[test]
    fn a_frame_split_across_chunks_still_parses() {
        let capture = UsageCapture::default();
        let mut scan = ChatUsageScan::new(capture.clone());
        scan.feed(b"data: {\"usage\":{\"prompt_tokens\":10,");
        scan.feed(b"\"completion_tokens\":2,\"total_tokens\":12}}\n");
        let snap = capture.snapshot();
        assert_eq!(snap.input_tokens, 10);
        assert_eq!(snap.output_tokens, 2);
    }
}
