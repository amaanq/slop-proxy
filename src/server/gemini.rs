use axum::body::{Body, Bytes};
use axum::response::Response;
use futures_util::{StreamExt, stream};

use super::auth::AuthInfo;
use super::error::{Dialect, error_response, pool_error_response, pool_error_status};
use super::relay::forwarded_response;
use super::{AppState, LogGuard, log_error};
use crate::db::usage::UsageRecord;
use crate::gemini::client::GeminiProtocol;
use crate::gemini::native::NativeStream;
use crate::gemini::types::{GenerateContentRequest, GenerateContentResponse, UsageMetadata};
use crate::pool::Route;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::chat::{ChatEnvelope, ChatRequest, StreamOptions};

const DIALECT: Dialect = Dialect::OpenAi;

/// Google's OpenAI-compatible surface speaks the dialect the caller already
/// sent, so the body is relayed rather than translated and only usage is read
/// back out.
pub async fn chat_completions(
    state: AppState,
    auth: AuthInfo,
    mut body: ChatRequest,
    model: String,
    facts: super::facts::RequestFacts,
) -> Response {
    let started = std::time::Instant::now();
    let streaming = body.stream.unwrap_or(false);
    if let Some(effort) = &body.reasoning_effort {
        body.reasoning_effort =
            Some(crate::translate::gemini_bridge::gemini_effort(effort).to_string());
    }
    // Without this the terminal chunk carries no usage and the request bills
    // as zero tokens.
    if streaming {
        body.stream_options = Some(StreamOptions {
            include_usage: true,
        });
    }

    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        provider: Some(Provider::Gemini),
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
        .pools
        .gemini
        .execute(
            Route {
                session_key: &session_key,
                model: &model,
                prefer_trusted: false,
            },
            crate::pool::gemini::Call::OpenAi(Box::new(body)),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, &state.cfg.models, e);
        }
    };
    let protocol = upstream.protocol;
    let resp = upstream.response;
    let mut record = record;
    record.account_id = account_id;
    record.status = resp.status().as_u16() as i64;

    let builder = forwarded_response(&resp);
    let ok = resp.status().is_success();
    if !ok {
        return upstream_rejected(&state, record, builder, resp, started).await;
    }
    if streaming && protocol == GeminiProtocol::Native {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
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
        let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
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
            log_error(&state, record, 502, "upstream_read");
            return error_response(DIALECT, 502, "api_error", &e.to_string());
        }
    };
    let bytes = if protocol == GeminiProtocol::Native {
        match crate::gemini::native::response(&bytes, &model)
            .map_err(|e| e.to_string())
            .and_then(|c| serde_json::to_vec(&c).map_err(|e| e.to_string()))
        {
            Ok(body) => Bytes::from(body),
            Err(error) => {
                log_error(&state, record, 502, "upstream_decode");
                return error_response(DIALECT, 502, "api_error", &error);
            }
        }
    } else {
        bytes
    };
    if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
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
    super::log_usage(&state, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
}

/// A non-2xx carries no SSE frames, so the usage scanner logged a phantom
/// `client_disconnect` and dropped the body.
async fn upstream_rejected(
    state: &AppState,
    mut record: UsageRecord,
    builder: axum::http::response::Builder,
    resp: reqwest::Response,
    started: std::time::Instant,
) -> Response {
    let bytes = resp.bytes().await.unwrap_or_default();
    tracing::warn!(
        user = %record.user,
        model = %record.requested_model,
        dialect = record.dialect,
        status = record.status,
        body = %String::from_utf8_lossy(&bytes).chars().take(2000).collect::<String>(),
        "gemini rejected the request"
    );
    record.error_kind = Some("upstream_rejected".into());
    record.response_bytes = bytes.len() as i64;
    record.duration_ms = Some(started.elapsed().as_millis() as i64);
    super::log_usage(state, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
}

/// Pins a conversation to one account.
fn session_key(user: &str, body: &ChatRequest) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(user.as_bytes());
    if let Some(first) = body.messages.first() {
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
    body: Bytes,
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
    let parsed = match serde_json::from_slice::<GenerateContentRequest>(&body) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                DIALECT,
                400,
                "invalid_request_error",
                &format!("invalid request: {e}"),
            );
        }
    };
    let mut parsed = parsed;
    let body = match crate::gemini::signatures::restore(&mut parsed.contents) {
        true => match serde_json::to_vec(&parsed) {
            Ok(patched) => Bytes::from(patched),
            Err(_) => body,
        },
        false => body,
    };
    let key = native_session_key(&auth.user, &parsed);
    let facts = super::facts::RequestFacts::from_native(&parsed, &headers);
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        provider: Some(Provider::Gemini),
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
        model: model.to_string(),
        action: action.to_string(),
        query,
        body,
    };
    let (account_id, upstream) = match state
        .pools
        .gemini
        .execute(
            Route {
                session_key: &key,
                model,
                prefer_trusted: false,
            },
            call,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(&state, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, &state.cfg.models, e);
        }
    };
    let resp = upstream.response;
    let mut record = record;
    record.account_id = account_id;
    record.status = resp.status().as_u16() as i64;
    let ok = resp.status().is_success();
    let builder = forwarded_response(&resp);
    if !ok {
        return upstream_rejected(&state, record, builder, resp, started).await;
    }

    if streaming {
        let capture = UsageCapture::default();
        let guard = LogGuard::new(state.clone(), capture.clone(), record, started);
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
            log_error(&state, record, 502, "upstream_read");
            return error_response(DIALECT, 502, "api_error", &e.to_string());
        }
    };
    if let Ok(v) = serde_json::from_slice::<GenerateContentResponse>(&bytes) {
        for content in v.candidates.iter().filter_map(|c| c.content.as_ref()) {
            crate::gemini::signatures::remember(&content.parts);
        }
        if let Some(reason) = finish_reason(&v) {
            record.stop_reason = reason.to_string();
        }
        if let Some(u) = &v.usage_metadata {
            apply_native_usage(&mut record, u);
        }
    }
    super::log_usage(&state, record);
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| error_response(DIALECT, 502, "api_error", &e.to_string()))
}

fn finish_reason(chunk: &GenerateContentResponse) -> Option<&'static str> {
    chunk.candidates.first()?.finish_reason.map(|r| r.as_str())
}

fn apply_native_usage(record: &mut UsageRecord, usage: &UsageMetadata) {
    let capture = UsageCapture::default();
    capture.record(&crate::gemini::native::chat_usage(usage).into());
    let snap = capture.snapshot();
    record.input_tokens = snap.input_tokens;
    record.output_tokens = snap.output_tokens;
    record.cache_read_tokens = snap.cache_read_tokens;
    record.reasoning_tokens = snap.reasoning_tokens;
}

/// The native request nests its first turn under `contents`, where the chat
/// dialect uses `messages`.
fn native_session_key(user: &str, body: &GenerateContentRequest) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(user.as_bytes());
    if let Some(first) = body.contents.first() {
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
            let Ok(v) = serde_json::from_slice::<GenerateContentResponse>(data) else {
                continue;
            };
            for content in v.candidates.iter().filter_map(|c| c.content.as_ref()) {
                crate::gemini::signatures::remember(&content.parts);
            }
            if let Some(reason) = finish_reason(&v) {
                self.capture.note_stop_reason(reason);
            }
            if let Some(u) = &v.usage_metadata {
                self.capture
                    .record(&crate::gemini::native::chat_usage(u).into());
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
