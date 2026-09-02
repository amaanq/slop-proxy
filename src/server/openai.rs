use std::collections::VecDeque;
use std::convert::Infallible;

use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

use super::auth::AuthInfo;
use super::error::{Dialect, pool_error_response, pool_error_status, translation_error};
use super::{AppState, LogGuard, cache_key, log_error, log_rejected, log_usage};
use crate::codex::sse::{EventStream, event_stream};
use crate::codex::types::Usage;
use crate::db::usage::UsageRecord;
use crate::provider::Provider;
use crate::translate::openai_req::{self, OpenAiRequest};
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
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let raw = body.clone();
    let facts = super::facts::RequestFacts::extract(&raw, &headers);
    let req = match serde_json::from_value::<OpenAiRequest>(body) {
        Ok(r) => r,
        Err(e) => {
            log_rejected(&state.db, &auth, "chat", "unknown");
            return translation_error(DIALECT, &format!("invalid request: {e}"));
        }
    };
    match state.cfg.models.route(&req.model) {
        Provider::Anthropic => {
            log_rejected(&state.db, &auth, "chat", &req.model);
            return translation_error(
                DIALECT,
                "this model is relayed to Anthropic and only available on /v1/messages",
            );
        }
        Provider::Gemini => {
            let model = req.model.clone();
            return super::gemini::chat_completions(state, auth, raw, model, facts).await;
        }
        Provider::OpenAi => {}
    }
    let mut upstream_req = match openai_req::to_responses(&req, &state.cfg) {
        Ok(r) => r,
        Err(e) => {
            log_rejected(&state.db, &auth, "chat", &req.model);
            return translation_error(DIALECT, &e);
        }
    };
    upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));

    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "chat",
        requested_model: req.model.clone(),
        upstream_model: upstream_req.model.clone(),
        effort: upstream_req
            .reasoning
            .as_ref()
            .map(|r| r.effort.clone())
            .unwrap_or_default(),
        status: 200,
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };

    let req_value = match serde_json::to_value(&upstream_req) {
        Ok(v) => v,
        Err(e) => {
            log_error(&state.db, record, 400, "invalid_request");
            return translation_error(DIALECT, &format!("serializing request: {e}"));
        }
    };
    let session_key = upstream_req.prompt_cache_key.clone().unwrap_or_default();
    let (account_id, resp) = match state
        .codex
        .execute(&req_value, auth.prefer_trusted, &session_key)
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
    record.session_key = session_key;

    let capture = UsageCapture::default();
    let events = event_stream(resp);

    if req.stream.unwrap_or(false) {
        let translator = OpenAiStream::new(req.model.clone(), req.include_usage(), capture.clone());
        let guard = LogGuard::new(state.db.clone(), state.prices.clone(), capture, record);
        stream_response(events, translator, guard)
    } else {
        let agg = aggregate(events, &capture).await;
        let snap = capture.snapshot();
        record.input_tokens = snap.input_tokens;
        record.output_tokens = snap.output_tokens;
        record.cache_read_tokens = snap.cache_read_tokens;
        record.reasoning_tokens = snap.reasoning_tokens;
        if agg.stop == StopKind::Error {
            let msg = agg
                .error_message
                .unwrap_or_else(|| "upstream failure".into());
            log_error(&state.db, record, 502, "upstream_failed");
            return super::error::error_response(DIALECT, 502, "api_error", &msg);
        }
        record.error_kind = snap.error_kind;
        log_usage(&state.db, &state.prices, record);
        Json(render_aggregated(&agg, &req.model)).into_response()
    }
}

fn stream_response(upstream: EventStream, translator: OpenAiStream, guard: LogGuard) -> Response {
    struct St {
        upstream: EventStream,
        translator: OpenAiStream,
        queue: VecDeque<Event>,
        finished: bool,
        done_sent: bool,
        _guard: LogGuard,
    }
    let st = St {
        upstream,
        translator,
        queue: VecDeque::new(),
        finished: false,
        done_sent: false,
        _guard: guard,
    };
    let stream = futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(ev) = st.queue.pop_front() {
                return Some((Ok::<_, Infallible>(ev), st));
            }
            if st.finished {
                if !st.done_sent {
                    st.done_sent = true;
                    return Some((Ok(Event::default().data("[DONE]")), st));
                }
                return None;
            }
            match st.upstream.next().await {
                Some(ev) => {
                    for chunk in st.translator.handle(ev) {
                        st.queue.push_back(Event::default().data(chunk));
                    }
                }
                None => {
                    st.finished = true;
                    for chunk in st.translator.finalize() {
                        st.queue.push_back(Event::default().data(chunk));
                    }
                }
            }
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Codex opens a WebSocket to this path before falling back to HTTP, and only
/// 426 short-circuits that.
pub async fn responses_upgrade_required() -> Response {
    (
        axum::http::StatusCode::UPGRADE_REQUIRED,
        "this proxy serves the responses API over HTTP only",
    )
        .into_response()
}

/// Codex asks with a `client_version` query and reads its context window out
/// of the reply, so it gets the backend payload untouched.
pub async fn models(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<ModelsQuery>,
) -> Response {
    // Both harnesses ask the same path for a catalog, and each only
    // understands its own. `anthropic-version` is required on every Anthropic
    // API call, so its presence identifies the caller.
    if headers.contains_key("anthropic-version") {
        return match state.anthropic.models_raw().await {
            Ok(body) => ([("content-type", "application/json")], body).into_response(),
            Err(e) => super::error::error_response(
                super::error::Dialect::Anthropic,
                503,
                "api_error",
                &format!("reading the model catalog: {e}"),
            ),
        };
    }
    if q.client_version.is_some() {
        return match state.catalog_raw().await {
            Some(body) => ([("content-type", "application/json")], body).into_response(),
            None => super::error::error_response(
                DIALECT,
                503,
                "api_error",
                "no usable codex account to read the model catalog from",
            ),
        };
    }

    let created = crate::clock::unix_now();

    let live = state.catalog().await;

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

    let mut data = match live {
        Some(models) => models
            .iter()
            .filter(|m| m.listed())
            .map(|m| ModelEntry {
                id: m.slug.clone(),
                object: "model",
                created,
                owned_by: "openai",
                context_window: m.context_window,
            })
            .collect::<Vec<ModelEntry>>(),
        None => {
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
        }
    };

    data.extend(state.gemini.models().await.into_iter().filter_map(|id| {
        state
            .cfg
            .models
            .route(&id)
            .eq(&crate::provider::Provider::Gemini)
            .then_some(ModelEntry {
                id,
                object: "model",
                created,
                owned_by: "google",
                context_window: None,
            })
    }));

    Json(ModelList {
        object: "list",
        data,
    })
    .into_response()
}

/// The fields the proxy rewrites on a `/v1/responses` request
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
    #[serde(flatten)]
    rest: serde_json::Map<String, Value>,
}

#[derive(Default, serde::Deserialize, serde::Serialize)]
struct ReasoningPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(flatten)]
    rest: serde_json::Map<String, Value>,
}

pub async fn responses_passthrough(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let mut req = match serde_json::from_value::<PassthroughRequest>(body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };

    let requested_model = req
        .model
        .unwrap_or_else(|| state.cfg.models.default.clone());
    let resolved = model_map::resolve(&state.cfg.models, &requested_model);
    req.model = Some(resolved.model.clone());
    if let Some(effort) = model_map::suffix_effort(&requested_model) {
        req.reasoning.get_or_insert_default().effort =
            Some(model_map::clamp_effort(&resolved.model, &effort));
    }

    let client_streams = req.stream.unwrap_or(false);
    req.store = Some(false);
    req.stream = Some(true);
    if req.instructions.is_none() {
        req.instructions = Some(state.cfg.codex.instructions());
    }
    let body = match serde_json::to_value(&req) {
        Ok(v) => v,
        Err(e) => return translation_error(DIALECT, &format!("serializing request: {e}")),
    };

    // Codex sends its own prompt_cache_key, which is stable per conversation.
    let session_key = body
        .pointer("/prompt_cache_key")
        .and_then(Value::as_str)
        .unwrap_or(&auth.user)
        .to_string();

    let facts = super::facts::RequestFacts::extract(&body, &headers);
    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "responses",
        requested_model,
        upstream_model: resolved.model,
        effort: req
            .reasoning
            .as_ref()
            .and_then(|r| r.effort.clone())
            .unwrap_or_default(),
        service_tier: req.service_tier.clone().unwrap_or_default(),
        status: 200,
        session_key: session_key.clone(),
        turn_index: facts.turn_index,
        tools_declared: facts.tools_declared,
        thinking_budget: facts.thinking_budget,
        image_count: facts.image_count,
        request_bytes: facts.request_bytes,
        ..Default::default()
    };
    let (account_id, resp) = match state
        .codex
        .execute(&body, auth.prefer_trusted, &session_key)
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
    let capture = UsageCapture::default();

    if client_streams {
        let limits = rate_limit_headers(&state.codex.pool_windows().await);
        return relay_stream(
            resp,
            LogGuard::new(
                state.db.clone(),
                state.prices.clone(),
                capture.clone(),
                record,
            ),
            capture,
            limits,
        );
    }

    {
        // Upstream only streams; recover the final response object from the
        // terminal event for non-streaming clients.
        let mut raw_events = resp.bytes_stream().eventsource();
        let mut final_response = Option::<Value>::None;
        while let Some(ev) = raw_events.next().await {
            let Ok(ev) = ev else { break };
            let terminal = match serde_json::from_str::<TerminalEvent>(&ev.data) {
                Ok(TerminalEvent::Completed(p))
                | Ok(TerminalEvent::Incomplete(p))
                | Ok(TerminalEvent::Failed(p)) => p,
                _ => continue,
            };
            if let Ok(env) = serde_json::from_str::<UsageEnvelope>(&ev.data)
                && let Some(usage) = env.response.usage
            {
                capture.record(&usage);
            }
            final_response = Some(terminal.response);
        }
        let snap = capture.snapshot();
        record.input_tokens = snap.input_tokens;
        record.output_tokens = snap.output_tokens;
        record.cache_read_tokens = snap.cache_read_tokens;
        record.reasoning_tokens = snap.reasoning_tokens;
        log_usage(&state.db, &state.prices, record);
        match final_response {
            Some(v) => Json(v).into_response(),
            None => super::error::error_response(
                DIALECT,
                502,
                "api_error",
                "upstream stream ended unexpectedly",
            ),
        }
    }
}

/// The stream's terminal events; `response` stays raw because it is handed
/// back to the client verbatim.
#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum TerminalEvent {
    #[serde(rename = "response.completed")]
    Completed(TerminalPayload),
    #[serde(rename = "response.incomplete")]
    Incomplete(TerminalPayload),
    #[serde(rename = "response.failed")]
    Failed(TerminalPayload),
    #[serde(other)]
    Other,
}

#[derive(serde::Deserialize)]
struct TerminalPayload {
    response: Value,
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
    item: OutputItem,
}

#[derive(serde::Deserialize)]
struct OutputItem {
    #[serde(default)]
    name: Option<String>,
}

/// What codex reads for `/status`.
fn rate_limit_headers(windows: &[crate::pool::UsageWindow]) -> Vec<(String, String)> {
    let mut sorted: Vec<_> = windows.iter().collect();
    sorted.sort_by_key(|w| crate::pool::window_seconds(&w.name).unwrap_or(i64::MAX));
    let mut out = Vec::new();
    for (tier, w) in ["primary", "secondary"].iter().zip(sorted) {
        let Some(minutes) = crate::pool::window_seconds(&w.name).map(|s| s / 60) else {
            continue;
        };
        out.push((
            format!("x-codex-{tier}-window-minutes"),
            minutes.to_string(),
        ));
        out.push((
            format!("x-codex-{tier}-used-percent"),
            ((w.utilization * 100.0).round() as i64).to_string(),
        ));
        if let Some(resets_at) = w.resets_at {
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
    let eof_capture = capture.clone();
    let stream = resp.bytes_stream().eventsource().filter_map(move |ev| {
        let capture = capture.clone();
        async move {
            match ev {
                Ok(ev) => {
                    if !ev.event.is_empty() {
                        capture.note_event(&ev.event);
                    }
                    capture.note_bytes(ev.data.len());
                    if (ev.event == "response.completed"
                        || ev.data.contains("\"response.completed\""))
                        && let Ok(env) = serde_json::from_str::<UsageEnvelope>(&ev.data)
                    {
                        if let Some(usage) = env.response.usage {
                            capture.record(&usage);
                        }
                        let reason = env
                            .response
                            .incomplete_details
                            .and_then(|d| d.reason)
                            .or(env.response.status);
                        if let Some(reason) = reason {
                            capture.note_stop_reason(&reason);
                        }
                    }
                    if ev.event == "response.output_item.done"
                        && let Ok(done) = serde_json::from_str::<OutputItemDone>(&ev.data)
                        && let Some(name) = done.item.name
                    {
                        capture.note_tool_call(&name);
                    }
                    let mut out = Event::default().data(ev.data);
                    if !ev.event.is_empty() && ev.event != "message" {
                        out = out.event(ev.event);
                    }
                    Some(Ok::<_, Infallible>(out))
                }
                Err(e) => {
                    tracing::warn!("passthrough SSE error: {e}");
                    capture.fail("upstream_sse_error");
                    None
                }
            }
        }
    });
    struct Held<S> {
        inner: S,
        capture: UsageCapture,
        _guard: LogGuard,
    }
    impl<S: futures_util::Stream + Unpin> futures_util::Stream for Held<S> {
        type Item = S::Item;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            let polled = std::pin::Pin::new(&mut self.inner).poll_next(cx);
            // Never reached if the caller went away first.
            if matches!(polled, std::task::Poll::Ready(None)) {
                self.capture.note_upstream_eof();
            }
            polled
        }
    }
    let held = Held {
        inner: Box::pin(stream),
        capture: eof_capture,
        _guard: guard,
    };
    let mut response = Sse::new(held)
        .keep_alive(KeepAlive::default())
        .into_response();
    for (name, value) in limits {
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::try_from(name),
            axum::http::HeaderValue::try_from(value),
        ) {
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

    #[test]
    fn a_standard_turn_carries_no_tier() {
        let body = serde_json::json!({"model": "gpt-5.6-sol", "input": []});
        let req: PassthroughRequest = serde_json::from_value(body).unwrap();
        assert_eq!(req.service_tier, None);
        let out = serde_json::to_value(&req).unwrap();
        assert!(out.get("service_tier").is_none());
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
        let get = |k: &str| {
            out.iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.as_str())
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
                .any(|(n, v)| n == "x-codex-primary-used-percent" && v == "80")
        );
        assert!(!out.iter().any(|(n, _)| n == "x-codex-primary-reset-at"));
    }
}
