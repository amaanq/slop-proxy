use std::collections::VecDeque;
use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};

use super::anthropic::pool_error_status;
use super::auth::AuthInfo;
use super::error::{pool_error_response, translation_error, Dialect};
use super::{cache_key, log_error, AppState, LogGuard};
use crate::codex::sse::{event_stream, EventStream};
use crate::db::usage::UsageRecord;
use crate::translate::openai_req::{self, OpenAiRequest};
use crate::translate::openai_stream::{render_aggregated, OpenAiStream};
use crate::translate::{aggregate, model_map, StopKind, UsageCapture};

const DIALECT: Dialect = Dialect::OpenAi;

pub async fn chat_completions(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    Json(body): Json<Value>,
) -> Response {
    let req: OpenAiRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };
    let mut upstream_req = match openai_req::to_responses(&req, &state.cfg) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &e),
    };
    upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));

    let record = UsageRecord {
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "openai",
        requested_model: req.model.clone(),
        upstream_model: upstream_req.model.clone(),
        status: 200,
        ..Default::default()
    };

    let req_value = match serde_json::to_value(&upstream_req) {
        Ok(v) => v,
        Err(e) => return translation_error(DIALECT, &format!("serializing request: {e}")),
    };
    let (account_id, resp) = match state.pool.execute(&state.client, &req_value).await {
        Ok(r) => r,
        Err(e) => {
            log_error(&state.db, record, pool_error_status(&e), "pool");
            return pool_error_response(DIALECT, e);
        }
    };
    let mut record = record;
    record.account_id = Some(account_id);

    let capture = UsageCapture::default();
    let events = event_stream(resp);

    if req.stream.unwrap_or(false) {
        let translator = OpenAiStream::new(req.model.clone(), req.include_usage(), capture.clone());
        let guard = LogGuard::new(state.db.clone(), capture, record);
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
        let db = state.db.clone();
        tokio::spawn(async move {
            let _ = db.log_usage(&record).await;
        });
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
                        st.queue.push_back(Event::default().data(chunk.to_string()));
                    }
                }
                None => {
                    st.finished = true;
                    for chunk in st.translator.finalize() {
                        st.queue.push_back(Event::default().data(chunk.to_string()));
                    }
                }
            }
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn models(State(state): State<AppState>) -> Response {
    let created = chrono::Utc::now().timestamp();

    let live = match state.models.get() {
        Some(cached) => Some(cached),
        None => match state.pool.any_active_credentials().await {
            Some((access, account_id)) => {
                match state.client.list_models(&access, &account_id).await {
                    Ok(models) => {
                        state.models.put(models.clone());
                        Some(models)
                    }
                    Err(e) => {
                        tracing::warn!("fetching models from codex backend: {e}");
                        None
                    }
                }
            }
            None => None,
        },
    };

    let data: Vec<Value> = match live {
        Some(models) => models
            .iter()
            .filter(|m| m.listed())
            .map(|m| {
                json!({
                    "id": m.slug,
                    "object": "model",
                    "created": created,
                    "owned_by": "openai",
                    "context_window": m.context_window
                })
            })
            .collect(),
        None => {
            let mut ids = state.cfg.models.known.clone();
            let default = &state.cfg.models.default;
            if !default.is_empty() && !ids.contains(default) {
                ids.push(default.clone());
            }
            ids.iter()
                .map(|id| {
                    json!({
                        "id": id,
                        "object": "model",
                        "created": created,
                        "owned_by": "slop-proxy"
                    })
                })
                .collect()
        }
    };

    Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn responses_passthrough(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    Json(mut body): Json<Value>,
) -> Response {
    let Some(obj) = body.as_object_mut() else {
        return translation_error(DIALECT, "request body must be a JSON object");
    };

    let requested_model = obj
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.cfg.models.default)
        .to_string();
    let resolved = model_map::resolve(&state.cfg.models, &requested_model);
    obj.insert("model".into(), Value::String(resolved.model.clone()));

    let client_streams = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    obj.insert("store".into(), Value::Bool(false));
    obj.insert("stream".into(), Value::Bool(true));
    if !obj.contains_key("instructions") {
        obj.insert(
            "instructions".into(),
            Value::String(state.cfg.codex.instructions()),
        );
    }

    let record = UsageRecord {
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "responses",
        requested_model,
        upstream_model: resolved.model,
        status: 200,
        ..Default::default()
    };

    let (account_id, resp) = match state.pool.execute(&state.client, &body).await {
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
        return relay_stream(
            resp,
            LogGuard::new(state.db.clone(), capture.clone(), record),
            capture,
        );
    }

    {
        // Upstream only streams; recover the final response object from the
        // terminal event for non-streaming clients.
        let mut raw_events = resp.bytes_stream().eventsource();
        let mut final_response: Option<Value> = None;
        while let Some(ev) = raw_events.next().await {
            let Ok(ev) = ev else { break };
            let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
                continue;
            };
            match v.get("type").and_then(Value::as_str) {
                Some("response.completed")
                | Some("response.incomplete")
                | Some("response.failed") => {
                    if let Some(r) = v.get("response") {
                        if let Some(u) = r.get("usage") {
                            if let Ok(u) = serde_json::from_value(u.clone()) {
                                capture.record(&u);
                            }
                        }
                        final_response = Some(r.clone());
                    }
                }
                _ => {}
            }
        }
        let snap = capture.snapshot();
        record.input_tokens = snap.input_tokens;
        record.output_tokens = snap.output_tokens;
        record.cache_read_tokens = snap.cache_read_tokens;
        record.reasoning_tokens = snap.reasoning_tokens;
        let db = state.db.clone();
        tokio::spawn(async move {
            let _ = db.log_usage(&record).await;
        });
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

fn relay_stream(resp: reqwest::Response, guard: LogGuard, capture: UsageCapture) -> Response {
    let stream = resp.bytes_stream().eventsource().filter_map(move |ev| {
        let capture = capture.clone();
        async move {
            match ev {
                Ok(ev) => {
                    if ev.event == "response.completed"
                        || ev.data.contains("\"response.completed\"")
                    {
                        if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                            if let Some(usage) = v.get("response").and_then(|r| r.get("usage")) {
                                if let Ok(u) = serde_json::from_value(usage.clone()) {
                                    capture.record(&u);
                                }
                            }
                        }
                    }
                    let mut out = Event::default().data(ev.data);
                    if !ev.event.is_empty() && ev.event != "message" {
                        out = out.event(ev.event);
                    }
                    Some(Ok::<_, Infallible>(out))
                }
                Err(e) => {
                    tracing::warn!("passthrough SSE error: {e}");
                    None
                }
            }
        }
    });
    struct Held<S> {
        inner: S,
        _guard: LogGuard,
    }
    impl<S: futures_util::Stream + Unpin> futures_util::Stream for Held<S> {
        type Item = S::Item;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::pin::Pin::new(&mut self.inner).poll_next(cx)
        }
    }
    let held = Held {
        inner: Box::pin(stream),
        _guard: guard,
    };
    Sse::new(held)
        .keep_alive(KeepAlive::default())
        .into_response()
}
