use std::collections::VecDeque;
use std::convert::Infallible;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures_util::StreamExt;

use super::auth::AuthInfo;
use super::error::{Dialect, pool_error_response, pool_error_status, translation_error};
use super::{AppState, LogGuard, cache_key, log_error, log_rejected, log_usage};
use crate::codex::sse::{EventStream, event_stream};
use crate::db::usage::UsageRecord;
use crate::pool::Route;
use crate::provider::Provider;
use crate::translate::anthropic_req::{self, AnthropicRequest};
use crate::translate::anthropic_stream::{AnthropicStream, render_aggregated};
use crate::translate::{StopKind, UsageCapture, aggregate, count_tokens, gemini_bridge};

const DIALECT: Dialect = Dialect::Anthropic;

pub async fn messages(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let started = std::time::Instant::now();
    let peek = super::relay::Peek::from_slice(&body);
    // An effort suffix is part of what the caller typed, not part of the model
    // name a pattern matches, so routing the raw string sent muse:high to
    // codex and burned the pool on a model it cannot serve.
    let provider = state
        .cfg
        .models
        .route(&crate::translate::model_map::resolve(&state.cfg.models, &peek.model).model);
    if !auth.may_use(provider) {
        log_rejected(&state, &auth, "messages", &peek.model);
        return super::error::out_of_scope(DIALECT, provider);
    }
    match provider {
        Provider::Anthropic => {
            return super::relay::messages(state, auth, headers, body, peek).await;
        }
        // Z.ai speaks this dialect, so the body it needs is the one that
        // arrived and the reply needs no translating back.
        Provider::Glm => {
            return super::relay::glm(state, auth, headers, body, peek).await;
        }
        Provider::Gemini => {}
        Provider::Zen => {}
        Provider::OpenAi => {}
    }
    let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
        Ok(r) => r,
        Err(e) => {
            log_rejected(&state, &auth, "messages", &peek.model);
            return translation_error(DIALECT, &format!("invalid request: {e}"));
        }
    };
    let mut upstream_req = match anthropic_req::to_responses(&req, &state.cfg) {
        Ok(r) => r,
        Err(e) => {
            log_rejected(&state, &auth, "messages", &req.model);
            return translation_error(DIALECT, &e);
        }
    };
    upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));
    let est_input = count_tokens::estimate(&upstream_req);

    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        provider: Some(provider),
        dialect: "messages",
        requested_model: req.model.clone(),
        upstream_model: upstream_req.model.clone(),
        effort: upstream_req
            .reasoning
            .as_ref()
            .map(|r| r.effort.clone())
            .unwrap_or_default(),
        status: 200,
        ..Default::default()
    };

    let req_bytes = match serde_json::to_vec(&upstream_req) {
        Ok(v) => Bytes::from(v),
        Err(e) => {
            log_error(&state, record, 400, "invalid_request");
            return translation_error(DIALECT, &format!("serializing request: {e}"));
        }
    };
    let session_key = upstream_req.prompt_cache_key.clone().unwrap_or_default();
    let route = state.cfg.models.route(&upstream_req.model);
    let gemini = route == Provider::Gemini;
    let dispatched = match route {
        Provider::Gemini => {
            let chat = gemini_bridge::to_chat(&upstream_req);
            state
                .pools
                .gemini
                .execute(
                    Route {
                        session_key: &session_key,
                        model: &req.model,
                        prefer_trusted: false,
                    },
                    crate::pool::gemini::Call::OpenAi(Box::new(chat)),
                )
                .await
                .map(|(id, upstream)| (id, upstream.protocol, upstream.response))
        }
        // Zen already speaks the Responses API, so the body it is handed is
        // the one codex would have received.
        Provider::Zen => state
            .pools
            .zen
            .execute(
                Route {
                    session_key: &session_key,
                    model: &req.model,
                    prefer_trusted: false,
                },
                req_bytes.clone(),
            )
            .await
            .map(|(id, resp)| (id, crate::gemini::client::GeminiProtocol::OpenAi, resp)),
        _ => state
            .pools
            .codex
            .execute(
                Route {
                    session_key: &session_key,
                    model: &req.model,
                    prefer_trusted: auth.prefer_trusted,
                },
                req_bytes.clone(),
            )
            .await
            .map(|(id, resp)| (id, crate::gemini::client::GeminiProtocol::OpenAi, resp)),
    };
    let (account_id, protocol, resp) = match dispatched {
        Ok(r) => r,
        Err(e) => {
            let status = pool_error_status(&e);
            log_error(&state, record, status, "pool");
            return pool_error_response(DIALECT, &state.cfg.models, e);
        }
    };
    let mut record = record;
    record.account_id = account_id;

    let capture = UsageCapture::default();
    let events = if gemini {
        gemini_bridge::event_stream(
            resp,
            protocol,
            &upstream_req.model,
            Default::default(),
            capture.clone(),
        )
    } else {
        event_stream(resp)
    };
    let emit_thinking = req.thinking_enabled();

    if req.stream.unwrap_or(false) {
        let translator =
            AnthropicStream::new(req.model.clone(), est_input, emit_thinking, capture.clone());
        let guard = LogGuard::new(state.clone(), capture, record, started);
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
            log_error(&state, record, 502, "upstream_failed");
            return super::error::error_response(DIALECT, 502, "api_error", &msg);
        }
        record.error_kind = snap.error_kind;
        log_usage(&state, record);
        Json(render_aggregated(&agg, &req.model, emit_thinking)).into_response()
    }
}

fn stream_response(
    upstream: EventStream,
    translator: AnthropicStream,
    guard: LogGuard,
) -> Response {
    struct St {
        upstream: EventStream,
        translator: AnthropicStream,
        queue: VecDeque<Event>,
        finished: bool,
        _guard: LogGuard,
    }
    let st = St {
        upstream,
        translator,
        queue: VecDeque::new(),
        finished: false,
        _guard: guard,
    };
    let stream = futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(ev) = st.queue.pop_front() {
                return Some((Ok::<_, Infallible>(ev), st));
            }
            if st.finished {
                return None;
            }
            match st.upstream.next().await {
                Some(ev) => {
                    for (name, data) in st.translator.handle(ev) {
                        st.queue.push_back(Event::default().event(name).data(data));
                    }
                }
                None => {
                    st.finished = true;
                    for (name, data) in st.translator.finalize() {
                        st.queue.push_back(Event::default().event(name).data(data));
                    }
                }
            }
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn count_tokens(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let peek = super::relay::Peek::from_slice(&body);
    let provider = state.cfg.models.route(&peek.model);
    if !auth.may_use(provider) {
        return super::error::out_of_scope(DIALECT, provider);
    }
    match provider {
        Provider::Anthropic => {
            return super::relay::count_tokens(state, auth, headers, body, peek).await;
        }
        Provider::Gemini => {}
        Provider::Zen => {}
        Provider::Glm => {}
        Provider::OpenAi => {}
    }
    let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };
    #[derive(serde::Serialize)]
    struct TokenCount {
        input_tokens: i64,
    }

    match anthropic_req::to_responses(&req, &state.cfg) {
        Ok(upstream_req) => Json(TokenCount {
            input_tokens: count_tokens::estimate(&upstream_req),
        })
        .into_response(),
        Err(e) => translation_error(DIALECT, &e),
    }
}
