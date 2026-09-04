use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Event;
use axum::response::{IntoResponse as _, Response};
use axum::{Extension, Json};

use super::auth::AuthInfo;
use super::error::{Dialect, translation_error};
use super::pipeline::{self, apply_snapshot, dispatch_failed, translated};
use super::{AppState, LogGuard, cache_key, log_error, log_rejected, log_usage};
use crate::pool::Route;
use crate::pool::pools::Dispatched;
use crate::provider::Provider;
use crate::translate::anthropic_req::{self, AnthropicRequest};
use crate::translate::anthropic_stream::{AnthropicStream, render_aggregated};
use crate::translate::{StopKind, UsageCapture, aggregate, count_tokens};

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
        Provider::Gemini | Provider::Zen | Provider::OpenAi => {}
    }
    let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(near = %super::error::body_at(&body, &e), "anthropic request did not parse");
            log_rejected(&state, &auth, "messages", &peek.model);
            return translation_error(DIALECT, &format!("invalid request: {e}"));
        }
    };
    let mut upstream_req = anthropic_req::to_responses(&req, &state.cfg);
    upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));
    let est_input = count_tokens::estimate(&upstream_req);

    let mut record = pipeline::record(
        &auth,
        "messages",
        provider,
        req.model.clone(),
        upstream_req.model.clone(),
        super::facts::RequestFacts::from_anthropic(&req, &headers),
    );
    record.effort = upstream_req
        .reasoning
        .as_ref()
        .map(|r| r.effort.clone())
        .unwrap_or_default();

    let session_key = upstream_req.prompt_cache_key.clone().unwrap_or_default();
    record.session_key = session_key.clone();
    let route = Route {
        session_key: &session_key,
        model: &req.model,
        prefer_trusted: auth.limits.prefer_trusted,
    };
    let Dispatched {
        account_id,
        upstream,
    } = match state.pools.responses(provider, route, &upstream_req).await {
        Ok(d) => d,
        Err(e) => return dispatch_failed(&state, record, DIALECT, e),
    };
    record.account_id = account_id;

    let capture = UsageCapture::default();
    let events = upstream.events(&upstream_req.model, capture.clone());
    let emit_thinking = req.thinking_enabled();

    if req.stream.unwrap_or(false) {
        let mut translator =
            AnthropicStream::new(req.model.clone(), est_input, emit_thinking, capture.clone());
        let guard = LogGuard::new(state.clone(), capture, record, started);
        translated(events, guard, move |ev| {
            let frames = match ev {
                Some(ev) => translator.handle(ev),
                None => translator.finalize(),
            };
            frames
                .into_iter()
                .map(|(name, data)| Event::default().event(name).data(data))
                .collect()
        })
    } else {
        let agg = aggregate(events, &capture).await;
        let snap = capture.snapshot();
        apply_snapshot(&mut record, &snap);
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

pub async fn count_tokens(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    #[derive(serde::Serialize)]
    struct TokenCount {
        input_tokens: i64,
    }

    let peek = super::relay::Peek::from_slice(&body);
    let provider = state.cfg.models.route(&peek.model);
    if !auth.may_use(provider) {
        return super::error::out_of_scope(DIALECT, provider);
    }
    match provider {
        Provider::Anthropic => {
            return super::relay::count_tokens(state, auth, headers, body, peek).await;
        }
        Provider::Gemini | Provider::Zen | Provider::Glm | Provider::OpenAi => {}
    }
    let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };
    Json(TokenCount {
        input_tokens: count_tokens::estimate(&anthropic_req::to_responses(&req, &state.cfg)),
    })
    .into_response()
}
