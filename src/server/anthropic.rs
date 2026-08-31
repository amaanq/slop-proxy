use std::collections::VecDeque;
use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures_util::StreamExt;
use serde_json::Value;

use super::auth::AuthInfo;
use super::error::{pool_error_response, translation_error, Dialect};
use super::{cache_key, log_error, AppState, LogGuard};
use crate::codex::sse::{event_stream, EventStream};
use crate::db::usage::UsageRecord;
use crate::translate::anthropic_req::{self, AnthropicRequest};
use crate::translate::anthropic_stream::{render_aggregated, AnthropicStream};
use crate::translate::{aggregate, count_tokens, StopKind, UsageCapture};

const DIALECT: Dialect = Dialect::Anthropic;

pub async fn messages(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthInfo>,
    Json(body): Json<Value>,
) -> Response {
    let req: AnthropicRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };
    let mut upstream_req = match anthropic_req::to_responses(&req, &state.cfg) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &e),
    };
    upstream_req.prompt_cache_key = Some(cache_key(&auth.user, &upstream_req));
    let est_input = count_tokens::estimate(&upstream_req);

    let record = UsageRecord {
        meter_id: Some(auth.meter_id),
        token_id: Some(auth.token_id),
        user: auth.user.clone(),
        dialect: "anthropic",
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
            let status = pool_error_status(&e);
            log_error(&state.db, record, status, "pool");
            return pool_error_response(DIALECT, e);
        }
    };
    let mut record = record;
    record.account_id = Some(account_id);

    let capture = UsageCapture::default();
    let events = event_stream(resp);
    let emit_thinking = req.thinking_enabled();

    if req.stream.unwrap_or(false) {
        let translator =
            AnthropicStream::new(req.model.clone(), est_input, emit_thinking, capture.clone());
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
                        st.queue
                            .push_back(Event::default().event(name).data(data.to_string()));
                    }
                }
                None => {
                    st.finished = true;
                    for (name, data) in st.translator.finalize() {
                        st.queue
                            .push_back(Event::default().event(name).data(data.to_string()));
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
    Extension(_auth): Extension<AuthInfo>,
    Json(body): Json<Value>,
) -> Response {
    let req: AnthropicRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return translation_error(DIALECT, &format!("invalid request: {e}")),
    };
    match anthropic_req::to_responses(&req, &state.cfg) {
        Ok(upstream_req) => Json(serde_json::json!({
            "input_tokens": count_tokens::estimate(&upstream_req)
        }))
        .into_response(),
        Err(e) => translation_error(DIALECT, &e),
    }
}

pub(crate) fn pool_error_status(e: &crate::pool::PoolError) -> i64 {
    use crate::pool::PoolError::*;
    match e {
        NoAccounts => 503,
        AllCoolingDown { .. } => 429,
        BadRequest(_) => 400,
        Upstream(_) => 502,
    }
}
