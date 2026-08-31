use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::pool::PoolError;

#[derive(Clone, Copy, Debug)]
pub enum Dialect {
    Anthropic,
    OpenAi,
}

pub fn error_response(dialect: Dialect, status: u16, err_type: &str, message: &str) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = match dialect {
        Dialect::Anthropic => json!({
            "type": "error",
            "error": {"type": err_type, "message": message}
        }),
        Dialect::OpenAi => json!({
            "error": {"type": err_type, "message": message, "code": null, "param": null}
        }),
    };
    (status, Json(body)).into_response()
}

pub fn pool_error_response(dialect: Dialect, err: PoolError) -> Response {
    match err {
        PoolError::NoAccounts => error_response(
            dialect,
            503,
            "api_error",
            "no usable codex accounts; an admin must run `slop-proxy login`",
        ),
        PoolError::AllCoolingDown { retry_after } => {
            let err_type = match dialect {
                Dialect::Anthropic => "rate_limit_error",
                Dialect::OpenAi => "rate_limit_exceeded",
            };
            let mut resp = error_response(
                dialect,
                429,
                err_type,
                "all upstream accounts are rate limited; retry later",
            );
            if let Ok(v) = HeaderValue::from_str(&retry_after.to_string()) {
                resp.headers_mut().insert("retry-after", v);
            }
            resp
        }
        PoolError::BadRequest(body) => {
            tracing::warn!("upstream rejected request: {body}");
            error_response(
                dialect,
                400,
                "invalid_request_error",
                &format!("upstream rejected the translated request: {body}"),
            )
        }
        PoolError::Upstream(msg) => error_response(dialect, 502, "api_error", &msg),
    }
}

pub fn translation_error(dialect: Dialect, msg: &str) -> Response {
    error_response(dialect, 400, "invalid_request_error", msg)
}
