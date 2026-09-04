use axum::Json;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::config::ModelsConfig;
use crate::pool::PoolError;

#[derive(Clone, Copy, Debug)]
pub enum Dialect {
    Anthropic,
    OpenAi,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    err_type: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
struct AnthropicErrorBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct OpenAiErrorDetail<'a> {
    #[serde(flatten)]
    detail: ErrorDetail<'a>,
    code: Option<&'a str>,
    param: Option<&'a str>,
}

#[derive(Serialize)]
struct OpenAiErrorBody<'a> {
    error: OpenAiErrorDetail<'a>,
}

pub fn error_response(dialect: Dialect, status: u16, err_type: &str, message: &str) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let detail = ErrorDetail { err_type, message };
    let body = match dialect {
        Dialect::Anthropic => Json(AnthropicErrorBody {
            kind: "error",
            error: detail,
        })
        .into_response(),
        Dialect::OpenAi => Json(OpenAiErrorBody {
            error: OpenAiErrorDetail {
                detail,
                code: None,
                param: None,
            },
        })
        .into_response(),
    };
    (status, body).into_response()
}

pub fn pool_error_response(dialect: Dialect, models: &ModelsConfig, err: PoolError) -> Response {
    match err {
        PoolError::NoAccounts(provider) => error_response(
            dialect,
            503,
            "api_error",
            &format!("no usable {provider} accounts; an admin must run `slop-proxy login`"),
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
        PoolError::BadRequest {
            provider,
            model,
            body,
        } => {
            tracing::warn!(%provider, %model, "upstream rejected request: {body}");
            let message = match models.matched(&model) {
                Some(_) => format!("the {provider} backend rejected {model}: {body}"),
                None => {
                    let hint = models
                        .suggest(&model)
                        .map(|m| format!(", did you mean {m}?"))
                        .unwrap_or_default();
                    format!(
                        "no backend is configured for {model}{hint} it fell through to {provider}, which said: {body}"
                    )
                }
            };
            error_response(dialect, 400, "invalid_request_error", &message)
        }
        PoolError::Upstream(msg) => error_response(dialect, 502, "api_error", &msg),
    }
}

pub fn translation_error(dialect: Dialect, msg: &str) -> Response {
    tracing::warn!("rejected a {dialect:?} request: {msg}");
    error_response(dialect, 400, "invalid_request_error", msg)
}

pub fn pool_error_status(e: &PoolError) -> i64 {
    match e {
        PoolError::NoAccounts(_) => 503,
        PoolError::AllCoolingDown { .. } => 429,
        PoolError::BadRequest { .. } => 400,
        PoolError::Upstream(_) => 502,
    }
}

/// A 403 that says which provider the token lacks, so a scoped key does not
/// read as the model being broken for everyone.
pub fn out_of_scope(dialect: Dialect, provider: crate::provider::Provider) -> Response {
    error_response(
        dialect,
        403,
        "permission_error",
        &format!(
            "this token is not scoped to the {} backend",
            provider.as_str()
        ),
    )
}
