use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use axum::http::{HeaderName, HeaderValue};

use super::error::{error_response, Dialect};
use super::AppState;

#[derive(Clone, Debug)]
pub struct AuthInfo {
    pub token_id: i64,
    pub user: String,
    pub meter_id: i64,
}

pub async fn require_token(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let dialect = if req.uri().path().starts_with("/v1/messages") {
        Dialect::Anthropic
    } else {
        Dialect::OpenAi
    };

    let raw = req
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            req.headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::to_string)
        });

    let Some(raw) = raw else {
        return error_response(
            dialect,
            401,
            "authentication_error",
            "missing API token (x-api-key or Authorization: Bearer)",
        );
    };

    match state.db.auth_token(&raw).await {
        Ok(Some(token)) => {
            let admission = match state.db.admit_token(token.id, &token.limits).await {
                Ok(Ok(admission)) => admission,
                Ok(Err(err)) => {
                    let (message, retry_after) = match err {
                        crate::db::usage::AdmissionError::RequestLimit { retry_after } =>
                            ("API token request limit exceeded", retry_after),
                        crate::db::usage::AdmissionError::TokenLimit { retry_after } =>
                            ("API token token limit exceeded", retry_after),
                    };
                    let mut response = error_response(dialect, 429, "rate_limit_error", message);
                    insert_header(&mut response, "retry-after", retry_after);
                    return response;
                }
                Err(e) => {
                    tracing::error!("token metering failed: {e}");
                    return error_response(dialect, 500, "api_error", "internal error");
                }
            };
            if admission.slowdown_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(
                    admission.slowdown_ms as u64,
                ))
                .await;
            }
            req.extensions_mut().insert(AuthInfo {
                token_id: token.id,
                user: token.user,
                meter_id: admission.meter_id,
            });
            let mut response = next.run(req).await;
            if let Some(limit) = admission.request_limit {
                insert_header(&mut response, "x-ratelimit-limit-requests", limit);
            }
            if let Some(remaining) = admission.requests_remaining {
                insert_header(&mut response, "x-ratelimit-remaining-requests", remaining);
            }
            if let Some(limit) = admission.token_limit {
                insert_header(&mut response, "x-ratelimit-limit-tokens", limit);
            }
            if let Some(remaining) = admission.tokens_remaining {
                insert_header(&mut response, "x-ratelimit-remaining-tokens", remaining);
            }
            insert_header(&mut response, "x-ratelimit-reset", admission.reset_after);
            if admission.slowdown_ms > 0 {
                insert_header(&mut response, "x-slop-slowdown-ms", admission.slowdown_ms);
            }
            response
        }
        Ok(None) => error_response(
            dialect,
            401,
            "authentication_error",
            "invalid or revoked API token",
        ),
        Err(e) => {
            tracing::error!("token lookup failed: {e}");
            error_response(dialect, 500, "api_error", "internal error")
        }
    }
}

fn insert_header(response: &mut Response, name: &'static str, value: i64) {
    if let Ok(value) = HeaderValue::from_str(&value.to_string()) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(name), value);
    }
}
