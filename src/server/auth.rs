use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;

use super::error::{error_response, Dialect};
use super::AppState;

#[derive(Clone, Debug)]
pub struct AuthInfo {
    pub token_id: i64,
    pub user: String,
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
        Ok(Some((token_id, user))) => {
            req.extensions_mut().insert(AuthInfo { token_id, user });
            next.run(req).await
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
