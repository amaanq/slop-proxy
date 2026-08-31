use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Ten years out. Codex refreshes when it believes the grant is near expiry,
/// and the refresh would go to OpenAI rather than here, so the claim is dated
/// far enough ahead that it never fires.
const LIFETIME_SECS: i64 = 10 * 365 * 24 * 3600;

/// Codex only asks a provider for its model catalog when it is in ChatGPT-auth
/// mode, which reads the bearer out of `auth.json` instead of the environment.
/// Minting that file here is what lets a client learn its own context window
/// rather than restating it, and it is scoped to whatever CODEX_HOME the
/// caller drops it in, so it never displaces a real codex login.
pub async fn codex_auth(headers: HeaderMap) -> Response {
    let Some(token) = bearer(&headers) else {
        return super::error::error_response(
            super::error::Dialect::OpenAi,
            401,
            "authentication_error",
            "missing api token",
        );
    };

    let now = crate::clock::unix_now();
    let claims = json!({
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "slop-proxy",
            "chatgpt_plan_type": "pro",
        },
        "email": "slop-proxy",
        "iat": now,
        "exp": now + LIFETIME_SECS,
    });

    Json(json!({
        "OPENAI_API_KEY": null,
        "tokens": {
            "id_token": jwt(&claims),
            "access_token": token,
            "refresh_token": token,
            "account_id": "slop-proxy",
        },
        "last_refresh": crate::clock::rfc3339(now),
    }))
    .into_response()
}

/// The provider block that puts codex in the auth mode above. `env_key` is
/// deliberately absent: with it codex authenticates from the environment,
/// never asks for the catalog, and is left guessing its context window.
pub async fn codex_config(headers: HeaderMap) -> Response {
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("https");

    let body = format!(
        "model_provider = \"slop\"\n\
         \n\
         [model_providers.slop]\n\
         name = \"slop\"\n\
         base_url = \"{scheme}://{host}/v1\"\n\
         wire_api = \"responses\"\n\
         requires_openai_auth = true\n"
    );
    ([("content-type", "text/plain; charset=utf-8")], body).into_response()
}

/// Unsigned JWT. Codex reads the claims without verifying them, and the proxy
/// is the only party that ever sees this file.
fn jwt(claims: &serde_json::Value) -> String {
    let part = |v: &serde_json::Value| {
        data_encoding::BASE64URL_NOPAD.encode(v.to_string().as_bytes())
    };
    format!(
        "{}.{}.slop",
        part(&json!({"alg": "none", "typ": "JWT"})),
        part(claims)
    )
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    let raw = headers
        .get("x-api-key")
        .or_else(|| headers.get("authorization"))?
        .to_str()
        .ok()?;
    Some(raw.strip_prefix("Bearer ").unwrap_or(raw).to_string())
}
