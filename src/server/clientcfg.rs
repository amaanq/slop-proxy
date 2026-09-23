use axum::Json;
use axum::http::HeaderMap;
use axum::response::IntoResponse as _;
use axum::response::Response;
use serde::Serialize;

use crate::clock;

/// Ten years out. Codex refreshes when it believes the grant is near expiry,
/// and the refresh would go to `OpenAI` rather than here, so the claim is dated
/// far enough ahead that it never fires.
const LIFETIME_SECS: i64 = 10 * 365 * 24 * 3600;

const ACCOUNT_ID: &str = "slop-proxy";

#[derive(Serialize)]
struct Claims {
   #[serde(rename = "https://api.openai.com/auth")]
   auth: AuthClaim,
   email: &'static str,
   iat: i64,
   exp: i64,
}

#[derive(Serialize)]
struct AuthClaim {
   chatgpt_account_id: &'static str,
   chatgpt_plan_type: &'static str,
}

#[derive(Serialize)]
struct AuthFile {
   #[serde(rename = "OPENAI_API_KEY")]
   openai_api_key: Option<()>,
   tokens: Tokens,
   last_refresh: String,
}

#[derive(Serialize)]
struct Tokens {
   id_token: String,
   access_token: String,
   refresh_token: String,
   account_id: &'static str,
}

#[derive(Serialize)]
struct AccountsCheck {
   accounts: [AccountEntry; 1],
}

#[derive(Serialize)]
struct AccountEntry {
   id: &'static str,
   workspace_backend_origin: &'static str,
   account_routing_override: &'static str,
}

#[derive(Serialize)]
struct Header {
   alg: &'static str,
   typ: &'static str,
}

/// Codex only asks a provider for its catalog in ChatGPT-auth mode, which
/// reads the bearer from `auth.json` rather than `env_key`.
pub async fn codex_auth(headers: HeaderMap) -> Response {
   let Some(token) = bearer(&headers) else {
      return super::error::error_response(
         super::error::Dialect::OpenAi,
         401,
         "authentication_error",
         "missing api token",
      );
   };

   let now = clock::unix_now();
   let claims = Claims {
      auth: AuthClaim {
         chatgpt_account_id: ACCOUNT_ID,
         chatgpt_plan_type: "pro",
      },
      email: "slop-proxy",
      iat: now,
      exp: now + LIFETIME_SECS,
   };

   Json(AuthFile {
      openai_api_key: None,
      tokens: Tokens {
         id_token: jwt(&claims),
         access_token: token.clone(),
         refresh_token: token,
         account_id: ACCOUNT_ID,
      },
      last_refresh: clock::rfc3339(now),
   })
   .into_response()
}

/// Codex 0.156 refuses to start a ChatGPT-auth session until this workspace
/// discovery succeeds for the `auth.json` account id. `NO_CONSTRAINT` keeps
/// requests on the `chatgpt_base_url` origin, which is the proxy.
pub async fn codex_accounts() -> Response {
   Json(AccountsCheck {
      accounts: [AccountEntry {
         id: ACCOUNT_ID,
         workspace_backend_origin: "NO_CONSTRAINT",
         account_routing_override: "NO_CONSTRAINT",
      }],
   })
   .into_response()
}

/// Overriding the base url keeps `model_provider_id` as `openai`, which the
/// resume picker filters threads by, so a custom provider would hide every
/// existing session. The apps connector is off because it authenticates with
/// a `ChatGPT` session cookie the proxy has no way to mint, and fails loudly at
/// startup with `no_biscuit_no_service`.
pub async fn codex_config(headers: HeaderMap) -> Response {
   let host = headers
      .get("host")
      .and_then(|value| value.to_str().ok())
      .unwrap_or("localhost");
   let scheme = headers
      .get("x-forwarded-proto")
      .and_then(|value| value.to_str().ok())
      .unwrap_or("https");

   let body = format!(
      "openai_base_url = \"{scheme}://{host}/v1\"\n\
         chatgpt_base_url = \"{scheme}://{host}/backend-api\"\n\
         \n\
         [features]\n\
         apps = false\n"
   );
   ([("content-type", "text/plain; charset=utf-8")], body).into_response()
}

/// Unsigned JWT. Codex reads the claims without verifying them, and the proxy
/// is the only party that ever sees this file.
fn jwt<T>(claims: &T) -> String
where
   T: Serialize,
{
   let part = |json: String| data_encoding::BASE64URL_NOPAD.encode(json.as_bytes());
   let header = Header {
      alg: "none",
      typ: "JWT",
   };
   format!(
      "{}.{}.slop",
      part(serde_json::to_string(&header).unwrap_or_default()),
      part(serde_json::to_string(claims).unwrap_or_default())
   )
}

fn bearer(headers: &HeaderMap) -> Option<String> {
   let raw = headers
      .get("x-api-key")
      .or_else(|| headers.get("authorization"))?
      .to_str()
      .ok()?;
   Some(raw.strip_prefix("Bearer ").unwrap_or(raw).to_owned())
}
