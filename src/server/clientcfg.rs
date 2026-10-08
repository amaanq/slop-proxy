use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::IntoResponse as _;
use axum::response::Response;
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};

use crate::clock;
use crate::server::AppState;
use crate::server::auth::{AuthInfo, bearer_token};
use crate::server::error::{Dialect, error_response};
use crate::server::relay::header_str;

/// Ten years out. Codex refreshes when it believes the grant is near expiry,
/// and the refresh goes to `OpenAI` unless `CODEX_REFRESH_TOKEN_URL_OVERRIDE`
/// points it at [`refresh`], so the claim is dated far enough ahead that it
/// never fires.
const LIFETIME_SECS: i64 = 10 * 365 * 24 * 3600;

const JWT_HEADER: &str = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0";

pub const ACCOUNT_ID: &str = "slop-proxy";

/// Every token reads as this plan, whichever accounts serve it.
pub const PLAN: &str = "pro";

/// The user each token is to codex, so two people sharing the proxy are
/// two users of the one workspace rather than the same login.
pub fn user_id(token_id: i64) -> String {
   format!("slop-proxy-token-{token_id}")
}

#[derive(Serialize)]
struct Claims<'a> {
   #[serde(rename = "https://api.openai.com/auth")]
   auth: AuthClaim<'a>,
   #[serde(rename = "https://api.openai.com/profile")]
   profile: ProfileClaim<'a>,
   sub: &'a str,
   email: &'a str,
   name: &'a str,
   iat: i64,
   exp: i64,
}

#[derive(Serialize)]
struct AuthClaim<'a> {
   chatgpt_account_id: &'static str,
   chatgpt_plan_type: &'static str,
   user_id: &'a str,
   chatgpt_user_id: &'a str,
   chatgpt_account_user_id: &'a str,
}

#[derive(Serialize)]
struct ProfileClaim<'a> {
   email: &'a str,
   name: &'a str,
}

/// The desktop reads its identity from the access token rather than the id
/// token, so the access token is the same claims with the API key inside.
#[derive(Serialize)]
struct AccessClaims<'a> {
   #[serde(flatten)]
   claims: &'a Claims<'a>,
   api_key: &'a str,
}

#[derive(Deserialize)]
struct AccessKey {
   api_key: String,
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

impl Tokens {
   /// The refresh token is the access token, so a refresh only has to find
   /// the API key inside it to mint the pair again.
   fn mint(token_id: i64, user: &str, api_key: &str, now: i64) -> Self {
      let user_id = user_id(token_id);
      let claims = Claims {
         auth: AuthClaim {
            chatgpt_account_id: ACCOUNT_ID,
            chatgpt_plan_type: PLAN,
            user_id: &user_id,
            chatgpt_user_id: &user_id,
            chatgpt_account_user_id: &user_id,
         },
         profile: ProfileClaim {
            email: user,
            name: user,
         },
         sub: &user_id,
         email: user,
         name: user,
         iat: now,
         exp: now + LIFETIME_SECS,
      };
      let access_token = jwt(&AccessClaims {
         claims: &claims,
         api_key,
      });
      Self {
         id_token: jwt(&claims),
         refresh_token: access_token.clone(),
         access_token,
         account_id: ACCOUNT_ID,
      }
   }
}

#[derive(Deserialize)]
pub struct RefreshRequest {
   refresh_token: String,
}

#[derive(Serialize)]
struct RefreshResponse {
   id_token: String,
   access_token: String,
   refresh_token: String,
}

/// Codex only asks a provider for its catalog in ChatGPT-auth mode, which
/// reads the bearer from `auth.json` rather than `env_key`.
pub async fn codex_auth(
   Extension(auth): Extension<AuthInfo>,
   uri: Uri,
   headers: HeaderMap,
) -> Response {
   let token = bearer_token(&headers, uri.query()).expect("require_token admitted the request");

   let now = clock::unix_now();
   Json(AuthFile {
      openai_api_key: None,
      tokens: Tokens::mint(auth.token_id, &auth.user, &token, now),
      last_refresh: clock::rfc3339(now),
   })
   .into_response()
}

/// Where `CODEX_REFRESH_TOKEN_URL_OVERRIDE` sends codex, outside
/// `require_token` since the credential rides in the body. A refresh is only
/// as good as the API key inside it, so a revoked key stops refreshing.
pub async fn refresh(State(state): State<AppState>, Json(req): Json<RefreshRequest>) -> Response {
   let api_key = api_key(req.refresh_token);
   let token = match state.db.auth_token(&api_key).await {
      Ok(Some(token)) => token,
      Ok(None) => {
         return error_response(
            Dialect::OpenAi,
            StatusCode::UNAUTHORIZED,
            "invalid_grant",
            "invalid or revoked API token",
         );
      },
      Err(err) => {
         tracing::error!("token lookup failed: {err}");
         return error_response(
            Dialect::OpenAi,
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "internal error",
         );
      },
   };
   let tokens = Tokens::mint(token.id, &token.user, &api_key, clock::unix_now());
   Json(RefreshResponse {
      id_token: tokens.id_token,
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
   })
   .into_response()
}

/// The API key a bearer carries: the access token [`codex_auth`] minted
/// wraps it, anything else is taken as the key itself.
pub fn api_key(bearer: String) -> String {
   let Some(payload) = bearer
      .strip_prefix(JWT_HEADER)
      .and_then(|rest| rest.strip_prefix('.'))
      .and_then(|rest| rest.split('.').next())
   else {
      return bearer;
   };
   data_encoding::BASE64URL_NOPAD
      .decode(payload.as_bytes())
      .ok()
      .and_then(|claims| serde_json::from_slice::<AccessKey>(&claims).ok())
      .map_or(bearer, |claims| claims.api_key)
}

/// Overriding the base url keeps `model_provider_id` as `openai`, which the
/// resume picker filters threads by, so a custom provider would hide every
/// existing session. The apps connector is off because it authenticates with
/// a `ChatGPT` session cookie the proxy has no way to mint, and fails loudly at
/// startup with `no_biscuit_no_service`.
pub async fn codex_config(headers: HeaderMap) -> Response {
   let host = header_str(&headers, "host").unwrap_or("localhost");
   let scheme = header_str(&headers, "x-forwarded-proto").unwrap_or("https");

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
   let payload = serde_json::to_vec(claims).expect("static claims serialize");
   format!(
      "{JWT_HEADER}.{}.slop",
      data_encoding::BASE64URL_NOPAD.encode(&payload)
   )
}
