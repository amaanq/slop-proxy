use std::time::Duration;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use tokio::time;

use crate::db::tokens::TokenLimits;
use crate::db::usage::{Admission, AdmissionError};
use crate::pool::Route;
use crate::server::AppState;
use crate::server::error::{Dialect, error_response};

#[derive(Clone, Debug)]
pub struct AuthInfo {
   pub token_id: i64,
   pub user: String,
   pub meter_id: Option<i64>,
   pub limits: TokenLimits,
}

impl AuthInfo {
   pub fn route<'route>(
      &'route self,
      session_key: &'route str,
      model: &'route str,
   ) -> Route<'route> {
      Route {
         session_key,
         model,
         service_tier: None,
         user: &self.user,
         pinned_account: self.limits.pinned_account,
         prefer_trusted: self.limits.prefer_trusted,
         reserved_only: self.limits.reserved_only,
      }
   }
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

   let Some(raw) = bearer_token(req.headers(), req.uri().query()) else {
      return error_response(
         dialect,
         StatusCode::UNAUTHORIZED,
         "authentication_error",
         "missing API token (x-api-key or Authorization: Bearer)",
      );
   };

   if req.method() == Method::GET && req.uri().path() == "/v1/responses" {
      return match authenticate(&state, dialect, &raw).await {
         Ok(auth) => {
            req.extensions_mut().insert(auth);
            next.run(req).await
         },
         Err(response) => response,
      };
   }
   let (auth, admission) = match admit_token(&state, dialect, &raw).await {
      Ok(admitted) => admitted,
      Err(response) => return response,
   };
   req.extensions_mut().insert(auth);
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

async fn authenticate(state: &AppState, dialect: Dialect, raw: &str) -> Result<AuthInfo, Response> {
   match state.db.auth_token(raw).await {
      Ok(Some(token)) => Ok(AuthInfo {
         token_id: token.id,
         user: token.user,
         meter_id: None,
         limits: token.limits,
      }),
      Ok(None) => {
         tracing::warn!(
            prefix = %raw.chars().take(12).collect::<String>(),
            "rejected an unknown API token"
         );
         Err(error_response(
            dialect,
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or revoked API token",
         ))
      },
      Err(err) => {
         tracing::error!("token lookup failed: {err}");
         Err(error_response(
            dialect,
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "internal error",
         ))
      },
   }
}

pub async fn admit_token(
   state: &AppState,
   dialect: Dialect,
   raw: &str,
) -> Result<(AuthInfo, Admission), Response> {
   let mut auth = authenticate(state, dialect, raw).await?;
   let admission = match state.db.admit_token(auth.token_id, &auth.limits).await {
      Ok(Ok(admission)) => admission,
      Ok(Err(err)) => {
         let (message, retry_after) = match err {
            AdmissionError::RequestLimit { retry_after } => {
               ("API token request limit exceeded", retry_after)
            },
            AdmissionError::TokenLimit { retry_after } => {
               ("API token token limit exceeded", retry_after)
            },
         };
         let mut response = error_response(
            dialect,
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            message,
         );
         insert_header(&mut response, "retry-after", retry_after);
         return Err(response);
      },
      Err(err) => {
         tracing::error!("token metering failed: {err}");
         return Err(error_response(
            dialect,
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "internal error",
         ));
      },
   };
   if admission.slowdown_ms > 0 {
      time::sleep(Duration::from_millis(admission.slowdown_ms as u64)).await;
   }
   auth.meter_id = Some(admission.meter_id);
   Ok((auth, admission))
}

fn insert_header(response: &mut Response, name: &'static str, value: i64) {
   response
      .headers_mut()
      .insert(HeaderName::from_static(name), HeaderValue::from(value));
}

/// Gemini CLI sends its key as `x-goog-api-key`, and the raw REST form puts it
/// in a `key` query parameter, so neither of the other two headers is present.
pub fn bearer_token(headers: &HeaderMap, query: Option<&str>) -> Option<String> {
   let header = |name: &str| headers.get(name)?.to_str().ok().map(str::to_owned);
   header("x-api-key")
      .or_else(|| header("x-goog-api-key"))
      .or_else(|| {
         headers
            .get("authorization")?
            .to_str()
            .ok()?
            .strip_prefix("Bearer ")
            .map(str::to_owned)
      })
      .or_else(|| {
         query?
            .split('&')
            .find_map(|part| part.strip_prefix("key="))
            .map(str::to_owned)
      })
}
