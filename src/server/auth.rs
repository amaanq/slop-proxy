use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;

use super::AppState;
use super::error::{Dialect, error_response};
use crate::db::usage::AdmissionError;

#[derive(Clone, Debug)]
pub struct AuthInfo {
   pub token_id: i64,
   pub user: String,
   pub meter_id: i64,
   pub limits: crate::db::tokens::TokenLimits,
}

impl AuthInfo {
   pub fn may_use(&self, provider: crate::provider::Provider) -> bool {
      self.limits.may_use(provider)
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

   let raw = bearer_token(req.headers(), req.uri().query());

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
                  AdmissionError::RequestLimit { retry_after } => {
                     ("API token request limit exceeded", retry_after)
                  },
                  AdmissionError::TokenLimit { retry_after } => {
                     ("API token token limit exceeded", retry_after)
                  },
               };
               let mut response = error_response(dialect, 429, "rate_limit_error", message);
               insert_header(&mut response, "retry-after", retry_after);
               return response;
            },
            Err(e) => {
               tracing::error!("token metering failed: {e}");
               return error_response(dialect, 500, "api_error", "internal error");
            },
         };
         if admission.slowdown_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(
               admission.slowdown_ms as u64,
            ))
            .await;
         }
         req.extensions_mut().insert(AuthInfo {
            token_id: token.id,
            user: token.user.clone(),
            meter_id: admission.meter_id,
            limits: token.limits.clone(),
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
      },
      Ok(None) => error_response(
         dialect,
         401,
         "authentication_error",
         "invalid or revoked API token",
      ),
      Err(e) => {
         tracing::error!("token lookup failed: {e}");
         error_response(dialect, 500, "api_error", "internal error")
      },
   }
}

fn insert_header(response: &mut Response, name: &'static str, value: i64) {
   if let Ok(value) = HeaderValue::from_str(&value.to_string()) {
      response
         .headers_mut()
         .insert(HeaderName::from_static(name), value);
   }
}

/// Gemini CLI sends its key as `x-goog-api-key`, and the raw REST form puts it
/// in a `key` query parameter, so neither of the other two headers is present.
fn bearer_token(headers: &axum::http::HeaderMap, query: Option<&str>) -> Option<String> {
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
            .find_map(|p| p.strip_prefix("key="))
            .map(str::to_owned)
      })
}
