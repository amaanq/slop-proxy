use axum::Json;
use axum::http::{HeaderValue, StatusCode};
use axum::response::IntoResponse as _;
use axum::response::Response;
use serde::Serialize;

use crate::config::ModelsConfig;
use crate::pool::PoolError;
use crate::provider::Provider;

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
         if let Ok(value) = HeaderValue::from_str(&retry_after.to_string()) {
            resp.headers_mut().insert("retry-after", value);
         }
         resp
      },
      PoolError::BadRequest {
         provider,
         model,
         body,
      } => {
         tracing::warn!(%provider, %model, "upstream rejected request: {body}");
         let message = if models.matched(&model).is_some() {
            format!("the {provider} backend rejected {model}: {body}")
         } else {
            let hint = models
               .suggest(&model)
               .map(|model| format!(", did you mean {model}?"))
               .unwrap_or_default();
            format!(
               "no backend is configured for {model}{hint} it fell through to {provider}, which said: {body}"
            )
         };
         error_response(dialect, 400, "invalid_request_error", &message)
      },
      PoolError::Upstream(msg) => error_response(dialect, 502, "api_error", &msg),
   }
}

pub fn translation_error(dialect: Dialect, msg: &str) -> Response {
   tracing::warn!("rejected a {dialect:?} request: {msg}");
   error_response(dialect, 400, "invalid_request_error", msg)
}

/// An untagged enum reports only that nothing matched, so the column serde
/// stopped at is the sole evidence of which block the caller actually sent.
pub fn body_at(body: &[u8], error: &serde_json::Error) -> String {
   let column = error.column();
   if column == 0 || column > body.len() {
      return String::new();
   }
   let start = column.saturating_sub(60);
   let end = (column + 600).min(body.len());
   String::from_utf8_lossy(&body[start..end]).into_owned()
}

pub const fn pool_error_status(err: &PoolError) -> i64 {
   match *err {
      PoolError::NoAccounts(_) => 503,
      PoolError::AllCoolingDown { .. } => 429,
      PoolError::BadRequest { .. } => 400,
      PoolError::Upstream(_) => 502,
   }
}

/// A 403 that says which provider the token lacks, so a scoped key does not
/// read as the model being broken for everyone.
pub fn out_of_scope(dialect: Dialect, provider: Provider) -> Response {
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
