use std::mem;

use axum::http::Response;
use reqwest::header::HeaderMap;
use thiserror::Error;

use crate::clock;

#[derive(Debug, Error)]
pub enum SendError {
   #[error("upstream auth failed: {0}")]
   Auth(String),
   #[error("upstream rate limited")]
   RateLimited {
      retry_after: Option<i64>,
      body: String,
   },
   #[error("upstream rate limited this model")]
   ModelLimited {
      retry_after: Option<i64>,
      body: String,
   },
   #[error("upstream error {status}: {body}")]
   Upstream { status: u16, body: String },
   #[error("bad request upstream: {0}")]
   BadRequest(String),
   #[error("network error: {0}")]
   Network(String),
}

/// Seconds until the caller may retry, from `retry-after` or the first of the
/// backend-specific reset headers that parses.
pub fn retry_after_secs(headers: &HeaderMap, reset_headers: &[&str]) -> Option<i64> {
   let get = |name: &str| headers.get(name)?.to_str().ok();
   if let Some(retry) = get("retry-after").and_then(|value| value.parse::<i64>().ok()) {
      return Some(retry);
   }
   let reset = reset_headers.iter().find_map(|header| get(header))?;
   let now = clock::unix_now();
   if let Ok(timestamp) = reset.parse::<jiff::Timestamp>() {
      return Some((timestamp.as_second() - now).max(1));
   }
   let secs = reset.parse::<f64>().ok()? as i64;
   // Reset headers have been observed both as an absolute epoch and as
   // seconds-from-now.
   Some(if secs > now { secs - now } else { secs }.max(1))
}

/// How one backend's statuses read. Every client used to spell this out
/// by hand and they only ever differed in these three fields.
#[derive(Clone, Copy)]
pub struct Classify {
   /// Non-2xx statuses handed back as a response, for a relay that wants
   /// the body verbatim.
   pub pass: fn(u16) -> bool,
   pub auth: &'static [u16],
   pub reset_headers: &'static [&'static str],
   /// Substrings of a 400 body that say this account cannot serve anyone,
   /// so it is benched like a rate limit instead of failing the caller.
   pub account_faults: &'static [&'static str],
}

/// A fault like an empty balance ends when a human acts, which no header
/// predicts, so the account is re-probed on this schedule until it serves.
const ACCOUNT_FAULT_RETRY_SECS: i64 = 3600;

impl Classify {
   pub const STRICT: Self = Self {
      pass: |_| false,
      auth: &[401, 403],
      reset_headers: &[],
      account_faults: &[],
   };
}

pub async fn classify(
   mut resp: reqwest::Response,
   rules: Classify,
) -> Result<reqwest::Response, SendError> {
   let status = resp.status().as_u16();
   let passed = (rules.pass)(status);
   let inspect = status == 400 && !rules.account_faults.is_empty();
   if resp.status().is_success() || (passed && !inspect) {
      return Ok(resp);
   }

   let retry_after = retry_after_secs(resp.headers(), rules.reset_headers);
   let status_code = resp.status();
   let headers = resp.headers().clone();
   let extensions = mem::take(resp.extensions_mut());
   let bytes = resp.bytes().await.unwrap_or_default();
   let body = String::from_utf8_lossy(&bytes)
      .chars()
      .take(2000)
      .collect::<String>();
   if inspect
      && rules
         .account_faults
         .iter()
         .any(|fault| body.contains(fault))
   {
      return Err(SendError::RateLimited {
         retry_after: Some(ACCOUNT_FAULT_RETRY_SECS),
         body,
      });
   }
   if passed {
      let mut rebuilt = Response::new(bytes);
      *rebuilt.status_mut() = status_code;
      *rebuilt.headers_mut() = headers;
      *rebuilt.extensions_mut() = extensions;
      return Ok(rebuilt.into());
   }

   Err(if rules.auth.contains(&status) {
      SendError::Auth(body)
   } else {
      match status {
         407 => SendError::Network("proxy authentication failed".into()),
         429 => SendError::RateLimited { retry_after, body },
         400 => SendError::BadRequest(body),
         code => SendError::Upstream { status: code, body },
      }
   })
}

#[cfg(test)]
mod tests {
   use super::*;

   fn response(status: u16, body: &'static str) -> reqwest::Response {
      Response::builder()
         .status(status)
         .header("retry-after", "7")
         .body(body)
         .unwrap()
         .into()
   }

   #[tokio::test]
   async fn a_passed_status_keeps_its_body_for_the_relay() {
      let rules = Classify {
         pass: |status| !matches!(status, 401 | 429 | 500..=599),
         ..Classify::STRICT
      };
      assert_eq!(
         classify(response(404, "x"), rules).await.unwrap().status(),
         404
      );
      assert!(matches!(
         classify(response(429, "slow"), rules).await,
         Err(SendError::RateLimited {
            retry_after: Some(7),
            ..
         })
      ));
   }

   #[tokio::test]
   async fn the_auth_list_decides_what_a_403_means() {
      assert!(matches!(
         classify(response(403, "no"), Classify::STRICT).await,
         Err(SendError::Auth(_))
      ));
      let cloudflare = Classify {
         auth: &[401],
         ..Classify::STRICT
      };
      assert!(matches!(
         classify(response(403, "no"), cloudflare).await,
         Err(SendError::Upstream { status: 403, .. })
      ));
   }
}
