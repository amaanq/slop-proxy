use serde::Deserialize;
use thiserror::Error;

use crate::clock;
use crate::oauth::{TokenSet, http, jwt};

#[derive(Debug, Error)]
pub enum RefreshError {
   /// The refresh token is dead; the account needs a fresh `login`.
   #[error("refresh token no longer valid: {0}")]
   Terminal(String),
   #[error("token refresh failed: {0}")]
   Transient(String),
}

const TERMINAL_CODES: &[&str] = &[
   "refresh_token_expired",
   "refresh_token_reused",
   "refresh_token_invalidated",
   "invalid_grant",
];

#[derive(serde::Serialize)]
struct RefreshRequest<'a> {
   client_id: &'a str,
   grant_type: &'a str,
   refresh_token: &'a str,
}

#[derive(Deserialize)]
pub struct TokenResponse {
   pub access_token: String,
   pub refresh_token: Option<String>,
   pub id_token: Option<String>,
   pub expires_in: Option<i64>,
   #[serde(default)]
   pub account: Option<Account>,
}

#[derive(Deserialize, Default)]
pub struct Account {
   pub uuid: Option<String>,
   pub email_address: Option<String>,
}

impl TokenResponse {
   /// Anthropic does not always rotate the refresh token, so the prior one
   /// carries over when the response omits it.
   pub fn into_token_set(self, prior_refresh: Option<&str>) -> Option<TokenSet> {
      let expires_at = jwt::exp(&self.access_token)
         .or_else(|| self.expires_in.map(|secs| clock::unix_now() + secs));
      Some(TokenSet {
         access_token: self.access_token,
         refresh_token: self
            .refresh_token
            .or_else(|| prior_refresh.map(String::from))?,
         id_token: self.id_token,
         expires_at,
      })
   }
}

/// The `OpenAI` endpoint rotates refresh tokens. Anthropic does not always, so
/// it passes the one it holds as `prior` to carry over.
pub async fn refresh_at(
   url: &str,
   client_id: &str,
   refresh_token: &str,
   prior: Option<&str>,
) -> Result<TokenSet, RefreshError> {
   let resp = http()
      .post(url)
      .json(&RefreshRequest {
         client_id,
         grant_type: "refresh_token",
         refresh_token,
      })
      .send()
      .await
      .map_err(|err| RefreshError::Transient(err.to_string()))?;
   let status = resp.status().as_u16();
   let body = resp
      .text()
      .await
      .map_err(|err| RefreshError::Transient(err.to_string()))?;
   if !(200..300).contains(&status) {
      let detail = format!("{status}: {body}");
      let dead = status == 403 || TERMINAL_CODES.iter().any(|code| body.contains(code));
      return Err(if dead {
         RefreshError::Terminal(detail)
      } else {
         RefreshError::Transient(detail)
      });
   }
   serde_json::from_str::<TokenResponse>(&body)
      .map_err(|err| RefreshError::Transient(format!("bad token response: {err}")))?
      .into_token_set(prior)
      .ok_or_else(|| RefreshError::Transient("token response missing refresh_token".into()))
}
