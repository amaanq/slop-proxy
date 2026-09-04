use serde::Deserialize;
use thiserror::Error;

use super::{CLIENT_ID, TOKEN_URL, TokenSet, jwt};

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
pub struct RefreshRequest<'a> {
   pub client_id: &'a str,
   pub grant_type: &'a str,
   pub refresh_token: &'a str,
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
         .or_else(|| self.expires_in.map(|s| crate::clock::unix_now() + s));
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

pub async fn post_token<B>(url: &str, body: &B) -> Result<(u16, String), RefreshError>
where
   B: serde::Serialize + Sync,
{
   let resp = super::http()
      .post(url)
      .json(body)
      .send()
      .await
      .map_err(|e| RefreshError::Transient(e.to_string()))?;
   let status = resp.status().as_u16();
   let text = resp
      .text()
      .await
      .map_err(|e| RefreshError::Transient(e.to_string()))?;
   Ok((status, text))
}

pub fn terminal(status: u16, body: &str) -> Option<String> {
   (status == 403 || TERMINAL_CODES.iter().any(|c| body.contains(c)))
      .then(|| format!("{status}: {body}"))
}

pub fn token_set(
   status: u16,
   body: &str,
   prior_refresh: Option<&str>,
) -> Result<TokenSet, RefreshError> {
   if !(200..300).contains(&status) {
      return Err(terminal(status, body).map_or_else(
         || RefreshError::Transient(format!("{status}: {body}")),
         RefreshError::Terminal,
      ));
   }
   serde_json::from_str::<TokenResponse>(body)
      .map_err(|e| RefreshError::Transient(format!("bad token response: {e}")))?
      .into_token_set(prior_refresh)
      .ok_or_else(|| RefreshError::Transient("token response missing refresh_token".into()))
}

/// Note: auth.openai.com rotates refresh tokens.
pub async fn refresh(refresh_token: &str) -> Result<TokenSet, RefreshError> {
   let (status, body) = post_token(
      TOKEN_URL,
      &RefreshRequest {
         client_id: CLIENT_ID,
         grant_type: "refresh_token",
         refresh_token,
      },
   )
   .await?;
   token_set(status, &body, None)
}
