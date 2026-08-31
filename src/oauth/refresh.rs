use thiserror::Error;

use super::{CLIENT_ID, TOKEN_URL, TokenSet};

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

/// Note: auth.openai.com rotates refresh tokens.
pub async fn refresh(refresh_token: &str) -> Result<TokenSet, RefreshError> {
    #[derive(serde::Serialize)]
    struct RefreshRequest<'a> {
        client_id: &'a str,
        grant_type: &'a str,
        refresh_token: &'a str,
    }

    let resp = super::http()
        .post(TOKEN_URL)
        .json(&RefreshRequest {
            client_id: CLIENT_ID,
            grant_type: "refresh_token",
            refresh_token,
        })
        .send()
        .await
        .map_err(|e| RefreshError::Transient(e.to_string()))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| RefreshError::Transient(e.to_string()))?;

    if !status.is_success() {
        let code = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.get("error").and_then(|e| {
                    e.as_str()
                        .map(String::from)
                        .or_else(|| e.get("code").and_then(|c| c.as_str()).map(String::from))
                        .or_else(|| e.get("type").and_then(|c| c.as_str()).map(String::from))
                })
            })
            .unwrap_or_default();
        let msg = format!("{status}: {body}");
        if TERMINAL_CODES
            .iter()
            .any(|c| code.contains(c) || body.contains(c))
        {
            return Err(RefreshError::Terminal(msg));
        }
        return Err(RefreshError::Transient(msg));
    }

    let parsed = serde_json::from_str(&body)
        .map_err(|e| RefreshError::Transient(format!("bad token response: {e}")))?;
    Ok(TokenSet::from_response(parsed))
}
