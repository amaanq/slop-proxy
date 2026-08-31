use eyre::{Result, WrapErr, bail, eyre};
use rand::RngCore;
use reqwest::Url;
use serde::Deserialize;

use super::TokenSet;
use super::refresh::RefreshError;
use crate::db::Db;
use crate::provider::Provider;

pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
pub const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
pub const SCOPES: &str = "org:create_api_key user:profile user:inference";

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    account: Option<AccountInfo>,
}

#[derive(Deserialize)]
struct AccountInfo {
    uuid: Option<String>,
    email_address: Option<String>,
}

impl TokenResponse {
    /// Anthropic does not always rotate the refresh token, so the prior one
    /// carries over when the response omits it.
    fn into_token_set(self, prior_refresh: Option<&str>) -> Option<TokenSet> {
        Some(TokenSet {
            expires_at: self
                .expires_in
                .map(|s| crate::clock::unix_now() + s),
            access_token: self.access_token,
            refresh_token: self
                .refresh_token
                .or_else(|| prior_refresh.map(String::from))?,
            id_token: None,
        })
    }
}

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
    let mut raw = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut raw);
    let verifier = data_encoding::BASE64URL_NOPAD.encode(&raw);
    let challenge = data_encoding::BASE64URL_NOPAD.encode(&hmac_sha256::Hash::hash(verifier.as_bytes()));

    let mut url = Url::parse(AUTHORIZE_URL).expect("static url");
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &verifier);
    println!("To authorize, open this URL in a browser:\n\n    {url}\n");
    println!("then paste the code shown on the callback page (looks like code#state):");

    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .wrap_err("reading authorization code")?;
    let pasted = line.trim();
    if pasted.is_empty() {
        bail!("no authorization code provided");
    }
    let (code, state) = pasted
        .split_once('#')
        .unwrap_or((pasted, verifier.as_str()));

    #[derive(serde::Serialize)]
    struct ExchangeRequest<'a> {
        grant_type: &'a str,
        code: &'a str,
        state: &'a str,
        client_id: &'a str,
        redirect_uri: &'a str,
        code_verifier: &'a str,
    }

    let resp = super::http()
        .post(TOKEN_URL)
        .json(&ExchangeRequest {
            grant_type: "authorization_code",
            code,
            state,
            client_id: CLIENT_ID,
            redirect_uri: REDIRECT_URI,
            code_verifier: &verifier,
        })
        .send()
        .await
        .wrap_err("token exchange request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("token exchange failed: {status}: {body}");
    }
    let parsed = resp
        .json::<TokenResponse>()
        .await
        .wrap_err("parsing token response")?;

    let account = parsed.account.as_ref();
    let account_id = account
        .and_then(|a| a.uuid.clone())
        .ok_or_else(|| eyre!("token response has no account uuid"))?;
    let email = account.and_then(|a| a.email_address.clone());
    let tokens = parsed
        .into_token_set(None)
        .ok_or_else(|| eyre!("no refresh_token in token response"))?;

    let db_id = db
        .upsert_account(
            Provider::Anthropic,
            &account_id,
            email.as_deref(),
            label.as_deref(),
            None,
            &tokens,
        )
        .await?;
    println!(
        "logged in: anthropic account {} ({})",
        db_id,
        email.as_deref().unwrap_or("unknown email")
    );
    Ok(())
}

pub async fn refresh(refresh_token: &str) -> Result<TokenSet, RefreshError> {
    #[derive(serde::Serialize)]
    struct RefreshRequest<'a> {
        grant_type: &'a str,
        refresh_token: &'a str,
        client_id: &'a str,
    }

    let resp = super::http()
        .post(TOKEN_URL)
        .json(&RefreshRequest {
            grant_type: "refresh_token",
            refresh_token,
            client_id: CLIENT_ID,
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
        let msg = format!("{status}: {body}");
        if body.contains("invalid_grant") || status == reqwest::StatusCode::FORBIDDEN {
            return Err(RefreshError::Terminal(msg));
        }
        return Err(RefreshError::Transient(msg));
    }

    serde_json::from_str::<TokenResponse>(&body)
        .map_err(|e| RefreshError::Transient(format!("bad token response: {e}")))?
        .into_token_set(Some(refresh_token))
        .ok_or_else(|| RefreshError::Transient("token response missing refresh_token".into()))
}
