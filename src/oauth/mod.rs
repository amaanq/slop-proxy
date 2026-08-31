pub mod anthropic;
pub mod jwt;
pub mod refresh;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::db::Db;

/// One shared client so token refreshes reuse connections instead of paying
/// TLS setup per call (refreshes run while a slot mutex is held).
pub(crate) fn http() -> &'static reqwest::Client {
    static HTTP: std::sync::LazyLock<reqwest::Client> =
        std::sync::LazyLock::new(reqwest::Client::new);
    &HTTP
}

pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const DEVICE_USERCODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
pub const DEVICE_VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";
pub const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const DEVICE_TIMEOUT_SECS: u64 = 15 * 60;

#[derive(Debug, Clone)]
pub struct TokenSet {
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: Option<String>,
    pub expires_at: Option<i64>,
}

#[derive(Deserialize)]
pub(crate) struct TokenResponse {
    access_token: String,
    refresh_token: String,
    id_token: Option<String>,
    expires_in: Option<i64>,
}

impl TokenSet {
    pub(crate) fn from_response(r: TokenResponse) -> Self {
        let expires_at = jwt::exp(&r.access_token)
            .or_else(|| r.expires_in.map(|s| chrono::Utc::now().timestamp() + s));
        Self {
            access_token: r.access_token,
            refresh_token: r.refresh_token,
            id_token: r.id_token,
            expires_at,
        }
    }
}

#[derive(serde::Serialize)]
struct DeviceCodeRequest<'a> {
    client_id: &'a str,
}

#[derive(serde::Serialize)]
struct DevicePollRequest<'a> {
    device_auth_id: &'a str,
    user_code: &'a str,
}

#[derive(Deserialize)]
struct UserCodeResp {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default)]
    interval: Option<Value>,
}

#[derive(Deserialize)]
struct DeviceTokenResp {
    authorization_code: String,
    code_verifier: String,
}

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
    let client = reqwest::Client::new();

    let resp = client
        .post(DEVICE_USERCODE_URL)
        .json(&DeviceCodeRequest {
            client_id: CLIENT_ID,
        })
        .send()
        .await
        .context("requesting device user code")?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!(
            "device login is not enabled for this account. A workspace admin must enable Codex device auth."
        );
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("device user code request failed: {status}: {body}");
    }
    let uc = resp
        .json::<UserCodeResp>()
        .await
        .context("parsing user code response")?;
    let interval = parse_interval(uc.interval.as_ref());

    println!(
        "To authorize, open this URL on any device:\n\n    {DEVICE_VERIFICATION_URL}\n\nand enter this code:\n\n    {}\n",
        uc.user_code
    );
    println!("Waiting for authorization (up to 15 minutes)...");

    let success = poll_for_code(&client, &uc.device_auth_id, &uc.user_code, interval).await?;

    let resp = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", &success.authorization_code),
            ("redirect_uri", DEVICE_REDIRECT_URI),
            ("client_id", CLIENT_ID),
            ("code_verifier", &success.code_verifier),
        ])
        .send()
        .await
        .context("token exchange request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("token exchange failed: {status}: {body}");
    }

    let tokens = TokenSet::from_response(resp.json().await.context("parsing token response")?);
    let id_token = tokens
        .id_token
        .as_deref()
        .context("no id_token in token response")?;
    let info = jwt::parse_id_token(id_token)?;
    let account_id = info
        .chatgpt_account_id
        .context("id_token has no chatgpt account id")?;

    let db_id = db
        .upsert_account(
            crate::provider::Provider::Codex,
            &account_id,
            info.email.as_deref(),
            label.as_deref(),
            info.plan_type.as_deref(),
            &tokens,
        )
        .await?;

    println!(
        "logged in: account {} ({}, plan {})",
        db_id,
        info.email.as_deref().unwrap_or("unknown email"),
        info.plan_type.as_deref().unwrap_or("unknown")
    );
    Ok(())
}

async fn poll_for_code(
    client: &reqwest::Client,
    device_auth_id: &str,
    user_code: &str,
    interval: u64,
) -> Result<DeviceTokenResp> {
    let started = std::time::Instant::now();
    let max = std::time::Duration::from_secs(DEVICE_TIMEOUT_SECS);
    loop {
        let resp = client
            .post(DEVICE_TOKEN_URL)
            .json(&DevicePollRequest {
                device_auth_id,
                user_code,
            })
            .send()
            .await
            .context("polling device token")?;
        match resp.status().as_u16() {
            200 => {
                return resp
                    .json::<DeviceTokenResp>()
                    .await
                    .context("parsing device token response");
            }
            // 403/404 mean the user has not finished authorizing yet.
            403 | 404 => {}
            other => {
                let body = resp.text().await.unwrap_or_default();
                bail!("device token poll failed: {other}: {body}");
            }
        }
        if started.elapsed() > max {
            bail!("device authorization timed out after 15 minutes");
        }
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
    }
}

fn parse_interval(v: Option<&Value>) -> u64 {
    let secs = match v {
        Some(Value::Number(n)) => n.as_u64(),
        Some(Value::String(s)) => s.parse().ok(),
        _ => None,
    };
    secs.unwrap_or(5).clamp(1, 60)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn interval_accepts_string_number_and_missing() {
        assert_eq!(parse_interval(Some(&json!("7"))), 7);
        assert_eq!(parse_interval(Some(&json!(3))), 3);
        assert_eq!(parse_interval(None), 5);
        assert_eq!(parse_interval(Some(&json!(0))), 1);
        assert_eq!(parse_interval(Some(&json!(999))), 60);
    }
}
