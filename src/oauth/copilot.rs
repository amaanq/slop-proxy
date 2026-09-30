use std::time::{Duration, Instant};

use eyre::{Result, WrapErr as _, bail};
use serde::Deserialize;
use tokio::time::sleep;

use super::TokenSet;
use super::http;
use super::refresh::RefreshError;
use crate::copilot::client::{USER_AGENT, editor};
use crate::db::Db;
use crate::provider::Provider;

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

#[derive(Deserialize)]
struct DeviceCode {
   device_code: String,
   user_code: String,
   verification_uri: String,
   interval: u64,
}

#[derive(Deserialize)]
struct AccessGrant {
   access_token: Option<String>,
   error: Option<String>,
   error_description: Option<String>,
}

#[derive(Deserialize)]
struct GithubUser {
   login: String,
}

#[derive(Deserialize)]
struct CopilotToken {
   token: String,
   expires_at: i64,
}

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
   let code: DeviceCode = http()
      .post(DEVICE_CODE_URL)
      .header("accept", "application/json")
      .json(&serde_json::json!({"client_id": CLIENT_ID, "scope": "read:user"}))
      .send()
      .await
      .wrap_err("requesting device code")?
      .json()
      .await
      .wrap_err("parsing device code response")?;
   println!(
      "To authorize, open this URL in a browser:\n\n    {}\n\nand enter this code:\n\n    {}\n",
      code.verification_uri, code.user_code
   );
   println!("Waiting for authorization (up to 15 minutes)...");

   let deadline = Instant::now() + Duration::from_mins(15);
   let interval = Duration::from_secs(code.interval.clamp(5, 30));
   let github_token = loop {
      if Instant::now() > deadline {
         bail!("device authorization timed out");
      }
      sleep(interval).await;
      let grant: AccessGrant = http()
         .post(ACCESS_TOKEN_URL)
         .header("accept", "application/json")
         .json(&serde_json::json!({
            "client_id": CLIENT_ID,
            "device_code": code.device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
         }))
         .send()
         .await
         .wrap_err("polling device authorization")?
         .json()
         .await
         .wrap_err("parsing device token response")?;
      if let Some(token) = grant.access_token {
         break token;
      }
      match grant.error.as_deref().unwrap_or("authorization_pending") {
         "authorization_pending" | "slow_down" => {},
         "expired_token" => bail!("device code expired, run login again"),
         "access_denied" => bail!("authorization denied"),
         other => bail!(
            "device authorization failed: {other}: {}",
            grant.error_description.unwrap_or_default()
         ),
      }
   };

   let login = github_login(&github_token).await?;
   let tokens = TokenSet {
      access_token: github_token.clone(),
      refresh_token: github_token,
      id_token: None,
      expires_at: None,
   };
   super::finish_login(
      db,
      Provider::Copilot,
      &login,
      Some(&login),
      label.as_deref(),
      None,
      &tokens,
   )
   .await
}

pub async fn github_login(github_token: &str) -> Result<String> {
   let resp = http()
      .get("https://api.github.com/user")
      .header("authorization", format!("token {github_token}"))
      .header("user-agent", USER_AGENT)
      .send()
      .await
      .wrap_err("reading github user")?;
   if !resp.status().is_success() {
      let status = resp.status();
      let text = resp.text().await.unwrap_or_default();
      bail!("reading github user: {status}: {text}");
   }
   let user: GithubUser = resp.json().await.wrap_err("parsing github user")?;
   Ok(user.login)
}

/// Exchanges the stored GitHub grant for a short-lived Copilot token. The
/// grant stays in the refresh half, so it is never rotated.
pub async fn mint(github_token: &str) -> Result<TokenSet, RefreshError> {
   let resp = editor(http().get(COPILOT_TOKEN_URL))
      .header("authorization", format!("token {github_token}"))
      .send()
      .await
      .map_err(|err| RefreshError::Transient(err.to_string()))?;
   let status = resp.status();
   if !status.is_success() {
      let text = resp.text().await.unwrap_or_default();
      return Err(match status.as_u16() {
         401 | 403 | 404 => RefreshError::Terminal(format!("github grant rejected: {text}")),
         _ => RefreshError::Transient(format!("{status}: {text}")),
      });
   }
   let minted: CopilotToken = resp
      .json()
      .await
      .map_err(|err| RefreshError::Transient(format!("parsing copilot token: {err}")))?;
   Ok(TokenSet {
      access_token: minted.token,
      refresh_token: github_token.to_owned(),
      id_token: None,
      expires_at: Some(minted.expires_at),
   })
}
