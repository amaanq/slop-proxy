use std::time::{Duration, Instant};

use eyre::{Result, WrapErr as _, bail};
use serde::Deserialize;
use tokio::time::sleep;

use super::TokenSet;
use super::http;
use super::refresh::RefreshError;
use crate::clock;
use crate::copilot::client::CopilotClient;
use crate::db::Db;
use crate::provider::Provider;
use crate::upstream::SendError;

pub const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const SCOPES: &str = "read:user";

#[derive(Deserialize)]
struct DeviceCode {
   #[serde(default)]
   device_code: String,
   #[serde(default)]
   user_code: String,
   #[serde(default)]
   verification_uri: String,
   #[serde(default)]
   interval: i64,
}

#[derive(Deserialize)]
struct AccessGrant {
   #[serde(default)]
   access_token: Option<String>,
   #[serde(default)]
   error: Option<String>,
   #[serde(default)]
   error_description: Option<String>,
}

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
   let code: DeviceCode = http()
      .post(DEVICE_CODE_URL)
      .header("accept", "application/json")
      .json(&serde_json::json!({"client_id": CLIENT_ID, "scope": SCOPES}))
      .send()
      .await
      .wrap_err("requesting device code")?
      .json()
      .await
      .wrap_err("parsing device code response")?;
   if code.device_code.is_empty() || code.user_code.is_empty() {
      bail!("github device flow returned no code");
   }
   println!(
      "To authorize, open this URL in a browser:\n\n    {}\n\nand enter this code:\n\n    {}\n",
      if code.verification_uri.is_empty() {
         "https://github.com/login/device".to_owned()
      } else {
         code.verification_uri
      },
      code.user_code
   );
   println!("Waiting for authorization (up to 15 minutes)...");

   let deadline = Instant::now() + Duration::from_mins(15);
   let interval = Duration::from_secs(code.interval.clamp(5, 30) as u64);
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
      if let Some(token) = grant.access_token.filter(|token| !token.is_empty()) {
         break token;
      }
      match grant.error.as_deref().unwrap_or("authorization_pending") {
         "authorization_pending" | "slow_down" => {},
         "expired_token" => bail!("device code expired, run login again"),
         "access_denied" => bail!("authorization denied"),
         other => bail!(
            "device authorization failed: {other}{}",
            grant
               .error_description
               .map(|detail| format!(": {detail}"))
               .unwrap_or_default()
         ),
      }
   };

   let login = CopilotClient::login(&github_token).await?;
   // The grant lives in both columns until the first mint replaces the
   // access half with a Copilot token; the refresh half stays the grant.
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

/// Mint a short-lived Copilot token from the stored GitHub grant. The grant
/// itself is carried back in the set so the slot persists the rotation the
/// same way the other OAuth backends do.
pub async fn mint(github_token: &str) -> Result<TokenSet, RefreshError> {
   if github_token.is_empty() {
      return Err(RefreshError::Terminal("no github grant stored".into()));
   }
   match CopilotClient::mint_token(github_token).await {
      Ok(grant) => {
         if grant.token.is_empty() {
            return Err(RefreshError::Transient(
               "empty copilot token response".into(),
            ));
         }
         let now = clock::unix_now();
         let expires_at = usable_expiry(grant.expires_at, now)
            .or_else(|| {
               grant
                  .refresh_in
                  .filter(|&secs| secs > 0)
                  .map(|secs| now + secs)
            })
            // A response with no usable timestamp mints anew every call
            // rather than caching a token of unknown age.
            .unwrap_or(now);
         Ok(TokenSet {
            access_token: grant.token,
            refresh_token: github_token.to_owned(),
            id_token: None,
            expires_at: Some(expires_at),
         })
      },
      Err(SendError::Auth(text)) => Err(RefreshError::Terminal(format!(
         "github grant rejected: {text}"
      ))),
      Err(SendError::RateLimited { .. }) => Err(RefreshError::Transient(
         "copilot token endpoint rate limited".into(),
      )),
      Err(err) => Err(RefreshError::Transient(err.to_string())),
   }
}

/// An expiry in the past, or one in milliseconds, is not a timestamp the slot
/// can cache against.
fn usable_expiry(expires_at: Option<i64>, now: i64) -> Option<i64> {
   let timestamp = expires_at?;
   let timestamp = if timestamp > 4_102_444_800 {
      timestamp / 1000
   } else {
      timestamp
   };
   (timestamp > now).then_some(timestamp)
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn millisecond_expiries_are_normalized() {
      let now = 1_700_000_000;
      assert_eq!(usable_expiry(Some(now + 60), now), Some(now + 60));
      assert_eq!(usable_expiry(Some((now + 60) * 1000), now), Some(now + 60));
      assert_eq!(usable_expiry(None, now), None);
   }
}
