pub mod anthropic;
pub mod copilot;
pub mod glm;
pub mod jwt;
pub mod refresh;

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use eyre::{Result, WrapErr as _, bail, eyre};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::time;

use crate::db::Db;
use crate::db::accounts::NewAccount;
use crate::provider::{AuthMode, Provider};
use refresh::TokenResponse;

/// One shared client so token refreshes reuse connections instead of paying
/// TLS setup per call (refreshes run while a slot mutex is held).
pub fn http() -> &'static reqwest::Client {
   static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
      reqwest::Client::builder()
         .timeout(Duration::from_secs(30))
         .build()
         .expect("oauth client")
   });
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

#[derive(serde::Serialize)]
struct DeviceCodeRequest<'a> {
   client_id: &'a str,
}

#[derive(serde::Serialize)]
struct DevicePollRequest<'a> {
   device_auth_id: &'a str,
   user_code: &'a str,
}

/// The endpoint has been observed sending the poll interval both as a number
/// and as a string.
#[derive(Deserialize)]
#[serde(untagged)]
enum Interval {
   Seconds(u64),
   Text(String),
}

#[derive(Deserialize)]
struct UserCodeResp {
   device_auth_id: String,
   #[serde(alias = "usercode")]
   user_code: String,
   #[serde(default)]
   interval: Option<Interval>,
}

#[derive(Deserialize)]
struct DeviceTokenResp {
   authorization_code: String,
   code_verifier: String,
}

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
   let resp = http()
      .post(DEVICE_USERCODE_URL)
      .json(&DeviceCodeRequest {
         client_id: CLIENT_ID,
      })
      .send()
      .await
      .wrap_err("requesting device user code")?;
   if resp.status() == reqwest::StatusCode::NOT_FOUND {
      bail!(
         "device login is not enabled for this account. A workspace admin must enable Codex device auth."
      );
   }
   let user_code: UserCodeResp = ok_json(resp, "device user code request").await?;
   let interval = parse_interval(user_code.interval.as_ref());

   println!(
      "To authorize, open this URL on any device:\n\n    {DEVICE_VERIFICATION_URL}\n\nand enter this code:\n\n    {}\n",
      user_code.user_code
   );
   println!("Waiting for authorization (up to 15 minutes)...");

   let started = Instant::now();
   let success = loop {
      let poll = http()
         .post(DEVICE_TOKEN_URL)
         .json(&DevicePollRequest {
            device_auth_id: &user_code.device_auth_id,
            user_code: &user_code.user_code,
         })
         .send()
         .await
         .wrap_err("polling device token")?;
      match poll.status().as_u16() {
         200 => {
            break poll
               .json::<DeviceTokenResp>()
               .await
               .wrap_err("parsing device token response")?;
         },
         // 403/404 mean the user has not finished authorizing yet.
         403 | 404 => {},
         other => {
            let body = poll.text().await.unwrap_or_default();
            bail!("device token poll failed: {other}: {body}");
         },
      }
      if started.elapsed() > Duration::from_secs(DEVICE_TIMEOUT_SECS) {
         bail!("device authorization timed out after 15 minutes");
      }
      time::sleep(Duration::from_secs(interval)).await;
   };

   let token_resp = http()
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
      .wrap_err("token exchange request failed")?;
   let tokens = ok_json::<TokenResponse>(token_resp, "token exchange")
      .await?
      .into_token_set(None)
      .ok_or_else(|| eyre!("no refresh_token in token response"))?;
   let id_token = tokens
      .id_token
      .as_deref()
      .ok_or_else(|| eyre!("no id_token in token response"))?;
   let info = jwt::parse_id_token(id_token)?;
   let account_id = info
      .chatgpt_account_id
      .ok_or_else(|| eyre!("id_token has no chatgpt account id"))?;

   finish_login(
      db,
      Provider::OpenAi,
      &account_id,
      info.email.as_deref(),
      label.as_deref(),
      info.plan_type.as_deref(),
      &tokens,
   )
   .await
}

#[derive(Deserialize)]
struct CodexAuth {
   tokens: Option<CodexTokens>,
}

#[derive(Deserialize)]
struct CodexTokens {
   id_token: String,
   access_token: String,
   refresh_token: String,
}

/// Takes over the login codex keeps in `auth.json`. Both then hold one
/// rotating refresh token, so whichever refreshes first signs the other out.
pub async fn import_codex(db: &Db, auth_json: &[u8], label: Option<String>) -> Result<()> {
   let auth: CodexAuth = serde_json::from_slice(auth_json).wrap_err("parsing codex auth.json")?;
   let tokens = auth
      .tokens
      .ok_or_else(|| eyre!("auth.json holds no ChatGPT login"))?;
   let info = jwt::parse_id_token(&tokens.id_token)?;
   let account_id = info
      .chatgpt_account_id
      .ok_or_else(|| eyre!("id_token has no chatgpt account id"))?;
   let tokens = TokenSet {
      expires_at: jwt::exp(&tokens.access_token),
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      id_token: Some(tokens.id_token),
   };

   finish_login(
      db,
      Provider::OpenAi,
      &account_id,
      info.email.as_deref(),
      label.as_deref(),
      info.plan_type.as_deref(),
      &tokens,
   )
   .await
}

pub async fn ok_json<T>(resp: reqwest::Response, what: &str) -> Result<T>
where
   T: DeserializeOwned,
{
   if !resp.status().is_success() {
      let status = resp.status();
      let body = resp.text().await.unwrap_or_default();
      bail!("{what} failed: {status}: {body}");
   }
   resp
      .json()
      .await
      .wrap_err_with(|| format!("parsing {what} response"))
}

async fn finish_login(
   db: &Db,
   provider: Provider,
   id: &str,
   email: Option<&str>,
   label: Option<&str>,
   plan: Option<&str>,
   tokens: &TokenSet,
) -> Result<()> {
   let db_id = db
      .upsert_account(NewAccount {
         provider,
         id,
         email,
         label,
         plan,
         tokens,
         auth_mode: AuthMode::OAuth,
      })
      .await?;
   let plan = plan
      .map(|tier| format!(", plan {tier}"))
      .unwrap_or_default();
   println!(
      "logged in: {provider} account {db_id} ({}{plan})",
      email.unwrap_or("unknown email")
   );
   Ok(())
}

fn parse_interval(value: Option<&Interval>) -> u64 {
   let secs = match value {
      Some(&Interval::Seconds(count)) => Some(count),
      Some(&Interval::Text(ref text)) => text.parse().ok(),
      None => None,
   };
   secs.unwrap_or(5).clamp(1, 60)
}

#[cfg(test)]
mod tests {
   use std::env;

   use super::*;

   #[test]
   fn interval_accepts_string_number_and_missing() {
      assert_eq!(parse_interval(Some(&Interval::Text("7".into()))), 7);
      assert_eq!(parse_interval(Some(&Interval::Seconds(3))), 3);
      assert_eq!(parse_interval(None), 5);
      assert_eq!(parse_interval(Some(&Interval::Seconds(0))), 1);
      assert_eq!(parse_interval(Some(&Interval::Seconds(999))), 60);
   }

   #[tokio::test]
   async fn import_codex_stores_the_chatgpt_login() {
      let path = env::temp_dir().join(format!("slop-test-{}.db", uuid::Uuid::new_v4()));
      let db = Db::open(&path).unwrap();
      let claims = serde_json::json!({
         "email": "alice@example.com",
         "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1", "chatgpt_plan_type": "plus"},
      });
      let id_token = format!(
         "e30.{}.sig",
         data_encoding::BASE64URL_NOPAD.encode(claims.to_string().as_bytes())
      );
      let auth = serde_json::json!({
         "auth_mode": "chatgpt",
         "tokens": {"id_token": id_token, "access_token": "at", "refresh_token": "rt", "account_id": "acct-1"},
      });
      import_codex(&db, auth.to_string().as_bytes(), Some("laptop".into()))
         .await
         .unwrap();

      let accounts = db.list_accounts().await.unwrap();
      assert_eq!(accounts.len(), 1);
      assert_eq!(accounts[0].provider_account_id, "acct-1");
      assert_eq!(accounts[0].email.as_deref(), Some("alice@example.com"));
      assert_eq!(accounts[0].plan_type.as_deref(), Some("plus"));
      assert_eq!(accounts[0].refresh_token, "rt");

      let api_key = br#"{"auth_mode":"apikey","OPENAI_API_KEY":"sk-x","tokens":null}"#;
      assert!(import_codex(&db, api_key, None).await.is_err());
   }
}
