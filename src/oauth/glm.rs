use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use eyre::{Result, WrapErr as _, bail, eyre};
use rand::RngCore as _;
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::clock;
use crate::oauth::http;

const CLI_OAUTH: &str = "https://zcode.z.ai/api/v1/oauth/cli";
const BIZ_HOST: &str = "https://api.z.ai";
const KEY_NAME: &str = "zcode-api-key";
#[expect(
   clippy::non_ascii_literal,
   reason = "the name z.ai gives every default org"
)]
const DEFAULT_ORG: &str = "默认机构";
#[expect(
   clippy::non_ascii_literal,
   reason = "the name z.ai gives every default project"
)]
const DEFAULT_PROJECT: &str = "默认项目";

/// Every z.ai business response, including the zcode token exchange.
#[derive(Deserialize)]
struct Envelope<T> {
   code: Option<Code>,
   msg: Option<String>,
   data: Option<T>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Code {
   Number(i64),
   Text(String),
}

#[derive(Serialize)]
struct FlowStart<'body> {
   provider: &'body str,
}

#[derive(Deserialize)]
struct Flow {
   flow_id: String,
   authorize_url: String,
   expires_at: i64,
   poll_interval_sec: u64,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum FlowState {
   Pending,
   Failed,
   Ready { zai: Grant },
}

#[derive(Deserialize)]
struct Grant {
   access_token: String,
}

#[derive(Serialize)]
struct BizLogin<'body> {
   token: &'body str,
}

#[derive(Deserialize)]
struct BizSession {
   #[serde(alias = "accessToken")]
   access_token: String,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Customer {
   organizations: Vec<Organization>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Organization {
   organization_id: String,
   #[serde(default)]
   organization_name: String,
   #[serde(default)]
   projects: Vec<Project>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Project {
   project_id: String,
   #[serde(default)]
   project_name: String,
}

#[derive(Serialize)]
struct NewKey<'body> {
   name: &'body str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiKey {
   name: String,
   api_key: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Secret {
   secret_key: String,
}

/// Resolves a coding-plan key the way the `ZCode` CLI does, a polled OAuth
/// flow into a business session, then the `zcode-api-key` of the default
/// project.
pub async fn login() -> Result<String> {
   let mut raw = [0_u8; 32];
   rand::thread_rng().fill_bytes(&mut raw);
   let poll_token = data_encoding::HEXLOWER.encode(&raw);
   let flow: Flow = biz(
      http()
         .post(format!("{CLI_OAUTH}/init"))
         .bearer_auth(&poll_token)
         .json(&FlowStart { provider: "zai" }),
      "oauth init",
   )
   .await?;
   println!(
      "To authorize, open this URL in a browser:\n\n    {}\n",
      flow.authorize_url
   );
   let opener = if cfg!(target_os = "macos") {
      "open"
   } else {
      "xdg-open"
   };
   let _ = Command::new(opener)
      .arg(&flow.authorize_url)
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .spawn();
   println!("Waiting for authorization...");

   let remaining = u64::try_from(flow.expires_at - clock::unix_now()).unwrap_or(0);
   let deadline = Instant::now() + Duration::from_secs(remaining.min(300));
   let interval = Duration::from_secs(flow.poll_interval_sec.max(1));
   let grant = loop {
      if Instant::now() > deadline {
         bail!("authorization timed out");
      }
      sleep(interval).await;
      let resp = http()
         .get(format!("{CLI_OAUTH}/poll/{}", flow.flow_id))
         .bearer_auth(&poll_token)
         .send()
         .await
         .wrap_err("oauth poll request failed")?;
      match resp.status() {
         StatusCode::TOO_MANY_REQUESTS => continue,
         StatusCode::BAD_REQUEST => bail!("the sign-in link expired, run login again"),
         _ => {},
      }
      let state: FlowState = envelope(resp, "oauth poll").await?;
      match state {
         FlowState::Pending => {},
         FlowState::Failed => bail!("authorization failed"),
         FlowState::Ready { zai } => break zai,
      }
   };

   let session: BizSession = biz(
      http()
         .post(format!("{BIZ_HOST}/api/auth/z/login"))
         .json(&BizLogin {
            token: &grant.access_token,
         }),
      "z.ai login",
   )
   .await?;
   let bearer = session.access_token;

   let customer: Customer = biz(
      http()
         .get(format!("{BIZ_HOST}/api/biz/customer/getCustomerInfo"))
         .bearer_auth(&bearer),
      "customer info",
   )
   .await?;
   let org = customer
      .organizations
      .iter()
      .find(|org| org.organization_name.contains(DEFAULT_ORG))
      .or_else(|| customer.organizations.first())
      .ok_or_else(|| eyre!("no organizations on this account"))?;
   let project = org
      .projects
      .iter()
      .find(|project| project.project_name.contains(DEFAULT_PROJECT))
      .or_else(|| org.projects.first())
      .ok_or_else(|| eyre!("no projects in organization {}", org.organization_id))?;

   let keys = format!(
      "{BIZ_HOST}/api/biz/v1/organization/{}/projects/{}/api_keys",
      org.organization_id, project.project_id
   );
   let listed: Vec<ApiKey> = biz(http().get(&keys).bearer_auth(&bearer), "api key list").await?;
   let api_key = if let Some(key) = listed.into_iter().find(|key| key.name == KEY_NAME) {
      key.api_key
   } else {
      let created: ApiKey = biz(
         http()
            .post(&keys)
            .bearer_auth(&bearer)
            .json(&NewKey { name: KEY_NAME }),
         "api key creation",
      )
      .await?;
      created.api_key
   };
   let secret: Secret = biz(
      http()
         .get(format!("{keys}/copy/{api_key}"))
         .bearer_auth(&bearer),
      "api key secret",
   )
   .await?;
   Ok(format!("{api_key}.{}", secret.secret_key))
}

async fn biz<T>(req: RequestBuilder, what: &str) -> Result<T>
where
   T: DeserializeOwned,
{
   let resp = req
      .send()
      .await
      .wrap_err_with(|| format!("{what} request failed"))?;
   envelope(resp, what).await
}

async fn envelope<T>(resp: Response, what: &str) -> Result<T>
where
   T: DeserializeOwned,
{
   let status = resp.status();
   if !status.is_success() {
      let body = resp.text().await.unwrap_or_default();
      bail!("{what} failed: {status}: {body}");
   }
   let envelope: Envelope<T> = resp
      .json()
      .await
      .wrap_err_with(|| format!("parsing {what} response"))?;
   let ok = match envelope.code {
      None => true,
      Some(Code::Number(code)) => matches!(code, 0 | 200),
      Some(Code::Text(code)) => matches!(code.as_str(), "0" | "200"),
   };
   if !ok {
      bail!("{what} failed: {}", envelope.msg.unwrap_or_default());
   }
   envelope
      .data
      .ok_or_else(|| eyre!("{what} response carried no data"))
}
