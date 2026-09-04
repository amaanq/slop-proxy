use eyre::{Result, WrapErr as _, bail, eyre};
use rand::RngCore as _;
use reqwest::Url;
use serde::Deserialize;

use super::TokenSet;
use super::refresh::{RefreshError, RefreshRequest, post_token, token_set};
use crate::db::Db;
use crate::provider::Provider;

pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
pub const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
pub const SCOPES: &str = "org:create_api_key user:profile user:inference";
pub const PROFILE_URL: &str = "https://api.anthropic.com/api/oauth/profile";

pub async fn login(db: &Db, label: Option<String>) -> Result<()> {
    #[derive(serde::Serialize)]
    struct ExchangeRequest<'a> {
        grant_type: &'a str,
        code: &'a str,
        state: &'a str,
        client_id: &'a str,
        redirect_uri: &'a str,
        code_verifier: &'a str,
    }

    let mut raw = [0_u8; 32];
    rand::thread_rng().fill_bytes(&mut raw);
    let verifier = data_encoding::BASE64URL_NOPAD.encode(&raw);
    let challenge =
        data_encoding::BASE64URL_NOPAD.encode(&hmac_sha256::Hash::hash(verifier.as_bytes()));

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
    if state != verifier {
        bail!("state mismatch, paste the whole code");
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
    let parsed = super::exchanged(resp).await?;

    let account = parsed.account.as_ref();
    let account_id = account
        .and_then(|a| a.uuid.clone())
        .ok_or_else(|| eyre!("token response has no account uuid"))?;
    let email = account.and_then(|a| a.email_address.clone());
    let plan = subscription_tier(&parsed.access_token).await;
    let tokens = parsed
        .into_token_set(None)
        .ok_or_else(|| eyre!("no refresh_token in token response"))?;

    super::finish_login(
        db,
        Provider::Anthropic,
        &account_id,
        email.as_deref(),
        label.as_deref(),
        plan.as_deref(),
        &tokens,
    )
    .await
}

/// The subscription tier lives on the profile rather than the token
/// response, and distinguishes a 5x plan from a 20x one.
async fn subscription_tier(access_token: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Profile {
        organization: Option<Org>,
    }
    #[derive(Deserialize)]
    struct Org {
        rate_limit_tier: Option<String>,
    }
    super::http()
        .get(PROFILE_URL)
        .bearer_auth(access_token)
        .header("anthropic-beta", "oauth-2025-04-20")
        .send()
        .await
        .ok()?
        .json::<Profile>()
        .await
        .ok()?
        .organization?
        .rate_limit_tier
}

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
    token_set(status, &body, Some(refresh_token))
}
