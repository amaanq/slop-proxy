use eyre::{Result, WrapErr, eyre};
use serde::Deserialize;

pub struct IdTokenInfo {
    pub email: Option<String>,
    pub chatgpt_account_id: Option<String>,
    pub plan_type: Option<String>,
}

#[derive(Deserialize, Default)]
struct Claims {
    #[serde(default)]
    exp: Option<i64>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default, rename = "https://api.openai.com/auth")]
    auth: Option<AuthClaims>,
    #[serde(default, rename = "https://api.openai.com/profile")]
    profile: Option<ProfileClaims>,
}

#[derive(Deserialize, Default)]
struct AuthClaims {
    #[serde(default)]
    chatgpt_account_id: Option<String>,
    #[serde(default)]
    chatgpt_plan_type: Option<String>,
}

#[derive(Deserialize, Default)]
struct ProfileClaims {
    #[serde(default)]
    email: Option<String>,
}

/// Decodes a JWT payload without signature verification.
fn claims(token: &str) -> Result<Claims> {
    let part = token.split('.').nth(1).ok_or_else(|| eyre!("not a JWT"))?;
    let bytes = data_encoding::BASE64URL_NOPAD
        .decode(part.trim_end_matches('=').as_bytes())
        .wrap_err("JWT payload is not base64url")?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn exp(token: &str) -> Option<i64> {
    claims(token).ok()?.exp
}

pub fn parse_id_token(token: &str) -> Result<IdTokenInfo> {
    let c = claims(token)?;
    let auth = c.auth.unwrap_or_default();
    Ok(IdTokenInfo {
        email: c.email.or_else(|| c.profile.unwrap_or_default().email),
        chatgpt_account_id: auth.chatgpt_account_id,
        plan_type: auth.chatgpt_plan_type,
    })
}
