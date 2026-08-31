use eyre::{Result, WrapErr, eyre};
use serde_json::Value;

pub struct IdTokenInfo {
    pub email: Option<String>,
    pub chatgpt_account_id: Option<String>,
    pub plan_type: Option<String>,
}

/// Decodes a JWT payload without signature verification.
pub fn payload(token: &str) -> Result<Value> {
    let part = token.split('.').nth(1).ok_or_else(|| eyre!("not a JWT"))?;
    let bytes = data_encoding::BASE64URL_NOPAD
        .decode(part.trim_end_matches('=').as_bytes())
        .wrap_err("JWT payload is not base64url")?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn exp(token: &str) -> Option<i64> {
    payload(token).ok()?.get("exp")?.as_i64()
}

pub fn parse_id_token(token: &str) -> Result<IdTokenInfo> {
    let p = payload(token)?;
    let auth = p.get("https://api.openai.com/auth");
    let email = p
        .get("email")
        .and_then(Value::as_str)
        .or_else(|| {
            p.get("https://api.openai.com/profile")
                .and_then(|v| v.get("email"))
                .and_then(Value::as_str)
        })
        .map(String::from);
    Ok(IdTokenInfo {
        email,
        chatgpt_account_id: auth
            .and_then(|a| a.get("chatgpt_account_id"))
            .and_then(Value::as_str)
            .map(String::from),
        plan_type: auth
            .and_then(|a| a.get("chatgpt_plan_type"))
            .and_then(Value::as_str)
            .map(String::from),
    })
}
