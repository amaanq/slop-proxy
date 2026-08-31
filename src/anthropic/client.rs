use serde_json::Value;

use crate::config::AnthropicConfig;
use crate::upstream::{SendError, retry_after_secs};

const OAUTH_BETA: &str = "oauth-2025-04-20";

const RESET_HEADERS: &[&str] = &[
    "anthropic-ratelimit-unified-reset",
    "anthropic-ratelimit-requests-reset",
];

/// Client headers worth carrying through to the upstream request.
#[derive(Debug, Default, Clone)]
pub struct RelayHeaders {
    pub version: Option<String>,
    pub beta: Option<String>,
    pub user_agent: Option<String>,
}

pub struct AnthropicClient {
    http: reqwest::Client,
    cfg: AnthropicConfig,
}

impl AnthropicClient {
    pub fn new(cfg: AnthropicConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("building http client");
        Self { http, cfg }
    }

    /// Statuses other than 401/429/5xx come back as `Ok` so the caller can
    /// forward them verbatim; only failures worth retrying on another
    /// account become errors.
    pub async fn send(
        &self,
        access_token: &str,
        path: &str,
        body: &Value,
        hdrs: &RelayHeaders,
    ) -> Result<reqwest::Response, SendError> {
        let beta = match &hdrs.beta {
            Some(b) if b.split(',').any(|p| p.trim() == OAUTH_BETA) => b.clone(),
            Some(b) => format!("{OAUTH_BETA},{b}"),
            None => OAUTH_BETA.into(),
        };
        let mut req = self
            .http
            .post(format!("{}{path}", self.cfg.base_url.trim_end_matches('/')))
            .bearer_auth(access_token)
            .header(
                "anthropic-version",
                hdrs.version.as_deref().unwrap_or("2023-06-01"),
            )
            .header("anthropic-beta", beta)
            .json(body);
        if let Some(ua) = &hdrs.user_agent {
            req = req.header("user-agent", ua);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| SendError::Network(e.to_string()))?;

        let status = resp.status().as_u16();
        if !matches!(status, 401 | 429 | 500..=599) {
            return Ok(resp);
        }
        let retry_after = retry_after_secs(resp.headers(), RESET_HEADERS);
        let body = resp.text().await.unwrap_or_default();
        let body = body.chars().take(2000).collect::<String>();
        Err(match status {
            401 => SendError::Auth(body),
            429 => SendError::RateLimited { retry_after, body },
            s => SendError::Upstream { status: s, body },
        })
    }
}
