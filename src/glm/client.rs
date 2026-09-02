//! Z.ai publishes an Anthropic-compatible endpoint, so a GLM request is the
//! one the caller already sent and the reply needs no translation.

use serde_json::Value;

use crate::config::GlmConfig;
use crate::upstream::{SendError, retry_after_secs};

pub struct GlmClient {
    http: reqwest::Client,
    cfg: GlmConfig,
}

impl GlmClient {
    pub fn new(cfg: GlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .build()
            .expect("building http client");
        Self { http, cfg }
    }

    pub async fn send(
        &self,
        key: &str,
        path: &str,
        body: &Value,
    ) -> Result<reqwest::Response, SendError> {
        let resp = self
            .http
            .post(format!("{}{path}", self.cfg.base_url.trim_end_matches('/')))
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| SendError::Network(e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }

        let retry_after = retry_after_secs(resp.headers(), &[]);
        let body = resp.text().await.unwrap_or_default();
        let body = body.chars().take(2000).collect::<String>();
        Err(match status.as_u16() {
            401 | 403 => SendError::Auth(body),
            429 if body.contains("Insufficient balance") => SendError::Auth(body),
            429 => SendError::RateLimited { retry_after, body },
            400 => SendError::BadRequest(body),
            s => SendError::Upstream { status: s, body },
        })
    }
}
