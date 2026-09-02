//! OpenCode Zen speaks the Responses API, so nothing here translates. The
//! request goes up as the caller wrote it and comes back as frames the codex
//! parser already understands.

use serde_json::Value;

use crate::config::ZenConfig;
use crate::upstream::{SendError, retry_after_secs};

pub struct ZenClient {
    http: reqwest::Client,
    cfg: ZenConfig,
}

impl ZenClient {
    pub fn new(cfg: ZenConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .build()
            .expect("building http client");
        Self { http, cfg }
    }

    /// The free contributor models answer without any credential at all, so
    /// the key is optional and only attached when an account supplies one.
    pub async fn send(&self, key: Option<&str>, req: &Value) -> Result<reqwest::Response, SendError> {
        let mut builder = self
            .http
            .post(format!(
                "{}/responses",
                self.cfg.base_url.trim_end_matches('/')
            ))
            .header("Accept", "text/event-stream");
        if let Some(key) = key.filter(|k| !k.is_empty()) {
            builder = builder.bearer_auth(key);
        }
        let resp = builder
            .json(req)
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
            429 => SendError::RateLimited { retry_after, body },
            400 => SendError::BadRequest(body),
            s => SendError::Upstream { status: s, body },
        })
    }

    pub async fn models(&self) -> Result<Vec<String>, String> {
        #[derive(serde::Deserialize)]
        struct Entry {
            id: String,
        }
        #[derive(serde::Deserialize)]
        struct Listing {
            data: Vec<Entry>,
        }
        let resp = self
            .http
            .get(format!(
                "{}/models",
                self.cfg.base_url.trim_end_matches('/')
            ))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(resp.status().to_string());
        }
        let listing: Listing = resp.json().await.map_err(|e| e.to_string())?;
        Ok(listing.data.into_iter().map(|e| e.id).collect())
    }
}
