use serde_json::Value;

use crate::config::CodexConfig;
use crate::upstream::{SendError, retry_after_secs};

pub struct CodexClient {
    http: reqwest::Client,
    cfg: CodexConfig,
}

impl CodexClient {
    pub fn new(cfg: CodexConfig) -> Self {
        let http = reqwest::Client::builder()
            .cookie_store(true)
            .connect_timeout(std::time::Duration::from_secs(30))
            .user_agent(cfg.user_agent.clone())
            .build()
            .expect("building http client");
        Self { http, cfg }
    }

    pub async fn send(
        &self,
        access_token: &str,
        chatgpt_account_id: &str,
        req: &Value,
    ) -> Result<reqwest::Response, SendError> {
        match self.send_once(access_token, chatgpt_account_id, req).await {
            Err(SendError::BadRequest(body))
                if body.contains("max_output_tokens") && req.get("max_output_tokens").is_some() =>
            {
                tracing::debug!("upstream rejected max_output_tokens; retrying without it");
                let mut retry = req.clone();
                retry.as_object_mut().map(|o| o.remove("max_output_tokens"));
                self.send_once(access_token, chatgpt_account_id, &retry)
                    .await
            }
            // Cloudflare occasionally 403s fresh headless clients; the cookie
            // jar picks up clearance on the first response, so retry once.
            Err(SendError::Upstream { status: 403, .. }) => {
                self.send_once(access_token, chatgpt_account_id, req).await
            }
            other => other,
        }
    }

    pub fn models_url(&self) -> String {
        format!(
            "{}/models?client_version={}",
            self.cfg.base_url.trim_end_matches('/'),
            self.cfg.version
        )
    }

    pub async fn models_raw(
        &self,
        access_token: &str,
        chatgpt_account_id: &str,
    ) -> Result<(reqwest::StatusCode, String), String> {
        let resp = self
            .http
            .get(self.models_url())
            .bearer_auth(access_token)
            .header("ChatGPT-Account-ID", chatgpt_account_id)
            .header("chatgpt-account-id", chatgpt_account_id)
            .header("originator", self.cfg.originator.clone())
            .header("version", self.cfg.version.clone())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| e.to_string())?;
        Ok((status, body))
    }

    pub async fn list_models(
        &self,
        access_token: &str,
        chatgpt_account_id: &str,
    ) -> Result<Vec<super::models::ModelInfo>, String> {
        let (status, body) = self.models_raw(access_token, chatgpt_account_id).await?;
        if !status.is_success() {
            return Err(format!(
                "{status}: {}",
                body.chars().take(400).collect::<String>()
            ));
        }
        let parsed = serde_json::from_str::<super::models::ModelsResponse>(&body)
            .map_err(|e| format!("parsing models response: {e}"))?;
        Ok(parsed.models)
    }

    async fn send_once(
        &self,
        access_token: &str,
        chatgpt_account_id: &str,
        req: &Value,
    ) -> Result<reqwest::Response, SendError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let resp = self
            .http
            .post(format!(
                "{}/responses",
                self.cfg.base_url.trim_end_matches('/')
            ))
            .bearer_auth(access_token)
            .header("chatgpt-account-id", chatgpt_account_id)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", self.cfg.originator.clone())
            .header("session_id", session_id.clone())
            .header("session-id", session_id)
            .header("Accept", "text/event-stream")
            .json(req)
            .send()
            .await
            .map_err(|e| SendError::Network(e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }

        let retry_after = retry_after_secs(resp.headers(), &["x-codex-primary-reset-at"]);
        let body = resp.text().await.unwrap_or_default();
        let body = body.chars().take(2000).collect::<String>();
        Err(match status.as_u16() {
            401 => SendError::Auth(body),
            429 => SendError::RateLimited { retry_after, body },
            400 => SendError::BadRequest(body),
            s => SendError::Upstream { status: s, body },
        })
    }
}
