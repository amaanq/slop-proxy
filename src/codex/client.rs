use std::time::Duration;

use axum::body::Bytes;
use axum::http::Response;
use futures_util::StreamExt as _;
use futures_util::stream;
use reqwest::header;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::CodexConfig;
use crate::upstream::{Classify, SendError, classify};

const RULES: Classify = Classify {
   pass: |_| false,
   auth: &[401],
   reset_headers: &["x-codex-primary-reset-at"],
};

/// One rolling limit window as the usage endpoint reports it.
#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct UsageWindow {
   #[serde(default)]
   pub used_percent: f64,
   #[serde(default)]
   pub limit_window_seconds: i64,
   #[serde(default)]
   pub reset_at: Option<i64>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct RateLimit {
   #[serde(default)]
   pub limit_reached: bool,
   #[serde(default)]
   pub primary_window: Option<UsageWindow>,
   #[serde(default)]
   pub secondary_window: Option<UsageWindow>,
}

impl RateLimit {
   pub fn windows(&self) -> impl Iterator<Item = &UsageWindow> {
      [&self.primary_window, &self.secondary_window]
         .into_iter()
         .flatten()
   }
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct Usage {
   #[serde(default)]
   pub rate_limit: RateLimit,
}

pub struct CodexClient {
   http: reqwest::Client,
   cfg: CodexConfig,
}

/// The one field the retry strips; everything else goes back out untouched.
#[derive(Serialize, Deserialize)]
struct Retry {
   #[serde(skip_serializing_if = "Option::is_none")]
   max_output_tokens: Option<u64>,
   #[serde(flatten)]
   rest: serde_json::Map<String, serde_json::Value>,
}

const EARLY_REFUSAL_COOLDOWN: i64 = 60;

enum Opening {
   Pending,
   Serve,
   Refused(String),
}

/// `response.created` always arrives first, even when the next event is
/// `error` with "Selected model is at capacity", so a status code never
/// shows the refusal and the verdict is read from the first event after the
/// lifecycle ones. The backend queues a busy model behind `keepalive`
/// frames for about thirty seconds before refusing, so those wait too. A
/// `response.failed` is left alone, since it arrives with usage to bill and
/// is relayed as the terminal status.
fn opening(head: &[u8]) -> Opening {
   let text = String::from_utf8_lossy(head);
   let Some((frames, _)) = text.rsplit_once("\n\n") else {
      return Opening::Pending;
   };
   for frame in frames.split("\n\n") {
      let data = frame
         .lines()
         .filter_map(|line| line.strip_prefix("data:"))
         .map(str::trim_start)
         .collect::<Vec<_>>()
         .join("\n");
      if data.is_empty() {
         continue;
      }
      let Ok(event) = serde_json::from_str::<Value>(&data) else {
         return Opening::Serve;
      };
      let error = match event
         .get("type")
         .and_then(Value::as_str)
         .unwrap_or_default()
      {
         "response.created" | "response.in_progress" | "keepalive" => continue,
         "error" => event.get("error"),
         _ => return Opening::Serve,
      };
      let field = |name: &str| {
         error
            .and_then(|error| error.get(name))
            .and_then(Value::as_str)
            .unwrap_or_default()
      };
      return if field("type") == "invalid_request_error" {
         Opening::Serve
      } else {
         Opening::Refused(data)
      };
   }
   Opening::Pending
}

async fn refuse_early(resp: reqwest::Response) -> Result<reqwest::Response, SendError> {
   let status = resp.status();
   let headers = resp.headers().clone();
   let mut stream = resp.bytes_stream();
   let mut head = Vec::new();
   loop {
      match opening(&head) {
         Opening::Serve => break,
         Opening::Refused(body) => {
            tracing::warn!(%body, "backend refused inside a 200, trying another account");
            return Err(SendError::RateLimited {
               retry_after: Some(EARLY_REFUSAL_COOLDOWN),
               body,
            });
         },
         Opening::Pending if head.len() > 256 * 1024 => break,
         Opening::Pending => match stream.next().await {
            Some(Ok(chunk)) => head.extend_from_slice(&chunk),
            Some(Err(err)) => return Err(SendError::Network(err.to_string())),
            None => break,
         },
      }
   }
   let replay =
      stream::once(async move { Ok::<Bytes, reqwest::Error>(Bytes::from(head)) }).chain(stream);
   let mut rebuilt = Response::new(reqwest::Body::wrap_stream(replay));
   *rebuilt.status_mut() = status;
   *rebuilt.headers_mut() = headers;
   Ok(reqwest::Response::from(rebuilt))
}

impl CodexClient {
   pub fn new(cfg: CodexConfig) -> Self {
      let http = reqwest::Client::builder()
         .cookie_store(true)
         .connect_timeout(Duration::from_secs(30))
         .tcp_keepalive(Duration::from_secs(30))
         .user_agent(cfg.user_agent.clone())
         .build()
         .expect("building http client");
      Self { http, cfg }
   }

   pub async fn post(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
      req: &Bytes,
      session_id: &str,
      model: &str,
   ) -> Result<reqwest::Response, SendError> {
      match self
         .send_once(access_token, chatgpt_account_id, req, session_id, model)
         .await
      {
         Err(SendError::BadRequest(body))
            if body.contains("max_output_tokens")
               && let Ok(mut retry) = serde_json::from_slice::<Retry>(req)
               && retry.max_output_tokens.take().is_some()
               && let Ok(retry) = serde_json::to_vec(&retry) =>
         {
            tracing::debug!("upstream rejected max_output_tokens; retrying without it");
            self
               .send_once(
                  access_token,
                  chatgpt_account_id,
                  &Bytes::from(retry),
                  session_id,
                  model,
               )
               .await
         },
         // Cloudflare occasionally 403s fresh headless clients; the cookie
         // jar picks up clearance on the first response, so retry once.
         Err(SendError::Upstream { status: 403, .. }) => {
            self
               .send_once(access_token, chatgpt_account_id, req, session_id, model)
               .await
         },
         other => other,
      }
   }

   /// Quota without spending an inference request. The same figures ride on
   /// response headers, but only for accounts that are actively serving.
   pub async fn usage(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
   ) -> Result<Usage, SendError> {
      let resp = self
         .http
         .get(format!("{}/usage", self.cfg.base_url.trim_end_matches('/')))
         .bearer_auth(access_token)
         .header("chatgpt-account-id", chatgpt_account_id)
         .header("originator", self.cfg.originator.clone())
         .header("version", self.cfg.version.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, Classify::STRICT).await?;
      let status = resp.status().as_u16();
      resp.json().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("parsing usage response: {err}"),
      })
   }

   pub const fn soft_utilization_limit(&self) -> f64 {
      self.cfg.soft_utilization_limit
   }

   pub fn models_url(&self) -> String {
      format!(
         "{}/models?client_version={}",
         self.cfg.base_url.trim_end_matches('/'),
         self.cfg.version
      )
   }

   async fn models_response(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
   ) -> Result<reqwest::Response, SendError> {
      self
         .http
         .get(self.models_url())
         .bearer_auth(access_token)
         .header("ChatGPT-Account-ID", chatgpt_account_id)
         .header("chatgpt-account-id", chatgpt_account_id)
         .header("originator", self.cfg.originator.clone())
         .header("version", self.cfg.version.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))
   }

   pub async fn models_raw(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
   ) -> Result<(reqwest::StatusCode, String), SendError> {
      let resp = self
         .models_response(access_token, chatgpt_account_id)
         .await?;
      let status = resp.status();
      let status_u16 = status.as_u16();
      let body = resp.text().await.map_err(|err| SendError::Upstream {
         status: status_u16,
         body: format!("reading models response: {err}"),
      })?;
      Ok((status, body))
   }

   pub async fn list_models(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
   ) -> Result<Vec<super::models::ModelInfo>, SendError> {
      let resp = self
         .models_response(access_token, chatgpt_account_id)
         .await?;
      let resp = classify(resp, Classify::STRICT).await?;
      let status = resp.status().as_u16();
      let parsed: super::models::ModelsResponse =
         resp.json().await.map_err(|err| SendError::Upstream {
            status,
            body: format!("parsing models response: {err}"),
         })?;
      Ok(parsed.models)
   }

   async fn send_once(
      &self,
      access_token: &str,
      chatgpt_account_id: &str,
      req: &Bytes,
      session_id: &str,
      model: &str,
   ) -> Result<reqwest::Response, SendError> {
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
         .header("version", self.cfg.version.clone())
         .header("session_id", session_id)
         .header("session-id", session_id)
         .header("thread_id", session_id)
         .header("thread-id", session_id)
         .header("x-codex-routing-hint", format!("model={model}"))
         .header("Accept", "text/event-stream")
         .header(header::CONTENT_TYPE, "application/json")
         .body(req.clone())
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, RULES).await?;
      refuse_early(resp).await
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   fn head(events: &[&str]) -> Vec<u8> {
      events
         .iter()
         .fold(String::new(), |mut out, data| {
            out.push_str("event: x\ndata: ");
            out.push_str(data);
            out.push_str("\n\n");
            out
         })
         .into_bytes()
   }

   #[test]
   fn a_refusal_after_the_lifecycle_events_is_read_as_one() {
      let created = r#"{"type":"response.created","response":{}}"#;
      let capacity = r#"{"type":"error","error":{"type":"server_error","code":"model_at_capacity","message":"Selected model is at capacity."}}"#;
      let bad = r#"{"type":"error","error":{"type":"invalid_request_error","code":"invalid_encrypted_content","message":"x"}}"#;
      let failed =
         r#"{"type":"response.failed","response":{"status":"failed","usage":{"output_tokens":5}}}"#;
      let output = r#"{"type":"response.output_item.added","item":{}}"#;
      let keepalive = r#"{"type":"keepalive"}"#;
      assert!(matches!(opening(&head(&[created])), Opening::Pending));
      assert!(matches!(
         opening(&head(&[created, keepalive, keepalive])),
         Opening::Pending
      ));
      assert!(matches!(
         opening(&head(&[created, keepalive, capacity])),
         Opening::Refused(_)
      ));
      assert!(matches!(
         opening(&head(&[created, capacity])),
         Opening::Refused(_)
      ));
      assert!(matches!(opening(&head(&[created, bad])), Opening::Serve));
      assert!(matches!(opening(&head(&[created, failed])), Opening::Serve));
      assert!(matches!(opening(&head(&[created, output])), Opening::Serve));
      assert!(matches!(
         opening(b"event: x\ndata: {\"type\":\"resp"),
         Opening::Pending
      ));
   }
}
