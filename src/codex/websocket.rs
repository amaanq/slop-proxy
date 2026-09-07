use std::time::Duration;

use axum::http::Response;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async_with_config, tungstenite};

use super::client::CodexClient;
use crate::upstream::{Classify, SendError, classify};

pub const MAX_MESSAGE_SIZE: usize = 192 * 1024 * 1024;
pub type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub struct Connection {
   pub socket: Socket,
   pub headers: HeaderMap,
}

/// What an error frame says about the account that produced it, which is a
/// different question from whether the request may be retried.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fault {
   /// The request itself, so every other account would refuse it too.
   Caller,
   /// This account has spent its allowance and stays spent until the window
   /// rolls over.
   Exhausted,
   /// Upstream weather. Worth another account now and worth coming back to.
   Transient,
}

pub struct ResponseError {
   pub status: u16,
   pub fault: Fault,
}

impl ResponseError {
   /// Whether to hide the upstream code behind `upstream_unavailable`. Only a
   /// 5xx earns that: a 429 names a real condition the client should read.
   pub const fn transient(&self) -> bool {
      matches!(self.fault, Fault::Transient) && self.status >= 500
   }
}

impl ResponseError {
   pub fn normalize(self, event: &mut Value) {
      let bare = event.get("type").and_then(Value::as_str) == Some("error");
      if bare {
         event["status"] = Value::from(self.status);
         if let Some(event) = event.as_object_mut() {
            event.remove("status_code");
         }
      }
      let error = if bare {
         event.get_mut("error")
      } else {
         event
            .get_mut("response")
            .and_then(|response| response.get_mut("error"))
      };
      if self.transient()
         && let Some(error) = error
         && matches!(
            error.get("code").and_then(Value::as_str),
            Some("server_is_overloaded" | "slow_down")
         )
      {
         error["upstream_code"] = error["code"].take();
         error["code"] = Value::from("upstream_unavailable");
      }
   }
}

pub fn response_error(event: &Value) -> Option<ResponseError> {
   let error = match event.get("type")?.as_str()? {
      "error" => event.get("error"),
      "response.failed" => event.get("response")?.get("error"),
      _ => None,
   }?;
   let code = error
      .get("code")
      .and_then(Value::as_str)
      .unwrap_or_default();
   let kind = error
      .get("type")
      .and_then(Value::as_str)
      .unwrap_or_default();
   let (fallback, fault) = match code {
      "cyber_policy"
      | "invalid_prompt"
      | "context_length_exceeded"
      | "invalid_encrypted_content"
      | "previous_response_not_found" => (400, Fault::Caller),
      "insufficient_quota" | "usage_limit_reached" | "usage_not_included" => {
         (429, Fault::Exhausted)
      },
      "rate_limit_exceeded" => (429, Fault::Transient),
      "server_is_overloaded" | "slow_down" | "model_at_capacity" => (503, Fault::Transient),
      _ => match kind {
         "invalid_request_error" => (400, Fault::Caller),
         "authentication_error" => (401, Fault::Caller),
         "permission_error" => (403, Fault::Caller),
         "usage_limit_reached" | "usage_not_included" | "insufficient_quota" => {
            (429, Fault::Exhausted)
         },
         "rate_limit_error" | "rate_limit_exceeded" => (429, Fault::Transient),
         "server_error" | "api_error" | "internal_server_error" => (500, Fault::Transient),
         _ if code == "server_error" => (500, Fault::Transient),
         _ => (0, Fault::Transient),
      },
   };
   let status = event
      .get("status")
      .and_then(Value::as_u64)
      .or_else(|| event.get("status_code").and_then(Value::as_u64))
      .unwrap_or(fallback);
   // A 5xx is the only status worth calling weather. Anything else that
   // claimed to be transient was guessing from a code it did not recognise.
   let fault = match fault {
      Fault::Transient if status < 500 && status != 429 => Fault::Caller,
      Fault::Caller => Fault::Caller,
      Fault::Exhausted => Fault::Exhausted,
      Fault::Transient => Fault::Transient,
   };
   (400..600)
      .contains(&status)
      .then_some(ResponseError { status: status as u16, fault })
}

impl CodexClient {
   pub fn responses_url(&self) -> String {
      format!("{}/responses", self.config().base_url.trim_end_matches('/'))
   }

   pub fn responses_headers(
      &self,
      token: &str,
      account: &str,
      session: &str,
      model: &str,
      incoming: &HeaderMap,
   ) -> Result<HeaderMap, SendError> {
      let caller_session = incoming
         .get("session-id")
         .or_else(|| incoming.get("session_id"))
         .and_then(|value| value.to_str().ok())
         .unwrap_or(session);
      let thread = incoming
         .get("thread-id")
         .or_else(|| incoming.get("thread_id"))
         .and_then(|value| value.to_str().ok())
         .unwrap_or(caller_session);
      let authorization = format!("Bearer {token}");
      let routing = format!("model={model}");
      let routing = incoming
         .get("x-codex-routing-hint")
         .and_then(|value| value.to_str().ok())
         .map(|hint| {
            hint
               .split(',')
               .map(|part| {
                  if part.trim().starts_with("model=") {
                     routing.as_str()
                  } else {
                     part
                  }
               })
               .collect::<Vec<_>>()
               .join(",")
         })
         .unwrap_or(routing);
      let mut headers = HeaderMap::new();
      for (name, value) in [
         ("authorization", authorization.as_str()),
         ("chatgpt-account-id", account),
         ("openai-beta", "responses=experimental"),
         ("originator", self.config().originator.as_str()),
         ("version", self.config().version.as_str()),
         ("user-agent", self.config().user_agent.as_str()),
         ("session_id", caller_session),
         ("session-id", caller_session),
         ("thread_id", thread),
         ("thread-id", thread),
         ("x-codex-routing-hint", routing.as_str()),
      ] {
         let value =
            HeaderValue::from_str(value).map_err(|err| SendError::Network(err.to_string()))?;
         headers.insert(name, value);
      }
      for name in [
         "openai-beta",
         "originator",
         "version",
         "user-agent",
         "session_id",
         "session-id",
         "thread_id",
         "thread-id",
         "x-codex-turn-metadata",
         "x-codex-turn-state",
         "x-codex-beta-features",
         "x-codex-window-id",
         "x-client-request-id",
      ] {
         if let Some(value) = incoming.get(name) {
            headers.insert(name, value.clone());
         }
      }
      Ok(headers)
   }

   pub async fn connect_websocket(
      &self,
      token: &str,
      account: &str,
      session: &str,
      model: &str,
      incoming: &HeaderMap,
   ) -> Result<Connection, SendError> {
      let mut url = reqwest::Url::parse(&self.responses_url())
         .map_err(|err| SendError::Network(err.to_string()))?;
      let scheme = match url.scheme() {
         "https" => "wss",
         "http" => "ws",
         _ => {
            return Err(SendError::Network(
               "unsupported WebSocket upstream scheme".into(),
            ));
         },
      };
      url.set_scheme(scheme)
         .map_err(|()| SendError::Network("invalid WebSocket upstream URL".into()))?;
      let mut request = url
         .as_str()
         .into_client_request()
         .map_err(|err| SendError::Network(err.to_string()))?;
      let mut headers = self.responses_headers(token, account, session, model, incoming)?;
      if !incoming.contains_key("openai-beta") {
         headers.insert(
            "openai-beta",
            HeaderValue::from_static("responses_websockets=2026-02-06"),
         );
      }
      request.headers_mut().extend(headers);
      let config = WebSocketConfig::default()
         .max_message_size(Some(MAX_MESSAGE_SIZE))
         .max_frame_size(Some(MAX_MESSAGE_SIZE));
      let result = timeout(
         Duration::from_secs(30),
         connect_async_with_config(request, Some(config), false),
      )
      .await
      .map_err(|_| SendError::Network("WebSocket handshake timed out".into()))?;
      match result {
         Ok((socket, response)) => Ok(Connection {
            socket,
            headers: response.headers().clone(),
         }),
         Err(tungstenite::Error::Http(response)) => {
            let (parts, body) = response.into_parts();
            let rejected = Response::from_parts(parts, body.unwrap_or_default());
            let classified = classify(
               rejected.into(),
               Classify {
                  pass: |_| false,
                  auth: &[401],
                  reset_headers: &["x-codex-primary-reset-at"],
               },
            )
            .await;
            Err(classified.err().unwrap_or_else(|| {
               SendError::Network("upstream did not upgrade to WebSocket".into())
            }))
         },
         Err(err) => Err(SendError::Network(err.to_string())),
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::config::CodexConfig;

   #[test]
   fn routing_hints_use_the_resolved_model_and_keep_other_fields() {
      let client = CodexClient::new(CodexConfig::default());
      let mut incoming = HeaderMap::new();
      incoming.insert(
         "x-codex-routing-hint",
         "model=alias,feature=enabled".parse().unwrap(),
      );
      let headers = client
         .responses_headers("token", "account", "session", "gpt-6-astra", &incoming)
         .unwrap();
      assert_eq!(
         headers["x-codex-routing-hint"],
         "model=gpt-6-astra,feature=enabled"
      );
      assert_eq!(headers["version"], "0.153.4");
      assert_eq!(headers["session-id"], "session");
      assert_eq!(headers["thread-id"], "session");
   }
}
