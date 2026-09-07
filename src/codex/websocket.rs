use std::time::Duration;

use axum::http::Response;
use reqwest::header::{HeaderMap, HeaderValue};
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
