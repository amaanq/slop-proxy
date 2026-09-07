use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use axum::Extension;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{OriginalUri, State};
use axum::http::HeaderMap;
use axum::response::Response;
use futures_util::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::time::{MissedTickBehavior, interval, sleep, timeout};
use tokio_tungstenite::tungstenite::Message as UpstreamMessage;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as UpstreamCloseFrame;

use super::{DIALECT, PassthroughRequest, prepare_request, restore_reserved_namespace};
use crate::codex::types::{ResponsesEvent, ResponsesRequest};
use crate::codex::websocket::{
   Fault, MAX_MESSAGE_SIZE, Socket, response_error as upstream_error,
};
use crate::db::usage::AdmissionError;
use crate::pool::Route;
use crate::provider::Provider;
use crate::server::auth::{AuthInfo, bearer_token};
use crate::server::error::{error_response, out_of_scope, pool_error_response, translation_error};
use crate::server::facts::RequestFacts;
use crate::server::{AppState, LogGuard, pipeline};
use crate::translate::{UsageCapture, model_map};

const SEND_TIMEOUT: Duration = Duration::from_secs(30);
const PING_INTERVAL: Duration = Duration::from_secs(25);

pub async fn responses(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   OriginalUri(uri): OriginalUri,
   headers: HeaderMap,
   upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
   let requested = headers
      .get("x-codex-routing-hint")
      .and_then(|value| value.to_str().ok())
      .and_then(|hint| {
         hint
            .split(',')
            .find_map(|part| part.trim().strip_prefix("model="))
      })
      .unwrap_or(&state.cfg.models.default);
   let resolved = model_map::resolve(&state.cfg.models, requested);
   let provider = state.cfg.models.route(&resolved.model);
   if !auth.may_use(provider) {
      return out_of_scope(DIALECT, provider);
   }
   if provider != Provider::OpenAi {
      return super::responses_upgrade_required();
   }
   let upgrade = match upgrade {
      Ok(upgrade) => upgrade,
      Err(reason) => {
         tracing::warn!(%reason, "WebSocket upgrade unavailable, falling back to HTTP");
         return super::responses_upgrade_required();
      },
   };
   let session_key = ["session-id", "session_id", "thread-id", "thread_id"]
      .into_iter()
      .find_map(|name| headers.get(name)?.to_str().ok())
      .unwrap_or(&auth.user)
      .to_owned();
   let route = Route {
      session_key: &session_key,
      model: &resolved.model,
      user: &auth.user,
      pinned_account: auth.limits.pinned_account,
      prefer_trusted: auth.limits.prefer_trusted,
   };
   let (account_id, upstream) = match state.pools.codex.websocket(route, headers.clone()).await {
      Ok(connection) => connection,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   let relay = Relay {
      state,
      token: bearer_token(&headers, uri.query()).unwrap_or_default(),
      headers,
      auth,
      account_id,
      session_key,
      pending: BTreeMap::new(),
      upstream_failed: false,
   };
   let mut response = upgrade
      .max_message_size(MAX_MESSAGE_SIZE)
      .max_frame_size(MAX_MESSAGE_SIZE)
      .on_upgrade(move |socket| relay.run(socket, upstream.socket));
   for name in [
      "x-codex-turn-state",
      "x-reasoning-included",
      "x-models-etag",
      "openai-model",
      "x-request-id",
   ] {
      if let Some(value) = upstream.headers.get(name) {
         response.headers_mut().insert(name, value.clone());
      }
   }
   response
}

struct Pending {
   capture: UsageCapture,
   _guard: LogGuard,
   generated: bool,
}

struct Relay {
   state: AppState,
   token: String,
   headers: HeaderMap,
   auth: AuthInfo,
   account_id: Option<i64>,
   session_key: String,
   pending: BTreeMap<String, VecDeque<Pending>>,
   upstream_failed: bool,
}

impl Relay {
   async fn authenticate(&self) -> Result<AuthInfo, Response> {
      let token = self
         .state
         .db
         .auth_token(&self.token)
         .await
         .map_err(|_| error_response(DIALECT, 500, "api_error", "token lookup failed"))?
         .ok_or_else(|| {
            error_response(
               DIALECT,
               401,
               "authentication_error",
               "invalid or revoked API token",
            )
         })?;
      if !token.limits.may_use(Provider::OpenAi) {
         return Err(out_of_scope(DIALECT, Provider::OpenAi));
      }
      if token
         .limits
         .pinned_account
         .is_some_and(|id| Some(id) != self.account_id)
      {
         return Err(error_response(
            DIALECT,
            403,
            "permission_error",
            "reconnect to use the pinned account",
         ));
      }
      let admission = self
         .state
         .db
         .admit_token(token.id, &token.limits)
         .await
         .map_err(|_| error_response(DIALECT, 500, "api_error", "token metering failed"))?
         .map_err(|err| {
            let (message, retry_after) = match err {
               AdmissionError::RequestLimit { retry_after } => {
                  ("API token request limit exceeded", retry_after)
               },
               AdmissionError::TokenLimit { retry_after } => {
                  ("API token token limit exceeded", retry_after)
               },
            };
            let mut response = error_response(DIALECT, 429, "rate_limit_error", message);
            if let Ok(value) = retry_after.to_string().parse() {
               response.headers_mut().insert("retry-after", value);
            }
            response
         })?;
      if admission.slowdown_ms > 0 {
         sleep(Duration::from_millis(admission.slowdown_ms as u64)).await;
      }
      Ok(AuthInfo {
         token_id: token.id,
         user: token.user,
         meter_id: Some(admission.meter_id),
         limits: token.limits,
      })
   }

   async fn request(&mut self, text: &str) -> Result<String, Response> {
      let value: Value = serde_json::from_str(text)
         .map_err(|err| translation_error(DIALECT, &format!("invalid request {err}")))?;
      if value.get("type").and_then(Value::as_str) != Some("response.create") {
         return Ok(text.to_owned());
      }
      let started = Instant::now();
      let stream = stream_id(&value).to_owned();
      let req: PassthroughRequest = serde_json::from_value(value)
         .map_err(|err| translation_error(DIALECT, &format!("invalid request {err}")))?;
      let auth = self.authenticate().await?;
      let (mut req, requested_model, provider) =
         prepare_request(&self.state, &auth, req).map_err(|response| *response)?;
      if provider != Provider::OpenAi {
         return Err(translation_error(
            DIALECT,
            "this connection only serves OpenAI models",
         ));
      }
      req.stream = None;
      req.rest.remove("background");
      let encoded = serde_json::to_string(&req)
         .map_err(|err| translation_error(DIALECT, &format!("serializing request {err}")))?;
      let facts = serde_json::from_str::<ResponsesRequest>(&encoded)
         .ok()
         .map(|typed| RequestFacts::from_responses(&typed, &self.headers))
         .unwrap_or_default();
      let mut record = pipeline::record(
         &auth,
         "responses",
         provider,
         requested_model,
         req.model.unwrap_or_default(),
         facts,
      );
      record.account_id = self.account_id;
      record.request_bytes = text.len() as i64;
      record.session_key = req
         .prompt_cache_key
         .unwrap_or_else(|| self.session_key.clone());
      record.effort = req
         .reasoning
         .and_then(|reasoning| reasoning.effort)
         .unwrap_or_default();
      record.service_tier = req.service_tier.unwrap_or_default();
      let capture = UsageCapture::default();
      let guard = LogGuard::new(self.state.clone(), capture.clone(), record, started);
      self.pending.entry(stream).or_default().push_back(Pending {
         capture,
         _guard: guard,
         generated: req.rest.get("generate").and_then(Value::as_bool) != Some(false),
      });
      self.auth = auth;
      Ok(encoded)
   }

   async fn observe(&mut self, text: String) -> String {
      let Ok(mut value) = serde_json::from_str::<Value>(&text) else {
         return text;
      };
      if value.get("type").and_then(Value::as_str) == Some("codex.rate_limits") {
         self
            .state
            .pools
            .codex
            .rewrite_rate_limits(
               self.account_id,
               &self.auth.user,
               self.auth.limits.pinned_account,
               &mut value,
            )
            .await;
         return value.to_string();
      }
      let stream = stream_id(&value);
      let kind = value
         .get("type")
         .and_then(Value::as_str)
         .unwrap_or_default();
      let error = upstream_error(&value);
      match error.as_ref().map(|error| error.fault) {
         Some(Fault::Exhausted) => self.exhaust_upstream().await,
         Some(Fault::Transient) => self.fail_upstream().await,
         Some(Fault::Caller) | None => {},
      }
      if matches!(kind, "error" | "response.failed") {
         tracing::warn!(
            account = ?self.account_id,
            frame = %text.chars().take(600).collect::<String>(),
            "upstream error frame on websocket"
         );
      }
      let mut completed = false;
      if let Some(queue) = self.pending.get_mut(stream) {
         if let Some(pending) = queue.front() {
            pending.capture.note_bytes(text.len());
            if let Ok(event) = serde_json::from_value::<ResponsesEvent>(value.clone()) {
               pending.capture.observe(&event);
            }
            if kind == "error" {
               pending.capture.fail("upstream_rejected");
               pending.capture.note_stop_reason("error");
            }
            completed = kind == "response.completed" && pending.generated;
         }
         if matches!(
            kind,
            "response.completed" | "response.failed" | "response.incomplete" | "error"
         ) {
            queue.pop_front();
            if queue.is_empty() {
               self.pending.remove(stream);
            }
         }
      }
      if completed && !self.upstream_failed {
         self
            .state
            .pools
            .codex
            .websocket_completed(self.account_id)
            .await;
      }
      if let Some(error) = error {
         error.normalize(&mut value);
         return value.to_string();
      }
      text
   }

   /// Separate from the failure counter: an exhausted account is not flaky,
   /// it is spent, and coming back in a minute only spends the next turn.
   async fn exhaust_upstream(&mut self) {
      self.upstream_failed = true;
      self
         .state
         .pools
         .codex
         .websocket_exhausted(self.account_id)
         .await;
   }

   async fn fail_upstream(&mut self) {
      if !self.upstream_failed {
         self.upstream_failed = true;
         self
            .state
            .pools
            .codex
            .websocket_failed(self.account_id)
            .await;
      }
   }

   async fn upstream_closed(&mut self) {
      for pending in self.pending.values().flatten() {
         pending.capture.note_upstream_eof();
      }
      if !self.pending.is_empty() {
         self.fail_upstream().await;
      }
   }

   #[tracing::instrument(skip_all, fields(account_id = ?self.account_id, session = %self.session_key))]
   async fn run(mut self, mut client: WebSocket, mut upstream: Socket) {
      let mut keepalive = interval(PING_INTERVAL);
      keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);
      keepalive.tick().await;
      loop {
         tokio::select! {
            _ = keepalive.tick() => {
               if !send_upstream(&mut upstream, UpstreamMessage::Ping(Bytes::default())).await {
                  self.upstream_closed().await;
                  break;
               }
               if !send_client(&mut client, Message::Ping(Bytes::default())).await {
                  break;
               }
            },
            message = client.recv() => {
               let message = match message {
                  Some(Ok(message)) => message,
                  Some(Err(error)) => {
                     tracing::warn!(%error, "WebSocket client receive failed");
                     break;
                  },
                  None => {
                     tracing::info!("WebSocket client stream ended");
                     break;
                  },
               };
               let message = match message {
                  Message::Text(text) => match self.request(&text).await {
                     Ok(text) => UpstreamMessage::Text(text.into()),
                     Err(response) => {
                        let lane = serde_json::from_str::<Value>(&text).ok()
                           .and_then(|value| value.get("stream_id").cloned());
                        if !send_client(&mut client, response_error(response, lane).await).await { break; }
                        continue;
                     },
                  },
                  Message::Binary(_) => {
                     let _ = send_client(&mut client, Message::Close(Some(CloseFrame {
                        code: 1003, reason: "Responses requests must be text messages".into(),
                     }))).await;
                     let _ = send_upstream(&mut upstream, UpstreamMessage::Close(None)).await;
                     return;
                  },
                  Message::Ping(_) | Message::Pong(_) => continue,
                  Message::Close(frame) => {
                     tracing::info!(
                        code = ?frame.as_ref().map(|frame| frame.code),
                        reason = ?frame.as_ref().map(|frame| frame.reason.as_str()),
                        "WebSocket client closed"
                     );
                     let frame = frame.map(|frame| UpstreamCloseFrame {
                        code: frame.code.into(), reason: frame.reason.as_str().to_owned().into(),
                     });
                     let _ = tokio::join!(
                        timeout(SEND_TIMEOUT, client.flush()),
                        send_upstream(&mut upstream, UpstreamMessage::Close(frame)),
                     );
                     return;
                  },
               };
               if !send_upstream(&mut upstream, message).await {
                  self.upstream_closed().await;
                  break;
               }
            },
            message = upstream.next() => {
               let message = match message {
                  Some(Ok(message)) => message,
                  Some(Err(error)) => {
                     tracing::warn!(%error, "WebSocket upstream receive failed");
                     self.upstream_closed().await;
                     break;
                  },
                  None => {
                     tracing::warn!("WebSocket upstream stream ended");
                     self.upstream_closed().await;
                     break;
                  },
               };
               let message = match message {
                  UpstreamMessage::Text(text) => {
                     let text = self.observe(text.to_string()).await;
                     Message::Text(restore_reserved_namespace(text).into())
                  },
                  UpstreamMessage::Binary(bytes) => Message::Binary(bytes),
                  UpstreamMessage::Ping(_) | UpstreamMessage::Pong(_) | UpstreamMessage::Frame(_) => continue,
                  UpstreamMessage::Close(frame) => {
                     tracing::info!(
                        code = ?frame.as_ref().map(|frame| u16::from(frame.code)),
                        reason = ?frame.as_ref().map(|frame| frame.reason.as_str()),
                        "WebSocket upstream closed"
                     );
                     self.upstream_closed().await;
                     let frame = frame.map(|frame| CloseFrame {
                        code: frame.code.into(), reason: frame.reason.as_str().to_owned().into(),
                     });
                     let _ = tokio::join!(
                        timeout(SEND_TIMEOUT, upstream.flush()),
                        send_client(&mut client, Message::Close(frame)),
                     );
                     return;
                  },
               };
               if !send_client(&mut client, message).await { break; }
            },
         }
      }
      let _ = send_client(
         &mut client,
         Message::Close(Some(CloseFrame {
            code: 1011,
            reason: "WebSocket connection ended".into(),
         })),
      )
      .await;
      let _ = send_upstream(&mut upstream, UpstreamMessage::Close(None)).await;
   }
}

fn stream_id(value: &Value) -> &str {
   value
      .get("stream_id")
      .and_then(Value::as_str)
      .unwrap_or_default()
}

async fn send_client(client: &mut WebSocket, message: Message) -> bool {
   match timeout(SEND_TIMEOUT, client.send(message)).await {
      Ok(Ok(())) => true,
      Ok(Err(error)) => {
         tracing::warn!(%error, "WebSocket client send failed");
         false
      },
      Err(_) => {
         tracing::warn!("WebSocket client send timed out");
         false
      },
   }
}

async fn send_upstream(upstream: &mut Socket, message: UpstreamMessage) -> bool {
   match timeout(SEND_TIMEOUT, upstream.send(message)).await {
      Ok(Ok(())) => true,
      Ok(Err(error)) => {
         tracing::warn!(%error, "WebSocket upstream send failed");
         false
      },
      Err(_) => {
         tracing::warn!("WebSocket upstream send timed out");
         false
      },
   }
}

async fn response_error(response: Response, stream: Option<Value>) -> Message {
   let status = response.status();
   let headers: BTreeMap<_, _> = response
      .headers()
      .iter()
      .filter_map(|(name, value)| Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned())))
      .collect();
   let body = to_bytes(response.into_body(), 64 * 1024)
      .await
      .unwrap_or_default();
   let mut value: Value = serde_json::from_slice(&body).unwrap_or_else(|_| {
      json!({
         "error": { "type": "api_error", "message": String::from_utf8_lossy(&body) }
      })
   });
   value["type"] = json!("error");
   value["status"] = json!(status.as_u16());
   value["headers"] = json!(headers);
   if let Some(stream) = stream {
      value["stream_id"] = stream;
   }
   Message::Text(value.to_string().into())
}
