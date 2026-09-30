use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Display;
use std::time::{Duration, Instant};

use axum::Extension;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures_util::{Sink, SinkExt as _, StreamExt as _};
use serde::{Deserialize as _, Serialize};
use serde_json::{Map, Value};
use tokio::time::{MissedTickBehavior, interval, timeout};
use tokio_tungstenite::tungstenite::Message as UpstreamMessage;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as UpstreamCloseFrame;

use crate::codex::turn_state::TurnState;
use crate::codex::types::{ResponsesEvent, ResponsesRequest};
use crate::codex::websocket::{Fault, MAX_MESSAGE_SIZE, Socket, response_error as upstream_error};
use crate::pool::{Route, Served};
use crate::provider::Provider;
use crate::server::auth::{AuthInfo, admit_token, bearer_token};
use crate::server::error::{error_response, pool_error_response, translation_error};
use crate::server::facts::RequestFacts;
use crate::server::openai::{
   DIALECT, PassthroughRequest, passthrough_record, prepare_request, responses_upgrade_required,
   restore_reserved_namespace,
};
use crate::server::relay::header_str;
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
   let requested = header_str(&headers, "x-codex-routing-hint")
      .and_then(|hint| {
         hint
            .split(',')
            .find_map(|part| part.trim().strip_prefix("model="))
      })
      .unwrap_or(&state.cfg.models.default);
   let resolved = model_map::resolve(&state.cfg.models, requested);
   match pipeline::admit(
      &state,
      &auth,
      DIALECT,
      "responses",
      requested,
      &resolved.model,
   ) {
      Ok(Provider::OpenAi) => {},
      Ok(_) => return responses_upgrade_required(),
      Err(response) => return *response,
   }
   let upgrade = match upgrade {
      Ok(upgrade) => upgrade,
      Err(reason) => {
         tracing::warn!(%reason, "WebSocket upgrade unavailable, falling back to HTTP");
         return responses_upgrade_required();
      },
   };
   let session_key = ["session-id", "session_id", "thread-id", "thread_id"]
      .into_iter()
      .find_map(|name| headers.get(name)?.to_str().ok())
      .unwrap_or(&auth.user)
      .to_owned();
   let route = auth.route(&session_key, &resolved.model);
   let Served {
      account_id,
      response: upstream,
      attempts,
   } = match state.pools.codex.websocket(route, headers.clone()).await {
      Ok(served) => served,
      Err(err) => return pool_error_response(DIALECT, &state.cfg.models, err),
   };
   let turn_state_blocks = observe_turn_state(&state, account_id, &upstream.headers).await;
   let relay = Relay {
      state,
      token: bearer_token(&headers, uri.query()).unwrap_or_default(),
      headers,
      auth,
      account_id,
      attempts,
      turn_state_blocks,
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

async fn observe_turn_state(
   state: &AppState,
   account_id: Option<i64>,
   headers: &HeaderMap,
) -> Option<i64> {
   let observed = TurnState::from_headers(headers)?;
   let blocks = observed.blocks as i64;
   state
      .pools
      .codex
      .note_turn_state(account_id, observed)
      .await;
   Some(blocks)
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
   attempts: u32,
   turn_state_blocks: Option<i64>,
   session_key: String,
   pending: BTreeMap<String, VecDeque<Pending>>,
   upstream_failed: bool,
}

impl Relay {
   async fn request(&mut self, text: &str, upstream: &mut Socket) -> Result<String, Response> {
      let value: Value = serde_json::from_str(text)
         .map_err(|err| translation_error(DIALECT, &format!("invalid request {err}")))?;
      if value.get("type").and_then(Value::as_str) != Some("response.create") {
         return Ok(text.to_owned());
      }
      let started = Instant::now();
      let stream = stream_id(&value).to_owned();
      let req: PassthroughRequest = serde_json::from_value(value)
         .map_err(|err| translation_error(DIALECT, &format!("invalid request {err}")))?;
      let (auth, _) = admit_token(&self.state, DIALECT, &self.token).await?;
      if auth
         .limits
         .pinned_account
         .is_some_and(|id| Some(id) != self.account_id)
      {
         return Err(error_response(
            DIALECT,
            StatusCode::FORBIDDEN,
            "permission_error",
            "reconnect to use the pinned account",
         ));
      }
      let (mut req, requested_model, provider) =
         prepare_request(&self.state, &auth, req).map_err(|response| *response)?;
      if provider != Provider::OpenAi {
         return Err(translation_error(
            DIALECT,
            "this connection only serves OpenAI models",
         ));
      }
      let model = req
         .model
         .as_deref()
         .unwrap_or(&self.state.cfg.models.default);
      let route = Route {
         service_tier: req.service_tier.as_deref(),
         ..auth.route(&self.session_key, model)
      };
      let serves = self
         .state
         .pools
         .codex
         .websocket_serves(self.account_id, route)
         .await
         .map_err(|error| pool_error_response(DIALECT, &self.state.cfg.models, error))?;
      if !serves {
         if !self.pending.is_empty()
            || req
               .rest
               .get("previous_response_id")
               .is_some_and(|previous| !previous.is_null())
         {
            return Err(translation_error(
               DIALECT,
               "changing service tier requires a new connection with the complete input history",
            ));
         }
         let redialed = self
            .state
            .pools
            .codex
            .websocket(route, self.headers.clone())
            .await
            .map_err(|error| pool_error_response(DIALECT, &self.state.cfg.models, error))?;
         let _ = send(upstream, UpstreamMessage::Close(None), "upstream").await;
         *upstream = redialed.response.socket;
         self.account_id = redialed.account_id;
         self.attempts = redialed.attempts;
         self.turn_state_blocks = Box::pin(observe_turn_state(
            &self.state,
            redialed.account_id,
            &redialed.response.headers,
         ))
         .await;
         self.upstream_failed = false;
      }
      req.stream = None;
      req.rest.remove("background");
      let encoded = serde_json::to_string(&req)
         .map_err(|err| translation_error(DIALECT, &format!("serializing request {err}")))?;
      let facts = serde_json::from_str::<ResponsesRequest>(&encoded)
         .ok()
         .map(|typed| RequestFacts::from_responses(&typed, &self.headers))
         .unwrap_or_default();
      let mut record = passthrough_record(
         &auth,
         provider,
         requested_model,
         &req,
         facts,
         &self.session_key,
      );
      record.account_id = self.account_id;
      record.request_bytes = text.len() as i64;
      record.attempts = i64::from(self.attempts);
      record.turn_state_blocks = self.turn_state_blocks;
      self.attempts = u32::from(self.account_id.is_some());
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
         // Bypasses the failure counter: spent is not flaky.
         Some(Fault::Exhausted) => {
            self.upstream_failed = true;
            self
               .state
               .pools
               .codex
               .websocket_exhausted(self.account_id)
               .await;
         },
         Some(Fault::Transient) => {
            let status = error.as_ref().map_or(0, |error| error.status);
            self
               .fail_upstream(&format!(
                  "websocket error frame {status}: {}",
                  text.chars().take(300).collect::<String>()
               ))
               .await;
         },
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
            if let Ok(event) = ResponsesEvent::deserialize(&value) {
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

   async fn fail_upstream(&mut self, why: &str) {
      if !self.upstream_failed {
         self.upstream_failed = true;
         self
            .state
            .pools
            .codex
            .websocket_failed(self.account_id, why)
            .await;
      }
   }

   async fn upstream_closed(&mut self) {
      for pending in self.pending.values().flatten() {
         pending.capture.note_upstream_eof();
      }
      if !self.pending.is_empty() {
         self
            .fail_upstream(&format!(
               "websocket closed with {} streams pending",
               self.pending.len()
            ))
            .await;
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
               if !send(&mut upstream, UpstreamMessage::Ping(Bytes::default()), "upstream").await {
                  self.upstream_closed().await;
                  break;
               }
               if !send(&mut client, Message::Ping(Bytes::default()), "client").await {
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
                  Message::Text(text) => match self.request(&text, &mut upstream).await {
                     Ok(text) => UpstreamMessage::Text(text.into()),
                     Err(response) => {
                        let lane = serde_json::from_str::<Value>(&text).ok()
                           .and_then(|value| value.get("stream_id").cloned());
                        let error = response_error(response, lane).await;
                        if !send(&mut client, error, "client").await { break; }
                        continue;
                     },
                  },
                  Message::Binary(_) => {
                     let _ = send(&mut client, Message::Close(Some(CloseFrame {
                        code: 1003, reason: "Responses requests must be text messages".into(),
                     })), "client").await;
                     let _ = send(&mut upstream, UpstreamMessage::Close(None), "upstream").await;
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
                        send(&mut upstream, UpstreamMessage::Close(frame), "upstream"),
                     );
                     return;
                  },
               };
               if !send(&mut upstream, message, "upstream").await {
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
                        send(&mut client, Message::Close(frame), "client"),
                     );
                     return;
                  },
               };
               if !send(&mut client, message, "client").await { break; }
            },
         }
      }
      let _ = send(
         &mut client,
         Message::Close(Some(CloseFrame {
            code: 1011,
            reason: "WebSocket connection ended".into(),
         })),
         "client",
      )
      .await;
      let _ = send(&mut upstream, UpstreamMessage::Close(None), "upstream").await;
   }
}

fn stream_id(value: &Value) -> &str {
   value
      .get("stream_id")
      .and_then(Value::as_str)
      .unwrap_or_default()
}

async fn send<S, M>(sink: &mut S, message: M, side: &str) -> bool
where
   S: Sink<M> + Unpin,
   S::Error: Display,
{
   match timeout(SEND_TIMEOUT, sink.send(message)).await {
      Ok(Ok(())) => true,
      Ok(Err(error)) => {
         tracing::warn!(%error, "WebSocket {side} send failed");
         false
      },
      Err(_) => {
         tracing::warn!("WebSocket {side} send timed out");
         false
      },
   }
}

#[derive(Serialize)]
struct ErrorBody<'a> {
   #[serde(rename = "type")]
   kind: &'static str,
   message: Cow<'a, str>,
}

#[derive(Serialize)]
struct ErrorFrame {
   #[serde(flatten)]
   body: Map<String, Value>,
   #[serde(rename = "type")]
   kind: &'static str,
   status: u16,
   headers: BTreeMap<String, String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   stream_id: Option<Value>,
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
   let body = serde_json::from_slice(&body).unwrap_or_else(|_| {
      let error = ErrorBody {
         kind: "api_error",
         message: String::from_utf8_lossy(&body),
      };
      Map::from_iter([(
         "error".to_owned(),
         serde_json::to_value(error).unwrap_or_default(),
      )])
   });
   let frame = ErrorFrame {
      body,
      kind: "error",
      status: status.as_u16(),
      headers,
      stream_id: stream,
   };
   Message::Text(serde_json::to_string(&frame).unwrap_or_default().into())
}
