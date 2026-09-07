use axum::extract::ws::{CloseFrame, Message as ServerMessage, WebSocketUpgrade};
use axum::http::{HeaderMap, Request};
use axum::routing::get;
use futures_util::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio::task::yield_now;
use tokio::time::timeout;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as UpstreamCloseFrame;
use tokio_tungstenite::tungstenite::{Error, Message};

use super::*;
use crate::codex::websocket::Socket;

async fn serve_upstream(app: Router) -> (String, Db) {
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let base = format!("http://{}", listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(listener, app).await.unwrap();
   });
   spawn_proxy_at(ModelsConfig::default(), None, base).await
}

async fn mock() -> (String, Db, mpsc::Receiver<Value>, Arc<Mutex<HeaderMap>>) {
   let (sender, receiver) = mpsc::channel(16);
   let seen = Arc::new(Mutex::new(HeaderMap::new()));
   let headers = Arc::clone(&seen);
   let app = Router::new().route(
      "/responses",
      get(move |upgrade: WebSocketUpgrade, incoming: HeaderMap| {
         let sender = sender.clone();
         *headers.lock().unwrap() = incoming;
         async move {
            let mut response = upgrade.on_upgrade(move |mut socket| async move {
               while let Some(Ok(message)) = socket.recv().await {
                  let ServerMessage::Text(text) = message else {
                     continue;
                  };
                  let value: Value = serde_json::from_str(&text).unwrap();
                  let _ = sender.send(value.clone()).await;
                  if value["hold"] == true {
                     continue;
                  }
                  let finish = value["finish"].as_str().unwrap_or("completed");
                  let mut event = json!({
                     "type": format!("response.{finish}"),
                     "response": {
                        "id": value["label"].as_str().unwrap_or("r"),
                        "status": finish,
                        "output": [],
                        "usage": {"input_tokens": 12_i32, "output_tokens": 5_i32},
                        "future_field": {"preserved": true}
                     }
                  });
                  if let Some(lane) = value.get("stream_id") {
                     event["stream_id"] = lane.clone();
                  }
                  if socket
                     .send(ServerMessage::Text(event.to_string().into()))
                     .await
                     .is_err()
                  {
                     break;
                  }
               }
            });
            response
               .headers_mut()
               .insert("x-codex-turn-state", "state-from-upstream".parse().unwrap());
            response
               .headers_mut()
               .insert("x-reasoning-included", "true".parse().unwrap());
            response
               .headers_mut()
               .insert("x-models-etag", "catalog-v1".parse().unwrap());
            response
         }
      }),
   );
   let (base, db) = serve_upstream(app).await;
   (base, db, receiver, seen)
}

fn upgrade_request(base: &str) -> Request<()> {
   let mut request = format!("{}/v1/responses", base.replacen("http", "ws", 1))
      .into_client_request()
      .unwrap();
   for (name, value) in [
      ("authorization", "Bearer sp-test"),
      ("chatgpt-account-id", "untrusted-account"),
      ("cookie", "do-not-forward=secret"),
      ("openai-beta", "responses_websockets=2026-02-06"),
      ("version", "0.153.4"),
      ("originator", "codex_exec"),
      ("session-id", "session-from-client"),
      ("thread-id", "thread-from-client"),
      ("x-codex-routing-hint", "model=gpt-6-astra"),
      (
         "x-codex-turn-metadata",
         r#"{"request_kind":"prewarm","installation_id":"test"}"#,
      ),
      ("x-codex-beta-features", "remote_compaction_v2"),
      ("x-codex-window-id", "window-from-client"),
      ("x-client-request-id", "request-from-client"),
   ] {
      request.headers_mut().insert(name, value.parse().unwrap());
   }
   request
}

async fn send(socket: &mut Socket, value: Value) {
   socket
      .send(Message::Text(value.to_string().into()))
      .await
      .unwrap();
}

async fn event(socket: &mut Socket) -> Value {
   let message = timeout(Duration::from_secs(3), socket.next())
      .await
      .unwrap()
      .unwrap()
      .unwrap();
   serde_json::from_str(message.to_text().unwrap()).unwrap()
}

#[tokio::test]
async fn websocket_preserves_prewarm_continuations_and_handshake_metadata() {
   let (base, db, mut requests, headers) = mock().await;
   let request = upgrade_request(&base);
   let original = request.headers().clone();
   let (mut socket, response) = connect_async(request).await.unwrap();
   assert_eq!(response.status(), 101);
   assert_eq!(
      response.headers()["x-codex-turn-state"],
      "state-from-upstream"
   );
   assert_eq!(response.headers()["x-reasoning-included"], "true");
   assert_eq!(response.headers()["x-models-etag"], "catalog-v1");
   assert_eq!(
      db.token_meter("sp-test").await.unwrap().unwrap().requests,
      0
   );
   let received = headers.lock().unwrap().clone();
   for name in [
      "openai-beta",
      "version",
      "originator",
      "session-id",
      "thread-id",
      "x-codex-routing-hint",
      "x-codex-turn-metadata",
      "x-codex-beta-features",
      "x-codex-window-id",
      "x-client-request-id",
   ] {
      assert_eq!(received[name], original[name], "{name}");
   }
   assert_eq!(received["authorization"], "Bearer at");
   assert_eq!(received["chatgpt-account-id"], "acct-1");
   assert_eq!(received["session_id"], received["session-id"]);
   assert_eq!(received["thread_id"], received["thread-id"]);
   assert!(!received.contains_key("cookie"));
   assert_ne!(received["sec-websocket-key"], original["sec-websocket-key"]);
   send(&mut socket, json!({"type":"response.create", "model":"gpt-6-astra", "generate":false, "label":"warm", "input":[]})).await;
   assert_eq!(event(&mut socket).await["response"]["id"], "warm");
   let prewarm = requests.recv().await.unwrap();
   assert_eq!(prewarm["generate"], false);
   assert!(prewarm.get("stream").is_none());
   assert_eq!(prewarm["store"], false);
   send(&mut socket, json!({"type":"response.create", "model":"gpt-6-astra", "previous_response_id":"warm", "input":[{"role":"user", "content":"continue"}], "future_request_field":true})).await;
   let completed = event(&mut socket).await;
   assert_eq!(completed["response"]["future_field"]["preserved"], true);
   let turn = requests.recv().await.unwrap();
   assert_eq!(turn["previous_response_id"], "warm");
   assert_eq!(turn["input"].as_array().unwrap().len(), 1);
   assert_eq!(turn["future_request_field"], true);
   db.flush().await.unwrap();
   let totals = db.usage_totals(0, i64::MAX).await.unwrap();
   assert_eq!(totals.requests, 2);
   assert_eq!(totals.input_tokens, 24);
   assert_eq!(totals.output_tokens, 10);
   socket.close(None).await.unwrap();
}

#[tokio::test]
async fn websocket_enforces_each_turn_limit_and_revocation() {
   let (base, db, mut requests, _) = mock().await;
   db.set_token_limits(
      "sp-test",
      &TokenLimits {
         requests: Some(1),
         window_seconds: 3600,
         ..TokenLimits::default()
      },
   )
   .await
   .unwrap();
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   let turn = json!({"type":"response.create", "model":"gpt-6-astra", "input":[]});
   send(&mut socket, turn.clone()).await;
   assert_eq!(event(&mut socket).await["type"], "response.completed");
   requests.recv().await.unwrap();
   send(&mut socket, turn.clone()).await;
   let error = event(&mut socket).await;
   assert_eq!(error["status"], 429_i32);
   assert!(error["headers"]["retry-after"].is_string());
   requests.try_recv().unwrap_err();
   db.revoke_token("sp-test").await.unwrap();
   send(&mut socket, turn).await;
   assert_eq!(event(&mut socket).await["status"], 401_i32);
}

#[tokio::test]
async fn websocket_scopes_models_on_every_turn() {
   let (base, db, mut requests, _) = mock().await;
   db.set_token_limits(
      "sp-test",
      &TokenLimits {
         providers: vec![Provider::OpenAi],
         ..TokenLimits::default()
      },
   )
   .await
   .unwrap();
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   send(
      &mut socket,
      json!({"type":"response.create", "model":"gemini-test", "input":[]}),
   )
   .await;
   assert_eq!(event(&mut socket).await["status"], 403_i32);
   requests.try_recv().unwrap_err();
}

#[tokio::test]
async fn websocket_terminal_statuses_log_usage_per_lane() {
   let (base, db, _, _) = mock().await;
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   for finish in ["completed", "incomplete", "failed"] {
      send(&mut socket, json!({"type":"response.create", "model":"gpt-6-astra", "stream_id":finish, "finish":finish, "input":[]})).await;
   }
   for finish in ["completed", "incomplete", "failed"] {
      let response = event(&mut socket).await;
      assert_eq!(response["stream_id"], finish);
      assert_eq!(response["response"]["status"], finish);
   }
   db.flush().await.unwrap();
   let totals = db.usage_totals(0, i64::MAX).await.unwrap();
   assert_eq!(totals.requests, 3);
   assert_eq!(totals.output_tokens, 15);
   let rows = db.usage_metrics().await.unwrap();
   assert_eq!(rows[0].errors, 1);
}

#[tokio::test]
async fn websocket_upgrade_rejects_bad_auth_and_preserves_http_fallback() {
   let (base, _, _, _) = mock().await;
   let mut request = upgrade_request(&base);
   request.headers_mut().remove("authorization");
   let Err(Error::Http(response)) = connect_async(request).await else {
      panic!("expected rejected upgrade")
   };
   assert_eq!(response.status(), 401);
   let mut fallback = upgrade_request(&base);
   fallback
      .headers_mut()
      .insert("x-codex-routing-hint", "model=gemini-test".parse().unwrap());
   let Err(Error::Http(rejected)) = connect_async(fallback).await else {
      panic!("expected HTTP fallback")
   };
   assert_eq!(rejected.status(), 426);
}

#[tokio::test]
async fn http_responses_forward_the_same_client_metadata() {
   let seen = Arc::new(Mutex::new(HeaderMap::new()));
   let capture = Arc::clone(&seen);
   let app = Router::new().route(
      "/responses",
      post(move |headers: HeaderMap| async move {
         *capture.lock().unwrap() = headers;
         ([("content-type", "text/event-stream")], MOCK_SSE)
      }),
   );
   let (base, _) = serve_upstream(app).await;
   let incoming = upgrade_request(&base);
   let mut headers = incoming.headers().clone();
   for name in [
      "connection",
      "upgrade",
      "sec-websocket-key",
      "sec-websocket-version",
      "host",
   ] {
      headers.remove(name);
   }
   let response = reqwest::Client::new()
      .post(format!("{base}/v1/responses"))
      .headers(headers)
      .json(&json!({"model":"gpt-6-astra", "input":[]}))
      .send()
      .await
      .unwrap();
   assert_eq!(response.status(), 200);
   response.bytes().await.unwrap();
   let received = seen.lock().unwrap();
   for name in [
      "version",
      "originator",
      "session-id",
      "thread-id",
      "x-codex-routing-hint",
      "x-codex-turn-metadata",
      "x-codex-window-id",
      "x-client-request-id",
   ] {
      assert_eq!(received[name], incoming.headers()[name], "{name}");
   }
   assert_eq!(received["authorization"], "Bearer at");
   assert_eq!(received["chatgpt-account-id"], "acct-1");
   assert!(!received.contains_key("cookie"));
}

#[tokio::test]
async fn websocket_relays_upstream_errors_and_close_codes() {
   let app = Router::new().route(
      "/responses",
      get(|upgrade: WebSocketUpgrade| async move {
         upgrade.on_upgrade(|mut socket| async move {
            socket.recv().await.unwrap().unwrap();
            socket
               .send(ServerMessage::Text(
                  json!({
                     "type":"error", "status":429_i32,
                     "error":{"type":"server_error", "code":"slow_down", "message":"busy"},
                     "headers":{"retry-after":"17"}
                  })
                  .to_string()
                  .into(),
               ))
               .await
               .unwrap();
            socket
               .send(ServerMessage::Close(Some(CloseFrame {
                  code: 1013,
                  reason: "retry later".into(),
               })))
               .await
               .unwrap();
         })
      }),
   );
   let (base, db) = serve_upstream(app).await;
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   send(
      &mut socket,
      json!({"type":"response.create", "model":"gpt-6-astra", "input":[]}),
   )
   .await;
   let error = event(&mut socket).await;
   assert_eq!(error["status"], 429_i32);
   assert_eq!(error["error"]["code"], "slow_down");
   assert_eq!(error["headers"]["retry-after"], "17");
   let Some(Ok(Message::Close(Some(frame)))) = timeout(Duration::from_secs(3), socket.next())
      .await
      .unwrap()
   else {
      panic!("expected upstream close")
   };
   assert_eq!(u16::from(frame.code), 1013);
   assert_eq!(frame.reason, "retry later");
   db.flush().await.unwrap();
   assert_eq!(
      db.error_metrics().await.unwrap()[0].kind,
      "upstream_rejected"
   );
}

#[tokio::test]
async fn websocket_ping_and_client_disconnect_close_the_upstream() {
   let (sender, receiver) = oneshot::channel();
   let notify = Arc::new(Mutex::new(Some(sender)));
   let app = Router::new().route(
      "/responses",
      get(move |upgrade: WebSocketUpgrade| {
         let notify = Arc::clone(&notify);
         async move {
            upgrade.on_upgrade(move |mut socket| async move {
               let mut pong = false;
               socket
                  .send(ServerMessage::Ping("upstream-ping".into()))
                  .await
                  .unwrap();
               while let Some(Ok(message)) = socket.recv().await {
                  match message {
                     ServerMessage::Pong(bytes) => {
                        assert_eq!(bytes.as_ref(), b"upstream-ping");
                        pong = true;
                     },
                     ServerMessage::Text(_) => {
                        socket
                           .send(ServerMessage::Text(
                              json!({"type":"response.created", "response":{"id":"r"}})
                                 .to_string()
                                 .into(),
                           ))
                           .await
                           .unwrap();
                     },
                     ServerMessage::Close(frame) => {
                        assert_eq!(frame.unwrap().code, 1000);
                        notify.lock().unwrap().take().unwrap().send(pong).unwrap();
                        break;
                     },
                     _ => {},
                  }
               }
            })
         }
      }),
   );
   let (base, db) = serve_upstream(app).await;
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   socket
      .send(Message::Ping("client-ping".into()))
      .await
      .unwrap();
   let Some(Ok(Message::Pong(bytes))) = timeout(Duration::from_secs(3), socket.next())
      .await
      .unwrap()
   else {
      panic!("expected client pong")
   };
   assert_eq!(bytes.as_ref(), b"client-ping");
   send(
      &mut socket,
      json!({"type":"response.create", "model":"gpt-6-astra", "input":[]}),
   )
   .await;
   assert_eq!(event(&mut socket).await["type"], "response.created");
   socket
      .close(Some(UpstreamCloseFrame {
         code: 1000.into(),
         reason: "finished".into(),
      }))
      .await
      .unwrap();
   assert!(
      timeout(Duration::from_secs(3), receiver)
         .await
         .unwrap()
         .unwrap()
   );
   timeout(Duration::from_secs(3), async {
      loop {
         db.flush().await.unwrap();
         if let Some(error) = db.error_metrics().await.unwrap().first() {
            assert_eq!(error.kind, "client_disconnect");
            break;
         }
         yield_now().await;
      }
   })
   .await
   .unwrap();
}

#[tokio::test]
async fn websocket_rejects_binary_requests_before_dispatch() {
   let (base, db, mut requests, _) = mock().await;
   let (mut socket, _) = connect_async(upgrade_request(&base)).await.unwrap();
   socket
      .send(Message::Binary(
         json!({"type":"response.create", "model":"gpt-6-astra", "input":[]})
            .to_string()
            .into(),
      ))
      .await
      .unwrap();
   let Some(Ok(Message::Close(Some(frame)))) = timeout(Duration::from_secs(3), socket.next())
      .await
      .unwrap()
   else {
      panic!("expected unsupported message close")
   };
   assert_eq!(u16::from(frame.code), 1003);
   requests.try_recv().unwrap_err();
   assert_eq!(
      db.token_meter("sp-test").await.unwrap().unwrap().requests,
      0
   );
}

#[tokio::test]
async fn websocket_handshake_rate_limits_are_returned_before_upgrade() {
   let app = Router::new().route(
      "/responses",
      get(|| async {
         (
            StatusCode::TOO_MANY_REQUESTS,
            [("retry-after", "17")],
            "busy",
         )
      }),
   );
   let (base, _) = serve_upstream(app).await;
   let Err(Error::Http(response)) = connect_async(upgrade_request(&base)).await else {
      panic!("expected rejected handshake")
   };
   assert_eq!(response.status(), 429);
   assert!(response.headers().contains_key("retry-after"));
}
