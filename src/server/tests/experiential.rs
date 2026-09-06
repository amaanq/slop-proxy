use super::*;
use crate::config::ModelAlias;
use axum::body::Bytes;
use axum::http::HeaderMap;
use std::time::Duration;

type Requests = Arc<Mutex<Vec<(HeaderMap, Bytes)>>>;

async fn gateway(
   status: StatusCode,
   content_type: &'static str,
   reply: String,
) -> (String, Db, Requests) {
   let requests = Requests::default();
   let seen = Arc::clone(&requests);
   let upstream = Router::new().route(
      "/v1/messages",
      post(move |headers: HeaderMap, body: Bytes| {
         seen.lock().unwrap().push((headers, body));
         let reply = reply.clone();
         async move {
            (
               status,
               [
                  ("content-type", content_type),
                  ("request-id", "gateway-request"),
                  ("retry-after", "7"),
               ],
               reply,
            )
         }
      }),
   );
   let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let upstream_url = format!("http://{}", upstream_listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(upstream_listener, upstream).await.unwrap();
   });
   let db_path = env::temp_dir().join(format!("slop-gateway-{}.db", uuid::Uuid::new_v4()));
   let db = Db::open(&db_path).unwrap();
   db.create_token("alice", "sp-test", "sp-test")
      .await
      .unwrap();
   db.upsert_account(NewAccount {
      provider: Provider::Experiential,
      id: "gateway-key",
      email: None,
      label: None,
      plan: None,
      tokens: &TokenSet {
         access_token: "upstream-key".into(),
         ..fresh_tokens()
      },
      auth_mode: AuthMode::ApiKey,
   })
   .await
   .unwrap();
   let cfg = Config {
      db_path: db_path.clone(),
      experiential: ExperientialConfig {
         base_url: upstream_url,
      },
      models: ModelsConfig {
         experiential_patterns: vec!["gateway-model".into()],
         aliases: [(
            "gateway".into(),
            ModelAlias {
               model: "gateway-model".into(),
               effort: None,
            },
         )]
         .into(),
         ..ModelsConfig::default()
      },
      ..Config::for_tests()
   };
   let pools = Pools::load(&db, &cfg).await.unwrap();
   let state = AppState(Arc::new(Inner {
      db: db.clone(),
      cfg,
      prices: Prices::new(&db_path, PricingConfig::default().url),
      models: super::super::ModelCache::new(),
      pools,
   }));
   let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let base = format!("http://{}", proxy_listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(proxy_listener, router(state)).await.unwrap();
   });
   (base, db, requests)
}

#[tokio::test]
async fn messages_preserve_payloads_usage_and_provider_scope() {
   let message = serde_json::json!({
      "id": "msg-test", "type": "message", "content": [], "stop_reason": "end_turn",
      "usage": {"input_tokens": 12_i32, "output_tokens": 5_i32}
   })
   .to_string();
   let sse = concat!(
      "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12}}}\n\n",
      "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5},\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
      "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
   );
   for (streaming, content_type, reply) in [
      (false, "application/json", message),
      (true, "text/event-stream", sse.to_owned()),
   ] {
      let (base, db, requests) = gateway(StatusCode::OK, content_type, reply.clone()).await;
      let client = reqwest::Client::builder()
         .timeout(Duration::from_secs(3))
         .build()
         .unwrap();
      let mut request = serde_json::json!({
         "model":"gateway", "stream":streaming, "max_tokens":32_i32,
         "messages":[{"role":"user","content":"hello"}], "unknown_field":{"keep":true}
      });
      let response = client
         .post(format!("{base}/v1/messages"))
         .bearer_auth("sp-test")
         .json(&request)
         .send()
         .await
         .unwrap();
      assert_eq!(response.status(), 200);
      assert_eq!(response.headers()["request-id"], "gateway-request");
      assert_eq!(response.text().await.unwrap(), reply);
      db.flush().await.unwrap();
      let totals = db.usage_totals(0, i64::MAX).await.unwrap();
      assert_eq!(
         (totals.requests, totals.input_tokens, totals.output_tokens),
         (1, 12, 5)
      );
      {
         let seen = requests.lock().unwrap();
         assert_eq!(seen.len(), 1);
         assert_eq!(seen[0].0["authorization"], "Bearer upstream-key");
         request["model"] = "gateway-model".into();
         assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&seen[0].1).unwrap(),
            request
         );
      }
      db.set_token_limits(
         "sp-test",
         &TokenLimits {
            providers: vec![Provider::OpenAi],
            ..TokenLimits::default()
         },
      )
      .await
      .unwrap();
      let denied_response = client
         .post(format!("{base}/v1/messages"))
         .bearer_auth("sp-test")
         .json(&request)
         .send()
         .await
         .unwrap();
      assert_eq!(denied_response.status(), 403);
      assert_eq!(requests.lock().unwrap().len(), 1);
   }
}

#[tokio::test]
async fn quota_exhaustion_keeps_the_retry_header() {
   let (base, _, _) = gateway(
      StatusCode::TOO_MANY_REQUESTS,
      "application/json",
      "quota exhausted".into(),
   )
   .await;
   let response = reqwest::Client::new()
      .post(format!("{base}/v1/messages"))
      .bearer_auth("sp-test")
      .json(&serde_json::json!({"model":"gateway-model","messages":[],"max_tokens":32_i32}))
      .timeout(Duration::from_secs(3))
      .send()
      .await
      .unwrap();
   assert_eq!(response.status(), 429);
   assert!(
      response.headers()["retry-after"]
         .to_str()
         .unwrap()
         .parse::<u64>()
         .unwrap()
         > 0
   );
}
