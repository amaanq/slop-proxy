use super::*;
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::routing::get;
use std::time::Duration;

type Requests = Arc<Mutex<Vec<(HeaderMap, Bytes)>>>;

async fn copilot_upstream(content_type: &'static str, reply: String) -> (String, Requests) {
   let requests = Requests::default();
   let seen = Arc::clone(&requests);
   let upstream = Router::new()
      .route(
         "/chat/completions",
         post(move |headers: HeaderMap, body: Bytes| {
            seen.lock().unwrap().push((headers, body));
            let reply = reply.clone();
            async move { (StatusCode::OK, [("content-type", content_type)], reply) }
         }),
      )
      .route(
         "/models",
         get(move || async move {
            (
               StatusCode::OK,
               [("content-type", "application/json")],
               r#"{"object":"list","data":[{"id":"gpt-5-copilot"}]}"#,
            )
         }),
      );
   let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let upstream_url = format!("http://{}", upstream_listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(upstream_listener, upstream).await.unwrap();
   });
   (upstream_url, requests)
}

async fn proxy(base_url: String) -> (String, Db) {
   let db_path = env::temp_dir().join(format!("slop-copilot-{}.db", uuid::Uuid::new_v4()));
   let db = Db::open(&db_path).unwrap();
   db.create_token("alice", "sp-test", "sp-test")
      .await
      .unwrap();
   // The slot caches the minted access half, so the test seeds one with a
   // live expiry and keeps the grant in the refresh half.
   db.upsert_account(NewAccount {
      provider: Provider::Copilot,
      id: "copilot-user",
      email: Some("copilot-user"),
      label: None,
      plan: None,
      tokens: &TokenSet {
         access_token: "minted-token".into(),
         refresh_token: "github-grant".into(),
         id_token: None,
         expires_at: Some(2_000_000_000),
      },
      auth_mode: AuthMode::OAuth,
   })
   .await
   .unwrap();
   let cfg = Config {
      db_path: db_path.clone(),
      copilot: CopilotConfig {
         base_url,
         ..CopilotConfig::default()
      },
      models: ModelsConfig {
         copilot_patterns: vec!["gpt-5-copilot".into()],
         aliases: [(
            "copilot".into(),
            ModelAlias {
               model: "gpt-5-copilot".into(),
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
      pools,
   }));
   let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let base = format!("http://{}", proxy_listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(proxy_listener, router(state)).await.unwrap();
   });
   (base, db)
}

#[tokio::test]
async fn chat_preserves_payloads_usage_and_provider_scope() {
   let completion = serde_json::json!({
      "id": "chatcmpl-test", "object": "chat.completion", "created": 1_i32,
      "model": "gpt-5-copilot",
      "choices": [{"index": 0_i32, "message": {"role": "assistant", "content": "hi"},
         "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 12_i32, "completion_tokens": 5_i32, "total_tokens": 17_i32}
   })
   .to_string();
   let sse = concat!(
      "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5-copilot\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
      "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5-copilot\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}\n\n",
      "data: [DONE]\n\n",
   );
   for (streaming, content_type, reply) in [
      (false, "application/json", completion),
      (true, "text/event-stream", sse.to_owned()),
   ] {
      let (upstream_url, requests) = copilot_upstream(content_type, reply.clone()).await;
      let (base, db) = proxy(upstream_url).await;
      let client = reqwest::Client::builder()
         .timeout(Duration::from_secs(3))
         .build()
         .unwrap();
      let request = serde_json::json!({
         "model": "copilot", "stream": streaming,
         "messages": [{"role": "user", "content": "hello"}]
      });
      let response = client
         .post(format!("{base}/v1/chat/completions"))
         .bearer_auth("sp-test")
         .json(&request)
         .send()
         .await
         .unwrap();
      assert_eq!(response.status(), 200);
      let text = response.text().await.unwrap();
      assert_eq!(text, reply);
      db.flush().await.unwrap();
      let totals = db.usage_totals(0, i64::MAX).await.unwrap();
      assert_eq!(
         (totals.requests, totals.input_tokens, totals.output_tokens),
         (1, 12, 5)
      );
      {
         let seen = requests.lock().unwrap();
         assert_eq!(seen.len(), 1);
         // The handler rewrote the alias to the upstream model and forced
         // usage on the wire.
         let sent: serde_json::Value = serde_json::from_slice(&seen[0].1).unwrap();
         assert_eq!(sent["model"], "gpt-5-copilot");
         if streaming {
            assert_eq!(sent["stream_options"]["include_usage"], true);
         }
         assert_eq!(
            seen[0].0["openai-intent"], "conversation-panel",
            "upstream headers must carry the client identity"
         );
         assert!(seen[0].0.contains_key("authorization"));
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
      let denied = client
         .post(format!("{base}/v1/chat/completions"))
         .bearer_auth("sp-test")
         .json(&request)
         .send()
         .await
         .unwrap();
      assert_eq!(denied.status(), 403);
      assert_eq!(requests.lock().unwrap().len(), 1);
   }
}
