use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse as _;
use axum::routing::post;
use tokio::net::TcpListener;

use super::{AppState, Inner, metrics, router};
use crate::clock;
use crate::codex::client::CodexClient;
use crate::config::{
   AnthropicConfig, CodexConfig, Config, ExperientialConfig, GeminiConfig, GlmConfig, ModelsConfig,
   PricingConfig, ZenConfig,
};
use crate::db::Db;
use crate::db::accounts::NewAccount;
use crate::db::tokens::TokenLimits;
use crate::db::usage::UsageDim;
use crate::oauth::TokenSet;
use crate::pool::Pools;
use crate::pool::codex::CodexPool;
use crate::pricing::Prices;
use crate::provider::{AuthMode, Provider};

const MOCK_SSE: &str = concat!(
   "event: response.created\n",
   "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\"}}\n\n",
   "event: response.output_item.added\n",
   "data: {\"output_index\":0,\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
   "event: response.reasoning_summary_part.added\n",
   "data: {\"output_index\":0,\"type\":\"response.reasoning_summary_part.added\"}\n\n",
   "event: response.reasoning_summary_text.delta\n",
   "data: {\"output_index\":0,\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"pondering\"}\n\n",
   "event: response.output_item.done\n",
   "data: {\"output_index\":0,\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"pondering\"}],\"encrypted_content\":\"ENC\"}}\n\n",
   "event: response.output_item.added\n",
   "data: {\"output_index\":1,\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"role\":\"assistant\"}}\n\n",
   "event: response.output_text.delta\n",
   "data: {\"output_index\":1,\"type\":\"response.output_text.delta\",\"delta\":\"Hello \"}\n\n",
   "event: response.output_text.delta\n",
   "data: {\"output_index\":1,\"type\":\"response.output_text.delta\",\"delta\":\"world\"}\n\n",
   "event: response.output_item.done\n",
   "data: {\"output_index\":1,\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello world\"}]}}\n\n",
   "event: response.output_item.added\n",
   "data: {\"output_index\":2,\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n",
   "event: response.function_call_arguments.delta\n",
   "data: {\"output_index\":2,\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"city\\\":\"}\n\n",
   "event: response.function_call_arguments.delta\n",
   "data: {\"output_index\":2,\"type\":\"response.function_call_arguments.delta\",\"delta\":\"\\\"Oslo\\\"}\"}\n\n",
   "event: response.output_item.done\n",
   "data: {\"output_index\":2,\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}}\n\n",
   "event: response.completed\n",
   "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\",\"usage\":{\"input_tokens\":100,\"output_tokens\":25,\"input_tokens_details\":{\"cached_tokens\":20},\"output_tokens_details\":{\"reasoning_tokens\":5}}}}\n\n",
);

async fn spawn_mock_upstream(sse: String) -> String {
   let app = Router::new().route(
      "/responses",
      post(move || {
         let sse = sse.clone();
         async move { ([("content-type", "text/event-stream")], sse).into_response() }
      }),
   );
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let addr = listener.local_addr().unwrap();
   tokio::spawn(async move {
      axum::serve(listener, app).await.unwrap();
   });
   format!("http://{addr}")
}

fn fresh_tokens() -> TokenSet {
   TokenSet {
      access_token: "at".into(),
      refresh_token: "rt".into(),
      id_token: None,
      expires_at: Some(clock::unix_now() + 3600),
   }
}

async fn spawn_proxy_with(models: ModelsConfig, anthropic_base: Option<String>) -> (String, Db) {
   spawn_proxy_with_response(models, anthropic_base, MOCK_SSE.into()).await
}

async fn spawn_proxy_with_response(
   models: ModelsConfig,
   anthropic_base: Option<String>,
   sse: String,
) -> (String, Db) {
   let base_url = spawn_mock_upstream(sse).await;
   let db_path = env::temp_dir().join(format!("slop-test-{}.db", uuid::Uuid::new_v4()));
   let db = Db::open(&db_path).unwrap();
   db.create_token("alice", "sp-test", "sp-test")
      .await
      .unwrap();
   db.upsert_account(NewAccount {
      provider: Provider::OpenAi,
      id: "acct-1",
      email: Some("test@example.com"),
      label: None,
      plan: Some("plus"),
      tokens: &fresh_tokens(),
      auth_mode: AuthMode::OAuth,
   })
   .await
   .unwrap();
   if anthropic_base.is_some() {
      db.upsert_account(NewAccount {
         provider: Provider::Anthropic,
         id: "acct-a1",
         email: None,
         label: None,
         plan: None,
         tokens: &fresh_tokens(),
         auth_mode: AuthMode::OAuth,
      })
      .await
      .unwrap();
   }

   let cfg_db_path = db_path.clone();
   let cfg = Config {
      db_path,
      bind: String::new(),
      metrics_bind: None,
      codex: CodexConfig {
         base_url,
         ..CodexConfig::default()
      },
      anthropic: AnthropicConfig {
         base_url: anthropic_base.unwrap_or_default(),
         ..AnthropicConfig::default()
      },
      gemini: GeminiConfig::default(),
      zen: ZenConfig::default(),
      glm: GlmConfig::default(),
      experiential: ExperientialConfig::default(),
      pricing: PricingConfig::default(),
      models,
   };
   let pools = Pools::load(&db, &cfg).await.unwrap();
   let state = AppState(Arc::new(Inner {
      db: db.clone(),
      cfg,
      prices: Prices::new(&cfg_db_path, PricingConfig::default().url),
      models: super::ModelCache::new(),
      pools,
   }));
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let addr = listener.local_addr().unwrap();
   tokio::spawn(async move {
      axum::serve(listener, router(state)).await.unwrap();
   });
   (format!("http://{addr}"), db)
}

/// Translation tests use claude-* model names against the codex mock, so
/// relay routing is switched off for them.
async fn spawn_proxy() -> (String, Db) {
   let models = ModelsConfig {
      anthropic_patterns: Vec::new(),
      ..ModelsConfig::default()
   };
   spawn_proxy_with(models, None).await
}

#[tokio::test]
async fn anthropic_streaming_end_to_end() {
   let (base, db) = spawn_proxy().await;
   let body = serde_json::json!({
       "model": "claude-sonnet-4",
       "max_tokens": 1000_u64,
       "stream": true,
       "thinking": {"type": "enabled", "budget_tokens": 8000_u64},
       "messages": [{"role": "user", "content": "hi"}],
       "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}]
   });
   let resp = reqwest::Client::new()
      .post(format!("{base}/v1/messages"))
      .header("x-api-key", "sp-test")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(resp.status(), 200);
   let text = resp.text().await.unwrap();

   let order = [
      "event: message_start",
      "\"type\":\"thinking\"",
      "\"thinking\":\"pondering\"",
      "signature_delta",
      "\"type\":\"text\"",
      "\"text\":\"Hello \"",
      "\"name\":\"get_weather\"",
      "\"type\":\"tool_use\"",
      "input_json_delta",
      "\"stop_reason\":\"tool_use\"",
      "event: message_stop",
   ];
   let mut pos = 0;
   for needle in order {
      let haystack = text.get(pos..).unwrap_or("");
      let found = haystack.find(needle);
      assert!(
         found.is_some(),
         "missing {needle:?} after byte {pos} in:\n{text}"
      );
      pos += found.unwrap();
   }
   assert!(text.contains("\"input_tokens\":80"));
   assert!(text.contains("\"cache_read_input_tokens\":20"));

   db.flush().await.unwrap();
   let totals = db.usage_totals(0, i64::MAX).await.unwrap();
   assert_eq!(totals.requests, 1);
   // 100 prompt tokens of which 20 were cached, so 80 are freshly billed.
   assert_eq!(totals.input_tokens, 80);
   assert_eq!(totals.cache_read_tokens, 20);
   assert_eq!(totals.output_tokens, 25);
   assert_eq!(totals.reasoning_tokens, 5);
}

#[tokio::test]
async fn openai_non_streaming_end_to_end() {
   let (base, db) = spawn_proxy().await;
   let body = serde_json::json!({
       "model": "gpt-5-codex",
       "messages": [
           {"role": "system", "content": "be brief"},
           {"role": "user", "content": "hi"}
       ],
       "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
   });
   let resp = reqwest::Client::new()
      .post(format!("{base}/v1/chat/completions"))
      .bearer_auth("sp-test")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(resp.status(), 200);
   let value = resp.json::<serde_json::Value>().await.unwrap();

   assert_eq!(value["object"], "chat.completion");
   let msg = &value["choices"][0]["message"];
   assert_eq!(msg["content"], "Hello world");
   assert_eq!(msg["reasoning_content"], "pondering");
   assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
   assert_eq!(
      msg["tool_calls"][0]["function"]["arguments"],
      "{\"city\":\"Oslo\"}"
   );
   assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
   assert_eq!(value["usage"]["prompt_tokens"], 100_u64);
   assert_eq!(value["usage"]["completion_tokens"], 25_u64);

   db.flush().await.unwrap();
   let by_user = db.usage_by(UsageDim::User, 0, i64::MAX).await.unwrap();
   assert_eq!(by_user[0].key, "alice");
   assert_eq!(by_user[0].input_tokens, 80);
   let metrics = db.insight_metrics().await.unwrap();
   assert_eq!(
      metrics[0].response_bytes,
      serde_json::to_vec(&value).unwrap().len() as i64
   );
   assert_eq!(metrics[0].ttft_samples, 1);
}

#[tokio::test]
async fn openai_streaming_end_to_end() {
   let (base, _db) = spawn_proxy().await;
   let body = serde_json::json!({
       "model": "gpt-5-codex",
       "stream": true,
       "stream_options": {"include_usage": true},
       "messages": [{"role": "user", "content": "hi"}]
   });
   let resp = reqwest::Client::new()
      .post(format!("{base}/v1/chat/completions"))
      .bearer_auth("sp-test")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(resp.status(), 200);
   let text = resp.text().await.unwrap();

   assert!(text.contains("\"role\":\"assistant\""));
   assert!(text.contains("\"content\":\"Hello \""));
   assert!(text.contains("\"reasoning_content\":\"pondering\""));
   assert!(text.contains("\"finish_reason\":\"tool_calls\""));
   assert!(text.contains("\"prompt_tokens\":100"));
   assert!(text.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn thinking_disabled_hides_thinking_blocks() {
   let (base, _db) = spawn_proxy().await;
   let body = serde_json::json!({
       "model": "claude-sonnet-4",
       "max_tokens": 1000_u64,
       "messages": [{"role": "user", "content": "hi"}]
   });
   let resp = reqwest::Client::new()
      .post(format!("{base}/v1/messages"))
      .header("x-api-key", "sp-test")
      .json(&body)
      .send()
      .await
      .unwrap();
   let value = resp.json::<serde_json::Value>().await.unwrap();
   let types = value["content"]
      .as_array()
      .unwrap()
      .iter()
      .map(|block| block["type"].as_str().unwrap())
      .collect::<Vec<&str>>();
   assert_eq!(types, vec!["text", "tool_use"]);
   assert_eq!(value["stop_reason"], "tool_use");
}

const MOCK_ANTHROPIC_SSE: &str = concat!(
   "event: message_start\n",
   "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":40,\"output_tokens\":1}}}\n\n",
   "event: content_block_delta\n",
   "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
   "event: message_delta\n",
   "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
   "event: message_stop\n",
   "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn anthropic_relay_passthrough() {
   use axum::http::HeaderMap;

   let seen = Arc::new(Mutex::new(HeaderMap::new()));
   let seen2 = Arc::clone(&seen);
   let app = Router::new().route(
      "/v1/messages",
      post(move |headers: HeaderMap| async move {
         *seen2.lock().unwrap() = headers;
         ([("content-type", "text/event-stream")], MOCK_ANTHROPIC_SSE).into_response()
      }),
   );
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let upstream = format!("http://{}", listener.local_addr().unwrap());
   tokio::spawn(async move {
      axum::serve(listener, app).await.unwrap();
   });

   let (base, db) = spawn_proxy_with(ModelsConfig::default(), Some(upstream)).await;
   let body = serde_json::json!({
       "model": "claude-opus-5",
       "max_tokens": 100_u64,
       "stream": true,
       "metadata": {"user_id": "session-1"},
       "messages": [{"role": "user", "content": "hi"}]
   });
   let resp = reqwest::Client::new()
      .post(format!("{base}/v1/messages"))
      .header("x-api-key", "sp-test")
      .header(
         "anthropic-beta",
         "claude-code-20250219,context-1m-2025-08-07",
      )
      .header("user-agent", "claude-cli/2.1.252 (external, cli)")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(resp.status(), 200);
   let text = resp.text().await.unwrap();
   assert_eq!(text, MOCK_ANTHROPIC_SSE);

   let upstream_headers = seen.lock().unwrap().clone();
   let beta = upstream_headers["anthropic-beta"].to_str().unwrap();
   assert!(beta.contains("oauth-2025-04-20"));
   assert!(beta.contains("context-1m-2025-08-07"));
   assert_eq!(
      upstream_headers["authorization"].to_str().unwrap(),
      "Bearer at"
   );

   db.flush().await.unwrap();
   let totals = db.usage_totals(0, i64::MAX).await.unwrap();
   assert_eq!(totals.requests, 1);
   assert_eq!(totals.input_tokens, 50);
   assert_eq!(totals.output_tokens, 7);
}

#[tokio::test]
async fn metrics_render_accounts_and_usage() {
   let (base, db) = spawn_proxy().await;
   let body = serde_json::json!({
       "model": "gpt-5-codex",
       "messages": [{"role": "user", "content": "hi"}]
   });
   reqwest::Client::new()
      .post(format!("{base}/v1/chat/completions"))
      .bearer_auth("sp-test")
      .json(&body)
      .send()
      .await
      .unwrap();
   db.flush().await.unwrap();

   let cfg = Config {
      db_path: PathBuf::new(),
      bind: String::new(),
      metrics_bind: None,
      codex: CodexConfig::default(),
      anthropic: AnthropicConfig::default(),
      gemini: GeminiConfig::default(),
      zen: ZenConfig::default(),
      glm: GlmConfig::default(),
      experiential: ExperientialConfig::default(),
      pricing: PricingConfig::default(),
      models: ModelsConfig::default(),
   };
   let pools = Pools::load(&db, &cfg).await.unwrap();
   let state = AppState(Arc::new(Inner {
      db: db.clone(),
      cfg,
      prices: Prices::new(&PathBuf::new(), PricingConfig::default().url),
      models: super::ModelCache::new(),
      pools,
   }));
   let resp = metrics::metrics(State(state)).await;
   let bytes = body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
   let text = String::from_utf8(bytes.to_vec()).unwrap();

   assert!(
      text.contains("slop_account_status{provider=\"openai\",account=\"test@example.com\"} 0")
   );
   assert!(text.contains("slop_requests_total{user=\"alice\",account=\"test@example.com\",provider=\"openai\",requested_model=\"gpt-5-codex\",model=\"gpt-5-codex\",effort=\"medium\",service_tier=\"none\",dialect=\"chat\"} 1"));
   assert!(text.contains("kind=\"input\"} 80"));
   assert!(text.contains("kind=\"cache_read\"} 20"));
}

#[tokio::test]
async fn pool_reload_picks_up_new_logins() {
   let db_path = env::temp_dir().join(format!("slop-reload-{}.db", uuid::Uuid::new_v4()));
   let db = Db::open(&db_path).unwrap();
   let pool = CodexPool::load(db.clone(), CodexClient::new(CodexConfig::default()))
      .await
      .unwrap();
   assert_eq!(pool.len().await, 0);

   db.upsert_account(NewAccount {
      provider: Provider::OpenAi,
      id: "acct-r1",
      email: None,
      label: None,
      plan: None,
      tokens: &fresh_tokens(),
      auth_mode: AuthMode::OAuth,
   })
   .await
   .unwrap();
   pool.reload().await.unwrap();
   assert_eq!(pool.len().await, 1);
}

#[tokio::test]
async fn token_request_limit_enforced() {
   let (base, db) = spawn_proxy().await;
   let id = db.create_token("marc", "sp-marc", "sp-marc").await.unwrap();
   db.set_token_limits(
      &id.to_string(),
      &TokenLimits {
         providers: Vec::new(),
         requests: Some(1),
         tokens: None,
         window_seconds: 3600,
         slowdown_ms: 0,
         prefer_trusted: false,
         pinned_account: None,
      },
   )
   .await
   .unwrap();

   let body = serde_json::json!({
       "model": "gpt-5-codex",
       "messages": [{"role": "user", "content": "hi"}]
   });
   let client = reqwest::Client::new();
   let first = client
      .post(format!("{base}/v1/chat/completions"))
      .bearer_auth("sp-marc")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(first.status(), 200);
   assert_eq!(
      first
         .headers()
         .get("x-ratelimit-remaining-requests")
         .unwrap(),
      "0"
   );

   let second = client
      .post(format!("{base}/v1/chat/completions"))
      .bearer_auth("sp-marc")
      .json(&body)
      .send()
      .await
      .unwrap();
   assert_eq!(second.status(), 429);
   assert!(second.headers().contains_key("retry-after"));
}

const GEMINI_REJECTION: &str = "{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"contents is not specified\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}\n";

async fn spawn_proxy_with_gemini() -> (String, Db) {
   spawn_proxy_with_gemini_reply(
      StatusCode::BAD_REQUEST,
      "application/json",
      GEMINI_REJECTION.into(),
   )
   .await
}

async fn spawn_proxy_with_gemini_reply(
   status: StatusCode,
   content_type: &'static str,
   body: String,
) -> (String, Db) {
   let app = Router::new().fallback(move || {
      let body = body.clone();
      async move { (status, [("content-type", content_type)], body).into_response() }
   });
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let addr = listener.local_addr().unwrap();
   tokio::spawn(async move {
      axum::serve(listener, app).await.unwrap();
   });

   let db_path = env::temp_dir().join(format!("slop-test-{}.db", uuid::Uuid::new_v4()));
   let db = Db::open(&db_path).unwrap();
   db.create_token("alice", "sp-test", "sp-test")
      .await
      .unwrap();
   db.upsert_account(NewAccount {
      provider: Provider::Gemini,
      id: "gem-1",
      email: None,
      label: Some("gem1"),
      plan: None,
      tokens: &fresh_tokens(),
      auth_mode: AuthMode::ApiKey,
   })
   .await
   .unwrap();

   let cfg_db_path = db_path.clone();
   let cfg = Config {
      db_path,
      bind: String::new(),
      metrics_bind: None,
      codex: CodexConfig::default(),
      anthropic: AnthropicConfig::default(),
      gemini: GeminiConfig {
         base_url: format!("http://{addr}"),
         ..GeminiConfig::default()
      },
      zen: ZenConfig::default(),
      glm: GlmConfig::default(),
      experiential: ExperientialConfig::default(),
      pricing: PricingConfig::default(),
      models: ModelsConfig::default(),
   };
   let pools = Pools::load(&db, &cfg).await.unwrap();
   let state = AppState(Arc::new(Inner {
      db: db.clone(),
      cfg,
      prices: Prices::new(&cfg_db_path, PricingConfig::default().url),
      models: super::ModelCache::new(),
      pools,
   }));
   let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let proxy_addr = proxy_listener.local_addr().unwrap();
   tokio::spawn(async move {
      axum::serve(proxy_listener, router(state)).await.unwrap();
   });
   (format!("http://{proxy_addr}"), db)
}

#[tokio::test]
async fn a_rejected_gemini_request_keeps_googles_reason() {
   let (base, db) = spawn_proxy_with_gemini().await;
   let resp = reqwest::Client::new()
      .post(format!(
         "{base}/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
      ))
      .header("x-goog-api-key", "sp-test")
      .json(&serde_json::json!({"contents": []}))
      .send()
      .await
      .unwrap();

   assert_eq!(resp.status(), 400);
   assert_eq!(resp.text().await.unwrap(), GEMINI_REJECTION);

   db.flush().await.unwrap();
   let kinds = db.error_metrics().await.unwrap();
   assert_eq!(kinds[0].kind, "upstream_rejected");
   assert_eq!(kinds[0].provider, "gemini");

   let logged = db
      .call(|conn| {
         Ok(
            conn.query_row("SELECT response_bytes FROM usage_log", [], |row| {
               row.get::<_, i64>(0)
            })?,
         )
      })
      .await
      .unwrap();
   assert_eq!(logged as usize, GEMINI_REJECTION.len());
}

#[tokio::test]
async fn responses_preserve_terminal_status_and_bill_partial_usage() {
   for kind in ["completed", "incomplete", "failed"] {
      let terminal = serde_json::json!({
         "type": format!("response.{kind}"),
         "response": {"id":"r", "status":kind, "output":[], "usage":{"input_tokens":12,"output_tokens":5}}
      });
      let source = format!("event: response.{kind}\ndata: {terminal}\n\n");
      let (base, db) = spawn_proxy_with_response(ModelsConfig::default(), None, source).await;
      for streaming in [false, true] {
         let response = reqwest::Client::new()
            .post(format!("{base}/v1/responses"))
            .bearer_auth("sp-test")
            .json(&serde_json::json!({"model":"gpt-test","stream":streaming,"input":[]}))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .unwrap();
         assert_eq!(response.status(), 200);
         if streaming {
            assert!(
               response
                  .text()
                  .await
                  .unwrap()
                  .contains(&format!("response.{kind}"))
            );
         } else {
            let response: serde_json::Value = response.json().await.unwrap();
            assert_eq!(response["status"], kind);
            assert_eq!(response["usage"]["output_tokens"], 5);
         }
      }
      db.flush().await.unwrap();
      let rows = db.usage_metrics().await.unwrap();
      assert_eq!(rows[0].requests, 2);
      assert_eq!(rows[0].input_tokens, 24);
      assert_eq!(rows[0].output_tokens, 10);
      assert_eq!(rows[0].errors, if kind == "failed" { 2 } else { 0 });
   }
}

#[tokio::test]
async fn responses_eof_is_logged_as_failure() {
   let (base, db) = spawn_proxy_with_response(ModelsConfig::default(), None,
      "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n".into()).await;
   let response = reqwest::Client::new()
      .post(format!("{base}/v1/responses"))
      .bearer_auth("sp-test")
      .json(&serde_json::json!({"model":"gpt-test","input":[]}))
      .timeout(std::time::Duration::from_secs(2))
      .send()
      .await
      .unwrap();
   assert_eq!(response.status(), 502);
   response.bytes().await.unwrap();
   db.flush().await.unwrap();
   let rows = db.error_metrics().await.unwrap();
   assert_eq!(rows[0].kind, "upstream_eof");
}

#[tokio::test]
async fn aliases_are_resolved_before_provider_scope_on_every_endpoint() {
   let mut models = ModelsConfig::default();
   models.aliases.insert(
      "shortcut".into(),
      crate::config::ModelAlias {
         model: "gemini-test".into(),
         effort: Some("low".into()),
      },
   );
   let (base, db) = spawn_proxy_with(models, None).await;
   db.set_token_limits(
      "sp-test",
      &TokenLimits {
         providers: vec![Provider::OpenAi],
         ..TokenLimits::default()
      },
   )
   .await
   .unwrap();
   for endpoint in [
      "chat/completions",
      "messages",
      "messages/count_tokens",
      "responses",
   ] {
      let response = reqwest::Client::new()
         .post(format!("{base}/v1/{endpoint}"))
         .bearer_auth("sp-test")
         .json(
            &serde_json::json!({"model":"shortcut:high","input":[],"messages":[],"max_tokens":1}),
         )
         .timeout(std::time::Duration::from_secs(2))
         .send()
         .await
         .unwrap();
      assert_eq!(response.status(), 403, "{endpoint}");
   }
}

#[tokio::test]
async fn bridged_responses_preserve_status_usage_and_output_order() {
   for (finish, status) in [
      ("stop", "completed"),
      ("length", "incomplete"),
      ("error", "failed"),
   ] {
      let first = serde_json::json!({"id":"r","choices":[{"delta":{
         "content":"partial",
         "tool_calls":[{"index":0,"id":"upstream","function":{"name":"read","arguments":"{}"}}]
      }}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}});
      let last = if finish == "error" {
         serde_json::json!({"error":{"message":"boom","code":"UPSTREAM_FAILED"}})
      } else {
         serde_json::json!({"choices":[{"finish_reason":finish}]})
      };
      let (base, db) = spawn_proxy_with_gemini_reply(
         StatusCode::OK,
         "text/event-stream",
         format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n"),
      )
      .await;
      let response = reqwest::Client::new()
         .post(format!("{base}/v1/responses"))
         .bearer_auth("sp-test")
         .json(&serde_json::json!({"model":"gemini-test","input":"hello"}))
         .timeout(std::time::Duration::from_secs(2))
         .send()
         .await
         .unwrap();
      assert_eq!(response.status(), 200);
      let value: serde_json::Value = response.json().await.unwrap();
      assert_eq!(value["status"], status);
      assert_eq!(value["id"], "r");
      assert_eq!(value["usage"]["output_tokens"], 5);
      if status == "failed" {
         assert_eq!(value["error"]["message"], "boom");
      } else {
         assert_eq!(value["output"][0]["type"], "message");
         assert_eq!(value["output"][0]["content"][0]["text"], "partial");
         assert_eq!(value["output"][1]["name"], "read");
      }
      db.flush().await.unwrap();
      let rows = db.usage_metrics().await.unwrap();
      assert_eq!(rows[0].input_tokens, 12);
      assert_eq!(rows[0].output_tokens, 5);
      assert_eq!(rows[0].errors, i64::from(status == "failed"));
   }
}

mod experiential;
mod rate_limits;
