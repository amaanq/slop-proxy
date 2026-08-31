use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;

use super::{router, AppState};
use crate::codex::client::CodexClient;
use crate::config::{CodexConfig, Config, ModelsConfig};
use crate::db::Db;
use crate::oauth::TokenSet;
use crate::pool::AccountPool;

const MOCK_SSE: &str = concat!(
    "event: response.created\n",
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\"}}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n",
    "event: response.reasoning_summary_part.added\n",
    "data: {\"type\":\"response.reasoning_summary_part.added\"}\n\n",
    "event: response.reasoning_summary_text.delta\n",
    "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"pondering\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"pondering\"}],\"encrypted_content\":\"ENC\"}}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"role\":\"assistant\"}}\n\n",
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello \"}\n\n",
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"world\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello world\"}]}}\n\n",
    "event: response.output_item.added\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n",
    "event: response.function_call_arguments.delta\n",
    "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"city\\\":\"}\n\n",
    "event: response.function_call_arguments.delta\n",
    "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"\\\"Oslo\\\"}\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}}\n\n",
    "event: response.completed\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\",\"usage\":{\"input_tokens\":100,\"output_tokens\":25,\"input_tokens_details\":{\"cached_tokens\":20},\"output_tokens_details\":{\"reasoning_tokens\":5}}}}\n\n",
);

async fn spawn_mock_upstream() -> String {
    let app = Router::new().route(
        "/responses",
        post(|| async { ([("content-type", "text/event-stream")], MOCK_SSE).into_response() }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn spawn_proxy() -> (String, Db) {
    let base_url = spawn_mock_upstream().await;
    let db_path = std::env::temp_dir().join(format!("slop-test-{}.db", uuid::Uuid::new_v4()));
    let db = Db::open(&db_path).await.unwrap();
    db.create_token("alice", "sp-test", "sp-test")
        .await
        .unwrap();
    db.upsert_account(
        "acct-1",
        Some("test@example.com"),
        None,
        Some("plus"),
        &TokenSet {
            access_token: "at".into(),
            refresh_token: "rt".into(),
            id_token: None,
            expires_at: Some(chrono::Utc::now().timestamp() + 3600),
        },
    )
    .await
    .unwrap();

    let cfg = Config {
        db_path,
        bind: String::new(),
        codex: CodexConfig {
            base_url,
            ..CodexConfig::default()
        },
        models: ModelsConfig::default(),
    };
    let pool = AccountPool::load(db.clone()).await.unwrap();
    let state = AppState {
        db: db.clone(),
        pool: Arc::new(pool),
        client: Arc::new(CodexClient::new(cfg.codex.clone())),
        cfg: Arc::new(cfg),
        models: Arc::new(super::ModelCache::new()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router(state)).await.unwrap();
    });
    (format!("http://{addr}"), db)
}

#[tokio::test]
async fn anthropic_streaming_end_to_end() {
    let (base, db) = spawn_proxy().await;
    let body = serde_json::json!({
        "model": "claude-sonnet-4",
        "max_tokens": 1000,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
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
        let found = text[pos..]
            .find(needle)
            .unwrap_or_else(|| panic!("missing {needle:?} after byte {pos} in:\n{text}"));
        pos += found;
    }
    assert!(text.contains("\"input_tokens\":80"));
    assert!(text.contains("\"cache_read_input_tokens\":20"));

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let totals = db.usage_totals(0, i64::MAX).await.unwrap();
    assert_eq!(totals.requests, 1);
    assert_eq!(totals.input_tokens, 100);
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
    let v: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(v["object"], "chat.completion");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "Hello world");
    assert_eq!(msg["reasoning_content"], "pondering");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(
        msg["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Oslo\"}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["usage"]["prompt_tokens"], 100);
    assert_eq!(v["usage"]["completion_tokens"], 25);

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let by_user = db
        .usage_by(crate::db::usage::UsageDim::User, 0, i64::MAX)
        .await
        .unwrap();
    assert_eq!(by_user[0].key, "alice");
    assert_eq!(by_user[0].input_tokens, 100);
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
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "sp-test")
        .json(&body)
        .send()
        .await
        .unwrap();
    let v: serde_json::Value = resp.json().await.unwrap();
    let types: Vec<&str> = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    assert_eq!(types, vec!["text", "tool_use"]);
    assert_eq!(v["stop_reason"], "tool_use");
}
