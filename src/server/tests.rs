use std::sync::Arc;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::post;

use super::{AppState, router};
use crate::anthropic::client::AnthropicClient;
use crate::codex::client::CodexClient;
use crate::config::{AnthropicConfig, CodexConfig, Config, ModelsConfig};
use crate::db::Db;
use crate::oauth::TokenSet;
use crate::pool::anthropic::AnthropicPool;
use crate::pool::codex::CodexPool;
use crate::provider::Provider;

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

fn fresh_tokens() -> TokenSet {
    TokenSet {
        access_token: "at".into(),
        refresh_token: "rt".into(),
        id_token: None,
        expires_at: Some(crate::clock::unix_now() + 3600),
    }
}

async fn spawn_proxy_with(models: ModelsConfig, anthropic_base: Option<String>) -> (String, Db) {
    let base_url = spawn_mock_upstream().await;
    let db_path = std::env::temp_dir().join(format!("slop-test-{}.db", uuid::Uuid::new_v4()));
    let db = Db::open(&db_path).await.unwrap();
    db.create_token("alice", "sp-test", "sp-test")
        .await
        .unwrap();
    db.upsert_account(
        Provider::Codex,
        "acct-1",
        Some("test@example.com"),
        None,
        Some("plus"),
        &fresh_tokens(),
    )
    .await
    .unwrap();
    if anthropic_base.is_some() {
        db.upsert_account(
            Provider::Anthropic,
            "acct-a1",
            None,
            None,
            None,
            &fresh_tokens(),
        )
        .await
        .unwrap();
    }

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
        models,
    };
    let codex = CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone()))
        .await
        .unwrap();
    let anthropic = AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone()))
        .await
        .unwrap();
    let state = AppState {
        db: db.clone(),
        codex: Arc::new(codex),
        anthropic: Arc::new(anthropic),
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
    let v = resp.json::<serde_json::Value>().await.unwrap();

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
    assert_eq!(by_user[0].input_tokens, 80);
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
    let v = resp.json::<serde_json::Value>().await.unwrap();
    let types = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect::<Vec<&str>>();
    assert_eq!(types, vec!["text", "tool_use"]);
    assert_eq!(v["stop_reason"], "tool_use");
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

    let seen = Arc::new(std::sync::Mutex::new(HeaderMap::new()));
    let seen2 = seen.clone();
    let app = Router::new().route(
        "/v1/messages",
        post(move |headers: HeaderMap| async move {
            *seen2.lock().unwrap() = headers;
            ([("content-type", "text/event-stream")], MOCK_ANTHROPIC_SSE).into_response()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let (base, db) = spawn_proxy_with(ModelsConfig::default(), Some(upstream)).await;
    let body = serde_json::json!({
        "model": "claude-opus-5",
        "max_tokens": 100,
        "stream": true,
        "metadata": {"user_id": "session-1"},
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "sp-test")
        .header("anthropic-beta", "context-1m-2025-08-07")
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

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
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
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let cfg = Config {
        db_path: std::path::PathBuf::new(),
        bind: String::new(),
        metrics_bind: None,
        codex: CodexConfig::default(),
        anthropic: AnthropicConfig::default(),
        models: ModelsConfig::default(),
    };
    let state = AppState {
        db: db.clone(),
        codex: Arc::new(
            CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone()))
                .await
                .unwrap(),
        ),
        anthropic: Arc::new(
            AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone()))
                .await
                .unwrap(),
        ),
        cfg: Arc::new(cfg),
        models: Arc::new(super::ModelCache::new()),
    };
    let resp = super::metrics::metrics(axum::extract::State(state)).await;
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    assert!(
        text.contains("slop_account_status{provider=\"codex\",account=\"test@example.com\"} 0")
    );
    assert!(text.contains("slop_requests_total{user=\"alice\",account=\"test@example.com\",provider=\"codex\",requested_model=\"gpt-5-codex\",model=\"gpt-5-codex\",effort=\"medium\",dialect=\"openai\"} 1"));
    assert!(text.contains("kind=\"input\"} 80"));
    assert!(text.contains("kind=\"cache_read\"} 20"));
}

#[tokio::test]
async fn pool_reload_picks_up_new_logins() {
    let db_path = std::env::temp_dir().join(format!("slop-reload-{}.db", uuid::Uuid::new_v4()));
    let db = Db::open(&db_path).await.unwrap();
    let pool = CodexPool::load(db.clone(), CodexClient::new(CodexConfig::default()))
        .await
        .unwrap();
    assert_eq!(pool.len().await, 0);

    db.upsert_account(
        Provider::Codex,
        "acct-r1",
        None,
        None,
        None,
        &fresh_tokens(),
    )
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
        &crate::db::tokens::TokenLimits {
            requests: Some(1),
            tokens: None,
            window_seconds: 3600,
            slowdown_ms: 0,
            prefer_trusted: false,
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
