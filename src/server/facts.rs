use serde_json::Value;

/// Structural facts about a request, read the same way across the three body
/// shapes the proxy accepts. Deliberately names no message text, no tool
/// arguments and no paths, so the log stays metadata rather than a transcript.
#[derive(Debug, Default, Clone, Copy)]
pub struct RequestFacts {
    pub turn_index: i64,
    pub tools_declared: i64,
    pub thinking_budget: i64,
    pub image_count: i64,
    pub request_bytes: i64,
}

impl RequestFacts {
    pub fn extract(body: &Value, headers: &axum::http::HeaderMap) -> Self {
        let turns = messages(body);
        Self {
            request_bytes: request_bytes(headers),
            turn_index: turns.map_or(0, |t| t.len() as i64),
            tools_declared: tools_in(body)
                + turns.map_or(0, |t| t.iter().map(tools_in).sum::<i64>()),
            thinking_budget: thinking_budget(body),
            image_count: turns.map_or(0, |t| t.iter().map(image_parts).sum()),
        }
    }
}

/// The decompressor rewrites `content-length` to the decoded size, so this
/// measures what the handler parsed rather than what arrived on the wire.
fn request_bytes(headers: &axum::http::HeaderMap) -> i64 {
    headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Codex carries its catalog as an `AdditionalTools` item inside `input`
/// rather than a top-level array (protocol/src/models.rs:953), so a request
/// has to be searched in both places.
fn tools_in(node: &Value) -> i64 {
    node.get("tools").and_then(Value::as_array).map_or(0, |t| {
        t.iter()
            .map(|d| {
                d.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .map_or(1, Vec::len) as i64
            })
            .sum()
    })
}

/// `messages` is Anthropic and OpenAI chat, `input` the Responses API,
/// `contents` Gemini.
fn messages(body: &Value) -> Option<&Vec<Value>> {
    ["messages", "input", "contents"]
        .iter()
        .find_map(|k| body.get(k).and_then(Value::as_array))
}

/// Anthropic takes a token count, Gemini nests the same idea two levels down,
/// and the OpenAI shapes spend an effort level instead, which `effort`
/// already carries.
fn thinking_budget(body: &Value) -> i64 {
    let anthropic = body.pointer("/thinking/budget_tokens");
    let gemini = body.pointer("/generationConfig/thinkingConfig/thinkingBudget");
    anthropic.or(gemini).and_then(Value::as_i64).unwrap_or(0)
}

fn image_parts(message: &Value) -> i64 {
    let parts = ["content", "parts"]
        .iter()
        .find_map(|k| message.get(k).and_then(Value::as_array));
    parts.map_or(0, |parts| {
        parts
            .iter()
            .filter(|p| {
                let tagged = p
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| matches!(t, "image" | "image_url" | "input_image"));
                tagged || p.get("inline_data").is_some() || p.get("inlineData").is_some()
            })
            .count() as i64
    })
}

#[cfg(test)]
mod tests {
    use super::RequestFacts;
    use axum::http::HeaderMap;
    use serde_json::json;

    fn facts(body: serde_json::Value) -> RequestFacts {
        RequestFacts::extract(&body, &HeaderMap::new())
    }

    #[test]
    fn anthropic() {
        let f = facts(json!({
            "messages": [
                {"role": "user", "content": [{"type": "text"}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text"}]},
            ],
            "tools": [{"name": "Read"}, {"name": "Bash"}],
            "thinking": {"budget_tokens": 10000},
        }));
        assert_eq!((f.turn_index, f.tools_declared), (2, 2));
        assert_eq!((f.thinking_budget, f.image_count), (10000, 1));
    }

    #[test]
    fn openai_responses() {
        let f = facts(json!({
            "input": [{"role": "user", "content": [{"type": "input_image"}]}],
            "tools": [{"name": "shell"}],
        }));
        assert_eq!((f.turn_index, f.tools_declared, f.image_count), (1, 1, 1));
    }

    #[test]
    fn gemini_counts_declarations_not_wrappers() {
        let f = facts(json!({
            "contents": [{"role": "user", "parts": [{"inlineData": {}}]}],
            "tools": [{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 512}},
        }));
        assert_eq!((f.tools_declared, f.thinking_budget, f.image_count), (2, 512, 1));
    }

    #[test]
    fn codex_declares_its_tools_inside_input() {
        let f = facts(json!({
            "input": [
                {"role": "system", "tools": [{"name": "exec"}, {"name": "wait"}]},
                {"role": "user", "content": []},
            ],
        }));
        assert_eq!((f.turn_index, f.tools_declared), (2, 2));
    }

    #[test]
    fn an_unrecognised_body_yields_zeroes() {
        let f = facts(json!({"model": "m"}));
        assert_eq!((f.turn_index, f.tools_declared), (0, 0));
    }
}
