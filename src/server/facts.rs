use crate::codex::types::{ContentPart, InputItem, ResponsesRequest};
use crate::gemini::types::GenerateContentRequest;
use crate::translate::anthropic_req::{AnthropicRequest, ContentBlock, MessageContent};
use crate::translate::chat::{ChatContent, ChatPart, ChatRequest};

/// Structural facts about a request, read the same way across the body
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
    pub fn empty(headers: &axum::http::HeaderMap) -> Self {
        Self {
            request_bytes: request_bytes(headers),
            ..Default::default()
        }
    }

    pub fn from_chat(req: &ChatRequest, headers: &axum::http::HeaderMap) -> Self {
        Self {
            request_bytes: request_bytes(headers),
            turn_index: req.messages.len() as i64,
            tools_declared: req.tools.as_ref().map_or(0, |t| t.len() as i64),
            thinking_budget: 0,
            image_count: req
                .messages
                .iter()
                .filter_map(|m| match &m.content {
                    Some(ChatContent::Parts(parts)) => Some(parts),
                    _ => None,
                })
                .flatten()
                .filter(|p| matches!(p, ChatPart::ImageUrl { .. }))
                .count() as i64,
        }
    }

    pub fn from_responses(req: &ResponsesRequest, headers: &axum::http::HeaderMap) -> Self {
        let mut tools = req.tools.len() as i64;
        let mut images = 0;
        for item in &req.input {
            match item {
                InputItem::AdditionalTools { tools: t, .. } => tools += t.len() as i64,
                InputItem::Message { content, .. } => {
                    images += content
                        .iter()
                        .filter(|p| matches!(p, ContentPart::InputImage { .. }))
                        .count() as i64;
                }
                _ => {}
            }
        }
        Self {
            request_bytes: request_bytes(headers),
            turn_index: req.input.len() as i64,
            tools_declared: tools,
            thinking_budget: 0,
            image_count: images,
        }
    }

    pub fn from_anthropic(req: &AnthropicRequest, headers: &axum::http::HeaderMap) -> Self {
        Self {
            request_bytes: request_bytes(headers),
            turn_index: req.messages.len() as i64,
            tools_declared: req.tools.as_ref().map_or(0, |t| t.len() as i64),
            thinking_budget: req
                .thinking
                .as_ref()
                .and_then(|t| t.budget_tokens)
                .unwrap_or(0) as i64,
            image_count: req
                .messages
                .iter()
                .filter_map(|m| match &m.content {
                    MessageContent::Blocks(blocks) => Some(blocks),
                    MessageContent::Text(_) => None,
                })
                .flatten()
                .filter(|b| matches!(b, ContentBlock::Image { .. }))
                .count() as i64,
        }
    }

    pub fn from_native(req: &GenerateContentRequest, headers: &axum::http::HeaderMap) -> Self {
        Self {
            request_bytes: request_bytes(headers),
            turn_index: req.contents.len() as i64,
            tools_declared: req
                .tools
                .iter()
                .flatten()
                .map(|t| t.function_declarations.len() as i64)
                .sum(),
            thinking_budget: req
                .generation_config
                .as_ref()
                .and_then(|g| g.thinking_config.as_ref())
                .and_then(|t| t.thinking_budget)
                .unwrap_or(0),
            image_count: req
                .contents
                .iter()
                .flat_map(|c| &c.parts)
                .filter(|p| p.inline_data.is_some())
                .count() as i64,
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

#[cfg(test)]
mod tests {
    use super::RequestFacts;
    use axum::http::HeaderMap;
    use serde_json::json;

    #[test]
    fn anthropic() {
        let req = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text"}, {"type": "image",
                    "source": {"type": "url", "url": "u"}}]},
                {"role": "assistant", "content": [{"type": "text"}]},
            ],
            "tools": [{"name": "Read"}, {"name": "Bash"}],
            "thinking": {"budget_tokens": 10000},
        }))
        .unwrap();
        let f = RequestFacts::from_anthropic(&req, &HeaderMap::new());
        assert_eq!((f.turn_index, f.tools_declared), (2, 2));
        assert_eq!((f.thinking_budget, f.image_count), (10000, 1));
    }

    #[test]
    fn openai_responses() {
        let req = serde_json::from_value(json!({
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_image", "image_url": "u"}]}],
            "tools": [{"name": "shell"}],
        }))
        .unwrap();
        let f = RequestFacts::from_responses(&req, &HeaderMap::new());
        assert_eq!((f.turn_index, f.tools_declared, f.image_count), (1, 1, 1));
    }

    #[test]
    fn gemini_counts_declarations_not_wrappers() {
        let req = serde_json::from_value(json!({
            "contents": [{"role": "user", "parts": [{"inlineData": {}}]}],
            "tools": [{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 512}},
        }))
        .unwrap();
        let f = RequestFacts::from_native(&req, &HeaderMap::new());
        assert_eq!(
            (f.tools_declared, f.thinking_budget, f.image_count),
            (2, 512, 1)
        );
    }

    #[test]
    fn codex_declares_its_tools_inside_input() {
        let req = serde_json::from_value(json!({
            "input": [
                {"type": "additional_tools", "role": "system",
                 "tools": [{"name": "exec"}, {"name": "wait"}]},
                {"type": "message", "role": "user", "content": []},
            ],
        }))
        .unwrap();
        let f = RequestFacts::from_responses(&req, &HeaderMap::new());
        assert_eq!((f.turn_index, f.tools_declared), (2, 2));
    }

    #[test]
    fn a_bare_chat_request_yields_zeroes() {
        let req = serde_json::from_value(json!({"model": "m"})).unwrap();
        let f = RequestFacts::from_chat(&req, &HeaderMap::new());
        assert_eq!((f.turn_index, f.tools_declared), (0, 0));
    }
}
