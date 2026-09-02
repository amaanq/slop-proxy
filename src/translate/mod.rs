pub mod anthropic_req;
pub mod anthropic_stream;
pub mod count_tokens;
pub mod model_map;
pub mod openai_req;
pub mod openai_stream;

use std::sync::{Arc, Mutex};

use data_encoding::BASE64URL_NOPAD;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::codex::sse::EventStream;
use crate::codex::types::{OutputContentPart, OutputItem, ResponsesEvent, SummaryPart, Usage};

#[derive(Serialize, Deserialize)]
struct SignaturePayload {
    id: Option<String>,
    ec: Option<String>,
}

/// Anthropic thinking-block signatures are opaque round-tripped strings, so we
/// smuggle the Responses reasoning item id + encrypted_content through them.
pub fn encode_signature(id: Option<&str>, encrypted_content: &str) -> String {
    let payload = SignaturePayload {
        id: id.map(String::from),
        ec: Some(encrypted_content.to_string()),
    };
    let payload = serde_json::to_string(&payload).unwrap_or_default();
    BASE64URL_NOPAD.encode(payload.as_bytes())
}

pub fn decode_signature(sig: &str) -> (Option<String>, Option<String>) {
    if let Ok(bytes) = BASE64URL_NOPAD.decode(sig.as_bytes())
        && let Ok(p) = serde_json::from_slice::<SignaturePayload>(&bytes)
        && p.ec.is_some()
    {
        return (p.id, p.ec);
    }
    (None, Some(sig.to_string()))
}

#[derive(Default, Debug, Clone)]
pub struct CapturedUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub reasoning_tokens: i64,
    pub completed: bool,
    pub error_kind: Option<String>,
    /// Distinguishes upstream ending early from the caller hanging up.
    pub upstream_eof: bool,
    pub last_event: Option<String>,
}

#[derive(Default, Clone)]
pub struct UsageCapture(pub Arc<Mutex<CapturedUsage>>);

impl UsageCapture {
    /// Codex counts cached tokens inside `input_tokens`, Anthropic reports
    /// them alongside. Subtracting leaves it meaning freshly billed prompt on
    /// both. `reasoning_tokens` stays a subset of `output_tokens`.
    pub fn record(&self, usage: &Usage) {
        let mut c = self.0.lock().unwrap();
        let cached = usage.input_tokens_details.cached_tokens;
        c.input_tokens = (usage.input_tokens - cached).max(0);
        c.output_tokens = usage.output_tokens;
        c.cache_read_tokens = cached;
        c.reasoning_tokens = usage.output_tokens_details.reasoning_tokens;
        c.completed = true;
    }

    pub fn note_event(&self, name: &str) {
        let mut c = self.0.lock().unwrap();
        c.last_event = Some(name.to_string());
    }

    pub fn note_upstream_eof(&self) {
        self.0.lock().unwrap().upstream_eof = true;
    }

    pub fn fail(&self, kind: &str) {
        let mut c = self.0.lock().unwrap();
        if c.error_kind.is_none() {
            c.error_kind = Some(kind.to_string());
        }
    }

    pub fn snapshot(&self) -> CapturedUsage {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Debug)]
pub enum Block {
    Thinking {
        text: String,
        signature: Option<String>,
    },
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopKind {
    EndTurn,
    MaxTokens,
    ToolUse,
    Error,
}

#[derive(Debug)]
pub struct Aggregated {
    pub id: String,
    pub blocks: Vec<Block>,
    pub stop: StopKind,
    pub usage: Usage,
    pub error_message: Option<String>,
    pub completed: bool,
}

pub async fn aggregate(mut stream: EventStream, capture: &UsageCapture) -> Aggregated {
    let mut agg = Aggregated {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        blocks: Vec::new(),
        stop: StopKind::EndTurn,
        usage: Usage::default(),
        error_message: None,
        completed: false,
    };
    let mut saw_tool = false;

    while let Some(ev) = stream.next().await {
        match ev {
            ResponsesEvent::Created { response } => {
                if let Some(id) = response.id {
                    agg.id = id;
                }
            }
            ResponsesEvent::OutputItemDone { item } => match item {
                OutputItem::Message { content, .. } => {
                    let text = content
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|p| match p {
                            OutputContentPart::OutputText { text } => Some(text.as_str()),
                            OutputContentPart::Other => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    agg.blocks.push(Block::Text { text });
                }
                OutputItem::Reasoning {
                    id,
                    summary,
                    encrypted_content,
                } => {
                    let text = summary
                        .unwrap_or_default()
                        .iter()
                        .map(|SummaryPart::SummaryText { text }| text.clone())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    let signature =
                        encrypted_content.map(|ec| encode_signature(id.as_deref(), &ec));
                    agg.blocks.push(Block::Thinking { text, signature });
                }
                OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    saw_tool = true;
                    agg.blocks.push(Block::ToolCall {
                        id: call_id,
                        name,
                        arguments: arguments.unwrap_or_else(|| "{}".into()),
                    });
                }
                OutputItem::Other => {}
            },
            ResponsesEvent::Completed { response } => {
                if let Some(u) = &response.usage {
                    agg.usage = u.clone();
                    capture.record(u);
                }
                agg.stop = if saw_tool {
                    StopKind::ToolUse
                } else {
                    StopKind::EndTurn
                };
                agg.completed = true;
            }
            ResponsesEvent::Incomplete { response } => {
                if let Some(u) = &response.usage {
                    agg.usage = u.clone();
                    capture.record(u);
                }
                agg.stop = StopKind::MaxTokens;
                agg.completed = true;
            }
            ResponsesEvent::Failed { response } => {
                let msg = response
                    .error
                    .and_then(|e| e.message)
                    .unwrap_or_else(|| "upstream response failed".into());
                agg.error_message = Some(msg);
                agg.stop = StopKind::Error;
                agg.completed = true;
                capture.fail("upstream_failed");
            }
            _ => {}
        }
    }
    if !agg.completed {
        agg.error_message = Some("upstream stream ended unexpectedly".into());
        agg.stop = StopKind::Error;
        capture.fail("midstream");
    }
    agg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_roundtrip() {
        let sig = encode_signature(Some("rs_1"), "SECRET");
        assert_eq!(
            decode_signature(&sig),
            (Some("rs_1".into()), Some("SECRET".into()))
        );
        let sig = encode_signature(None, "SECRET");
        assert_eq!(decode_signature(&sig), (None, Some("SECRET".into())));
        assert_eq!(
            decode_signature("not-base64-json"),
            (None, Some("not-base64-json".into()))
        );
    }
}
