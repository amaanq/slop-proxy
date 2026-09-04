pub mod anthropic_req;
pub mod anthropic_stream;
pub mod chat;
pub mod count_tokens;
pub mod gemini_bridge;
pub mod model_map;
pub mod openai_req;
pub mod openai_stream;

use std::sync::{Arc, Mutex};

use data_encoding::BASE64URL_NOPAD;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::value::{RawValue, to_raw_value};

use crate::codex::sse::EventStream;
use crate::codex::types::{OutputContentPart, OutputItem, ResponsesEvent, SummaryPart, Usage};

/// A tagged or untagged enum, and anything behind a `flatten`, buffers the
/// map first, and `RawValue` cannot be read back out of that buffer.
pub fn buffered_raw<'de, D>(deserializer: D) -> Result<Option<Box<RawValue>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<serde_json::Value>::deserialize(deserializer)?
        .map(|value| to_raw_value(&value).map_err(serde::de::Error::custom))
        .transpose()
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("tool message without tool_call_id")]
    ToolMessageWithoutCallId,
    #[error("unsupported message role {0:?}")]
    UnsupportedRole(String),
}

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
    /// Separates a slow account from a long answer, which one duration cannot.
    pub first_byte_at: Option<std::time::Instant>,
    pub response_bytes: i64,
    pub stop_reason: Option<String>,
    /// Names only. An argument is the caller's shell command or source.
    pub tools_called: Vec<String>,
    /// What upstream opened with, the only evidence left when a 200 yields
    /// no events at all.
    pub upstream_head: Option<String>,
}

#[derive(Default, Clone)]
pub struct UsageCapture(pub Arc<Mutex<CapturedUsage>>);

impl UsageCapture {
    /// Codex counts cached tokens inside `input_tokens`, Anthropic reports
    /// them alongside. Subtracting leaves it meaning freshly billed prompt on
    /// both. `reasoning_tokens` stays a subset of `output_tokens`.
    pub fn record(&self, usage: &Usage) {
        self.record_partial(usage);
        self.0.lock().unwrap().completed = true;
    }

    pub fn record_partial(&self, usage: &Usage) {
        let mut c = self.0.lock().unwrap();
        let cached = usage.input_tokens_details.cached_tokens;
        c.input_tokens = (usage.input_tokens - cached).max(0);
        c.output_tokens = usage.output_tokens;
        c.cache_read_tokens = cached;
        c.reasoning_tokens = usage.output_tokens_details.reasoning_tokens;
    }

    pub fn note_event(&self, name: &str) {
        let mut c = self.0.lock().unwrap();
        c.last_event = Some(name.to_string());
        c.first_byte_at.get_or_insert_with(std::time::Instant::now);
    }

    pub fn note_bytes(&self, n: usize) {
        let mut c = self.0.lock().unwrap();
        c.response_bytes += n as i64;
        c.first_byte_at.get_or_insert_with(std::time::Instant::now);
    }

    pub fn note_stop_reason(&self, reason: &str) {
        self.0.lock().unwrap().stop_reason = Some(reason.to_string());
    }

    pub fn note_cutoff(&self, status: &str) {
        let mut c = self.0.lock().unwrap();
        c.error_kind = Some("upstream_cutoff".into());
        c.stop_reason = Some(status.to_ascii_lowercase());
    }

    pub fn note_tool_call(&self, name: &str) {
        let mut c = self.0.lock().unwrap();
        if !c.tools_called.iter().any(|t| t == name) {
            c.tools_called.push(name.to_string());
        }
    }

    pub fn note_upstream_head(&self, bytes: &[u8]) {
        let mut c = self.0.lock().unwrap();
        if c.upstream_head.is_none() {
            c.upstream_head = Some(String::from_utf8_lossy(bytes).chars().take(400).collect());
        }
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

impl StopKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::ToolUse => "tool_use",
            Self::Error => "error",
        }
    }
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

pub enum Step {
    Start {
        id: Option<String>,
    },
    OpenThinking,
    Thinking(String),
    Signature(String),
    OpenText,
    Text(String),
    OpenCall {
        id: String,
        name: String,
    },
    Args(String),
    CloseBlock,
    Stop {
        kind: StopKind,
        usage: Usage,
    },
    Failed {
        message: String,
        code: Option<String>,
    },
}

#[derive(PartialEq, Clone, Copy)]
enum Open {
    Thinking,
    Text,
    Call,
}

/// The one state machine every consumer of a Responses stream shares.
pub struct Walker {
    open: Option<Open>,
    saw_tool: bool,
    text_seen: bool,
    args_seen: bool,
    done: bool,
    capture: UsageCapture,
}

impl Walker {
    pub fn new(capture: UsageCapture) -> Self {
        Self {
            open: None,
            saw_tool: false,
            text_seen: false,
            args_seen: false,
            done: false,
            capture,
        }
    }

    pub fn step(&mut self, ev: ResponsesEvent) -> Vec<Step> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        match ev {
            ResponsesEvent::Created { response } => out.push(Step::Start { id: response.id }),
            ResponsesEvent::OutputItemAdded { item, .. } => match item {
                OutputItem::Reasoning { .. } => self.open(&mut out, Open::Thinking),
                OutputItem::FunctionCall { call_id, name, .. } => {
                    self.open(&mut out, Open::Call);
                    out.push(Step::OpenCall { id: call_id, name });
                }
                _ => {}
            },
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.open == Some(Open::Thinking) && self.text_seen {
                    out.push(Step::Thinking("\n\n".into()));
                }
            }
            ResponsesEvent::ReasoningSummaryTextDelta { delta }
            | ResponsesEvent::ReasoningTextDelta { delta } => {
                self.ensure(&mut out, Open::Thinking);
                self.text_seen = true;
                out.push(Step::Thinking(delta));
            }
            ResponsesEvent::OutputTextDelta { delta, .. } => {
                self.ensure(&mut out, Open::Text);
                self.text_seen = true;
                out.push(Step::Text(delta));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta, .. } => {
                if self.open == Some(Open::Call) {
                    self.args_seen = true;
                    out.push(Step::Args(delta));
                }
            }
            ResponsesEvent::OutputItemDone { item, .. } => match item {
                OutputItem::Reasoning {
                    id,
                    summary,
                    encrypted_content,
                } => {
                    self.ensure(&mut out, Open::Thinking);
                    let text = summary
                        .unwrap_or_default()
                        .iter()
                        .map(|SummaryPart::SummaryText { text }| text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    if !self.text_seen && !text.is_empty() {
                        out.push(Step::Thinking(text));
                    }
                    if let Some(ec) = encrypted_content {
                        out.push(Step::Signature(encode_signature(id.as_deref(), &ec)));
                    }
                    self.close(&mut out);
                }
                OutputItem::Message { content, .. } => {
                    let text = content
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|p| match p {
                            OutputContentPart::OutputText { text } => Some(text.as_str()),
                            OutputContentPart::Other => None,
                        })
                        .collect::<String>();
                    if self.open != Some(Open::Text) && !text.is_empty() {
                        self.open(&mut out, Open::Text);
                        out.push(Step::Text(text));
                    }
                    self.close(&mut out);
                }
                OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    if self.open != Some(Open::Call) {
                        self.open(&mut out, Open::Call);
                        out.push(Step::OpenCall { id: call_id, name });
                    }
                    if !self.args_seen
                        && let Some(args) = arguments.filter(|a| !a.is_empty())
                    {
                        out.push(Step::Args(args));
                    }
                    self.close(&mut out);
                }
                OutputItem::Other | OutputItem::CustomToolCall { .. } => {}
            },
            ResponsesEvent::Completed { response } => {
                let kind = if self.saw_tool {
                    StopKind::ToolUse
                } else {
                    StopKind::EndTurn
                };
                self.stop(&mut out, kind, response.usage);
            }
            ResponsesEvent::Incomplete { response } => {
                self.stop(&mut out, StopKind::MaxTokens, response.usage);
            }
            ResponsesEvent::Failed { response } => {
                let (message, code) = response
                    .error
                    .map(|e| (e.message, e.code))
                    .unwrap_or_default();
                let message = message.unwrap_or_else(|| "upstream response failed".into());
                self.capture.fail("upstream_failed");
                self.capture.note_stop_reason(StopKind::Error.as_str());
                self.done = true;
                out.push(Step::Failed { message, code });
            }
            _ => {}
        }
        out
    }

    pub fn eof(&mut self) -> Vec<Step> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        self.capture.fail("midstream");
        vec![Step::Failed {
            message: "upstream stream ended unexpectedly".into(),
            code: None,
        }]
    }

    fn ensure(&mut self, out: &mut Vec<Step>, kind: Open) {
        if self.open != Some(kind) {
            self.open(out, kind);
        }
    }

    fn open(&mut self, out: &mut Vec<Step>, kind: Open) {
        self.close(out);
        self.open = Some(kind);
        self.text_seen = false;
        match kind {
            Open::Thinking => out.push(Step::OpenThinking),
            Open::Text => out.push(Step::OpenText),
            Open::Call => {
                self.saw_tool = true;
                self.args_seen = false;
            }
        }
    }

    fn close(&mut self, out: &mut Vec<Step>) {
        if self.open.take().is_some() {
            out.push(Step::CloseBlock);
        }
    }

    fn stop(&mut self, out: &mut Vec<Step>, kind: StopKind, usage: Option<Usage>) {
        self.close(out);
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        self.capture.note_stop_reason(kind.as_str());
        self.done = true;
        out.push(Step::Stop { kind, usage });
    }
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
    let mut walker = Walker::new(capture.clone());
    let mut steps = Vec::new();
    while let Some(ev) = stream.next().await {
        steps.extend(walker.step(ev));
    }
    steps.extend(walker.eof());
    for step in steps {
        agg.fold(step);
    }
    agg
}

impl Aggregated {
    fn fold(&mut self, step: Step) {
        match (step, self.blocks.last_mut()) {
            (Step::Start { id: Some(id) }, _) => self.id = id,
            (Step::OpenThinking, _) => self.blocks.push(Block::Thinking {
                text: String::new(),
                signature: None,
            }),
            (Step::OpenText, _) => self.blocks.push(Block::Text {
                text: String::new(),
            }),
            (Step::OpenCall { id, name }, _) => self.blocks.push(Block::ToolCall {
                id,
                name,
                arguments: String::new(),
            }),
            (Step::Thinking(s), Some(Block::Thinking { text, .. }))
            | (Step::Text(s), Some(Block::Text { text }))
            | (
                Step::Args(s),
                Some(Block::ToolCall {
                    arguments: text, ..
                }),
            ) => text.push_str(&s),
            (Step::Signature(s), Some(Block::Thinking { signature, .. })) => *signature = Some(s),
            (Step::CloseBlock, Some(Block::ToolCall { arguments, .. })) if arguments.is_empty() => {
                arguments.push_str("{}");
            }
            (Step::Stop { kind, usage }, _) => {
                self.stop = kind;
                self.usage = usage;
                self.completed = true;
            }
            (Step::Failed { message, .. }, _) => {
                self.error_message = Some(message);
                self.stop = StopKind::Error;
            }
            _ => {}
        }
    }
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
