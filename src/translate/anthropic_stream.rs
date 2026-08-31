use serde::Serialize;
use serde_json::Value;

use super::{Aggregated, Block, StopKind, UsageCapture, encode_signature};
use crate::codex::types::{OutputItem, ResponsesEvent, Usage};

pub struct AnthropicStream {
    model: String,
    est_input_tokens: i64,
    emit_thinking: bool,
    next_index: usize,
    open: Option<OpenBlock>,
    saw_tool: bool,
    thinking_text_seen: bool,
    tool_args_seen: bool,
    pub done: bool,
    capture: UsageCapture,
}

#[derive(PartialEq)]
enum OpenBlock {
    Thinking,
    Text,
    Tool,
}

pub type OutEvent = (&'static str, Value);

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthEvent {
    MessageStart {
        message: MessageStart,
    },
    Ping,
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StopDelta,
        usage: AnthUsage,
    },
    MessageStop,
    Error {
        error: ErrorBody,
    },
}

impl AnthEvent {
    fn out(self) -> OutEvent {
        let name = match &self {
            AnthEvent::MessageStart { .. } => "message_start",
            AnthEvent::Ping => "ping",
            AnthEvent::ContentBlockStart { .. } => "content_block_start",
            AnthEvent::ContentBlockDelta { .. } => "content_block_delta",
            AnthEvent::ContentBlockStop { .. } => "content_block_stop",
            AnthEvent::MessageDelta { .. } => "message_delta",
            AnthEvent::MessageStop => "message_stop",
            AnthEvent::Error { .. } => "error",
        };
        (name, serde_json::to_value(self).expect("event serializes"))
    }
}

#[derive(Serialize)]
struct MessageStart {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: String,
    content: [(); 0],
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthUsage,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    ToolUse {
        id: String,
        name: String,
        input: EmptyObject,
    },
    Thinking {
        thinking: &'static str,
    },
    Text {
        text: &'static str,
    },
}

#[derive(Serialize)]
struct EmptyObject {}

#[derive(Serialize)]
#[serde(tag = "type")]
enum BlockDelta {
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
}

#[derive(Serialize)]
struct StopDelta {
    stop_reason: &'static str,
    stop_sequence: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: &'static str,
    message: String,
}

#[derive(Serialize)]
struct AnthUsage {
    input_tokens: i64,
    output_tokens: i64,
    cache_read_input_tokens: i64,
    cache_creation_input_tokens: i64,
}

fn anthropic_usage(usage: &Usage) -> AnthUsage {
    let cached = usage.input_tokens_details.cached_tokens;
    AnthUsage {
        input_tokens: (usage.input_tokens - cached).max(0),
        output_tokens: usage.output_tokens,
        cache_read_input_tokens: cached,
        cache_creation_input_tokens: 0,
    }
}

impl AnthropicStream {
    pub fn new(
        model: String,
        est_input_tokens: i64,
        emit_thinking: bool,
        capture: UsageCapture,
    ) -> Self {
        Self {
            model,
            est_input_tokens,
            emit_thinking,
            next_index: 0,
            open: None,
            saw_tool: false,
            thinking_text_seen: false,
            tool_args_seen: false,
            done: false,
            capture,
        }
    }

    pub fn handle(&mut self, ev: ResponsesEvent) -> Vec<OutEvent> {
        let mut out = Vec::new();
        match ev {
            ResponsesEvent::Created { response } => {
                let id = response
                    .id
                    .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple()));
                out.push(
                    AnthEvent::MessageStart {
                        message: MessageStart {
                            id,
                            kind: "message",
                            role: "assistant",
                            model: self.model.clone(),
                            content: [],
                            stop_reason: None,
                            stop_sequence: None,
                            usage: AnthUsage {
                                input_tokens: self.est_input_tokens,
                                output_tokens: 1,
                                cache_read_input_tokens: 0,
                                cache_creation_input_tokens: 0,
                            },
                        },
                    }
                    .out(),
                );
                out.push(AnthEvent::Ping.out());
            }
            ResponsesEvent::OutputItemAdded { item } => match item {
                OutputItem::Reasoning { .. } if self.emit_thinking => {
                    self.close_open(&mut out);
                    self.open_block(&mut out, OpenBlock::Thinking);
                }
                OutputItem::FunctionCall { call_id, name, .. } => {
                    self.close_open(&mut out);
                    let index = self.next_index;
                    self.next_index += 1;
                    self.open = Some(OpenBlock::Tool);
                    self.saw_tool = true;
                    self.tool_args_seen = false;
                    out.push(
                        AnthEvent::ContentBlockStart {
                            index,
                            content_block: ContentBlockStart::ToolUse {
                                id: call_id,
                                name,
                                input: EmptyObject {},
                            },
                        }
                        .out(),
                    );
                }
                _ => {}
            },
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.emit_thinking
                    && self.open == Some(OpenBlock::Thinking)
                    && self.thinking_text_seen
                {
                    out.push(self.delta(BlockDelta::Thinking {
                        thinking: "\n\n".into(),
                    }));
                }
            }
            ResponsesEvent::ReasoningSummaryTextDelta { delta }
            | ResponsesEvent::ReasoningTextDelta { delta } => {
                if self.emit_thinking {
                    if self.open != Some(OpenBlock::Thinking) {
                        self.close_open(&mut out);
                        self.open_block(&mut out, OpenBlock::Thinking);
                    }
                    self.thinking_text_seen = true;
                    out.push(self.delta(BlockDelta::Thinking { thinking: delta }));
                }
            }
            ResponsesEvent::OutputTextDelta { delta } => {
                if self.open != Some(OpenBlock::Text) {
                    self.close_open(&mut out);
                    self.open_block(&mut out, OpenBlock::Text);
                }
                out.push(self.delta(BlockDelta::Text { text: delta }));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta } => {
                if self.open == Some(OpenBlock::Tool) {
                    self.tool_args_seen = true;
                    out.push(self.delta(BlockDelta::InputJson {
                        partial_json: delta,
                    }));
                }
            }
            ResponsesEvent::OutputItemDone { item } => match item {
                OutputItem::Reasoning {
                    id,
                    encrypted_content,
                    ..
                } => {
                    if self.open == Some(OpenBlock::Thinking) {
                        if let Some(ec) = encrypted_content {
                            out.push(self.delta(BlockDelta::Signature {
                                signature: encode_signature(id.as_deref(), &ec),
                            }));
                        }
                        self.close_open(&mut out);
                    }
                }
                OutputItem::FunctionCall { arguments, .. } => {
                    if self.open == Some(OpenBlock::Tool) {
                        if !self.tool_args_seen
                            && let Some(args) = arguments.filter(|a| !a.is_empty())
                        {
                            out.push(self.delta(BlockDelta::InputJson {
                                partial_json: args,
                            }));
                        }
                        self.close_open(&mut out);
                    }
                }
                OutputItem::Message { .. } => self.close_open(&mut out),
                OutputItem::Other => {}
            },
            ResponsesEvent::Completed { response } => {
                self.finish(
                    &mut out,
                    response.usage,
                    if self.saw_tool {
                        "tool_use"
                    } else {
                        "end_turn"
                    },
                );
            }
            ResponsesEvent::Incomplete { response } => {
                self.finish(&mut out, response.usage, "max_tokens");
            }
            ResponsesEvent::Failed { response } => {
                let msg = response
                    .error
                    .as_ref()
                    .and_then(|e| e.message.clone())
                    .unwrap_or_else(|| "upstream response failed".into());
                let code = response.error.and_then(|e| e.code).unwrap_or_default();
                let err_type = if code.contains("rate_limit") || msg.contains("rate limit") {
                    "rate_limit_error"
                } else {
                    "api_error"
                };
                out.push(
                    AnthEvent::Error {
                        error: ErrorBody {
                            kind: err_type,
                            message: msg,
                        },
                    }
                    .out(),
                );
                self.capture.fail("upstream_failed");
                self.done = true;
            }
            _ => {}
        }
        out
    }

    pub fn finalize(&mut self) -> Vec<OutEvent> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        self.capture.fail("midstream");
        vec![
            AnthEvent::Error {
                error: ErrorBody {
                    kind: "overloaded_error",
                    message: "upstream stream ended unexpectedly".into(),
                },
            }
            .out(),
        ]
    }

    fn finish(&mut self, out: &mut Vec<OutEvent>, usage: Option<Usage>, stop_reason: &'static str) {
        self.close_open(out);
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        out.push(
            AnthEvent::MessageDelta {
                delta: StopDelta {
                    stop_reason,
                    stop_sequence: None,
                },
                usage: anthropic_usage(&usage),
            }
            .out(),
        );
        out.push(AnthEvent::MessageStop.out());
        self.done = true;
    }

    fn open_block(&mut self, out: &mut Vec<OutEvent>, kind: OpenBlock) {
        let index = self.next_index;
        self.next_index += 1;
        let content_block = match kind {
            OpenBlock::Thinking => {
                self.thinking_text_seen = false;
                ContentBlockStart::Thinking { thinking: "" }
            }
            OpenBlock::Text => ContentBlockStart::Text { text: "" },
            OpenBlock::Tool => unreachable!("tool blocks open in OutputItemAdded"),
        };
        self.open = Some(kind);
        out.push(
            AnthEvent::ContentBlockStart {
                index,
                content_block,
            }
            .out(),
        );
    }

    fn close_open(&mut self, out: &mut Vec<OutEvent>) {
        if self.open.take().is_some() {
            out.push(
                AnthEvent::ContentBlockStop {
                    index: self.next_index - 1,
                }
                .out(),
            );
        }
    }

    fn delta(&self, delta: BlockDelta) -> OutEvent {
        AnthEvent::ContentBlockDelta {
            index: self.next_index - 1,
            delta,
        }
        .out()
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RenderedBlock {
    Thinking { thinking: String, signature: String },
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Serialize)]
struct RenderedMessage {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: String,
    content: Vec<RenderedBlock>,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: AnthUsage,
}

pub fn render_aggregated(agg: &Aggregated, model: &str, emit_thinking: bool) -> Value {
    let mut content = Vec::new();
    for block in &agg.blocks {
        match block {
            Block::Thinking { text, signature } => {
                if emit_thinking {
                    content.push(RenderedBlock::Thinking {
                        thinking: text.clone(),
                        signature: signature.clone().unwrap_or_default(),
                    });
                }
            }
            Block::Text { text } => content.push(RenderedBlock::Text { text: text.clone() }),
            Block::ToolCall {
                id,
                name,
                arguments,
            } => {
                let input = serde_json::from_str(arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                content.push(RenderedBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
            }
        }
    }
    let stop_reason = match agg.stop {
        StopKind::ToolUse => "tool_use",
        StopKind::MaxTokens => "max_tokens",
        _ => "end_turn",
    };
    serde_json::to_value(RenderedMessage {
        id: agg.id.clone(),
        kind: "message",
        role: "assistant",
        model: model.to_string(),
        content,
        stop_reason,
        stop_sequence: None,
        usage: anthropic_usage(&agg.usage),
    })
    .expect("message serializes")
}
