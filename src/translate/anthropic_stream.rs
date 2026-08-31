use serde_json::{json, Value};

use super::{encode_signature, Aggregated, Block, StopKind, UsageCapture};
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
                out.push((
                    "message_start",
                    json!({
                        "type": "message_start",
                        "message": {
                            "id": id,
                            "type": "message",
                            "role": "assistant",
                            "model": self.model,
                            "content": [],
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {
                                "input_tokens": self.est_input_tokens,
                                "output_tokens": 1,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0
                            }
                        }
                    }),
                ));
                out.push(("ping", json!({"type": "ping"})));
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
                    out.push((
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}
                        }),
                    ));
                }
                _ => {}
            },
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.emit_thinking
                    && self.open == Some(OpenBlock::Thinking)
                    && self.thinking_text_seen
                {
                    out.push(self.delta(json!({"type": "thinking_delta", "thinking": "\n\n"})));
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
                    out.push(self.delta(json!({"type": "thinking_delta", "thinking": delta})));
                }
            }
            ResponsesEvent::OutputTextDelta { delta } => {
                if self.open != Some(OpenBlock::Text) {
                    self.close_open(&mut out);
                    self.open_block(&mut out, OpenBlock::Text);
                }
                out.push(self.delta(json!({"type": "text_delta", "text": delta})));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta } => {
                if self.open == Some(OpenBlock::Tool) {
                    self.tool_args_seen = true;
                    out.push(
                        self.delta(json!({"type": "input_json_delta", "partial_json": delta})),
                    );
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
                            out.push(self.delta(json!({
                                "type": "signature_delta",
                                "signature": encode_signature(id.as_deref(), &ec)
                            })));
                        }
                        self.close_open(&mut out);
                    }
                }
                OutputItem::FunctionCall { arguments, .. } => {
                    if self.open == Some(OpenBlock::Tool) {
                        if !self.tool_args_seen
                            && let Some(args) = arguments.filter(|a| !a.is_empty()) {
                                out.push(self.delta(
                                    json!({"type": "input_json_delta", "partial_json": args}),
                                ));
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
                out.push((
                    "error",
                    json!({"type": "error", "error": {"type": err_type, "message": msg}}),
                ));
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
        vec![(
            "error",
            json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "upstream stream ended unexpectedly"}
            }),
        )]
    }

    fn finish(&mut self, out: &mut Vec<OutEvent>, usage: Option<Usage>, stop_reason: &str) {
        self.close_open(out);
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        out.push((
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": anthropic_usage(&usage)
            }),
        ));
        out.push(("message_stop", json!({"type": "message_stop"})));
        self.done = true;
    }

    fn open_block(&mut self, out: &mut Vec<OutEvent>, kind: OpenBlock) {
        let index = self.next_index;
        self.next_index += 1;
        let content_block = match kind {
            OpenBlock::Thinking => {
                self.thinking_text_seen = false;
                json!({"type": "thinking", "thinking": ""})
            }
            OpenBlock::Text => json!({"type": "text", "text": ""}),
            OpenBlock::Tool => unreachable!("tool blocks open in OutputItemAdded"),
        };
        self.open = Some(kind);
        out.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": content_block}),
        ));
    }

    fn close_open(&mut self, out: &mut Vec<OutEvent>) {
        if self.open.take().is_some() {
            out.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.next_index - 1}),
            ));
        }
    }

    fn delta(&self, delta: Value) -> OutEvent {
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": self.next_index - 1,
                "delta": delta
            }),
        )
    }
}

fn anthropic_usage(usage: &Usage) -> Value {
    let cached = usage.input_tokens_details.cached_tokens;
    json!({
        "input_tokens": (usage.input_tokens - cached).max(0),
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0
    })
}

pub fn render_aggregated(agg: &Aggregated, model: &str, emit_thinking: bool) -> Value {
    let mut content = Vec::new();
    for block in &agg.blocks {
        match block {
            Block::Thinking { text, signature } => {
                if emit_thinking {
                    content.push(json!({
                        "type": "thinking",
                        "thinking": text,
                        "signature": signature.clone().unwrap_or_default()
                    }));
                }
            }
            Block::Text { text } => content.push(json!({"type": "text", "text": text})),
            Block::ToolCall {
                id,
                name,
                arguments,
            } => {
                let input = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
            }
        }
    }
    let stop_reason = match agg.stop {
        StopKind::ToolUse => "tool_use",
        StopKind::MaxTokens => "max_tokens",
        _ => "end_turn",
    };
    json!({
        "id": agg.id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": anthropic_usage(&agg.usage)
    })
}
