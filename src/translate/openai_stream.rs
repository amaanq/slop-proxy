use serde_json::{json, Value};

use super::{Aggregated, Block, StopKind, UsageCapture};
use crate::codex::types::{OutputItem, ResponsesEvent, Usage};

pub struct OpenAiStream {
    model: String,
    id: String,
    created: i64,
    include_usage: bool,
    tool_index: i64,
    tool_open: bool,
    tool_args_seen: bool,
    saw_tool: bool,
    reasoning_seen: bool,
    pub done: bool,
    capture: UsageCapture,
}

impl OpenAiStream {
    pub fn new(model: String, include_usage: bool, capture: UsageCapture) -> Self {
        Self {
            model,
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            created: chrono::Utc::now().timestamp(),
            include_usage,
            tool_index: -1,
            tool_open: false,
            tool_args_seen: false,
            saw_tool: false,
            reasoning_seen: false,
            done: false,
            capture,
        }
    }

    pub fn handle(&mut self, ev: ResponsesEvent) -> Vec<Value> {
        let mut out = Vec::new();
        match ev {
            ResponsesEvent::Created { response } => {
                if let Some(id) = response.id {
                    self.id = format!("chatcmpl-{id}");
                }
                out.push(self.chunk(json!({"role": "assistant", "content": ""}), None));
            }
            ResponsesEvent::OutputTextDelta { delta } => {
                out.push(self.chunk(json!({"content": delta}), None));
            }
            ResponsesEvent::ReasoningSummaryTextDelta { delta }
            | ResponsesEvent::ReasoningTextDelta { delta } => {
                self.reasoning_seen = true;
                out.push(self.chunk(json!({"reasoning_content": delta}), None));
            }
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.reasoning_seen {
                    out.push(self.chunk(json!({"reasoning_content": "\n\n"}), None));
                }
            }
            ResponsesEvent::OutputItemAdded {
                item: OutputItem::FunctionCall { call_id, name, .. },
            } => {
                self.tool_index += 1;
                self.tool_open = true;
                self.tool_args_seen = false;
                self.saw_tool = true;
                out.push(self.chunk(
                    json!({
                        "tool_calls": [{
                            "index": self.tool_index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""}
                        }]
                    }),
                    None,
                ));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta } => {
                if self.tool_open {
                    self.tool_args_seen = true;
                    out.push(self.chunk(
                        json!({
                            "tool_calls": [{
                                "index": self.tool_index,
                                "function": {"arguments": delta}
                            }]
                        }),
                        None,
                    ));
                }
            }
            ResponsesEvent::OutputItemDone {
                item: OutputItem::FunctionCall { arguments, .. },
            } => {
                if self.tool_open && !self.tool_args_seen {
                    if let Some(args) = arguments.filter(|a| !a.is_empty()) {
                        out.push(self.chunk(
                            json!({
                                "tool_calls": [{
                                    "index": self.tool_index,
                                    "function": {"arguments": args}
                                }]
                            }),
                            None,
                        ));
                    }
                }
                self.tool_open = false;
            }
            ResponsesEvent::Completed { response } => {
                let reason = if self.saw_tool { "tool_calls" } else { "stop" };
                self.finish(&mut out, response.usage, reason);
            }
            ResponsesEvent::Incomplete { response } => {
                self.finish(&mut out, response.usage, "length");
            }
            ResponsesEvent::Failed { response } => {
                let msg = response
                    .error
                    .and_then(|e| e.message)
                    .unwrap_or_else(|| "upstream response failed".into());
                out.push(json!({
                    "error": {"message": msg, "type": "api_error", "code": null}
                }));
                self.capture.fail("upstream_failed");
                self.done = true;
            }
            _ => {}
        }
        out
    }

    pub fn finalize(&mut self) -> Vec<Value> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        self.capture.fail("midstream");
        vec![json!({
            "error": {"message": "upstream stream ended unexpectedly", "type": "api_error", "code": null}
        })]
    }

    fn finish(&mut self, out: &mut Vec<Value>, usage: Option<Usage>, reason: &str) {
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        out.push(self.chunk(json!({}), Some(reason)));
        if self.include_usage {
            let mut v = json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
            });
            v["usage"] = openai_usage(&usage);
            out.push(v);
        }
        self.done = true;
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": null
            }]
        })
    }
}

pub fn openai_usage(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.input_tokens_details.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": usage.output_tokens_details.reasoning_tokens}
    })
}

pub fn render_aggregated(agg: &Aggregated, model: &str) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for block in &agg.blocks {
        match block {
            Block::Text { text: t } => text.push_str(t),
            Block::Thinking { text: t, .. } => {
                if !reasoning.is_empty() {
                    reasoning.push_str("\n\n");
                }
                reasoning.push_str(t);
            }
            Block::ToolCall {
                id,
                name,
                arguments,
            } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }));
            }
        }
    }

    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let finish_reason = match agg.stop {
        StopKind::ToolUse => "tool_calls",
        StopKind::MaxTokens => "length",
        _ => "stop",
    };
    json!({
        "id": format!("chatcmpl-{}", agg.id),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": null
        }],
        "usage": openai_usage(&agg.usage)
    })
}
