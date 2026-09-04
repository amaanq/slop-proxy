use serde::Serialize;

use super::chat::{
    ChatChoice, ChatChunk, ChatCompletion, ChatContent, ChatDelta, ChatError, ChatErrorBody,
    ChatMessage, ChatToolCall, ChatUsage, ChunkChoice, FinishReason, FunctionBody,
};
use super::{Aggregated, Block, StopKind, UsageCapture};
use crate::codex::types::{OutputItem, ResponsesEvent, Usage};

pub struct OpenAiStream {
    model: String,
    id: String,
    created: i64,
    include_usage: bool,
    tool_index: u64,
    tool_open: bool,
    tool_args_seen: bool,
    saw_tool: bool,
    reasoning_seen: bool,
    pub done: bool,
    capture: UsageCapture,
}

fn to_json<T: Serialize>(payload: T) -> String {
    serde_json::to_string(&payload).expect("chunk serializes")
}

impl OpenAiStream {
    pub fn new(model: String, include_usage: bool, capture: UsageCapture) -> Self {
        Self {
            model,
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            created: crate::clock::unix_now(),
            include_usage,
            tool_index: 0,
            tool_open: false,
            tool_args_seen: false,
            saw_tool: false,
            reasoning_seen: false,
            done: false,
            capture,
        }
    }

    pub fn handle(&mut self, ev: ResponsesEvent) -> Vec<String> {
        let mut out = Vec::new();
        match ev {
            ResponsesEvent::Created { response } => {
                if let Some(id) = response.id {
                    self.id = format!("chatcmpl-{id}");
                }
                out.push(self.chunk(
                    ChatDelta {
                        role: Some("assistant".into()),
                        content: Some(String::new()),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::OutputTextDelta { delta, .. } => {
                out.push(self.chunk(
                    ChatDelta {
                        content: Some(delta),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::ReasoningSummaryTextDelta { delta }
            | ResponsesEvent::ReasoningTextDelta { delta } => {
                self.reasoning_seen = true;
                out.push(self.chunk(
                    ChatDelta {
                        reasoning_content: Some(delta),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.reasoning_seen {
                    out.push(self.chunk(
                        ChatDelta {
                            reasoning_content: Some("\n\n".into()),
                            ..Default::default()
                        },
                        None,
                    ));
                }
            }
            ResponsesEvent::OutputItemAdded {
                item: OutputItem::FunctionCall { call_id, name, .. },
                ..
            } => {
                if self.saw_tool {
                    self.tool_index += 1;
                }
                self.tool_open = true;
                self.tool_args_seen = false;
                self.saw_tool = true;
                out.push(self.chunk(
                    ChatDelta {
                        tool_calls: Some(vec![ChatToolCall {
                            index: Some(self.tool_index),
                            id: Some(call_id),
                            kind: Some("function".into()),
                            function: FunctionBody {
                                name: Some(name),
                                arguments: Some(String::new()),
                            },
                            extra_content: None,
                        }]),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta, .. } => {
                if self.tool_open {
                    self.tool_args_seen = true;
                    out.push(self.args_chunk(delta));
                }
            }
            ResponsesEvent::OutputItemDone {
                item: OutputItem::FunctionCall { arguments, .. },
                ..
            } => {
                if self.tool_open
                    && !self.tool_args_seen
                    && let Some(args) = arguments.filter(|a| !a.is_empty())
                {
                    out.push(self.args_chunk(args));
                }
                self.tool_open = false;
            }
            ResponsesEvent::Completed { response } => {
                let reason = if self.saw_tool {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                };
                self.finish(&mut out, response.usage, reason);
            }
            ResponsesEvent::Incomplete { response } => {
                self.finish(&mut out, response.usage, FinishReason::Length);
            }
            ResponsesEvent::Failed { response } => {
                let msg = response
                    .error
                    .and_then(|e| e.message)
                    .unwrap_or_else(|| "upstream response failed".into());
                out.push(to_json(ChatError {
                    error: ChatErrorBody {
                        message: msg,
                        kind: Some("api_error".into()),
                        code: None,
                    },
                }));
                self.capture.fail("upstream_failed");
                self.done = true;
            }
            _ => {}
        }
        out
    }

    pub fn finalize(&mut self) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        self.capture.fail("midstream");
        vec![to_json(ChatError {
            error: ChatErrorBody {
                message: "upstream stream ended unexpectedly".into(),
                kind: Some("api_error".into()),
                code: None,
            },
        })]
    }

    fn finish(&mut self, out: &mut Vec<String>, usage: Option<Usage>, reason: FinishReason) {
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        out.push(self.chunk(ChatDelta::default(), Some(reason)));
        if self.include_usage {
            out.push(to_json(ChatChunk {
                id: self.id.clone(),
                object: "chat.completion.chunk".into(),
                created: self.created,
                model: self.model.clone(),
                choices: Vec::new(),
                usage: Some(ChatUsage::from(&usage)),
            }));
        }
        self.done = true;
    }

    fn args_chunk(&self, arguments: String) -> String {
        self.chunk(
            ChatDelta {
                tool_calls: Some(vec![ChatToolCall {
                    index: Some(self.tool_index),
                    function: FunctionBody {
                        name: None,
                        arguments: Some(arguments),
                    },
                    ..Default::default()
                }]),
                ..Default::default()
            },
            None,
        )
    }

    fn chunk(&self, delta: ChatDelta, finish_reason: Option<FinishReason>) -> String {
        to_json(ChatChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".into(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
                logprobs: None,
            }],
            usage: None,
        })
    }
}

pub fn render_aggregated(agg: &Aggregated, model: &str) -> ChatCompletion {
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
                tool_calls.push(ChatToolCall {
                    id: Some(id.clone()),
                    kind: Some("function".into()),
                    function: FunctionBody {
                        name: Some(name.clone()),
                        arguments: Some(arguments.clone()),
                    },
                    ..Default::default()
                });
            }
        }
    }

    let finish_reason = match agg.stop {
        StopKind::ToolUse => FinishReason::ToolCalls,
        StopKind::MaxTokens => FinishReason::Length,
        _ => FinishReason::Stop,
    };
    ChatCompletion {
        id: format!("chatcmpl-{}", agg.id),
        object: "chat.completion".into(),
        created: crate::clock::unix_now(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(ChatContent::Text(text)),
                reasoning_content: Some(reasoning).filter(|r| !r.is_empty()),
                tool_calls: Some(tool_calls).filter(|t| !t.is_empty()),
                ..Default::default()
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage: Some(ChatUsage::from(&agg.usage)),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::codex::types::{ResponseObj, UpstreamError};

    #[test]
    fn tool_call_indices_start_at_zero_and_count_up() {
        let mut stream = OpenAiStream::new("m".into(), false, UsageCapture::default());
        let mut indices = Vec::new();
        for _ in 0..3 {
            let frames = stream.handle(ResponsesEvent::OutputItemAdded {
                output_index: 0,
                item: OutputItem::FunctionCall {
                    id: None,
                    call_id: "c".into(),
                    name: "f".into(),
                    arguments: Some(String::new()),
                    status: None,
                },
            });
            assert_eq!(frames.len(), 1);
            let v: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
            indices.push(
                v["choices"][0]["delta"]["tool_calls"][0]["index"]
                    .as_u64()
                    .unwrap(),
            );
        }
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn the_failure_frame_names_its_type_and_a_null_code() {
        let mut stream = OpenAiStream::new("m".into(), false, UsageCapture::default());
        let frames = stream.handle(ResponsesEvent::Failed {
            response: ResponseObj {
                error: Some(UpstreamError {
                    code: None,
                    message: Some("boom".into()),
                }),
                ..Default::default()
            },
        });
        assert_eq!(frames.len(), 1);
        let v: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        assert_eq!(
            v,
            json!({"error": {"message": "boom", "type": "api_error", "code": null}})
        );
    }
}
