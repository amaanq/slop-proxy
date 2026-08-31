use serde::Serialize;

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

#[derive(Serialize)]
struct Chunk {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: i64,
    delta: ChunkDelta,
    finish_reason: Option<&'static str>,
    logprobs: Option<()>,
}

#[derive(Default, Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Serialize)]
struct ToolCallDelta {
    index: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: FunctionDelta,
}

#[derive(Serialize)]
struct FunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Serialize)]
struct StreamError {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    code: Option<()>,
}

#[derive(Serialize)]
pub struct OpenAiUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
    prompt_tokens_details: PromptTokensDetails,
    completion_tokens_details: CompletionTokensDetails,
}

#[derive(Serialize)]
pub struct PromptTokensDetails {
    cached_tokens: i64,
}

#[derive(Serialize)]
pub struct CompletionTokensDetails {
    reasoning_tokens: i64,
}

fn openai_usage(usage: &Usage) -> OpenAiUsage {
    OpenAiUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.input_tokens + usage.output_tokens,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: usage.input_tokens_details.cached_tokens,
        },
        completion_tokens_details: CompletionTokensDetails {
            reasoning_tokens: usage.output_tokens_details.reasoning_tokens,
        },
    }
}

/// Streamed chunks keep serde_json's sorted key order (structs serialize in
/// declaration order, but Value maps sort), which the e2e stream assertions
/// pin down.
fn to_json<T: Serialize>(payload: T) -> String {
    let value = serde_json::to_value(payload).expect("chunk serializes");
    value.to_string()
}

impl OpenAiStream {
    pub fn new(model: String, include_usage: bool, capture: UsageCapture) -> Self {
        Self {
            model,
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            created: crate::clock::unix_now(),
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

    pub fn handle(&mut self, ev: ResponsesEvent) -> Vec<String> {
        let mut out = Vec::new();
        match ev {
            ResponsesEvent::Created { response } => {
                if let Some(id) = response.id {
                    self.id = format!("chatcmpl-{id}");
                }
                out.push(self.chunk(
                    ChunkDelta {
                        role: Some("assistant"),
                        content: Some(String::new()),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::OutputTextDelta { delta } => {
                out.push(self.chunk(
                    ChunkDelta {
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
                    ChunkDelta {
                        reasoning_content: Some(delta),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::ReasoningSummaryPartAdded {} => {
                if self.reasoning_seen {
                    out.push(self.chunk(
                        ChunkDelta {
                            reasoning_content: Some("\n\n".into()),
                            ..Default::default()
                        },
                        None,
                    ));
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
                    ChunkDelta {
                        tool_calls: Some(vec![ToolCallDelta {
                            index: self.tool_index,
                            id: Some(call_id),
                            kind: Some("function"),
                            function: FunctionDelta {
                                name: Some(name),
                                arguments: Some(String::new()),
                            },
                        }]),
                        ..Default::default()
                    },
                    None,
                ));
            }
            ResponsesEvent::FunctionCallArgumentsDelta { delta } => {
                if self.tool_open {
                    self.tool_args_seen = true;
                    out.push(self.args_chunk(delta));
                }
            }
            ResponsesEvent::OutputItemDone {
                item: OutputItem::FunctionCall { arguments, .. },
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
                out.push(to_json(StreamError {
                    error: ErrorBody {
                        message: msg,
                        kind: "api_error",
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
        vec![to_json(StreamError {
            error: ErrorBody {
                message: "upstream stream ended unexpectedly".into(),
                kind: "api_error",
                code: None,
            },
        })]
    }

    fn finish(&mut self, out: &mut Vec<String>, usage: Option<Usage>, reason: &'static str) {
        let usage = usage.unwrap_or_default();
        self.capture.record(&usage);
        out.push(self.chunk(ChunkDelta::default(), Some(reason)));
        if self.include_usage {
            out.push(to_json(Chunk {
                id: self.id.clone(),
                object: "chat.completion.chunk",
                created: self.created,
                model: self.model.clone(),
                choices: Vec::new(),
                usage: Some(openai_usage(&usage)),
            }));
        }
        self.done = true;
    }

    fn args_chunk(&self, arguments: String) -> String {
        self.chunk(
            ChunkDelta {
                tool_calls: Some(vec![ToolCallDelta {
                    index: self.tool_index,
                    id: None,
                    kind: None,
                    function: FunctionDelta {
                        name: None,
                        arguments: Some(arguments),
                    },
                }]),
                ..Default::default()
            },
            None,
        )
    }

    fn chunk(&self, delta: ChunkDelta, finish_reason: Option<&'static str>) -> String {
        to_json(Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
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

#[derive(Serialize)]
pub struct RenderedCompletion {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<RenderedChoice>,
    usage: OpenAiUsage,
}

#[derive(Serialize)]
pub struct RenderedChoice {
    index: i64,
    message: RenderedMessage,
    finish_reason: &'static str,
    logprobs: Option<()>,
}

#[derive(Serialize)]
pub struct RenderedMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<RenderedToolCall>>,
}

#[derive(Serialize)]
pub struct RenderedToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: RenderedFunction,
}

#[derive(Serialize)]
pub struct RenderedFunction {
    name: String,
    arguments: String,
}

pub fn render_aggregated(agg: &Aggregated, model: &str) -> RenderedCompletion {
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
                tool_calls.push(RenderedToolCall {
                    id: id.clone(),
                    kind: "function",
                    function: RenderedFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
        }
    }

    let finish_reason = match agg.stop {
        StopKind::ToolUse => "tool_calls",
        StopKind::MaxTokens => "length",
        _ => "stop",
    };
    RenderedCompletion {
        id: format!("chatcmpl-{}", agg.id),
        object: "chat.completion",
        created: crate::clock::unix_now(),
        model: model.to_string(),
        choices: vec![RenderedChoice {
            index: 0,
            message: RenderedMessage {
                role: "assistant",
                content: text,
                reasoning_content: Some(reasoning).filter(|r| !r.is_empty()),
                tool_calls: Some(tool_calls).filter(|t| !t.is_empty()),
            },
            finish_reason,
            logprobs: None,
        }],
        usage: openai_usage(&agg.usage),
    }
}
