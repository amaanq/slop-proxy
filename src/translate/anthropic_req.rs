use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{decode_signature, model_map};
use crate::codex::types::{
    ContentPart, InputItem, ReasoningConfig, ResponsesRequest, SummaryPart, ToolChoice, ToolDef,
    ToolOutput,
};
use crate::config::Config;

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    pub messages: Vec<AnthMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub tools: Option<Vec<AnthToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<AnthToolChoice>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct AnthMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    RedactedThinking {},
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        #[serde(default = "default_media_type")]
        media_type: String,
        #[serde(default)]
        data: String,
    },
    Url {
        #[serde(default)]
        url: String,
    },
}

fn default_media_type() -> String {
    "image/png".into()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
    Other(Value),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Image {},
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
    Other(serde::de::IgnoredAny),
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthToolDef {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthToolChoice {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    disable_parallel_tool_use: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub budget_tokens: Option<u64>,
}

impl AnthropicRequest {
    pub fn thinking_enabled(&self) -> bool {
        self.thinking
            .as_ref()
            .and_then(|t| t.kind.as_deref())
            .map(|t| t == "enabled" || t == "adaptive")
            .unwrap_or(false)
    }
}

#[derive(Serialize)]
pub struct ObjectSchema {
    #[serde(rename = "type")]
    kind: &'static str,
    properties: BTreeMap<&'static str, PropertySchema>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    required: Vec<&'static str>,
}

#[derive(Serialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'static str>,
}

impl ObjectSchema {
    pub fn empty() -> Self {
        Self {
            kind: "object",
            properties: BTreeMap::new(),
            required: Vec::new(),
        }
    }

    pub fn one_string(name: &'static str, description: &'static str) -> Self {
        Self {
            kind: "object",
            properties: BTreeMap::from([(
                name,
                PropertySchema {
                    kind: "string",
                    description: Some(description),
                },
            )]),
            required: vec![name],
        }
    }
}

pub fn empty_schema() -> Value {
    serde_json::to_value(ObjectSchema::empty()).expect("schema serializes")
}

pub fn to_responses(req: &AnthropicRequest, cfg: &Config) -> Result<ResponsesRequest, String> {
    let resolved = model_map::resolve(&cfg.models, &req.model);
    let mut out = ResponsesRequest::new(resolved.model.clone(), cfg.codex.instructions());

    if let Some(system) = &req.system {
        let text = system_text(system);
        if !text.is_empty() {
            out.input.push(InputItem::Message {
                role: "developer".into(),
                content: vec![ContentPart::InputText { text }],
            });
        }
    }

    for msg in &req.messages {
        convert_message(msg, &mut out.input)?;
    }

    if let Some(tools) = &req.tools {
        for t in tools {
            let Some(name) = &t.name else {
                continue;
            };
            out.tools.push(ToolDef {
                kind: "function".into(),
                name: name.clone(),
                description: t.description.clone(),
                strict: false,
                parameters: Some(t.input_schema.clone().unwrap_or_else(empty_schema)),
            });
        }
    }

    if let Some(tc) = &req.tool_choice {
        out.tool_choice = Some(match tc.kind.as_deref().unwrap_or("auto") {
            "any" => ToolChoice::Mode("required".into()),
            "tool" => ToolChoice::function(tc.name.clone().unwrap_or_default()),
            "none" => ToolChoice::Mode("none".into()),
            _ => ToolChoice::Mode("auto".into()),
        });
        if tc.disable_parallel_tool_use == Some(true) {
            out.parallel_tool_calls = Some(false);
        }
    }

    let effort = req
        .thinking
        .as_ref()
        .and_then(|t| {
            if !req.thinking_enabled() {
                return Some("low".to_string());
            }
            t.budget_tokens.map(|b| {
                if b < 4096 {
                    "low".to_string()
                } else if b < 16384 {
                    "medium".to_string()
                } else {
                    "high".to_string()
                }
            })
        })
        .or(resolved.effort)
        .unwrap_or_else(|| "medium".into());
    out.reasoning = Some(ReasoningConfig {
        effort: model_map::clamp_effort(&out.model, &effort),
        summary: "auto".into(),
    });

    if cfg.codex.forward_max_tokens {
        out.max_output_tokens = req.max_tokens;
    }

    Ok(out)
}

fn system_text(system: &SystemPrompt) -> String {
    match system {
        SystemPrompt::Text(s) => s.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n"),
        SystemPrompt::Other(_) => String::new(),
    }
}

fn convert_message(msg: &AnthMessage, out: &mut Vec<InputItem>) -> Result<(), String> {
    let assistant = msg.role == "assistant";
    let role = if assistant { "assistant" } else { "user" };

    let text_block;
    let blocks = match &msg.content {
        MessageContent::Text(s) => {
            text_block = [ContentBlock::Text { text: s.clone() }];
            &text_block[..]
        }
        MessageContent::Blocks(b) => b.as_slice(),
    };

    let mut parts = Vec::<ContentPart>::new();
    let flush = |parts: &mut Vec<ContentPart>, out: &mut Vec<InputItem>| {
        if !parts.is_empty() {
            out.push(InputItem::Message {
                role: role.into(),
                content: std::mem::take(parts),
            });
        }
    };

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                let text = text.clone();
                parts.push(if assistant {
                    ContentPart::OutputText { text }
                } else {
                    ContentPart::InputText { text }
                });
            }
            ContentBlock::Image { source } => {
                let url = match source {
                    ImageSource::Base64 { media_type, data } => {
                        format!("data:{media_type};base64,{data}")
                    }
                    ImageSource::Url { url } => url.clone(),
                };
                parts.push(ContentPart::InputImage { image_url: url });
            }
            ContentBlock::ToolUse { id, name, input } => {
                flush(&mut parts, out);
                out.push(InputItem::FunctionCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                });
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                flush(&mut parts, out);
                let mut output = tool_result_text(content.as_ref());
                if *is_error == Some(true) {
                    output = format!("[tool error] {output}");
                }
                out.push(InputItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output: ToolOutput::Text(output),
                });
            }
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                let Some(sig) = signature else {
                    continue;
                };
                let (id, ec) = decode_signature(sig);
                if ec.is_none() {
                    continue;
                }
                flush(&mut parts, out);
                out.push(InputItem::Reasoning {
                    id,
                    summary: if thinking.is_empty() {
                        vec![]
                    } else {
                        vec![SummaryPart::SummaryText {
                            text: thinking.clone(),
                        }]
                    },
                    encrypted_content: ec,
                });
            }
            ContentBlock::RedactedThinking {} => {}
            ContentBlock::Other => {
                tracing::debug!("dropping unsupported anthropic block");
            }
        }
    }
    flush(&mut parts, out);
    Ok(())
}

fn tool_result_text(content: Option<&ToolResultContent>) -> String {
    match content {
        None => String::new(),
        Some(ToolResultContent::Text(s)) => s.clone(),
        Some(ToolResultContent::Blocks(blocks)) => blocks
            .iter()
            .filter_map(|b| match b {
                ToolResultBlock::Text { text } => Some(text.clone()),
                ToolResultBlock::Image {} => Some("[image omitted]".into()),
                ToolResultBlock::Other => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(ToolResultContent::Other(other)) => other.to_string(),
    }
}
