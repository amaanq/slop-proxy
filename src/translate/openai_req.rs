use serde::Deserialize;
use serde_json::Value;

use super::anthropic_req::empty_schema;
use super::model_map;
use crate::codex::types::{
    ContentPart, InputItem, ReasoningConfig, ResponsesRequest, ToolChoice, ToolDef,
};
use crate::config::Config;

#[derive(Debug, Deserialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub tools: Option<Vec<OpenAiToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<OpenAiToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiMessage {
    #[serde(default = "default_role")]
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

fn default_role() -> String {
    "user".into()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
    Other(Value),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    Text {
        #[serde(default)]
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    url: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiToolCall {
    id: String,
    #[serde(default)]
    function: FunctionBody,
}

#[derive(Debug, Default, Deserialize)]
pub struct FunctionBody {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiToolDef {
    #[serde(default)]
    function: Option<FunctionDef>,
    #[serde(flatten)]
    flat: FunctionDef,
}

#[derive(Debug, Default, Deserialize)]
pub struct FunctionDef {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenAiToolChoice {
    Mode(String),
    Named {
        #[serde(default)]
        function: Option<NamedFunction>,
    },
}

#[derive(Debug, Deserialize)]
pub struct NamedFunction {
    #[serde(default)]
    name: Option<String>,
}

impl OpenAiRequest {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false)
    }
}

pub fn to_responses(req: &OpenAiRequest, cfg: &Config) -> Result<ResponsesRequest, String> {
    let resolved = model_map::resolve(&cfg.models, &req.model);
    let mut out = ResponsesRequest::new(resolved.model.clone(), cfg.codex.instructions());

    for msg in &req.messages {
        convert_message(msg, &mut out.input)?;
    }

    if let Some(tools) = &req.tools {
        for t in tools {
            let f = t.function.as_ref().unwrap_or(&t.flat);
            let Some(name) = &f.name else {
                continue;
            };
            out.tools.push(ToolDef {
                kind: "function",
                name: name.clone(),
                description: f.description.clone(),
                strict: false,
                parameters: f.parameters.clone().unwrap_or_else(empty_schema),
            });
        }
    }

    if let Some(tc) = &req.tool_choice {
        out.tool_choice = Some(match tc {
            OpenAiToolChoice::Mode(s) => ToolChoice::Mode(s.clone()),
            OpenAiToolChoice::Named { function } => ToolChoice::function(
                function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default(),
            ),
        });
    }
    out.parallel_tool_calls = req.parallel_tool_calls;

    let effort = req
        .reasoning_effort
        .clone()
        .or(resolved.effort)
        .unwrap_or_else(|| "medium".into());
    out.reasoning = Some(ReasoningConfig {
        effort: model_map::clamp_effort(&out.model, &effort),
        summary: "auto".into(),
    });

    if cfg.codex.forward_max_tokens {
        out.max_output_tokens = req.max_completion_tokens.or(req.max_tokens);
    }

    Ok(out)
}

fn convert_message(msg: &OpenAiMessage, out: &mut Vec<InputItem>) -> Result<(), String> {
    match msg.role.as_str() {
        "system" | "developer" => {
            let text = content_text(msg.content.as_ref());
            if !text.is_empty() {
                out.push(InputItem::Message {
                    role: "developer".into(),
                    content: vec![ContentPart::InputText { text }],
                });
            }
        }
        "user" => {
            let parts = user_parts(msg.content.as_ref())?;
            if !parts.is_empty() {
                out.push(InputItem::Message {
                    role: "user".into(),
                    content: parts,
                });
            }
        }
        "assistant" => {
            let text = content_text(msg.content.as_ref());
            if !text.is_empty() {
                out.push(InputItem::Message {
                    role: "assistant".into(),
                    content: vec![ContentPart::OutputText { text }],
                });
            }
            if let Some(calls) = &msg.tool_calls {
                for call in calls {
                    out.push(InputItem::FunctionCall {
                        call_id: call.id.clone(),
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone().unwrap_or_else(|| "{}".into()),
                    });
                }
            }
        }
        "tool" => {
            out.push(InputItem::FunctionCallOutput {
                call_id: msg
                    .tool_call_id
                    .clone()
                    .ok_or("tool message without tool_call_id")?,
                output: content_text(msg.content.as_ref()),
            });
        }
        other => return Err(format!("unsupported message role {other:?}")),
    }
    Ok(())
}

fn content_text(content: Option<&MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                MessagePart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn user_parts(content: Option<&MessageContent>) -> Result<Vec<ContentPart>, String> {
    match content {
        Some(MessageContent::Text(s)) => Ok(vec![ContentPart::InputText { text: s.clone() }]),
        Some(MessageContent::Parts(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                match p {
                    MessagePart::Text { text } => out.push(ContentPart::InputText {
                        text: text.clone(),
                    }),
                    MessagePart::ImageUrl { image_url } => out.push(ContentPart::InputImage {
                        image_url: image_url.url.clone(),
                    }),
                    MessagePart::Other => {
                        tracing::debug!("dropping unsupported openai part");
                    }
                }
            }
            Ok(out)
        }
        None => Ok(Vec::new()),
        Some(MessageContent::Other(other)) => Err(format!("unsupported user content: {other}")),
    }
}
