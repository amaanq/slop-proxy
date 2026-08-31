use serde::Deserialize;
use serde_json::Value;

use super::model_map;
use crate::codex::types::{ContentPart, InputItem, ReasoningConfig, ResponsesRequest, ToolDef};
use crate::config::Config;

#[derive(Debug, Deserialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
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
    pub stream_options: Option<Value>,
}

impl OpenAiRequest {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
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
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(Value::as_str) else {
                continue;
            };
            out.tools.push(ToolDef {
                kind: "function",
                name: name.into(),
                description: f
                    .get("description")
                    .and_then(Value::as_str)
                    .map(String::from),
                strict: false,
                parameters: f
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            });
        }
    }

    if let Some(tc) = &req.tool_choice {
        out.tool_choice = Some(match tc {
            Value::String(s) => Value::String(s.clone()),
            obj => serde_json::json!({
                "type": "function",
                "name": obj
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            }),
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

fn convert_message(msg: &Value, out: &mut Vec<InputItem>) -> Result<(), String> {
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = msg.get("content");

    match role {
        "system" | "developer" => {
            let text = content_text(content);
            if !text.is_empty() {
                out.push(InputItem::Message {
                    role: "developer".into(),
                    content: vec![ContentPart::InputText { text }],
                });
            }
        }
        "user" => {
            let parts = user_parts(content)?;
            if !parts.is_empty() {
                out.push(InputItem::Message {
                    role: "user".into(),
                    content: parts,
                });
            }
        }
        "assistant" => {
            let text = content_text(content);
            if !text.is_empty() {
                out.push(InputItem::Message {
                    role: "assistant".into(),
                    content: vec![ContentPart::OutputText { text }],
                });
            }
            if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let f = call.get("function").cloned().unwrap_or_default();
                    out.push(InputItem::FunctionCall {
                        call_id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .ok_or("tool_call without id")?
                            .into(),
                        name: f
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .into(),
                        arguments: f
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}")
                            .into(),
                    });
                }
            }
        }
        "tool" => {
            out.push(InputItem::FunctionCallOutput {
                call_id: msg
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or("tool message without tool_call_id")?
                    .into(),
                output: content_text(content),
            });
        }
        other => return Err(format!("unsupported message role {other:?}")),
    }
    Ok(())
}

fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn user_parts(content: Option<&Value>) -> Result<Vec<ContentPart>, String> {
    match content {
        Some(Value::String(s)) => Ok(vec![ContentPart::InputText { text: s.clone() }]),
        Some(Value::Array(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => out.push(ContentPart::InputText {
                        text: p.get("text").and_then(Value::as_str).unwrap_or("").into(),
                    }),
                    "image_url" => out.push(ContentPart::InputImage {
                        image_url: p
                            .get("image_url")
                            .and_then(|i| i.get("url"))
                            .and_then(Value::as_str)
                            .ok_or("image_url part without url")?
                            .into(),
                    }),
                    other => tracing::debug!("dropping unsupported openai part {other:?}"),
                }
            }
            Ok(out)
        }
        None => Ok(Vec::new()),
        Some(other) => Err(format!("unsupported user content: {other}")),
    }
}
