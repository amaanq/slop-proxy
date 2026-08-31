use serde::Deserialize;
use serde_json::Value;

use super::{decode_signature, model_map};
use crate::codex::types::{
    ContentPart, InputItem, ReasoningConfig, ResponsesRequest, SummaryPart, ToolDef,
};
use crate::config::Config;

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    pub messages: Vec<AnthMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct AnthMessage {
    pub role: String,
    pub content: Value,
}

impl AnthropicRequest {
    pub fn thinking_enabled(&self) -> bool {
        self.thinking
            .as_ref()
            .and_then(|t| t.get("type"))
            .and_then(Value::as_str)
            .map(|t| t == "enabled" || t == "adaptive")
            .unwrap_or(false)
    }
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
            let Some(name) = t.get("name").and_then(Value::as_str) else {
                continue;
            };
            out.tools.push(ToolDef {
                kind: "function",
                name: name.into(),
                description: t
                    .get("description")
                    .and_then(Value::as_str)
                    .map(String::from),
                strict: false,
                parameters: t
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            });
        }
    }

    if let Some(tc) = &req.tool_choice {
        let kind = tc.get("type").and_then(Value::as_str).unwrap_or("auto");
        out.tool_choice = Some(match kind {
            "any" => Value::String("required".into()),
            "tool" => serde_json::json!({
                "type": "function",
                "name": tc.get("name").and_then(Value::as_str).unwrap_or_default(),
            }),
            "none" => Value::String("none".into()),
            _ => Value::String("auto".into()),
        });
        if tc.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
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
            t.get("budget_tokens").and_then(Value::as_u64).map(|b| {
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

fn system_text(system: &Value) -> String {
    match system {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn convert_message(msg: &AnthMessage, out: &mut Vec<InputItem>) -> Result<(), String> {
    let assistant = msg.role == "assistant";
    let role = if assistant { "assistant" } else { "user" };

    let blocks: Vec<Value> = match &msg.content {
        Value::String(s) => vec![serde_json::json!({"type": "text", "text": s})],
        Value::Array(a) => a.clone(),
        other => return Err(format!("unsupported message content: {other}")),
    };

    let mut parts: Vec<ContentPart> = Vec::new();
    let flush = |parts: &mut Vec<ContentPart>, out: &mut Vec<InputItem>| {
        if !parts.is_empty() {
            out.push(InputItem::Message {
                role: role.into(),
                content: std::mem::take(parts),
            });
        }
    };

    for block in blocks {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                parts.push(if assistant {
                    ContentPart::OutputText { text }
                } else {
                    ContentPart::InputText { text }
                });
            }
            "image" => {
                let source = block.get("source").cloned().unwrap_or_default();
                let url = match source.get("type").and_then(Value::as_str) {
                    Some("base64") => format!(
                        "data:{};base64,{}",
                        source
                            .get("media_type")
                            .and_then(Value::as_str)
                            .unwrap_or("image/png"),
                        source.get("data").and_then(Value::as_str).unwrap_or("")
                    ),
                    Some("url") => source
                        .get("url")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    _ => return Err("unsupported image source".into()),
                };
                parts.push(ContentPart::InputImage { image_url: url });
            }
            "tool_use" => {
                flush(&mut parts, out);
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or("tool_use block without id")?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("tool_use block without name")?;
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                out.push(InputItem::FunctionCall {
                    call_id: id.into(),
                    name: name.into(),
                    arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                });
            }
            "tool_result" => {
                flush(&mut parts, out);
                let call_id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .ok_or("tool_result block without tool_use_id")?;
                let mut output = tool_result_text(block.get("content"));
                if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                    output = format!("[tool error] {output}");
                }
                out.push(InputItem::FunctionCallOutput {
                    call_id: call_id.into(),
                    output,
                });
            }
            "thinking" => {
                let Some(sig) = block.get("signature").and_then(Value::as_str) else {
                    continue;
                };
                let (id, ec) = decode_signature(sig);
                if ec.is_none() {
                    continue;
                }
                flush(&mut parts, out);
                let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                out.push(InputItem::Reasoning {
                    id,
                    summary: if text.is_empty() {
                        vec![]
                    } else {
                        vec![SummaryPart::SummaryText { text: text.into() }]
                    },
                    encrypted_content: ec,
                });
            }
            "redacted_thinking" => {}
            other => {
                tracing::debug!("dropping unsupported anthropic block type {other:?}");
            }
        }
    }
    flush(&mut parts, out);
    Ok(())
}

fn tool_result_text(content: Option<&Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => b.get("text").and_then(Value::as_str).map(String::from),
                Some("image") => Some("[image omitted]".into()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
    }
}
