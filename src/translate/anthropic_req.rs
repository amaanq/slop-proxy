use std::collections::BTreeMap;
use std::mem;

use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};
use serde_json::value::{RawValue, to_raw_value};

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
   Empty,
}

/// A `null` where the API documents a string, which the API itself accepts.
fn nullable<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
   D: serde::Deserializer<'de>,
   T: Default + Deserialize<'de>,
{
   Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
   Text {
      #[serde(default, deserialize_with = "nullable")]
      text: String,
   },
   Image {
      #[serde(default)]
      source: Option<ImageSource>,
   },
   ToolUse {
      id: String,
      name: String,
      #[serde(default, deserialize_with = "crate::translate::buffered_raw")]
      input: Option<Box<RawValue>>,
   },
   ToolResult {
      tool_use_id: String,
      #[serde(default)]
      content: Option<ToolResultContent>,
      #[serde(default)]
      is_error: Option<bool>,
   },
   Thinking {
      #[serde(default, deserialize_with = "nullable")]
      thinking: String,
      #[serde(default)]
      signature: Option<String>,
   },
   RedactedThinking,
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
   #[serde(other)]
   Other,
}

fn default_media_type() -> String {
   "image/png".into()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
   Text(String),
   Blocks(Vec<ToolResultBlock>),
   Other(serde_json::Value),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
   Text {
      #[serde(default)]
      text: String,
   },
   Image,
   #[serde(other)]
   Other,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
   Text(String),
   Blocks(Vec<SystemBlock>),
   Other(IgnoredAny),
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
   input_schema: Option<Box<RawValue>>,
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
      self
         .thinking
         .as_ref()
         .and_then(|thinking| thinking.kind.as_deref())
         .is_some_and(|kind| kind == "enabled" || kind == "adaptive")
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
   pub const fn empty() -> Self {
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

pub fn empty_schema() -> Box<RawValue> {
   to_raw_value(&ObjectSchema::empty()).expect("schema serializes")
}

pub fn to_responses(req: &AnthropicRequest, cfg: &Config) -> ResponsesRequest {
   let resolved = model_map::resolve(&cfg.models, &req.model);
   let mut out = ResponsesRequest::new(resolved.model.clone(), cfg.codex.instructions());

   if let Some(system) = req.system.as_ref() {
      let text = system_text(system);
      if !text.is_empty() {
         out.input.push(InputItem::Message {
            role: "developer".into(),
            content: vec![ContentPart::InputText { text }],
         });
      }
   }

   for msg in &req.messages {
      convert_message(msg, &mut out.input);
   }

   if let Some(tools) = req.tools.as_ref() {
      for tool in tools {
         let Some(name) = tool.name.as_ref() else {
            continue;
         };
         out.tools.push(ToolDef {
            kind: "function".into(),
            name: name.clone(),
            description: tool.description.clone(),
            strict: false,
            parameters: Some(tool.input_schema.clone().unwrap_or_else(empty_schema)),
         });
      }
   }

   if let Some(tool_choice) = req.tool_choice.as_ref() {
      out.tool_choice = Some(match tool_choice.kind.as_deref().unwrap_or("auto") {
         "any" => ToolChoice::Mode("required".into()),
         "tool" => ToolChoice::function(tool_choice.name.clone().unwrap_or_default()),
         "none" => ToolChoice::Mode("none".into()),
         _ => ToolChoice::Mode("auto".into()),
      });
      if tool_choice.disable_parallel_tool_use == Some(true) {
         out.parallel_tool_calls = Some(false);
      }
   }

   let effort = req
      .thinking
      .as_ref()
      .and_then(|thinking| {
         if !req.thinking_enabled() {
            return Some("low".to_owned());
         }
         thinking.budget_tokens.map(|budget| {
            if budget < 4096 {
               "low".to_owned()
            } else if budget < 0x4000 {
               "medium".to_owned()
            } else {
               "high".to_owned()
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

   out
}

fn system_text(system: &SystemPrompt) -> String {
   match *system {
      SystemPrompt::Text(ref text) => text.clone(),
      SystemPrompt::Blocks(ref blocks) => blocks
         .iter()
         .filter_map(|block| block.text.as_deref())
         .collect::<Vec<_>>()
         .join("\n\n"),
      SystemPrompt::Other(_) => String::new(),
   }
}

fn convert_message(msg: &AnthMessage, out: &mut Vec<InputItem>) {
   let assistant = msg.role == "assistant";
   let role = if assistant { "assistant" } else { "user" };

   let text_block;
   let blocks = match msg.content {
      MessageContent::Text(ref text) => {
         text_block = [ContentBlock::Text { text: text.clone() }];
         &text_block[..]
      },
      MessageContent::Blocks(ref blocks) => blocks.as_slice(),
      MessageContent::Empty => &[],
   };

   let mut parts = Vec::<ContentPart>::new();
   let flush = |buffer: &mut Vec<ContentPart>, dst: &mut Vec<InputItem>| {
      if !buffer.is_empty() {
         dst.push(InputItem::Message {
            role: role.into(),
            content: mem::take(buffer),
         });
      }
   };

   for block in blocks {
      match *block {
         ContentBlock::Text { ref text } => {
            let text = text.clone();
            parts.push(if assistant {
               ContentPart::OutputText { text }
            } else {
               ContentPart::InputText { text }
            });
         },
         ContentBlock::Image { ref source } => {
            let url = match *source {
               Some(ImageSource::Base64 {
                  ref media_type,
                  ref data,
               }) => {
                  format!("data:{media_type};base64,{data}")
               },
               Some(ImageSource::Url { ref url }) => url.clone(),
               Some(ImageSource::Other) | None => continue,
            };
            parts.push(ContentPart::InputImage { image_url: url });
         },
         ContentBlock::ToolUse {
            ref id,
            ref name,
            ref input,
         } => {
            flush(&mut parts, out);
            out.push(InputItem::FunctionCall {
               call_id: id.clone(),
               name: name.clone(),
               arguments: input.as_ref().map_or("{}", |input| input.get()).to_owned(),
            });
         },
         ContentBlock::ToolResult {
            ref tool_use_id,
            ref content,
            ref is_error,
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
         },
         ContentBlock::Thinking {
            ref thinking,
            ref signature,
         } => {
            let Some(sig) = signature.as_ref() else {
               continue;
            };
            let (id, encrypted_content) = decode_signature(sig);
            if encrypted_content.is_none() {
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
               encrypted_content,
            });
         },
         ContentBlock::RedactedThinking => {},
         ContentBlock::Other => {
            tracing::debug!("dropping unsupported anthropic block");
         },
      }
   }
   flush(&mut parts, out);
}

fn tool_result_text(content: Option<&ToolResultContent>) -> String {
   match content {
      None => String::new(),
      Some(&ToolResultContent::Text(ref text)) => text.clone(),
      Some(&ToolResultContent::Blocks(ref blocks)) => blocks
         .iter()
         .filter_map(|block| match *block {
            ToolResultBlock::Text { ref text } => Some(text.clone()),
            ToolResultBlock::Image => Some("[image omitted]".into()),
            ToolResultBlock::Other => None,
         })
         .collect::<Vec<_>>()
         .join("\n"),
      Some(&ToolResultContent::Other(ref other)) => other.to_string(),
   }
}

#[cfg(test)]
mod tests {
   use super::AnthropicRequest;

   fn parse(content: &serde_json::Value) -> Result<AnthropicRequest, serde_json::Error> {
      serde_json::from_value(serde_json::json!({
          "model": "m", "max_tokens": 1,
          "messages": [{"role": "user", "content": content}]
      }))
   }

   /// A single block the API tolerates used to sink the whole request with
   /// "did not match any variant of untagged enum `MessageContent`".
   #[test]
   fn a_tolerated_block_does_not_sink_the_request() {
      for content in [
         serde_json::Value::Null,
         serde_json::json!([{"type": "image", "source": {"type": "file", "file_id": "f"}}]),
         serde_json::json!([{"type": "image"}]),
         serde_json::json!([{"type": "text", "text": null}]),
         serde_json::json!([{"type": "thinking", "thinking": null, "signature": "s"}]),
      ] {
         parse(&content).unwrap_or_else(|err| panic!("{content}: {err}"));
      }
   }
}

#[cfg(test)]
mod buffered_raw_tests {
   use super::*;

   #[test]
   fn a_replayed_tool_call_keeps_its_input() {
      let block = serde_json::from_str::<ContentBlock>(
         r#"{"id":"c1","input":{"file_path":"/x"},"name":"Read","type":"tool_use"}"#,
      )
      .expect("tool_use with an object input must parse");
      match block {
         ContentBlock::ToolUse { input, .. } => {
            assert_eq!(input.unwrap().get(), r#"{"file_path":"/x"}"#);
         },
         other => panic!("wrong variant: {other:?}"),
      }
   }

   #[test]
   fn a_tool_result_that_is_neither_text_nor_blocks_survives() {
      let block = serde_json::from_str::<ContentBlock>(
         r#"{"type":"tool_result","tool_use_id":"t","content":{"a":1}}"#,
      )
      .expect("an unmodelled tool_result body must not sink the request");
      assert!(matches!(block, ContentBlock::ToolResult { .. }));
   }
}
