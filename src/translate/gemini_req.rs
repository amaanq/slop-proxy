use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::value::{RawValue, to_raw_value};

use super::anthropic_req::empty_schema;
use super::chat::{
   ChatContent, ChatMessage, ChatPart, ChatRequest, ChatToolCall, ChatToolChoice, ChatToolDef,
   ExtraContent, FunctionBody, FunctionDef, ImageRef, StreamOptions,
};
use crate::codex::types::{ContentPart, InputItem, ResponsesRequest, ToolChoice};
use crate::gemini::signatures;

fn tool_call_message(call_id: &str, name: &str, arguments: String) -> ChatMessage {
   let call = ChatToolCall {
      id: Some(call_id.to_owned()),
      kind: Some("function".into()),
      function: FunctionBody {
         name: Some(name.to_owned()),
         arguments: Some(arguments),
      },
      extra_content: signatures::get(call_id)
         .as_deref()
         .map(ExtraContent::with_signature),
      ..Default::default()
   };
   ChatMessage {
      role: "assistant".into(),
      tool_calls: Some(vec![call]),
      ..Default::default()
   }
}

/// Codex's shell tool is a grammar-constrained freeform tool taking raw text.
/// Chat completions has no way to say that, so those are offered as a function
/// over a single string and their calls are turned back on the way out.
pub fn custom_tools(req: &ResponsesRequest) -> BTreeSet<String> {
   request_tools(req)
      .filter(|tool| tool.kind == "custom")
      .map(|tool| tool.name.clone())
      .collect()
}

fn request_tools(req: &ResponsesRequest) -> impl Iterator<Item = &crate::codex::types::ToolDef> {
   req.tools.iter().chain(
      req.input
         .iter()
         .filter_map(|item| match item {
            InputItem::AdditionalTools { tools, .. } => Some(tools.as_slice()),
            _ => None,
         })
         .flatten(),
   )
}

/// The single argument a custom tool is presented as taking.
pub(super) const FREEFORM_ARG: &str = "input";

#[derive(Serialize)]
struct Freeform<'a> {
   input: &'a str,
}

fn freeform_schema() -> Box<RawValue> {
   to_raw_value(&serde_json::json!({
      "type": "object",
      "properties": {"input": {"type": "string", "description": "The complete tool input, verbatim."}},
      "required": ["input"],
   }))
   .expect("schema serializes")
}

pub fn to_chat(req: &ResponsesRequest) -> ChatRequest {
   let mut messages = vec![ChatMessage {
      role: "system".into(),
      content: Some(ChatContent::Text(req.instructions.clone())),
      ..Default::default()
   }];
   for item in &req.input {
      match *item {
         InputItem::Message {
            ref role,
            ref content,
         } => messages.push(ChatMessage {
            role: chat_role(role).to_owned(),
            content: Some(parts(content)),
            ..Default::default()
         }),
         InputItem::FunctionCall {
            ref call_id,
            ref name,
            ref arguments,
         } => messages.push(tool_call_message(call_id, name, arguments.clone())),
         InputItem::FunctionCallOutput {
            ref call_id,
            ref output,
         }
         | InputItem::CustomToolCallOutput {
            ref call_id,
            ref output,
         } => messages.push(ChatMessage {
            role: "tool".into(),
            tool_call_id: Some(call_id.clone()),
            content: Some(ChatContent::Text(output.text())),
            ..Default::default()
         }),
         InputItem::CustomToolCall {
            ref call_id,
            ref name,
            ref input,
         } => messages.push(tool_call_message(
            call_id,
            name,
            serde_json::to_string(&Freeform { input }).unwrap_or_default(),
         )),
         // Gemini rejects an unknown role rather than ignoring it.
         InputItem::Reasoning { .. } | InputItem::AdditionalTools { .. } | InputItem::Other => {},
      }
   }

   let tools: Vec<ChatToolDef> = request_tools(req)
      .filter(|tool| !tool.name.is_empty())
      .map(|tool| {
         let parameters = if tool.kind == "custom" {
            freeform_schema()
         } else {
            tool.parameters.clone().unwrap_or_else(empty_schema)
         };
         ChatToolDef::function(FunctionDef {
            name: Some(tool.name.clone()),
            description: tool.description.clone(),
            parameters: Some(parameters),
            strict: None,
         })
      })
      .collect();

   ChatRequest {
      model: req.model.clone(),
      messages,
      stream: Some(true),
      // Without this the terminal chunk carries no usage and the request
      // bills as zero tokens.
      stream_options: Some(StreamOptions {
         include_usage: true,
      }),
      max_tokens: req.max_output_tokens,
      reasoning_effort: req
         .reasoning
         .as_ref()
         .filter(|reasoning| !reasoning.effort.is_empty())
         .map(|reasoning| gemini_effort(&reasoning.effort).to_owned()),
      tools: (!tools.is_empty()).then_some(tools),
      tool_choice: req.tool_choice.as_ref().map(|choice| match *choice {
         ToolChoice::Mode(ref mode) => ChatToolChoice::Mode(mode.clone()),
         ToolChoice::Function { ref name, .. } => ChatToolChoice::function(name.clone()),
      }),
      ..Default::default()
   }
}

/// Gemini takes none, low, medium or high and rejects anything else outright,
/// so codex asking for xhigh would 400 the whole turn.
pub fn gemini_effort(effort: &str) -> &str {
   match effort {
      "none" | "minimal" => "none",
      "low" => "low",
      "medium" => "medium",
      _ => "high",
   }
}

/// Chat completions has no `developer` role.
fn chat_role(role: &str) -> &str {
   match role {
      "developer" => "system",
      other => other,
   }
}

fn parts(content: &[ContentPart]) -> ChatContent {
   ChatContent::Parts(
      content
         .iter()
         .map(|part| match part {
            &ContentPart::InputImage { ref image_url } => ChatPart::ImageUrl {
               image_url: ImageRef::Url(image_url.clone()),
            },
            &ContentPart::InputText { ref text } | &ContentPart::OutputText { ref text } => {
               ChatPart::Text { text: text.clone() }
            },
            &ContentPart::Other => ChatPart::Text {
               text: String::new(),
            },
         })
         .collect(),
   )
}
