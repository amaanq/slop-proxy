use std::collections::BTreeSet;
use std::io::Cursor;

use data_encoding::BASE64;
use image_webp::WebPDecoder;
use png::{BitDepth, ColorType, Encoder};
use serde::Serialize;
use serde_json::value::RawValue;

use crate::codex::types::{ContentPart, InputItem, ResponsesRequest, ToolChoice, ToolDef};
use crate::gemini::signatures;
use crate::translate::chat::{
   ChatContent, ChatMessage, ChatPart, ChatRequest, ChatToolCall, ChatToolChoice, ChatToolDef,
   ExtraContent, FunctionBody, FunctionDef, ImageRef, StreamOptions,
};
use crate::translate::empty_schema;

/// One assistant turn becomes one message. Strict backends (zen's Console
/// upstream) 400 an assistant tool call that is not followed by its result,
/// and omp sends parallel calls and their narration as separate items.
fn push_tool_call(messages: &mut Vec<ChatMessage>, call_id: &str, name: &str, arguments: String) {
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

   if let Some(last) = messages.last_mut()
      && last.role == "assistant"
   {
      last.tool_calls.get_or_insert_default().push(call);
      return;
   }

   messages.push(ChatMessage {
      role: "assistant".into(),
      tool_calls: Some(vec![call]),
      ..Default::default()
   });
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

fn request_tools(req: &ResponsesRequest) -> impl Iterator<Item = &ToolDef> {
   req.tools.iter().chain(
      req.input
         .iter()
         .filter_map(|item| match *item {
            InputItem::AdditionalTools { ref tools, .. } => Some(tools.as_slice()),
            InputItem::Message { .. }
            | InputItem::FunctionCall { .. }
            | InputItem::FunctionCallOutput { .. }
            | InputItem::CustomToolCall { .. }
            | InputItem::CustomToolCallOutput { .. }
            | InputItem::Reasoning { .. }
            | InputItem::Other => None,
         })
         .flatten(),
   )
}

pub const FREEFORM_ARG: &str = "input";

#[derive(Serialize)]
struct Freeform<'a> {
   input: &'a str,
}

const FREEFORM_SCHEMA: &str = r#"{"properties":{"input":{"description":"The complete tool input, verbatim.","type":"string"}},"required":["input"],"type":"object"}"#;

fn freeform_schema() -> Box<RawValue> {
   RawValue::from_string(FREEFORM_SCHEMA.to_owned()).expect("schema is valid JSON")
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
         } if role == "assistant"
            && let Some(last) = messages.last_mut()
            && last.role == "assistant"
            && last.content.is_none() =>
         {
            last.content = Some(parts(content));
         },
         InputItem::Message {
            ref role,
            ref content,
         } => messages.push(ChatMessage {
            // Chat completions has no `developer` role.
            role: match role.as_str() {
               "developer" => "system",
               other => other,
            }
            .to_owned(),
            content: Some(parts(content)),
            ..Default::default()
         }),
         InputItem::FunctionCall {
            ref call_id,
            ref name,
            ref arguments,
         } => push_tool_call(&mut messages, call_id, name, arguments.clone()),
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
         } => push_tool_call(
            &mut messages,
            call_id,
            name,
            serde_json::to_string(&Freeform { input }).unwrap_or_default(),
         ),
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
         .map(|reasoning| clamped_effort(&req.model, &reasoning.effort).to_owned()),
      tools: (!tools.is_empty()).then_some(tools),
      tool_choice: req.tool_choice.as_ref().map(|choice| match *choice {
         ToolChoice::Mode(ref mode) => ChatToolChoice::Mode(mode.clone()),
         ToolChoice::Function { ref name, .. } => ChatToolChoice::function(name.clone()),
      }),
      ..Default::default()
   }
}

/// Gemini and zen both take none, low, medium or high and reject anything
/// else outright, so codex asking for xhigh would kill the whole turn.
/// fledge is the exception and 400s anything outside low, high or max.
pub fn clamped_effort<'effort>(model: &str, effort: &'effort str) -> &'effort str {
   if model.starts_with("fledge") {
      return match effort {
         "none" | "minimal" | "low" => "low",
         "medium" | "high" => "high",
         _ => "max",
      };
   }
   match effort {
      "none" | "minimal" => "none",
      "low" => "low",
      "medium" => "medium",
      _ => "high",
   }
}

#[derive(Debug, thiserror::Error)]
enum TranscodeError {
   #[error("base64")]
   Base64(#[source] data_encoding::DecodeError),
   #[error("webp decode")]
   Decode(#[source] image_webp::DecodingError),
   #[error("png encode")]
   Encode(#[source] png::EncodingError),
}

/// Zen's image sandbox 415s anything but PNG or JPEG.
fn accepted_image(url: &str) -> String {
   let Some(data) = url.strip_prefix("data:image/webp;base64,") else {
      return url.to_owned();
   };
   match webp_to_png(data) {
      Ok(png) => format!("data:image/png;base64,{png}"),
      Err(error) => {
         tracing::warn!(%error, "transcoding a webp image failed, sending it as is");
         url.to_owned()
      },
   }
}

fn webp_to_png(data: &str) -> Result<String, TranscodeError> {
   let bytes = BASE64.decode(data.as_bytes()).map_err(TranscodeError::Base64)?;
   let mut decoder = WebPDecoder::new(Cursor::new(bytes)).map_err(TranscodeError::Decode)?;
   let (width, height) = decoder.dimensions();
   let color = if decoder.has_alpha() {
      ColorType::Rgba
   } else {
      ColorType::Rgb
   };
   let mut pixels = vec![0; decoder.output_buffer_size().unwrap_or_default()];
   decoder.read_image(&mut pixels).map_err(TranscodeError::Decode)?;

   let mut out = Vec::new();
   let mut encoder = Encoder::new(&mut out, width, height);
   encoder.set_color(color);
   encoder.set_depth(BitDepth::Eight);
   let mut writer = encoder.write_header().map_err(TranscodeError::Encode)?;
   writer.write_image_data(&pixels).map_err(TranscodeError::Encode)?;
   writer.finish().map_err(TranscodeError::Encode)?;
   Ok(BASE64.encode(&out))
}

fn parts(content: &[ContentPart]) -> ChatContent {
   ChatContent::Parts(
      content
         .iter()
         .map(|part| match part {
            &ContentPart::InputImage { ref image_url } => ChatPart::ImageUrl {
               image_url: ImageRef::Object {
                  url: accepted_image(image_url),
               },
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
