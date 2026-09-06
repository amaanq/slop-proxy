use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponsesRequest {
   pub model: String,
   #[serde(skip_serializing_if = "String::is_empty")]
   pub instructions: String,
   #[serde(deserialize_with = "input_items")]
   pub input: Vec<InputItem>,
   #[serde(skip_serializing_if = "Vec::is_empty")]
   pub tools: Vec<ToolDef>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub tool_choice: Option<ToolChoice>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub parallel_tool_calls: Option<bool>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub reasoning: Option<ReasoningConfig>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub max_output_tokens: Option<u64>,
   pub store: bool,
   pub stream: bool,
   #[serde(skip_serializing_if = "Vec::is_empty")]
   pub include: Vec<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub prompt_cache_key: Option<String>,
   /// Codex sets this to `priority` for `/fast`.
   #[serde(skip_serializing_if = "Option::is_none")]
   pub service_tier: Option<String>,
}

fn empty_object() -> String {
   "{}".into()
}

fn function() -> String {
   "function".into()
}

impl ResponsesRequest {
   pub fn new(model: String, instructions: String) -> Self {
      Self {
         model,
         instructions,
         stream: true,
         include: vec!["reasoning.encrypted_content".into()],
         ..Self::default()
      }
   }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReasoningConfig {
   #[serde(skip_serializing_if = "String::is_empty")]
   pub effort: String,
   #[serde(skip_serializing_if = "String::is_empty")]
   pub summary: String,
}

/// The Responses API takes `input` as either a plain string or a list of
/// items, and an item omits `type` when it is a message.
fn input_items<'de, D>(deserializer: D) -> Result<Vec<InputItem>, D::Error>
where
   D: serde::Deserializer<'de>,
{
   #[derive(Deserialize)]
   #[serde(untagged)]
   enum Loose {
      Tagged(InputItem),
      Message {
         role: String,
         #[serde(default, deserialize_with = "content_parts")]
         content: Vec<ContentPart>,
      },
   }
   Ok(match StringOr::<Vec<Loose>>::deserialize(deserializer)? {
      StringOr::Text(text) => vec![InputItem::Message {
         role: "user".into(),
         content: vec![ContentPart::InputText { text }],
      }],
      StringOr::Items(items) => items
         .into_iter()
         .map(|item| match item {
            Loose::Tagged(item) => item,
            Loose::Message { role, content } => InputItem::Message { role, content },
         })
         .collect(),
   })
}

/// A message's content is a bare string as often as it is a list of parts.
fn content_parts<'de, D>(deserializer: D) -> Result<Vec<ContentPart>, D::Error>
where
   D: serde::Deserializer<'de>,
{
   Ok(match StringOr::deserialize(deserializer)? {
      StringOr::Text(text) => vec![ContentPart::InputText { text }],
      StringOr::Items(parts) => parts,
   })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOr<T> {
   Text(String),
   Items(T),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
   #[serde(rename = "message")]
   Message {
      role: String,
      #[serde(default, deserialize_with = "content_parts")]
      content: Vec<ContentPart>,
   },
   #[serde(rename = "function_call")]
   FunctionCall {
      call_id: String,
      name: String,
      #[serde(default = "empty_object")]
      arguments: String,
   },
   #[serde(rename = "function_call_output")]
   FunctionCallOutput {
      call_id: String,
      #[serde(default)]
      output: ToolOutput,
   },
   #[serde(rename = "custom_tool_call")]
   CustomToolCall {
      call_id: String,
      name: String,
      #[serde(default)]
      input: String,
   },
   #[serde(rename = "custom_tool_call_output")]
   CustomToolCallOutput {
      call_id: String,
      #[serde(default)]
      output: ToolOutput,
   },
   #[serde(rename = "reasoning")]
   Reasoning {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      id: Option<String>,
      #[serde(default)]
      summary: Vec<SummaryPart>,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      encrypted_content: Option<String>,
   },
   /// Codex carries its tool catalog as an input item rather than a top-level
   /// array (codex-rs/protocol/src/models.rs, `ResponseItem::AdditionalTools`).
   #[serde(rename = "additional_tools")]
   AdditionalTools {
      role: String,
      #[serde(default)]
      tools: Vec<ToolDef>,
   },
   #[serde(other)]
   Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
   Text(String),
   Parts(Vec<ContentPart>),
}

impl Default for ToolOutput {
   fn default() -> Self {
      Self::Text(String::new())
   }
}

impl ToolOutput {
   pub fn text(&self) -> String {
      match *self {
         Self::Text(ref text) => text.clone(),
         Self::Parts(ref parts) => parts
            .iter()
            .filter_map(|part| match *part {
               ContentPart::InputText { ref text } | ContentPart::OutputText { ref text } => {
                  Some(text.as_str())
               },
               ContentPart::InputImage { .. } | ContentPart::Other => None,
            })
            .collect::<Vec<_>>()
            .join(""),
      }
   }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
   #[serde(rename = "input_text")]
   InputText { text: String },
   #[serde(rename = "input_image")]
   InputImage { image_url: String },
   #[serde(rename = "output_text")]
   OutputText { text: String },
   #[serde(other)]
   Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SummaryPart {
   #[serde(rename = "summary_text")]
   SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
   Mode(String),
   Function {
      #[serde(rename = "type", default = "function")]
      kind: String,
      name: String,
   },
}

impl ToolChoice {
   pub fn function(name: String) -> Self {
      Self::Function {
         kind: "function".into(),
         name,
      }
   }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolDef {
   #[serde(rename = "type")]
   pub kind: String,
   pub name: String,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub description: Option<String>,
   pub strict: bool,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub parameters: Option<Box<RawValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesEvent {
   #[serde(rename = "response.created")]
   Created { response: ResponseObj },
   #[serde(rename = "response.in_progress")]
   InProgress,
   #[serde(rename = "response.output_item.added")]
   OutputItemAdded {
      #[serde(default)]
      output_index: u64,
      item: OutputItem,
   },
   #[serde(rename = "response.output_item.done")]
   OutputItemDone {
      #[serde(default)]
      output_index: u64,
      item: OutputItem,
   },
   #[serde(rename = "response.content_part.added")]
   ContentPartAdded {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      #[serde(default)]
      content_index: u64,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      part: Option<OutputContentPart>,
   },
   #[serde(rename = "response.content_part.done")]
   ContentPartDone,
   #[serde(rename = "response.output_text.delta")]
   OutputTextDelta {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      #[serde(default)]
      content_index: u64,
      delta: String,
   },
   #[serde(rename = "response.output_text.done")]
   OutputTextDone {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      #[serde(default)]
      content_index: u64,
      #[serde(default)]
      text: String,
   },
   #[serde(rename = "response.reasoning_summary_part.added")]
   ReasoningSummaryPartAdded {
      #[serde(default)]
      output_index: u64,
   },
   #[serde(rename = "response.reasoning_summary_part.done")]
   ReasoningSummaryPartDone,
   #[serde(rename = "response.reasoning_summary_text.delta")]
   ReasoningSummaryTextDelta {
      #[serde(default)]
      output_index: u64,
      delta: String,
   },
   #[serde(rename = "response.reasoning_summary_text.done")]
   ReasoningSummaryTextDone,
   #[serde(rename = "response.reasoning_text.delta")]
   ReasoningTextDelta {
      #[serde(default)]
      output_index: u64,
      delta: String,
   },
   #[serde(rename = "response.reasoning_text.done")]
   ReasoningTextDone,
   #[serde(rename = "response.function_call_arguments.delta")]
   FunctionCallArgumentsDelta {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      delta: String,
   },
   #[serde(rename = "response.function_call_arguments.done")]
   FunctionCallArgumentsDone {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      #[serde(default)]
      arguments: String,
   },
   #[serde(rename = "response.custom_tool_call_input.done")]
   CustomToolCallInputDone {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      item_id: Option<String>,
      #[serde(default)]
      output_index: u64,
      #[serde(default)]
      input: String,
   },
   #[serde(rename = "response.completed")]
   Completed { response: ResponseObj },
   #[serde(rename = "response.incomplete")]
   Incomplete { response: ResponseObj },
   #[serde(rename = "response.failed")]
   Failed { response: ResponseObj },
   #[serde(other)]
   Other,
}

impl ResponsesEvent {
   pub const fn terminal(&self) -> Option<(TerminalKind, &ResponseObj)> {
      match *self {
         Self::Completed { ref response } => Some((TerminalKind::Completed, response)),
         Self::Incomplete { ref response } => Some((TerminalKind::Incomplete, response)),
         Self::Failed { ref response } => Some((TerminalKind::Failed, response)),
         Self::Created { .. }
         | Self::InProgress
         | Self::OutputItemAdded { .. }
         | Self::OutputItemDone { .. }
         | Self::ContentPartAdded { .. }
         | Self::ContentPartDone
         | Self::OutputTextDelta { .. }
         | Self::OutputTextDone { .. }
         | Self::ReasoningSummaryPartAdded { .. }
         | Self::ReasoningSummaryPartDone
         | Self::ReasoningSummaryTextDelta { .. }
         | Self::ReasoningSummaryTextDone
         | Self::ReasoningTextDelta { .. }
         | Self::ReasoningTextDone
         | Self::FunctionCallArgumentsDelta { .. }
         | Self::FunctionCallArgumentsDone { .. }
         | Self::CustomToolCallInputDone { .. }
         | Self::Other => None,
      }
   }

   /// The wire name, which is also the SSE event name the stream carries.
   pub const fn kind(&self) -> &'static str {
      match *self {
         Self::Created { .. } => "response.created",
         Self::InProgress => "response.in_progress",
         Self::OutputItemAdded { .. } => "response.output_item.added",
         Self::OutputItemDone { .. } => "response.output_item.done",
         Self::ContentPartAdded { .. } => "response.content_part.added",
         Self::ContentPartDone => "response.content_part.done",
         Self::OutputTextDelta { .. } => "response.output_text.delta",
         Self::OutputTextDone { .. } => "response.output_text.done",
         Self::ReasoningSummaryPartAdded { .. } => "response.reasoning_summary_part.added",
         Self::ReasoningSummaryPartDone => "response.reasoning_summary_part.done",
         Self::ReasoningSummaryTextDelta { .. } => "response.reasoning_summary_text.delta",
         Self::ReasoningSummaryTextDone => "response.reasoning_summary_text.done",
         Self::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
         Self::ReasoningTextDone => "response.reasoning_text.done",
         Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
         Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
         Self::CustomToolCallInputDone { .. } => "response.custom_tool_call_input.done",
         Self::Completed { .. } => "response.completed",
         Self::Incomplete { .. } => "response.incomplete",
         Self::Failed { .. } => "response.failed",
         Self::Other => "message",
      }
   }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalKind {
   Completed,
   Incomplete,
   Failed,
}

impl TerminalKind {
   pub const fn as_str(self) -> &'static str {
      match self {
         Self::Completed => "completed",
         Self::Incomplete => "incomplete",
         Self::Failed => "failed",
      }
   }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseObj {
   #[serde(skip_serializing_if = "Option::is_none")]
   pub id: Option<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub status: Option<String>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub usage: Option<Usage>,
   #[serde(skip_serializing_if = "Option::is_none")]
   pub error: Option<UpstreamError>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UpstreamError {
   pub code: Option<String>,
   pub message: Option<String>,
}

/// A `ChatGPT` account refusing a model answers with a bare `detail`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ErrorEnvelope {
   pub detail: Option<String>,
   pub error: UpstreamError,
}

impl ErrorEnvelope {
   pub fn reason(body: String) -> String {
      match serde_json::from_str::<Self>(&body) {
         Ok(env) => env.detail.or(env.error.message).unwrap_or(body),
         Err(_) => body,
      }
   }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
   pub input_tokens: i64,
   pub output_tokens: i64,
   pub total_tokens: i64,
   pub input_tokens_details: TokenDetails,
   pub output_tokens_details: TokenDetails,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenDetails {
   pub cached_tokens: i64,
   pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
   #[serde(rename = "output_text")]
   OutputText { text: String },
   #[serde(other)]
   Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
   #[serde(rename = "message")]
   Message {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      id: Option<String>,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      role: Option<String>,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      status: Option<String>,
      #[serde(default)]
      content: Option<Vec<OutputContentPart>>,
   },
   #[serde(rename = "reasoning")]
   Reasoning {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      id: Option<String>,
      #[serde(default)]
      summary: Option<Vec<SummaryPart>>,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      encrypted_content: Option<String>,
   },
   #[serde(rename = "function_call")]
   FunctionCall {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      id: Option<String>,
      call_id: String,
      name: String,
      #[serde(default)]
      arguments: Option<String>,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      status: Option<String>,
   },
   #[serde(rename = "custom_tool_call")]
   CustomToolCall {
      #[serde(default, skip_serializing_if = "Option::is_none")]
      id: Option<String>,
      call_id: String,
      name: String,
      #[serde(default)]
      input: String,
      #[serde(default, skip_serializing_if = "Option::is_none")]
      status: Option<String>,
   },
   #[serde(other)]
   Other,
}

#[cfg(test)]
mod input_shape_tests {
   use super::*;

   fn parse(body: &str) -> ResponsesRequest {
      serde_json::from_str(body).expect("should type")
   }

   fn text_of(req: &ResponsesRequest) -> String {
      match req.input[0] {
         InputItem::Message { ref content, .. } => match content[0] {
            ContentPart::InputText { ref text } | ContentPart::OutputText { ref text } => {
               text.clone()
            },
            ContentPart::InputImage { .. } | ContentPart::Other => String::new(),
         },
         InputItem::FunctionCall { .. }
         | InputItem::FunctionCallOutput { .. }
         | InputItem::CustomToolCall { .. }
         | InputItem::CustomToolCallOutput { .. }
         | InputItem::Reasoning { .. }
         | InputItem::AdditionalTools { .. }
         | InputItem::Other => String::new(),
      }
   }

   /// Every spelling below is valid to the Responses API, and rejecting any
   /// of them turned a working request into "cannot be bridged to gemini".
   #[test]
   fn a_bare_string_input_becomes_a_user_message() {
      let req = parse(r#"{"model":"m","instructions":"i","input":"hi"}"#);
      assert_eq!(req.input.len(), 1);
      assert_eq!(text_of(&req), "hi");
   }

   #[test]
   fn a_bare_string_content_becomes_one_part() {
      let req = parse(
         r#"{"model":"m","instructions":"i","input":[{"type":"message","role":"user","content":"hi"}]}"#,
      );
      assert_eq!(text_of(&req), "hi");
   }

   #[test]
   fn an_item_without_a_type_is_a_message() {
      let req = parse(
         r#"{"model":"m","instructions":"i","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#,
      );
      assert_eq!(text_of(&req), "hi");
   }
}
