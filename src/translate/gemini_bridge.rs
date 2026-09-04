//! Claude Code only speaks the messages API and every Gemini surface speaks
//! chat completions.

use std::collections::{BTreeSet, HashMap};
use std::io::Error;
use std::mem;

use futures_util::stream;
use serde::Serialize;
use serde_json::value::{RawValue, to_raw_value};

use crate::codex::sse::EventStream;
use crate::codex::types::{
   ContentPart, InputItem, OutputContentPart, OutputItem, ResponseObj, ResponsesEvent,
   ResponsesRequest, ToolChoice, UpstreamError, Usage,
};
use crate::gemini::client::GeminiProtocol;
use crate::gemini::native::NativeStream;
use crate::gemini::signatures;
use crate::gemini::sse::Frames;
use crate::gemini::types::ApiError;
use crate::translate::UsageCapture;
use crate::translate::anthropic_req::{ObjectSchema, empty_schema};
use crate::translate::chat::{
   ChatChunk, ChatContent, ChatError, ChatErrorBody, ChatMessage, ChatPart, ChatRequest,
   ChatToolCall, ChatToolChoice, ChatToolDef, ErrorCode, ExtraContent, FunctionBody, FunctionDef,
   ImageRef, StreamOptions,
};

fn remember_signature(call_id: &str, signature: &str) {
   signatures::put(signatures::call_id_key(call_id), signature);
}

fn signature_for(call_id: &str) -> Option<String> {
   signatures::get(signatures::call_id_key(call_id))
}

fn tool_call_message(call_id: &str, name: &str, arguments: String) -> ChatMessage {
   let call = ChatToolCall {
      id: Some(call_id.to_owned()),
      kind: Some("function".into()),
      function: FunctionBody {
         name: Some(name.to_owned()),
         arguments: Some(arguments),
      },
      extra_content: signature_for(call_id)
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
   req.tools
      .iter()
      .filter(|tool| tool.kind == "custom")
      .map(|tool| tool.name.clone())
      .collect()
}

/// The single argument a custom tool is presented as taking.
const FREEFORM_ARG: &str = "input";

#[derive(Serialize)]
struct Freeform<'a> {
   input: &'a str,
}

fn freeform_schema() -> Box<RawValue> {
   to_raw_value(&ObjectSchema::one_string(
      FREEFORM_ARG,
      "The complete tool input, verbatim.",
   ))
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

   let tools: Vec<ChatToolDef> = req
      .tools
      .iter()
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

/// Tool calls arrive spread across chunks keyed by index, so a slot holds the
/// name until there is enough to open an item.
#[derive(Default)]
pub struct ChatToResponses {
   custom: BTreeSet<String>,
   opened: bool,
   calls: Vec<OpenCall>,
   text: String,
   msg_id: Option<String>,
   next_index: u64,
   finished: bool,
   usage: Option<Usage>,
   completed: bool,
   frames: usize,
   emitted: usize,
   first_frame: Option<String>,
}

/// A stream that produced nothing is the failure this path keeps hitting, and
/// the frame count separates nothing arriving from nothing being understood.
impl Drop for ChatToResponses {
   fn drop(&mut self) {
      if self.emitted == 0 {
         tracing::warn!(
            frames = self.frames,
            first = self.first_frame.as_deref().unwrap_or("<none>"),
            "gemini bridge produced no content"
         );
      }
   }
}

#[derive(Default)]
struct OpenCall {
   id: String,
   item_id: String,
   name: String,
   arguments: String,
   index: u64,
   announced: bool,
}

impl ChatToResponses {
   pub fn with_custom(custom: BTreeSet<String>) -> Self {
      let mut bridge = Self::default();
      bridge.custom = custom;
      bridge
   }

   #[expect(
      clippy::too_many_lines,
      reason = "one chunk in, every Responses event it implies out; the arms are the frame kinds"
   )]
   pub fn feed(&mut self, chunk: &ChatChunk) -> Vec<ResponsesEvent> {
      self.frames += 1;
      if self.first_frame.is_none() {
         self.first_frame = Some(
            serde_json::to_string(chunk)
               .unwrap_or_default()
               .chars()
               .take(400)
               .collect(),
         );
      }
      let response_id = (!chunk.id.is_empty()).then(|| chunk.id.clone());
      let mut out = Vec::new();
      if self.completed {
         return out;
      }
      if let Some(err) = chunk.error.as_ref() {
         self.completed = true;
         self.finished = true;
         self.calls.clear();
         self.text.clear();
         self.emitted += 1;
         out.push(ResponsesEvent::Failed {
            response: ResponseObj {
               id: response_id,
               status: Some("failed".into()),
               usage: None,
               error: Some(UpstreamError {
                  code: err.code.as_ref().map(|error_code| match *error_code {
                     ErrorCode::Text(ref code) => code.clone(),
                     ErrorCode::Number(code) => code.to_string(),
                  }),
                  message: Some(err.message.clone()),
               }),
            },
         });
         return out;
      }
      if !self.opened {
         self.opened = true;
         out.push(ResponsesEvent::Created {
            response: ResponseObj {
               id: response_id.clone(),
               ..Default::default()
            },
         });
      }
      let first = chunk.choices.first();
      let delta = first.map(|choice| &choice.delta);
      if let Some(text) = delta.and_then(|delta| delta.content.as_deref())
         && !text.is_empty()
      {
         // Codex attaches a delta to an item by id, so text streamed before
         // the item is announced is parsed and then dropped.
         let id = self
            .msg_id
            .get_or_insert_with(|| format!("msg_{}", uuid::Uuid::new_v4().simple()))
            .clone();
         if self.text.is_empty() {
            out.push(ResponsesEvent::OutputItemAdded {
               output_index: 0,
               item: OutputItem::Message {
                  id: Some(id.clone()),
                  role: Some("assistant".into()),
                  status: None,
                  content: Some(Vec::new()),
               },
            });
            out.push(ResponsesEvent::ContentPartAdded {
               item_id: Some(id.clone()),
               output_index: 0,
               content_index: 0,
               part: Some(OutputContentPart::OutputText {
                  text: String::new(),
               }),
            });
         }
         out.push(ResponsesEvent::OutputTextDelta {
            item_id: Some(id),
            output_index: 0,
            content_index: 0,
            delta: text.to_owned(),
         });
         self.text.push_str(text);
      }
      if let Some(calls) = delta.and_then(|delta| delta.tool_calls.as_ref()) {
         for call in calls {
            out.extend(self.tool_call(call));
         }
      }
      if first.is_some_and(|choice| choice.finish_reason.is_some()) {
         self.finished = true;
         for call in &self.calls {
            if !call.announced {
               continue;
            }
            // A freeform tool takes raw text, so the single string it was
            // offered as is unwrapped before the item is handed back.
            if self.custom.contains(&call.name) {
               let input = serde_json::from_str::<HashMap<String, String>>(&call.arguments)
                  .ok()
                  .and_then(|mut args| args.remove(FREEFORM_ARG))
                  .unwrap_or_else(|| call.arguments.clone());
               out.push(ResponsesEvent::CustomToolCallInputDone {
                  item_id: Some(call.item_id.clone()),
                  output_index: call.index,
                  input: input.clone(),
               });
               out.push(ResponsesEvent::OutputItemDone {
                  output_index: call.index,
                  item: OutputItem::CustomToolCall {
                     id: Some(call.item_id.clone()),
                     call_id: call.id.clone(),
                     name: call.name.clone(),
                     input,
                     status: Some("completed".into()),
                  },
               });
               continue;
            }
            out.push(ResponsesEvent::FunctionCallArgumentsDone {
               item_id: Some(call.item_id.clone()),
               output_index: call.index,
               arguments: call.arguments.clone(),
            });
            out.push(ResponsesEvent::OutputItemDone {
               output_index: call.index,
               item: OutputItem::FunctionCall {
                  id: Some(call.item_id.clone()),
                  call_id: call.id.clone(),
                  name: call.name.clone(),
                  arguments: Some(call.arguments.clone()),
                  status: Some("completed".into()),
               },
            });
         }
         self.calls.clear();
         // The streaming translator reads text deltas, but `aggregate`
         // builds its blocks only from finished items, so a non-streaming
         // turn needs the whole message restated as one.
         if !self.text.is_empty() {
            let id = self.msg_id.clone().unwrap_or_default();
            let text = mem::take(&mut self.text);
            out.push(ResponsesEvent::OutputTextDone {
               item_id: Some(id.clone()),
               output_index: 0,
               content_index: 0,
               text: text.clone(),
            });
            out.push(ResponsesEvent::OutputItemDone {
               output_index: 0,
               item: OutputItem::Message {
                  id: Some(id),
                  role: Some("assistant".into()),
                  status: Some("completed".into()),
                  content: Some(vec![OutputContentPart::OutputText { text }]),
               },
            });
         }
      }
      if let Some(usage) = chunk.usage.as_ref() {
         self.usage = Some(Usage::from(usage.clone()));
      }
      // Gemini reports usage on a chunk of its own, often before the one
      // carrying finish_reason, so emitting on usage alone closed the
      // response before its content and then closed it twice.
      if self.finished
         && !self.completed
         && let Some(usage) = self.usage.take()
      {
         self.completed = true;
         out.push(ResponsesEvent::Completed {
            response: ResponseObj {
               id: response_id,
               status: Some("completed".into()),
               usage: Some(usage),
               error: None,
            },
         });
      }
      // `created` and `completed` carry no answer, so they do not count as
      // content when deciding whether the bridge produced anything.
      self.emitted += out
         .iter()
         .filter(|event| {
            !matches!(
               event,
               ResponsesEvent::Created { .. } | ResponsesEvent::Completed { .. }
            )
         })
         .count();
      out
   }

   fn tool_call(&mut self, call: &ChatToolCall) -> Vec<ResponsesEvent> {
      let idx = call.index.unwrap_or(0) as usize;
      if self.calls.len() <= idx {
         self.calls.resize_with(idx + 1, OpenCall::default);
      }
      let mut out = Vec::new();
      let slot = &mut self.calls[idx];
      if let Some(id) = call.id.as_ref() {
         slot.id.clone_from(id);
      }
      if let Some(name) = call.function.name.as_ref() {
         slot.name.clone_from(name);
      }
      if let Some(sig) = call.thought_signature()
         && !slot.id.is_empty()
      {
         remember_signature(&slot.id, sig);
      }
      if !slot.announced && !slot.name.is_empty() {
         slot.announced = true;
         slot.item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
         slot.index = self.next_index;
         self.next_index += 1;
         let ready = &self.calls[idx];
         let item = if self.custom.contains(&ready.name) {
            OutputItem::CustomToolCall {
               id: Some(ready.item_id.clone()),
               call_id: ready.id.clone(),
               name: ready.name.clone(),
               input: String::new(),
               status: None,
            }
         } else {
            OutputItem::FunctionCall {
               id: Some(ready.item_id.clone()),
               call_id: ready.id.clone(),
               name: ready.name.clone(),
               arguments: Some(String::new()),
               status: None,
            }
         };
         out.push(ResponsesEvent::OutputItemAdded {
            output_index: ready.index,
            item,
         });
      }
      let args_slot = &mut self.calls[idx];
      if let Some(args) = call.function.arguments.as_deref()
         && !args.is_empty()
      {
         args_slot.arguments.push_str(args);
         out.push(ResponsesEvent::FunctionCallArgumentsDelta {
            item_id: Some(args_slot.item_id.clone()),
            output_index: args_slot.index,
            delta: args.to_owned(),
         });
      }
      out
   }
}

/// A referer-restricted key is served over the native surface, which answers
/// in Gemini's own frames, so that protocol is normalised before parsing.
pub fn event_stream(
   resp: reqwest::Response,
   protocol: GeminiProtocol,
   model: &str,
   custom: BTreeSet<String>,
   capture: UsageCapture,
) -> EventStream {
   use eventsource_stream::Eventsource as _;
   use futures_util::StreamExt as _;

   let mut native = (protocol == GeminiProtocol::Native).then(|| NativeStream::new(model));
   let mut frames = Frames::default();
   let mut cut = false;
   let head = capture;
   let chat_bytes = resp.bytes_stream().flat_map(move |item| {
      let chunks = match (&mut native, item) {
         (&mut Some(ref mut native), Ok(bytes)) => {
            head.note_upstream_head(&bytes);
            native.feed(&bytes)
         },
         (&mut None, Ok(bytes)) => {
            head.note_upstream_head(&bytes);
            let mut chunks: Vec<Vec<u8>> = frames
               .feed(&bytes)
               .into_iter()
               .map(|frame| format!("data: {}\n\n", String::from_utf8_lossy(&frame)).into_bytes())
               .collect();
            if !cut && let Some(error) = frames.cutoff() {
               cut = true;
               chunks.push(cutoff_frame(&error));
            }
            chunks
         },
         (_, Err(err)) => {
            tracing::warn!("gemini stream error: {err}");
            Vec::new()
         },
      };
      stream::iter(chunks.into_iter().map(Ok::<_, Error>))
   });

   let mut bridge = ChatToResponses::with_custom(custom);
   let stream = chat_bytes.eventsource().flat_map(move |event| {
      let events = match event {
         Ok(event) if event.data != "[DONE]" => {
            match serde_json::from_str::<ChatChunk>(&event.data) {
               Ok(chunk) => bridge.feed(&chunk),
               Err(error) => {
                  tracing::warn!(%error, frame = %event.data, "gemini chunk did not parse");
                  Vec::new()
               },
            }
         },
         Ok(_) => Vec::new(),
         Err(err) => {
            tracing::warn!("gemini SSE parse error: {err}");
            Vec::new()
         },
      };
      stream::iter(events)
   });
   Box::pin(stream)
}

fn cutoff_frame(error: &ApiError) -> Vec<u8> {
   let body = ChatError {
      error: ChatErrorBody {
         message: error.message.clone().unwrap_or_default(),
         kind: Some("server_error".into()),
         code: error.status.clone().map(ErrorCode::Text),
      },
   };
   format!(
      "data: {}\n\ndata: [DONE]\n\n",
      serde_json::to_string(&body).unwrap_or_default()
   )
   .into_bytes()
}

#[cfg(test)]
mod tests {
   use super::*;
   use serde_json::{Value, json};

   fn chunk(value: Value) -> ChatChunk {
      serde_json::from_value(value).unwrap()
   }

   fn request(value: Value) -> ResponsesRequest {
      serde_json::from_value(value).unwrap()
   }

   fn kinds(events: &[ResponsesEvent]) -> Vec<String> {
      events
         .iter()
         .map(|event| {
            serde_json::to_value(event).unwrap()["type"]
               .as_str()
               .unwrap()
               .to_owned()
         })
         .collect()
   }

   fn feed(bridge: &mut ChatToResponses, value: Value) -> Vec<String> {
      kinds(&bridge.feed(&chunk(value)))
   }

   fn item(event: &ResponsesEvent) -> Value {
      serde_json::to_value(event).unwrap()["item"].clone()
   }

   #[test]
   fn text_deltas_become_output_text() {
      let mut bridge = ChatToResponses::default();
      let first = feed(
         &mut bridge,
         json!({"choices": [{"delta": {"content": "hi"}}]}),
      );
      assert_eq!(
         first,
         [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta"
         ]
      );
      let next = feed(
         &mut bridge,
         json!({"choices": [{"delta": {"content": "!"}}]}),
      );
      assert_eq!(next, ["response.output_text.delta"]);
   }

   #[test]
   fn a_tool_call_split_across_chunks_opens_once() {
      let mut bridge = ChatToResponses::default();
      bridge.feed(&chunk(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0_u64, "id": "c1", "function": {"name": "Read", "arguments": "{\"p"}}]}}]})));
      let more = feed(
         &mut bridge,
         json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0_u64, "function": {"arguments": "ath\":1}"}}]}}]}),
      );
      assert_eq!(more, ["response.function_call_arguments.delta"]);
      let done = feed(
         &mut bridge,
         json!({"choices": [{"finish_reason": "tool_calls"}]}),
      );
      assert_eq!(
         done,
         [
            "response.function_call_arguments.done",
            "response.output_item.done"
         ]
      );
      // An item done without its arguments is a call the client cannot run.
      let last = bridge.feed(&chunk(
         json!({"choices": [{"finish_reason": "tool_calls"}]}),
      ));
      assert!(last.is_empty(), "the turn closed twice: {last:?}");
   }

   #[test]
   fn a_finished_tool_call_carries_its_arguments() {
      let mut bridge = ChatToResponses::default();
      bridge.feed(&chunk(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0_u64, "id": "c1", "function": {"name": "Read", "arguments": "{\"p"}}]}}]})));
      bridge.feed(&chunk(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0_u64, "function": {"arguments": "ath\":1}"}}]}}]})));
      let done = bridge.feed(&chunk(
         json!({"choices": [{"finish_reason": "tool_calls"}]}),
      ));
      let item = item(&done[1]);
      assert_eq!(item["arguments"], "{\"path\":1}");
   }

   /// Gemini puts `usage` on a chunk of its own, often ahead of the one with
   /// `finish_reason`. Closing on `usage` alone ended the response before its
   /// content and then ended it a second time, which codex rejects.
   #[test]
   fn the_response_closes_once_and_after_its_content() {
      let mut bridge = ChatToResponses::default();
      assert_eq!(
         feed(
            &mut bridge,
            json!({"choices": [{"delta": {"content": "hi"}}]})
         ),
         [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta"
         ]
      );
      assert_eq!(
         feed(
            &mut bridge,
            json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1_i64}})
         ),
         Vec::<String>::new()
      );
      assert_eq!(
         feed(
            &mut bridge,
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]})
         ),
         [
            "response.output_text.done",
            "response.output_item.done",
            "response.completed"
         ]
      );
      assert_eq!(
         feed(
            &mut bridge,
            json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1_i64}})
         ),
         Vec::<String>::new()
      );
   }

   #[test]
   fn an_error_chunk_fails_the_response_and_closes_it() {
      let mut bridge = ChatToResponses::default();
      let events = bridge.feed(&chunk(json!({"error": {
          "message": "high demand", "type": "server_error", "code": "UNAVAILABLE"
      }})));
      assert_eq!(kinds(&events), ["response.failed"]);
      let response = match &events[0] {
         &ResponsesEvent::Failed { ref response } => response,
         other => panic!("expected a failed event: {other:?}"),
      };
      let error = response.error.as_ref().unwrap();
      assert_eq!(error.message.as_deref(), Some("high demand"));
      assert_eq!(error.code.as_deref(), Some("UNAVAILABLE"));
      assert!(
         feed(
            &mut bridge,
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]})
         )
         .is_empty()
      );
   }

   #[test]
   fn the_developer_role_survives_as_system() {
      let req = request(json!({
          "model": "gemini-3.8-flash", "instructions": "sys", "stream": true,
          "input": [{"type": "message", "role": "developer",
                     "content": [{"type": "input_text", "text": "ctx"}]}],
      }));
      let body = serde_json::to_value(to_chat(&req)).unwrap();
      assert_eq!(body["messages"][1]["role"], "system");
   }

   /// Gemini rejects any effort outside none/low/medium/high, and codex asks
   /// for xhigh, so an unmapped value would 400 the turn rather than degrade.
   #[test]
   fn an_unsupported_effort_clamps_instead_of_failing() {
      let with = |effort: &str| {
         to_chat(&request(json!({
             "model": "gemini-3.8-flash", "instructions": "s", "stream": true,
             "input": [], "reasoning": {"effort": effort},
         })))
         .reasoning_effort
         .unwrap()
      };
      assert_eq!(with("xhigh"), "high");
      assert_eq!(with("minimal"), "none");
   }

   /// Codex's shell tool is `{"type":"custom","format":{"syntax":"lark"}}`
   /// with no parameters at all, so offering it verbatim gave gemini an empty
   /// schema and codex refused the arguments it invented.
   #[test]
   fn a_freeform_tool_round_trips_as_raw_text() {
      let req = request(json!({
          "model": "gemini-3.8-flash", "instructions": "s", "stream": true, "input": [],
          "tools": [{"type": "custom", "name": "exec",
                     "format": {"type": "grammar", "syntax": "lark"}}],
      }));
      let names = custom_tools(&req);

      let body = serde_json::to_value(to_chat(&req)).unwrap();
      let schema = &body["tools"][0]["function"]["parameters"];
      assert_eq!(schema["properties"]["input"]["type"], "string");

      let mut bridge = ChatToResponses::with_custom(names);
      bridge.feed(&chunk(
         json!({"choices": [{"delta": {"tool_calls": [{"index": 0_u64, "id": "c1",
            "function": {"name": "exec", "arguments": "{\"input\":\"ls -la\"}"}}]}}]}),
      ));
      let done = bridge.feed(&chunk(
         json!({"choices": [{"finish_reason": "tool_calls"}]}),
      ));
      let item = item(&done[1]);
      assert_eq!(item["type"], "custom_tool_call");
      assert_eq!(item["input"], "ls -la");
   }

   /// A prior turn comes back as the item type it was handed out as, so the
   /// call and its result have to map back onto chat messages.
   #[test]
   fn a_freeform_call_and_its_output_replay_as_messages() {
      let body = serde_json::to_value(to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [
              {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "ls"},
              {"type": "custom_tool_call_output", "call_id": "c1", "output": "a.txt"},
          ],
      }))))
      .unwrap();
      let call = &body["messages"][1]["tool_calls"][0]["function"];
      assert_eq!(call["arguments"], "{\"input\":\"ls\"}");
      assert_eq!(body["messages"][2]["content"], "a.txt");
   }

   /// Codex hands a tool result back as the content array it received, and a
   /// chat `tool` message takes a plain string, so forwarding the array made
   /// gemini reject the request and answer with an empty stream.
   #[test]
   fn a_tool_result_flattens_to_a_string() {
      let body = serde_json::to_value(to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [{"type": "custom_tool_call_output", "call_id": "c1", "output": [
              {"type": "input_text", "text": "Script completed\n"},
              {"type": "input_text", "text": "mango"},
          ]}],
      }))))
      .unwrap();
      assert_eq!(body["messages"][1]["content"], "Script completed\nmango");
   }

   #[test]
   fn the_bridge_streams_even_for_a_blocking_caller() {
      let out = to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": false,
          "input": [],
      })));
      assert_eq!(out.stream, Some(true));
      assert!(out.stream_options.is_some_and(|opts| opts.include_usage));
   }

   #[test]
   fn a_call_without_arguments_replays_as_an_empty_object() {
      let body = serde_json::to_value(to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [{"type": "function_call", "call_id": "c1", "name": "grep"}],
      }))))
      .unwrap();
      assert_eq!(
         body["messages"][1]["tool_calls"][0]["function"]["arguments"],
         "{}"
      );
   }

   #[test]
   fn a_tool_choice_without_a_type_still_names_the_function() {
      let body = serde_json::to_value(to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [], "tool_choice": {"name": "grep"},
      }))))
      .unwrap();
      assert_eq!(body["tool_choice"]["function"]["name"], "grep");
   }

   #[test]
   fn an_empty_effort_is_not_forwarded() {
      let out = to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [], "reasoning": {"effort": ""},
      })));
      assert!(out.reasoning_effort.is_none());
   }

   #[test]
   fn an_unknown_content_part_does_not_sink_the_request() {
      let req = request(json!({
          "model": "g", "instructions": "s", "stream": true,
          "input": [{"type": "message", "role": "user",
                     "content": [{"type": "input_file", "file_id": "f"},
                                 {"type": "input_text", "text": "hi"}]}],
      }));
      let body = serde_json::to_value(to_chat(&req)).unwrap();
      let content = body["messages"][1]["content"].as_array().unwrap();
      assert!(
         content
            .iter()
            .any(|part| part.get("text").is_some_and(|text| text == "hi"))
      );
   }

   #[test]
   fn a_tool_without_a_name_is_dropped() {
      let out = to_chat(&request(json!({
          "model": "g", "instructions": "s", "stream": true, "input": [],
          "tools": [{"type": "function", "parameters": {"type": "object"}}],
      })));
      assert!(out.tools.is_none());
   }
}

#[cfg(test)]
mod full_chain {
   use super::*;
   use crate::gemini::native::NativeStream;

   /// The whole native path offline: gemini frames in, Responses events out.
   #[test]
   fn a_captured_native_stream_yields_text() {
      let raw = include_bytes!("testdata/gemini_native.sse");
      let mut native = NativeStream::new("gemini-3.8-flash");
      let mut bridge = ChatToResponses::default();
      let mut kinds = Vec::new();
      let mut text = String::new();
      // The network delivers arbitrary chunks, not whole frames.
      let frames: Vec<Vec<u8>> = raw
         .chunks(64)
         .flat_map(|chunk| native.feed(chunk))
         .collect();
      for frame in frames {
         let line = String::from_utf8_lossy(&frame);
         for data in line.lines().filter_map(|line| line.strip_prefix("data:")) {
            let data = data.trim();
            if data == "[DONE]" {
               continue;
            }
            let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) else {
               continue;
            };
            for event in bridge.feed(&chunk) {
               if let &ResponsesEvent::OutputTextDelta { ref delta, .. } = &event {
                  text.push_str(delta);
               }
               kinds.push(serde_json::to_value(&event).unwrap()["type"].to_string());
            }
         }
      }
      assert!(!text.is_empty(), "no text recovered, events were {kinds:?}");
   }
}

#[cfg(test)]
mod real_chunk {
   use super::*;

   #[test]
   fn a_real_gemini_chunk_survives_deserialisation() {
      let chunk: ChatChunk = serde_json::from_str(r#"{"choices":[{"delta":{"content":"OK.","role":"assistant"},"index":0}],"created":1788373398,"id":"x","model":"gemini-3.8-flash","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":71}}"#).unwrap();
      let events = ChatToResponses::default().feed(&chunk);
      let kinds: Vec<_> = events
         .iter()
         .map(|event| {
            serde_json::to_value(event).unwrap()["type"]
               .as_str()
               .unwrap()
               .to_owned()
         })
         .collect();
      assert_eq!(
         kinds,
         [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta"
         ]
      );
      assert!(
         events
            .iter()
            .any(|event| matches!(event, ResponsesEvent::OutputTextDelta { .. })),
         "no text delta survived: {events:?}"
      );
   }
}

#[cfg(test)]
mod aggregate_path {
   use super::*;

   #[test]
   fn a_finished_turn_restates_its_text_as_an_item() {
      let mut bridge = ChatToResponses::default();
      let chunk = |value| serde_json::from_value::<ChatChunk>(value).unwrap();
      bridge.feed(&chunk(
         serde_json::json!({"choices": [{"delta": {"content": "ok"}}]}),
      ));
      let done = bridge.feed(&chunk(
         serde_json::json!({"choices": [{"finish_reason": "stop"}]}),
      ));
      let text = done.iter().find_map(|event| match *event {
         ResponsesEvent::OutputItemDone {
            item: OutputItem::Message { ref content, .. },
            ..
         } => content
            .as_ref()
            .and_then(|items| items.first())
            .map(|part| match *part {
               OutputContentPart::OutputText { ref text } => text.clone(),
               OutputContentPart::Other => String::new(),
            }),
         ResponsesEvent::OutputItemDone { .. }
         | ResponsesEvent::Created { .. }
         | ResponsesEvent::InProgress
         | ResponsesEvent::OutputItemAdded { .. }
         | ResponsesEvent::ContentPartAdded { .. }
         | ResponsesEvent::ContentPartDone
         | ResponsesEvent::OutputTextDelta { .. }
         | ResponsesEvent::OutputTextDone { .. }
         | ResponsesEvent::ReasoningSummaryPartAdded
         | ResponsesEvent::ReasoningSummaryPartDone
         | ResponsesEvent::ReasoningSummaryTextDelta { .. }
         | ResponsesEvent::ReasoningSummaryTextDone
         | ResponsesEvent::ReasoningTextDelta { .. }
         | ResponsesEvent::ReasoningTextDone
         | ResponsesEvent::FunctionCallArgumentsDelta { .. }
         | ResponsesEvent::FunctionCallArgumentsDone { .. }
         | ResponsesEvent::CustomToolCallInputDone { .. }
         | ResponsesEvent::Completed { .. }
         | ResponsesEvent::Incomplete { .. }
         | ResponsesEvent::Failed { .. }
         | ResponsesEvent::Other => None,
      });
      assert_eq!(text.as_deref(), Some("ok"));
   }
}
