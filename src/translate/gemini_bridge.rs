//! Claude Code only speaks the messages API and every Gemini surface speaks
//! chat completions.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use futures_util::stream;

use super::chat::{ChatChunk, ChatErrorBody, ChatToolCall, ErrorCode, FinishReason};
use super::gemini_req::FREEFORM_ARG;
use crate::codex::sse::EventStream;
use crate::codex::types::{
   OutputContentPart, OutputItem, ResponseObj, ResponsesEvent, SummaryPart, UpstreamError, Usage,
};
use crate::gemini::client::GeminiProtocol;
use crate::gemini::native::{NativeEvent, NativeStream};
use crate::gemini::signatures;
use crate::gemini::sse::Frames;
use crate::translate::UsageCapture;

/// Tool calls arrive spread across chunks keyed by index, so a slot holds the
/// name until there is enough to open an item.
#[derive(Default)]
pub struct ChatToResponses {
   custom: BTreeSet<String>,
   response_id: Option<String>,
   calls: BTreeMap<u64, OpenCall>,
   text: Option<TextOutput>,
   reasoning: Option<TextOutput>,
   next_index: u64,
   finish_reason: Option<FinishReason>,
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

struct TextOutput {
   id: String,
   index: u64,
   text: String,
}

impl TextOutput {
   fn new(prefix: &str, next_index: &mut u64) -> Self {
      let index = *next_index;
      *next_index += 1;
      Self {
         id: format!("{prefix}_{}", uuid::Uuid::new_v4().simple()),
         index,
         text: String::new(),
      }
   }
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
      let first_frame = self.response_id.is_none();
      let response_id = Some(
         self
            .response_id
            .get_or_insert_with(|| {
               if chunk.id.is_empty() {
                  format!("resp_{}", uuid::Uuid::new_v4().simple())
               } else {
                  chunk.id.clone()
               }
            })
            .clone(),
      );
      let mut out = Vec::new();
      if self.completed {
         return out;
      }
      if let Some(err) = chunk.error.as_ref() {
         self.completed = true;
         self.calls.clear();
         self.emitted += 1;
         out.push(ResponsesEvent::Failed {
            response: ResponseObj {
               id: response_id,
               status: Some("failed".into()),
               usage: chunk
                  .usage
                  .clone()
                  .map(Usage::from)
                  .or_else(|| self.usage.take()),
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
      if first_frame {
         out.push(ResponsesEvent::Created {
            response: ResponseObj {
               id: response_id.clone(),
               ..Default::default()
            },
         });
      }
      let first_choice = chunk.choices.first();
      let delta = first_choice.map(|choice| &choice.delta);
      if delta.is_some_and(|delta| {
         delta
            .images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
      }) {
         return self.fail(
            "image output cannot be represented on this endpoint",
            "unsupported_output",
         );
      }
      if let Some(text) = delta
         .and_then(|delta| delta.reasoning_content.as_deref())
         .filter(|text| !text.is_empty())
      {
         let first = self.reasoning.is_none();
         let reasoning = self
            .reasoning
            .get_or_insert_with(|| TextOutput::new("rs", &mut self.next_index));
         if first {
            out.push(ResponsesEvent::OutputItemAdded {
               output_index: reasoning.index,
               item: OutputItem::Reasoning {
                  id: Some(reasoning.id.clone()),
                  summary: Some(Vec::new()),
                  encrypted_content: None,
               },
            });
            out.push(ResponsesEvent::ReasoningSummaryPartAdded {
               output_index: reasoning.index,
            });
         }
         reasoning.text.push_str(text);
         out.push(ResponsesEvent::ReasoningSummaryTextDelta {
            output_index: reasoning.index,
            delta: text.to_owned(),
         });
      }
      if let Some(text) = delta.and_then(|delta| delta.content.as_deref())
         && !text.is_empty()
      {
         // Codex attaches a delta to an item by id, so text streamed before
         // the item is announced is parsed and then dropped.
         let first = self.text.is_none();
         let message = self
            .text
            .get_or_insert_with(|| TextOutput::new("msg", &mut self.next_index));
         let id = message.id.clone();
         if first {
            out.push(ResponsesEvent::OutputItemAdded {
               output_index: message.index,
               item: OutputItem::Message {
                  id: Some(id.clone()),
                  role: Some("assistant".into()),
                  status: None,
                  content: Some(Vec::new()),
               },
            });
            out.push(ResponsesEvent::ContentPartAdded {
               item_id: Some(id.clone()),
               output_index: message.index,
               content_index: 0,
               part: Some(OutputContentPart::OutputText {
                  text: String::new(),
               }),
            });
         }
         out.push(ResponsesEvent::OutputTextDelta {
            item_id: Some(id),
            output_index: message.index,
            content_index: 0,
            delta: text.to_owned(),
         });
         message.text.push_str(text);
      }
      if let Some(calls) = delta.and_then(|delta| delta.tool_calls.as_ref()) {
         for call in calls {
            out.extend(self.tool_call(call));
         }
      }
      if self.finish_reason.is_none()
         && first_choice.is_some_and(|choice| choice.finish_reason.is_some())
      {
         self.finish_reason = first_choice.and_then(|choice| choice.finish_reason);
         for call in self.calls.values() {
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
         // Responses clients recover final output from output_item.done.
         if let Some(TextOutput { id, index, text }) = self.text.take() {
            out.push(ResponsesEvent::OutputTextDone {
               item_id: Some(id.clone()),
               output_index: index,
               content_index: 0,
               text: text.clone(),
            });
            out.push(ResponsesEvent::OutputItemDone {
               output_index: index,
               item: OutputItem::Message {
                  id: Some(id),
                  role: Some("assistant".into()),
                  status: Some("completed".into()),
                  content: Some(vec![OutputContentPart::OutputText { text }]),
               },
            });
         }
         if let Some(TextOutput { id, index, text }) = self.reasoning.take() {
            out.push(ResponsesEvent::OutputItemDone {
               output_index: index,
               item: OutputItem::Reasoning {
                  id: Some(id),
                  summary: Some(vec![SummaryPart::SummaryText { text }]),
                  encrypted_content: None,
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
      if self.finish_reason.is_some()
         && !self.completed
         && let Some(usage) = self.usage.take()
      {
         self.completed = true;
         out.push(self.terminal(response_id, Some(usage)));
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
      let mut out = Vec::new();
      let slot = self.calls.entry(call.index.unwrap_or(0)).or_default();
      if slot.id.is_empty() {
         slot.id = format!("call_{}", uuid::Uuid::new_v4().simple());
      }
      if let Some(name) = call.function.name.as_ref() {
         slot.name.clone_from(name);
      }
      let was_announced = slot.announced;
      if let Some(args) = call.function.arguments.as_ref() {
         slot.arguments.push_str(args);
      }
      if let Some(sig) = call.thought_signature()
         && !slot.id.is_empty()
      {
         signatures::put(&slot.id, sig);
      }
      if !slot.announced && !slot.name.is_empty() {
         slot.announced = true;
         slot.item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
         slot.index = self.next_index;
         self.next_index += 1;
         let ready = &*slot;
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
      let args = if was_announced {
         call.function.arguments.as_deref().unwrap_or_default()
      } else {
         &slot.arguments
      };
      if slot.announced && !args.is_empty() {
         out.push(ResponsesEvent::FunctionCallArgumentsDelta {
            item_id: Some(slot.item_id.clone()),
            output_index: slot.index,
            delta: args.to_owned(),
         });
      }
      out
   }

   fn terminal(&self, id: Option<String>, usage: Option<Usage>) -> ResponsesEvent {
      let mut response = ResponseObj {
         id,
         usage,
         ..Default::default()
      };
      match self.finish_reason {
         Some(FinishReason::Stop | FinishReason::ToolCalls) => {
            response.status = Some("completed".into());
            ResponsesEvent::Completed { response }
         },
         Some(FinishReason::Length | FinishReason::ContentFilter) => {
            response.status = Some("incomplete".into());
            ResponsesEvent::Incomplete { response }
         },
         _ => {
            response.status = Some("failed".into());
            response.error = Some(UpstreamError {
               code: Some("upstream_eof".into()),
               message: Some("upstream ended without a supported finish reason".into()),
            });
            ResponsesEvent::Failed { response }
         },
      }
   }

   pub fn finalize(&mut self) -> Vec<ResponsesEvent> {
      if self.completed {
         return Vec::new();
      }
      self.completed = true;
      let usage = self.usage.take();
      vec![self.terminal(self.response_id.clone(), usage)]
   }

   fn fail(&mut self, message: &str, code: &str) -> Vec<ResponsesEvent> {
      self.feed(&ChatChunk {
         error: Some(ChatErrorBody {
            message: message.to_owned(),
            kind: Some("server_error".into()),
            code: Some(ErrorCode::Text(code.to_owned())),
         }),
         ..Default::default()
      })
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
   use futures_util::StreamExt as _;

   let mut native = (protocol == GeminiProtocol::Native).then(|| NativeStream::new(model));
   let mut frames = Frames::default();
   let mut bridge = ChatToResponses::with_custom(custom);
   let upstream = resp
      .bytes_stream()
      .map(Some)
      .chain(stream::once(async { None }));
   Box::pin(upstream.flat_map(move |item| {
      let Some(item) = item else {
         return stream::iter(bridge.finalize());
      };
      let payload = match item {
         Ok(payload) => payload,
         Err(error) => return stream::iter(bridge.fail(&error.to_string(), "upstream_read")),
      };
      capture.note_upstream_head(&payload);
      let chunks = if let Some(native) = native.as_mut() {
         native.events(&payload)
      } else {
         let mut chunks = Vec::new();
         for data in frames.feed(&payload) {
            if data == b"[DONE]" {
               continue;
            }
            match serde_json::from_slice::<ChatChunk>(&data) {
               Ok(chunk) => chunks.push(NativeEvent::Chunk(chunk)),
               Err(error) => {
                  return stream::iter(bridge.fail(&error.to_string(), "upstream_decode"));
               },
            }
         }
         if let Some(error) = frames.cutoff() {
            chunks.push(NativeEvent::Error(error));
         }
         chunks
      };
      let mut events = Vec::new();
      for chunk in chunks {
         match chunk {
            NativeEvent::Chunk(chunk) => events.extend(bridge.feed(&chunk)),
            NativeEvent::Error(error) => events.extend(bridge.fail(
               error.message.as_deref().unwrap_or("upstream failure"),
               error.status.as_deref().unwrap_or("upstream_error"),
            )),
            NativeEvent::Done => {},
         }
      }
      stream::iter(events)
   }))
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::codex::types::ResponsesRequest;
   use crate::translate::gemini_req::{custom_tools, to_chat};
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
      let mut terminals = Vec::new();
      for event in raw.chunks(64).flat_map(|chunk| native.events(chunk)) {
         match event {
            NativeEvent::Chunk(chunk) => {
               for response_event in bridge.feed(&chunk) {
                  if let &ResponsesEvent::OutputTextDelta { ref delta, .. } = &response_event {
                     text.push_str(delta);
                  }
                  kinds.push(response_event.kind().to_owned());
                  if let Some((_, response)) = response_event.terminal() {
                     terminals.push(response.clone());
                  }
               }
            },
            NativeEvent::Error(error) => panic!("{error:?}"),
            NativeEvent::Done => {},
         }
      }
      assert_eq!(text, "ok");
      assert_eq!(kinds.last().map(String::as_str), Some("response.completed"));
      assert_eq!(terminals.len(), 1);
      let usage = terminals[0].usage.as_ref().unwrap();
      assert_eq!(usage.input_tokens, 3);
      assert_eq!(usage.output_tokens, 89);
      assert_eq!(usage.output_tokens_details.reasoning_tokens, 88);
      assert!(bridge.finalize().is_empty());
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
         _ => None,
      });
      assert_eq!(text.as_deref(), Some("ok"));
   }
}
