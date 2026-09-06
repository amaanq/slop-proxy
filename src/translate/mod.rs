pub mod anthropic_req;
pub mod anthropic_stream;
pub mod chat;
pub mod count_tokens;
pub mod gemini_bridge;
pub mod gemini_req;
pub mod model_map;
pub mod openai_req;
pub mod openai_stream;

#[cfg(test)]
mod stream_tests;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use data_encoding::BASE64URL_NOPAD;
use futures_util::StreamExt as _;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use serde_json::value::{RawValue, to_raw_value};

use crate::codex::sse::EventStream;
use crate::codex::types::{
   OutputContentPart, OutputItem, ResponsesEvent, SummaryPart, TerminalKind, TokenDetails, Usage,
};

/// Zen's upstream 400s `max_output_tokens` below 16, and `CodexClient::post`
/// already salvages that by stripping the field rather than clamping it.
pub const MIN_MAX_OUTPUT_TOKENS: u64 = 16;

/// A cap the upstream will accept, or `None` to leave the field off entirely.
pub fn usable_cap(cap: Option<u64>) -> Option<u64> {
   cap.filter(|&cap| cap >= MIN_MAX_OUTPUT_TOKENS)
}

/// A tagged or untagged enum, and anything behind a `flatten`, buffers the
/// map first, and `RawValue` cannot be read back out of that buffer.
pub fn buffered_raw<'de, D>(deserializer: D) -> Result<Option<Box<RawValue>>, D::Error>
where
   D: serde::Deserializer<'de>,
{
   Option::<serde_json::Value>::deserialize(deserializer)?
      .map(|value| to_raw_value(&value).map_err(Error::custom))
      .transpose()
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
   #[error("tool message without tool_call_id")]
   ToolMessageWithoutCallId,
   #[error("unsupported message role {0:?}")]
   UnsupportedRole(String),
}

#[derive(Serialize, Deserialize)]
struct SignaturePayload {
   id: Option<String>,
   #[serde(rename = "ec")]
   encrypted_content: Option<String>,
}

/// Anthropic thinking-block signatures are opaque round-tripped strings, so we
/// smuggle the Responses reasoning item `id` + `encrypted_content` through them.
pub fn encode_signature(id: Option<&str>, encrypted_content: &str) -> String {
   let payload = SignaturePayload {
      id: id.map(String::from),
      encrypted_content: Some(encrypted_content.to_owned()),
   };
   let payload = serde_json::to_string(&payload).unwrap_or_default();
   BASE64URL_NOPAD.encode(payload.as_bytes())
}

pub fn decode_signature(sig: &str) -> (Option<String>, Option<String>) {
   if let Ok(bytes) = BASE64URL_NOPAD.decode(sig.as_bytes())
      && let Ok(payload) = serde_json::from_slice::<SignaturePayload>(&bytes)
      && payload.encrypted_content.is_some()
   {
      return (payload.id, payload.encrypted_content);
   }
   (None, Some(sig.to_owned()))
}

#[derive(Default, Debug, Clone)]
pub struct CapturedUsage {
   pub input_tokens: i64,
   pub output_tokens: i64,
   pub cache_read_tokens: i64,
   pub cache_write_tokens: i64,
   pub reasoning_tokens: i64,
   pub completed: bool,
   pub error_kind: Option<String>,
   /// Distinguishes upstream ending early from the caller hanging up.
   pub upstream_eof: bool,
   pub last_event: Option<String>,
   /// Separates a slow account from a long answer, which one duration cannot.
   pub first_byte_at: Option<Instant>,
   pub response_bytes: i64,
   pub stop_reason: Option<String>,
   /// Names only. An argument is the caller's shell command or source.
   pub tools_called: Vec<String>,
   /// What upstream opened with, the only evidence left when a 200 yields
   /// no events at all.
   pub upstream_head: Option<String>,
}

#[derive(Default, Clone)]
pub struct UsageCapture(pub Arc<Mutex<CapturedUsage>>);

impl UsageCapture {
   pub fn observe(&self, event: &ResponsesEvent) {
      self.note_event(event.kind());
      if let Some((kind, response)) = event.terminal() {
         if let Some(usage) = response.usage.as_ref() {
            self.record_partial(usage);
         }
         let mut captured = self.0.lock().unwrap();
         captured.completed = true;
         captured.stop_reason = Some(kind.as_str().into());
         if kind == TerminalKind::Failed {
            captured.error_kind = Some("upstream_failed".into());
         }
      }
      if let &ResponsesEvent::OutputItemDone {
         item:
            OutputItem::FunctionCall { ref name, .. } | OutputItem::CustomToolCall { ref name, .. },
         ..
      } = event
      {
         self.note_tool_call(name);
      }
   }

   /// Codex counts cached tokens inside `input_tokens`, Anthropic reports
   /// them alongside. Subtracting leaves it meaning freshly billed prompt on
   /// both. `reasoning_tokens` stays a subset of `output_tokens`.
   pub fn record(&self, usage: &Usage) {
      self.record_partial(usage);
      self.0.lock().unwrap().completed = true;
   }

   pub fn record_partial(&self, usage: &Usage) {
      let mut captured = self.0.lock().unwrap();
      let cached = usage.input_tokens_details.cached_tokens;
      captured.input_tokens = (usage.input_tokens - cached).max(0);
      captured.output_tokens = usage.output_tokens;
      captured.cache_read_tokens = cached;
      captured.reasoning_tokens = usage.output_tokens_details.reasoning_tokens;
   }

   pub fn note_event(&self, name: &str) {
      let mut captured = self.0.lock().unwrap();
      captured.last_event = Some(name.to_owned());
      captured.first_byte_at.get_or_insert_with(Instant::now);
   }

   pub fn note_bytes(&self, len: usize) {
      if len == 0 {
         return;
      }
      let mut captured = self.0.lock().unwrap();
      captured.response_bytes += len as i64;
      captured.first_byte_at.get_or_insert_with(Instant::now);
   }

   pub fn note_stop_reason(&self, reason: &str) {
      self.0.lock().unwrap().stop_reason = Some(reason.to_owned());
   }

   pub fn note_cutoff(&self, status: &str) {
      let mut captured = self.0.lock().unwrap();
      captured.error_kind = Some("upstream_cutoff".into());
      captured.stop_reason = Some(status.to_ascii_lowercase());
   }

   pub fn note_tool_call(&self, name: &str) {
      let mut captured = self.0.lock().unwrap();
      if !captured.tools_called.iter().any(|tool| tool == name) {
         captured.tools_called.push(name.to_owned());
      }
   }

   pub fn note_upstream_head(&self, bytes: &[u8]) {
      let mut captured = self.0.lock().unwrap();
      if captured.upstream_head.is_none() {
         captured.upstream_head = Some(String::from_utf8_lossy(bytes).chars().take(400).collect());
      }
   }

   pub fn note_upstream_eof(&self) {
      self.0.lock().unwrap().upstream_eof = true;
   }

   pub fn fail(&self, kind: &str) {
      let mut captured = self.0.lock().unwrap();
      if captured.error_kind.is_none() {
         captured.error_kind = Some(kind.to_owned());
      }
   }

   pub fn snapshot(&self) -> CapturedUsage {
      self.0.lock().unwrap().clone()
   }
}

#[derive(Debug)]
pub enum Block {
   Thinking {
      text: String,
      signature: Option<String>,
   },
   Text {
      text: String,
   },
   ToolCall {
      id: String,
      name: String,
      arguments: String,
   },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopKind {
   EndTurn,
   MaxTokens,
   ToolUse,
   Error,
}

impl StopKind {
   pub const fn as_str(self) -> &'static str {
      match self {
         Self::EndTurn => "end_turn",
         Self::MaxTokens => "max_tokens",
         Self::ToolUse => "tool_use",
         Self::Error => "error",
      }
   }
}

#[derive(Debug)]
pub struct Aggregated {
   pub id: String,
   pub blocks: Vec<Block>,
   pub stop: StopKind,
   pub usage: Usage,
   pub error_message: Option<String>,
   pub completed: bool,
}

pub enum Step {
   Start {
      id: Option<String>,
   },
   Block {
      index: usize,
      event: BlockEvent,
   },
   Stop {
      kind: StopKind,
      usage: Usage,
   },
   Failed {
      message: String,
      code: Option<String>,
   },
}

pub enum BlockEvent {
   Open(Block),
   Append(String),
   Signature(String),
   Close,
}

struct TrackedBlock {
   index: usize,
   seen: bool,
   closed: bool,
}

/// The one state machine every consumer of a Responses stream shares.
pub struct Walker {
   blocks: BTreeMap<(u64, u64), TrackedBlock>,
   saw_tool: bool,
   done: bool,
   capture: UsageCapture,
}

impl Walker {
   pub const fn new(capture: UsageCapture) -> Self {
      Self {
         blocks: BTreeMap::new(),
         saw_tool: false,
         done: false,
         capture,
      }
   }

   pub fn step(&mut self, event: ResponsesEvent) -> Vec<Step> {
      let mut out = Vec::new();
      if self.done {
         return out;
      }
      self.capture.observe(&event);
      match event {
         ResponsesEvent::Created { response } => out.push(Step::Start { id: response.id }),
         ResponsesEvent::OutputItemAdded { output_index, item } => match item {
            OutputItem::Reasoning { .. } => self.open(
               &mut out,
               (output_index, 0),
               Block::Thinking {
                  text: String::new(),
                  signature: None,
               },
            ),
            OutputItem::FunctionCall { call_id, name, .. } => {
               self.open(
                  &mut out,
                  (output_index, 0),
                  Block::ToolCall {
                     id: call_id,
                     name,
                     arguments: String::new(),
                  },
               );
            },
            OutputItem::Message { .. } | OutputItem::CustomToolCall { .. } | OutputItem::Other => {
            },
         },
         ResponsesEvent::ReasoningSummaryPartAdded { output_index } => {
            if self
               .blocks
               .get(&(output_index, 0))
               .is_some_and(|block| block.seen)
            {
               self.append(&mut out, (output_index, 0), "\n\n".into());
            }
         },
         ResponsesEvent::ReasoningSummaryTextDelta {
            output_index,
            delta,
         }
         | ResponsesEvent::ReasoningTextDelta {
            output_index,
            delta,
         } => {
            self.open(
               &mut out,
               (output_index, 0),
               Block::Thinking {
                  text: String::new(),
                  signature: None,
               },
            );
            self.append(&mut out, (output_index, 0), delta);
         },
         ResponsesEvent::OutputTextDelta {
            output_index,
            content_index,
            delta,
            ..
         } => {
            let key = (output_index, content_index);
            self.open(
               &mut out,
               key,
               Block::Text {
                  text: String::new(),
               },
            );
            self.append(&mut out, key, delta);
         },
         ResponsesEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
            ..
         } => {
            self.append(&mut out, (output_index, 0), delta);
         },
         ResponsesEvent::OutputItemDone { output_index, item } => {
            self.finish_item(&mut out, output_index, item);
         },
         ResponsesEvent::Completed { response } => {
            let kind = if self.saw_tool {
               StopKind::ToolUse
            } else {
               StopKind::EndTurn
            };
            self.stop(&mut out, kind, response.usage);
         },
         ResponsesEvent::Incomplete { response } => {
            self.stop(&mut out, StopKind::MaxTokens, response.usage);
         },
         ResponsesEvent::Failed { response } => {
            let (message, code) = response
               .error
               .map(|err| (err.message, err.code))
               .unwrap_or_default();
            let message = message.unwrap_or_else(|| "upstream response failed".into());
            self.capture.fail("upstream_failed");
            self.capture.note_stop_reason(StopKind::Error.as_str());
            self.done = true;
            out.push(Step::Failed { message, code });
         },
         ResponsesEvent::InProgress
         | ResponsesEvent::ContentPartAdded { .. }
         | ResponsesEvent::ContentPartDone
         | ResponsesEvent::OutputTextDone { .. }
         | ResponsesEvent::ReasoningSummaryPartDone
         | ResponsesEvent::ReasoningSummaryTextDone
         | ResponsesEvent::ReasoningTextDone
         | ResponsesEvent::FunctionCallArgumentsDone { .. }
         | ResponsesEvent::CustomToolCallInputDone { .. }
         | ResponsesEvent::Other => {},
      }
      out
   }

   fn finish_item(&mut self, out: &mut Vec<Step>, output_index: u64, item: OutputItem) {
      match item {
         OutputItem::Reasoning {
            id,
            summary,
            encrypted_content,
         } => {
            let key = (output_index, 0);
            self.open(
               out,
               key,
               Block::Thinking {
                  text: String::new(),
                  signature: None,
               },
            );
            let text = summary
               .unwrap_or_default()
               .iter()
               .map(|&SummaryPart::SummaryText { ref text }| text.as_str())
               .collect::<Vec<_>>()
               .join("\n\n");
            if !self.blocks[&key].seen && !text.is_empty() {
               self.append(out, key, text);
            }
            if let Some(content) = encrypted_content {
               out.push(Step::Block {
                  index: self.blocks[&key].index,
                  event: BlockEvent::Signature(encode_signature(id.as_deref(), &content)),
               });
            }
            self.close(out, key);
         },
         OutputItem::Message { content, .. } => {
            for (content_index, part) in content.unwrap_or_default().into_iter().enumerate() {
               if let OutputContentPart::OutputText { text } = part {
                  let key = (output_index, content_index as u64);
                  self.open(
                     out,
                     key,
                     Block::Text {
                        text: String::new(),
                     },
                  );
                  if !self.blocks[&key].seen {
                     self.append(out, key, text);
                  }
               }
            }
            let keys: Vec<_> = self
               .blocks
               .keys()
               .filter(|key| key.0 == output_index)
               .copied()
               .collect();
            for key in keys {
               self.close(out, key);
            }
         },
         OutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
         } => {
            let key = (output_index, 0);
            self.open(
               out,
               key,
               Block::ToolCall {
                  id: call_id,
                  name,
                  arguments: String::new(),
               },
            );
            if !self.blocks[&key].seen
               && let Some(args) = arguments.filter(|arg| !arg.is_empty())
            {
               self.append(out, key, args);
            }
            self.close(out, key);
         },
         OutputItem::Other | OutputItem::CustomToolCall { .. } => {},
      }
   }

   pub fn eof(&mut self) -> Vec<Step> {
      if self.done {
         return Vec::new();
      }
      self.done = true;
      self.capture.fail("upstream_eof");
      vec![Step::Failed {
         message: "upstream stream ended unexpectedly".into(),
         code: None,
      }]
   }

   fn open(&mut self, out: &mut Vec<Step>, key: (u64, u64), block: Block) {
      if self.blocks.contains_key(&key) {
         return;
      }
      if let Block::ToolCall { ref name, .. } = block {
         self.saw_tool = true;
         self.capture.note_tool_call(name);
      }
      let index = self.blocks.len();
      self.blocks.insert(
         key,
         TrackedBlock {
            index,
            seen: false,
            closed: false,
         },
      );
      out.push(Step::Block {
         index,
         event: BlockEvent::Open(block),
      });
   }

   fn append(&mut self, out: &mut Vec<Step>, key: (u64, u64), text: String) {
      if let Some(block) = self.blocks.get_mut(&key).filter(|block| !block.closed) {
         block.seen = true;
         out.push(Step::Block {
            index: block.index,
            event: BlockEvent::Append(text),
         });
      }
   }

   fn close(&mut self, out: &mut Vec<Step>, key: (u64, u64)) {
      if let Some(block) = self.blocks.get_mut(&key).filter(|block| !block.closed) {
         block.closed = true;
         out.push(Step::Block {
            index: block.index,
            event: BlockEvent::Close,
         });
      }
   }

   fn stop(&mut self, out: &mut Vec<Step>, kind: StopKind, usage: Option<Usage>) {
      for block in self.blocks.values_mut().filter(|block| !block.closed) {
         block.closed = true;
         out.push(Step::Block {
            index: block.index,
            event: BlockEvent::Close,
         });
      }
      let usage = usage.unwrap_or_else(|| {
         let snapshot = self.capture.snapshot();
         Usage {
            input_tokens: snapshot.input_tokens + snapshot.cache_read_tokens,
            output_tokens: snapshot.output_tokens,
            total_tokens: snapshot.input_tokens
               + snapshot.cache_read_tokens
               + snapshot.output_tokens,
            input_tokens_details: TokenDetails {
               cached_tokens: snapshot.cache_read_tokens,
               ..Default::default()
            },
            output_tokens_details: TokenDetails {
               reasoning_tokens: snapshot.reasoning_tokens,
               ..Default::default()
            },
         }
      });
      self.capture.note_stop_reason(kind.as_str());
      self.done = true;
      out.push(Step::Stop { kind, usage });
   }
}

pub async fn aggregate(mut stream: EventStream, capture: &UsageCapture) -> Aggregated {
   let mut agg = Aggregated {
      id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
      blocks: Vec::new(),
      stop: StopKind::EndTurn,
      usage: Usage::default(),
      error_message: None,
      completed: false,
   };
   let mut walker = Walker::new(capture.clone());
   while let Some(event) = stream.next().await {
      for step in walker.step(event) {
         agg.fold(step);
      }
   }
   for step in walker.eof() {
      agg.fold(step);
   }
   agg
}

impl Aggregated {
   fn fold(&mut self, step: Step) {
      match step {
         Step::Start { id: Some(id) } => self.id = id,
         Step::Block {
            event: BlockEvent::Open(block),
            ..
         } => self.blocks.push(block),
         Step::Block { index, event } => match (event, self.blocks.get_mut(index)) {
            (BlockEvent::Append(chunk), Some(block)) => match *block {
               Block::Thinking { ref mut text, .. }
               | Block::Text { ref mut text }
               | Block::ToolCall {
                  arguments: ref mut text,
                  ..
               } => text.push_str(&chunk),
            },
            (
               BlockEvent::Signature(sig),
               Some(&mut Block::Thinking {
                  ref mut signature, ..
               }),
            ) => {
               *signature = Some(sig);
            },
            (
               BlockEvent::Close,
               Some(&mut Block::ToolCall {
                  ref mut arguments, ..
               }),
            ) if arguments.is_empty() => {
               arguments.push_str("{}");
            },
            _ => {},
         },
         Step::Stop { kind, usage } => {
            self.stop = kind;
            self.usage = usage;
            self.completed = true;
         },
         Step::Failed { message, .. } => {
            self.error_message = Some(message);
            self.stop = StopKind::Error;
         },
         Step::Start { .. } => {},
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn a_cap_too_small_for_any_answer_is_dropped_not_clamped() {
      assert_eq!(usable_cap(None), None);
      assert_eq!(usable_cap(Some(8)), None);
      assert_eq!(usable_cap(Some(16)), Some(16));
   }

   #[test]
   fn signature_roundtrip() {
      let sig = encode_signature(Some("rs_1"), "SECRET");
      assert_eq!(
         decode_signature(&sig),
         (Some("rs_1".into()), Some("SECRET".into()))
      );
      let sig_no_id = encode_signature(None, "SECRET");
      assert_eq!(decode_signature(&sig_no_id), (None, Some("SECRET".into())));
      assert_eq!(
         decode_signature("not-base64-json"),
         (None, Some("not-base64-json".into()))
      );
   }
}
