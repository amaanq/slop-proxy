use serde::Serialize;
use serde_json::value::RawValue;

use super::{Aggregated, Block, Step, StopKind, UsageCapture, Walker};
use crate::codex::types::{ResponsesEvent, Usage};

pub struct AnthropicStream {
   model: String,
   est_input_tokens: i64,
   emit_thinking: bool,
   next_index: usize,
   hiding: bool,
   walker: Walker,
}

pub type OutEvent = (&'static str, String);

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthEvent {
   MessageStart {
      message: MessageStart,
   },
   Ping,
   ContentBlockStart {
      index: usize,
      content_block: ContentBlockStart,
   },
   ContentBlockDelta {
      index: usize,
      delta: BlockDelta,
   },
   ContentBlockStop {
      index: usize,
   },
   MessageDelta {
      delta: StopDelta,
      usage: AnthUsage,
   },
   MessageStop,
   Error {
      error: ErrorBody,
   },
}

impl AnthEvent {
   fn out(self) -> OutEvent {
      let name = match self {
         Self::MessageStart { .. } => "message_start",
         Self::Ping => "ping",
         Self::ContentBlockStart { .. } => "content_block_start",
         Self::ContentBlockDelta { .. } => "content_block_delta",
         Self::ContentBlockStop { .. } => "content_block_stop",
         Self::MessageDelta { .. } => "message_delta",
         Self::MessageStop => "message_stop",
         Self::Error { .. } => "error",
      };
      let value = serde_json::to_value(self).expect("event serializes");
      (name, value.to_string())
   }
}

#[derive(Serialize)]
struct MessageStart {
   id: String,
   #[serde(rename = "type")]
   kind: &'static str,
   role: &'static str,
   model: String,
   content: [(); 0],
   stop_reason: Option<String>,
   stop_sequence: Option<String>,
   usage: AnthUsage,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
   ToolUse {
      id: String,
      name: String,
      input: EmptyObject,
   },
   Thinking {
      thinking: &'static str,
   },
   Text {
      text: &'static str,
   },
}

#[derive(Serialize)]
struct EmptyObject;

#[derive(Serialize)]
#[serde(tag = "type")]
enum BlockDelta {
   #[serde(rename = "thinking_delta")]
   Thinking { thinking: String },
   #[serde(rename = "text_delta")]
   Text { text: String },
   #[serde(rename = "input_json_delta")]
   InputJson { partial_json: String },
   #[serde(rename = "signature_delta")]
   Signature { signature: String },
}

#[derive(Serialize)]
struct StopDelta {
   stop_reason: &'static str,
   stop_sequence: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
   #[serde(rename = "type")]
   kind: &'static str,
   message: String,
}

#[derive(Serialize)]
pub struct AnthUsage {
   input_tokens: i64,
   output_tokens: i64,
   cache_read_input_tokens: i64,
   cache_creation_input_tokens: i64,
}

fn anthropic_usage(usage: &Usage) -> AnthUsage {
   let cached = usage.input_tokens_details.cached_tokens;
   AnthUsage {
      input_tokens: (usage.input_tokens - cached).max(0),
      output_tokens: usage.output_tokens,
      cache_read_input_tokens: cached,
      cache_creation_input_tokens: 0,
   }
}

impl AnthropicStream {
   pub const fn new(
      model: String,
      est_input_tokens: i64,
      emit_thinking: bool,
      capture: UsageCapture,
   ) -> Self {
      Self {
         model,
         est_input_tokens,
         emit_thinking,
         next_index: 0,
         hiding: false,
         walker: Walker::new(capture),
      }
   }

   pub fn handle(&mut self, event: ResponsesEvent) -> Vec<OutEvent> {
      let mut out = Vec::new();
      for step in self.walker.step(event) {
         self.render(&mut out, step);
      }
      out
   }

   pub fn finalize(&mut self) -> Vec<OutEvent> {
      self
         .walker
         .eof()
         .into_iter()
         .filter_map(|step| match step {
            Step::Failed { message, .. } => Some(error("overloaded_error", message)),
            Step::Start { .. }
            | Step::OpenThinking
            | Step::Thinking(..)
            | Step::Signature(..)
            | Step::OpenText
            | Step::Text(..)
            | Step::OpenCall { .. }
            | Step::Args(..)
            | Step::CloseBlock
            | Step::Stop { .. } => None,
         })
         .collect()
   }

   fn render(&mut self, out: &mut Vec<OutEvent>, step: Step) {
      if self.hiding && !matches!(step, Step::Failed { .. }) {
         self.hiding = !matches!(step, Step::CloseBlock);
         return;
      }
      match step {
         Step::Start { id } => {
            let id = id.unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple()));
            out.push(
               AnthEvent::MessageStart {
                  message: MessageStart {
                     id,
                     kind: "message",
                     role: "assistant",
                     model: self.model.clone(),
                     content: [],
                     stop_reason: None,
                     stop_sequence: None,
                     usage: AnthUsage {
                        input_tokens: self.est_input_tokens,
                        output_tokens: 1,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                     },
                  },
               }
               .out(),
            );
            out.push(AnthEvent::Ping.out());
         },
         Step::OpenThinking if !self.emit_thinking => self.hiding = true,
         Step::OpenThinking => self.open(out, ContentBlockStart::Thinking { thinking: "" }),
         Step::OpenText => self.open(out, ContentBlockStart::Text { text: "" }),
         Step::OpenCall { id, name } => self.open(
            out,
            ContentBlockStart::ToolUse {
               id,
               name,
               input: EmptyObject,
            },
         ),
         Step::Thinking(thinking) => out.push(self.delta(BlockDelta::Thinking { thinking })),
         Step::Signature(signature) => {
            out.push(self.delta(BlockDelta::Signature { signature }));
         },
         Step::Text(text) => out.push(self.delta(BlockDelta::Text { text })),
         Step::Args(partial_json) => {
            out.push(self.delta(BlockDelta::InputJson { partial_json }));
         },
         Step::CloseBlock => out.push(
            AnthEvent::ContentBlockStop {
               index: self.next_index - 1,
            }
            .out(),
         ),
         Step::Stop { kind, usage } => {
            out.push(
               AnthEvent::MessageDelta {
                  delta: StopDelta {
                     stop_reason: kind.as_str(),
                     stop_sequence: None,
                  },
                  usage: anthropic_usage(&usage),
               }
               .out(),
            );
            out.push(AnthEvent::MessageStop.out());
         },
         Step::Failed { message, code } => {
            let rate_limited = code.is_some_and(|code| code.contains("rate_limit"))
               || message.contains("rate limit");
            let kind = if rate_limited {
               "rate_limit_error"
            } else {
               "api_error"
            };
            out.push(error(kind, message));
         },
      }
   }

   fn open(&mut self, out: &mut Vec<OutEvent>, content_block: ContentBlockStart) {
      let index = self.next_index;
      self.next_index += 1;
      out.push(
         AnthEvent::ContentBlockStart {
            index,
            content_block,
         }
         .out(),
      );
   }

   fn delta(&self, delta: BlockDelta) -> OutEvent {
      AnthEvent::ContentBlockDelta {
         index: self.next_index - 1,
         delta,
      }
      .out()
   }
}

fn error(kind: &'static str, message: String) -> OutEvent {
   AnthEvent::Error {
      error: ErrorBody { kind, message },
   }
   .out()
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RenderedBlock {
   Thinking {
      thinking: String,
      signature: String,
   },
   Text {
      text: String,
   },
   ToolUse {
      id: String,
      name: String,
      input: Box<RawValue>,
   },
}

#[derive(Serialize)]
pub struct RenderedMessage {
   id: String,
   #[serde(rename = "type")]
   kind: &'static str,
   role: &'static str,
   model: String,
   content: Vec<RenderedBlock>,
   stop_reason: &'static str,
   stop_sequence: Option<String>,
   usage: AnthUsage,
}

pub fn render_aggregated(agg: &Aggregated, model: &str, emit_thinking: bool) -> RenderedMessage {
   let mut content = Vec::new();
   for block in &agg.blocks {
      match *block {
         Block::Thinking {
            ref text,
            ref signature,
         } => {
            if emit_thinking {
               content.push(RenderedBlock::Thinking {
                  thinking: text.clone(),
                  signature: signature.clone().unwrap_or_default(),
               });
            }
         },
         Block::Text { ref text } => content.push(RenderedBlock::Text { text: text.clone() }),
         Block::ToolCall {
            ref id,
            ref name,
            ref arguments,
         } => {
            let input = RawValue::from_string(arguments.clone())
               .unwrap_or_else(|_| RawValue::from_string("{}".into()).expect("literal"));
            content.push(RenderedBlock::ToolUse {
               id: id.clone(),
               name: name.clone(),
               input,
            });
         },
      }
   }
   let stop_reason = match agg.stop {
      StopKind::ToolUse => "tool_use",
      StopKind::MaxTokens => "max_tokens",
      StopKind::EndTurn | StopKind::Error => "end_turn",
   };
   RenderedMessage {
      id: agg.id.clone(),
      kind: "message",
      role: "assistant",
      model: model.to_owned(),
      content,
      stop_reason,
      stop_sequence: None,
      usage: anthropic_usage(&agg.usage),
   }
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::codex::types::{OutputItem, ResponseObj};

   #[test]
   fn hidden_thinking_leaves_no_gap_in_block_indices() {
      let mut stream = AnthropicStream::new("m".into(), 1, false, UsageCapture::default());
      let reasoning = OutputItem::Reasoning {
         id: Some("rs".into()),
         summary: None,
         encrypted_content: Some("ENC".into()),
      };
      let mut frames = Vec::new();
      for event in [
         ResponsesEvent::OutputItemAdded {
            output_index: 0,
            item: reasoning.clone(),
         },
         ResponsesEvent::ReasoningSummaryTextDelta {
            delta: "secret".into(),
         },
         ResponsesEvent::OutputItemDone {
            output_index: 0,
            item: reasoning,
         },
         ResponsesEvent::OutputTextDelta {
            item_id: None,
            output_index: 1,
            content_index: 0,
            delta: "hi".into(),
         },
         ResponsesEvent::Completed {
            response: ResponseObj::default(),
         },
      ] {
         frames.extend(stream.handle(event));
      }
      let names = frames.iter().map(|&(name, _)| name).collect::<Vec<_>>();
      assert_eq!(
         names,
         [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
         ]
      );
      assert!(frames[0].1.contains("\"index\":0"));
      assert!(!frames.iter().any(|&(_, ref body)| body.contains("secret")));
   }
}
