use std::mem;

use serde::Serialize;

use super::chat::{
   ChatChoice, ChatChunk, ChatCompletion, ChatContent, ChatDelta, ChatError, ChatErrorBody,
   ChatMessage, ChatToolCall, ChatUsage, ChunkChoice, FinishReason, FunctionBody,
};
use super::{Aggregated, Block, BlockEvent, Step, StopKind, UsageCapture, Walker};
use crate::clock::unix_now;
use crate::codex::types::ResponsesEvent;

pub struct OpenAiStream {
   model: String,
   id: String,
   created: i64,
   include_usage: bool,
   blocks: Vec<Channel>,
   tool_count: u64,
   reasoning_seen: bool,
   separator_due: bool,
   walker: Walker,
}

enum Channel {
   Text,
   Thinking,
   Call(u64),
}

fn to_json<T>(payload: T) -> String
where
   T: Serialize,
{
   serde_json::to_string(&payload).expect("chunk serializes")
}

const fn finish_reason(kind: StopKind) -> FinishReason {
   match kind {
      StopKind::ToolUse => FinishReason::ToolCalls,
      StopKind::MaxTokens => FinishReason::Length,
      StopKind::EndTurn | StopKind::Error => FinishReason::Stop,
   }
}

fn error_chunk(message: String) -> String {
   to_json(ChatError {
      error: ChatErrorBody {
         message,
         kind: Some("api_error".into()),
         code: None,
      },
   })
}

impl OpenAiStream {
   pub fn new(model: String, include_usage: bool, capture: UsageCapture) -> Self {
      Self {
         model,
         id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
         created: unix_now(),
         include_usage,
         blocks: Vec::new(),
         tool_count: 0,
         reasoning_seen: false,
         separator_due: false,
         walker: Walker::new(capture),
      }
   }

   pub fn handle(&mut self, event: ResponsesEvent) -> Vec<String> {
      let mut out = Vec::new();
      for step in self.walker.step(event) {
         self.render(&mut out, step);
      }
      out
   }

   pub fn finalize(&mut self) -> Vec<String> {
      let mut out = Vec::new();
      for step in self.walker.eof() {
         self.render(&mut out, step);
      }
      out
   }

   fn render(&mut self, out: &mut Vec<String>, step: Step) {
      match step {
         Step::Start { id } => {
            if let Some(id) = id {
               self.id = format!("chatcmpl-{id}");
            }
            out.push(self.chunk(
               ChatDelta {
                  role: Some("assistant".into()),
                  content: Some(String::new()),
                  ..Default::default()
               },
               None,
            ));
         },
         Step::Block {
            event: BlockEvent::Open(Block::Text { .. }),
            ..
         } => self.blocks.push(Channel::Text),
         Step::Block {
            event: BlockEvent::Open(Block::Thinking { .. }),
            ..
         } => {
            self.blocks.push(Channel::Thinking);
            self.separator_due = self.reasoning_seen;
         },
         Step::Block {
            event: BlockEvent::Open(Block::ToolCall { id, name, .. }),
            ..
         } => {
            let index = self.tool_count;
            self.tool_count += 1;
            self.blocks.push(Channel::Call(index));
            out.push(self.chunk(
               ChatDelta {
                  tool_calls: Some(vec![ChatToolCall {
                     index: Some(index),
                     id: Some(id),
                     kind: Some("function".into()),
                     function: FunctionBody {
                        name: Some(name),
                        arguments: Some(String::new()),
                     },
                     extra_content: None,
                  }]),
                  ..Default::default()
               },
               None,
            ));
         },
         Step::Block {
            index,
            event: BlockEvent::Append(text),
         } => {
            let mut delta = ChatDelta::default();
            match self.blocks[index] {
               Channel::Text => delta.content = Some(text),
               Channel::Thinking => {
                  delta.reasoning_content = Some(if mem::take(&mut self.separator_due) {
                     format!("\n\n{text}")
                  } else {
                     text
                  });
                  self.reasoning_seen = true;
               },
               Channel::Call(index) => {
                  delta.tool_calls = Some(vec![ChatToolCall {
                     index: Some(index),
                     function: FunctionBody {
                        name: None,
                        arguments: Some(text),
                     },
                     ..Default::default()
                  }]);
               },
            }
            out.push(self.chunk(delta, None));
         },
         Step::Stop { kind, usage } => {
            out.push(self.chunk(ChatDelta::default(), Some(finish_reason(kind))));
            if self.include_usage {
               out.push(to_json(ChatChunk {
                  id: self.id.clone(),
                  object: "chat.completion.chunk".into(),
                  created: self.created,
                  model: self.model.clone(),
                  choices: Vec::new(),
                  usage: Some(ChatUsage::from(&usage)),
                  error: None,
               }));
            }
         },
         Step::Failed { message, .. } => out.push(error_chunk(message)),
         Step::Block { .. } => {},
      }
   }

   fn chunk(&self, delta: ChatDelta, finish_reason: Option<FinishReason>) -> String {
      to_json(ChatChunk {
         id: self.id.clone(),
         object: "chat.completion.chunk".into(),
         created: self.created,
         model: self.model.clone(),
         choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
            logprobs: None,
         }],
         usage: None,
         error: None,
      })
   }
}

pub fn render_aggregated(agg: &Aggregated, model: &str) -> ChatCompletion {
   let mut text = String::new();
   let mut reasoning = String::new();
   let mut tool_calls = Vec::new();
   for block in &agg.blocks {
      match *block {
         Block::Text { text: ref content } => text.push_str(content),
         Block::Thinking {
            text: ref thinking, ..
         } => {
            if !reasoning.is_empty() {
               reasoning.push_str("\n\n");
            }
            reasoning.push_str(thinking);
         },
         Block::ToolCall {
            ref id,
            ref name,
            ref arguments,
         } => {
            tool_calls.push(ChatToolCall {
               id: Some(id.clone()),
               kind: Some("function".into()),
               function: FunctionBody {
                  name: Some(name.clone()),
                  arguments: Some(arguments.clone()),
               },
               ..Default::default()
            });
         },
      }
   }

   ChatCompletion {
      id: format!("chatcmpl-{}", agg.id),
      object: "chat.completion".into(),
      created: unix_now(),
      model: model.to_owned(),
      choices: vec![ChatChoice {
         index: 0,
         message: ChatMessage {
            role: "assistant".into(),
            content: Some(ChatContent::Text(text)),
            reasoning_content: Some(reasoning).filter(|reason| !reason.is_empty()),
            tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
            ..Default::default()
         },
         finish_reason: Some(finish_reason(agg.stop)),
         logprobs: None,
      }],
      usage: Some(ChatUsage::from(&agg.usage)),
   }
}

#[cfg(test)]
mod tests {
   use serde_json::json;

   use super::*;
   use crate::codex::types::{OutputItem, ResponseObj, UpstreamError};

   #[test]
   fn tool_call_indices_start_at_zero_and_count_up() {
      let mut stream = OpenAiStream::new("m".into(), false, UsageCapture::default());
      let mut indices = Vec::new();
      for index in 0..3 {
         let frames = stream.handle(ResponsesEvent::OutputItemAdded {
            output_index: index,
            item: OutputItem::FunctionCall {
               id: None,
               call_id: "c".into(),
               name: "f".into(),
               arguments: Some(String::new()),
               status: None,
            },
         });
         assert_eq!(frames.len(), 1);
         let value: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
         indices.push(
            value["choices"][0]["delta"]["tool_calls"][0]["index"]
               .as_u64()
               .unwrap(),
         );
      }
      assert_eq!(indices, vec![0, 1, 2]);
   }

   #[test]
   fn the_failure_frame_names_its_type_and_a_null_code() {
      let mut stream = OpenAiStream::new("m".into(), false, UsageCapture::default());
      let frames = stream.handle(ResponsesEvent::Failed {
         response: ResponseObj {
            error: Some(UpstreamError {
               code: None,
               message: Some("boom".into()),
            }),
            ..Default::default()
         },
      });
      assert_eq!(frames.len(), 1);
      let value: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
      assert_eq!(
         value,
         json!({"error": {"message": "boom", "type": "api_error", "code": null}})
      );
   }
}
