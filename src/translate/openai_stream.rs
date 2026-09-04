use serde::Serialize;

use super::chat::{
   ChatChoice, ChatChunk, ChatCompletion, ChatContent, ChatDelta, ChatError, ChatErrorBody,
   ChatMessage, ChatToolCall, ChatUsage, ChunkChoice, FinishReason, FunctionBody,
};
use super::{Aggregated, Block, Step, StopKind, UsageCapture, Walker};
use crate::codex::types::ResponsesEvent;

pub struct OpenAiStream {
   model: String,
   id: String,
   created: i64,
   include_usage: bool,
   tool_index: Option<u64>,
   reasoning_seen: bool,
   separator_due: bool,
   walker: Walker,
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
         created: crate::clock::unix_now(),
         include_usage,
         tool_index: None,
         reasoning_seen: false,
         separator_due: false,
         walker: Walker::new(capture),
      }
   }

   pub fn handle(&mut self, ev: ResponsesEvent) -> Vec<String> {
      let mut out = Vec::new();
      for step in self.walker.step(ev) {
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
         Step::Text(content) => out.push(self.chunk(
            ChatDelta {
               content: Some(content),
               ..Default::default()
            },
            None,
         )),
         Step::OpenThinking => self.separator_due = self.reasoning_seen,
         Step::Thinking(delta) => {
            let reasoning_content = if std::mem::take(&mut self.separator_due) {
               format!("\n\n{delta}")
            } else {
               delta
            };
            self.reasoning_seen = true;
            out.push(self.chunk(
               ChatDelta {
                  reasoning_content: Some(reasoning_content),
                  ..Default::default()
               },
               None,
            ));
         },
         Step::OpenCall { id, name } => {
            let index = self.tool_index.map_or(0, |i| i + 1);
            self.tool_index = Some(index);
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
         Step::Args(arguments) => out.push(self.chunk(
            ChatDelta {
               tool_calls: Some(vec![ChatToolCall {
                  index: self.tool_index,
                  function: FunctionBody {
                     name: None,
                     arguments: Some(arguments),
                  },
                  ..Default::default()
               }]),
               ..Default::default()
            },
            None,
         )),
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
         Step::OpenText | Step::Signature(_) | Step::CloseBlock => {},
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
      match block {
         Block::Text { text: t } => text.push_str(t),
         Block::Thinking { text: t, .. } => {
            if !reasoning.is_empty() {
               reasoning.push_str("\n\n");
            }
            reasoning.push_str(t);
         },
         Block::ToolCall {
            id,
            name,
            arguments,
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
      created: crate::clock::unix_now(),
      model: model.to_owned(),
      choices: vec![ChatChoice {
         index: 0,
         message: ChatMessage {
            role: "assistant".into(),
            content: Some(ChatContent::Text(text)),
            reasoning_content: Some(reasoning).filter(|r| !r.is_empty()),
            tool_calls: Some(tool_calls).filter(|t| !t.is_empty()),
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
      for _ in 0..3 {
         let frames = stream.handle(ResponsesEvent::OutputItemAdded {
            output_index: 0,
            item: OutputItem::FunctionCall {
               id: None,
               call_id: "c".into(),
               name: "f".into(),
               arguments: Some(String::new()),
               status: None,
            },
         });
         assert_eq!(frames.len(), 1);
         let v: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
         indices.push(
            v["choices"][0]["delta"]["tool_calls"][0]["index"]
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
      let v: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
      assert_eq!(
         v,
         json!({"error": {"message": "boom", "type": "api_error", "code": null}})
      );
   }
}
