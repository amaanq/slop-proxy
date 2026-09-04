use super::TranslateError;
use super::anthropic_req::empty_schema;
use super::chat::{ChatContent, ChatMessage, ChatPart, ChatRequest, ChatToolChoice};
use super::model_map;
use crate::codex::types::{
   ContentPart, InputItem, ReasoningConfig, ResponsesRequest, ToolChoice, ToolDef, ToolOutput,
};
use crate::config::Config;

pub fn to_responses(req: &ChatRequest, cfg: &Config) -> Result<ResponsesRequest, TranslateError> {
   let resolved = model_map::resolve(&cfg.models, &req.model);
   let mut out = ResponsesRequest::new(resolved.model.clone(), cfg.codex.instructions());

   for msg in &req.messages {
      convert_message(msg, &mut out.input)?;
   }

   if let Some(tools) = &req.tools {
      for t in tools {
         let f = t.def();
         let Some(name) = &f.name else {
            continue;
         };
         out.tools.push(ToolDef {
            kind: "function".into(),
            name: name.clone(),
            description: f.description.clone(),
            strict: false,
            parameters: Some(f.parameters.clone().unwrap_or_else(empty_schema)),
         });
      }
   }

   if let Some(tc) = &req.tool_choice {
      out.tool_choice = Some(match tc {
         ChatToolChoice::Mode(s) => ToolChoice::Mode(s.clone()),
         ChatToolChoice::Named { function, .. } => ToolChoice::function(
            function
               .as_ref()
               .and_then(|f| f.name.clone())
               .unwrap_or_default(),
         ),
      });
   }
   out.parallel_tool_calls = req.parallel_tool_calls;

   let effort = req
      .reasoning_effort
      .clone()
      .or(resolved.effort)
      .unwrap_or_else(|| "medium".into());
   out.reasoning = Some(ReasoningConfig {
      effort: model_map::clamp_effort(&out.model, &effort),
      summary: "auto".into(),
   });

   if cfg.codex.forward_max_tokens {
      out.max_output_tokens = req.max_completion_tokens.or(req.max_tokens);
   }

   Ok(out)
}

fn convert_message(msg: &ChatMessage, out: &mut Vec<InputItem>) -> Result<(), TranslateError> {
   match msg.role.as_str() {
      "system" | "developer" => {
         let text = msg.text();
         if !text.is_empty() {
            out.push(InputItem::Message {
               role: "developer".into(),
               content: vec![ContentPart::InputText { text }],
            });
         }
      },
      "user" => {
         let parts = user_parts(msg.content.as_ref());
         if !parts.is_empty() {
            out.push(InputItem::Message {
               role: "user".into(),
               content: parts,
            });
         }
      },
      "assistant" => {
         let text = msg.text();
         if !text.is_empty() {
            out.push(InputItem::Message {
               role: "assistant".into(),
               content: vec![ContentPart::OutputText { text }],
            });
         }
         if let Some(calls) = &msg.tool_calls {
            for call in calls {
               out.push(InputItem::FunctionCall {
                  call_id: call.id.clone().unwrap_or_default(),
                  name: call.function.name.clone().unwrap_or_default(),
                  arguments: call
                     .function
                     .arguments
                     .clone()
                     .unwrap_or_else(|| "{}".into()),
               });
            }
         }
      },
      "tool" => {
         out.push(InputItem::FunctionCallOutput {
            call_id: msg
               .tool_call_id
               .clone()
               .ok_or(TranslateError::ToolMessageWithoutCallId)?,
            output: ToolOutput::Text(msg.text()),
         });
      },
      other => return Err(TranslateError::UnsupportedRole(other.to_owned())),
   }
   Ok(())
}

fn user_parts(content: Option<&ChatContent>) -> Vec<ContentPart> {
   match content {
      Some(ChatContent::Text(s)) => vec![ContentPart::InputText { text: s.clone() }],
      Some(ChatContent::Parts(parts)) => parts
         .iter()
         .filter_map(|p| match p {
            ChatPart::Text { text } => Some(ContentPart::InputText { text: text.clone() }),
            ChatPart::ImageUrl { image_url } => Some(ContentPart::InputImage {
               image_url: image_url.url().to_owned(),
            }),
            ChatPart::InputAudio { .. } | ChatPart::Other => {
               tracing::debug!("dropping unsupported openai part");
               None
            },
         })
         .collect(),
      None => Vec::new(),
   }
}
