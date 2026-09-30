use crate::codex::types::{
   ContentPart, InputItem, ResponsesRequest, ToolChoice, ToolDef, ToolOutput,
};
use crate::config::Config;
use crate::translate::TranslateError;
use crate::translate::chat::{ChatContent, ChatMessage, ChatPart, ChatRequest, ChatToolChoice};
use crate::translate::responses_request;

pub fn to_responses(req: &ChatRequest, cfg: &Config) -> Result<ResponsesRequest, TranslateError> {
   let mut out = responses_request(
      cfg,
      &req.model,
      req.reasoning_effort.clone(),
      req.max_completion_tokens.or(req.max_tokens),
   );

   for msg in &req.messages {
      convert_message(msg, &mut out.input)?;
   }

   if let Some(tools) = req.tools.as_ref() {
      for tool in tools {
         let def = tool.def();
         let Some(name) = def.name.as_ref() else {
            continue;
         };
         out.tools.push(ToolDef::function(
            name.clone(),
            def.description.clone(),
            def.parameters.clone(),
         ));
      }
   }

   if let Some(tool_choice) = req.tool_choice.as_ref() {
      out.tool_choice = Some(match *tool_choice {
         ChatToolChoice::Mode(ref mode) => ToolChoice::Mode(mode.clone()),
         ChatToolChoice::Named { ref function, .. } => ToolChoice::function(
            function
               .as_ref()
               .and_then(|named| named.name.clone())
               .unwrap_or_default(),
         ),
      });
   }
   out.parallel_tool_calls = req.parallel_tool_calls;

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
         if let Some(calls) = msg.tool_calls.as_ref() {
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
      Some(&ChatContent::Text(ref text)) => vec![ContentPart::InputText { text: text.clone() }],
      Some(&ChatContent::Parts(ref parts)) => parts
         .iter()
         .filter_map(|part| match *part {
            ChatPart::Text { ref text } => Some(ContentPart::InputText { text: text.clone() }),
            ChatPart::ImageUrl { ref image_url } => Some(ContentPart::InputImage {
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
