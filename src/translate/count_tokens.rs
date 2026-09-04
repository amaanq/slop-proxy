use crate::codex::types::{ContentPart, InputItem, ResponsesRequest, SummaryPart};

/// Rough chars/4 estimate.
pub fn estimate(req: &ResponsesRequest) -> i64 {
   let mut chars = req.instructions.len();
   let mut per_item_overhead = 0_i64;
   let mut image_tokens = 0_i64;

   for item in &req.input {
      per_item_overhead += 4;
      match *item {
         InputItem::Message { ref content, .. } => {
            for part in content {
               match *part {
                  ContentPart::InputText { ref text } | ContentPart::OutputText { ref text } => {
                     chars += text.len();
                  },
                  ContentPart::InputImage { .. } => image_tokens += 1000,
                  ContentPart::Other => {},
               }
            }
         },
         InputItem::FunctionCall {
            ref name,
            ref arguments,
            ..
         } => chars += name.len() + arguments.len(),
         InputItem::FunctionCallOutput { ref output, .. }
         | InputItem::CustomToolCallOutput { ref output, .. } => chars += output.text().len(),
         InputItem::CustomToolCall {
            ref name,
            ref input,
            ..
         } => chars += name.len() + input.len(),
         InputItem::AdditionalTools { ref tools, .. } => {
            for tool in tools {
               chars += serde_json::to_string(tool).map_or(0, |text| text.len());
            }
         },
         InputItem::Other => {},
         InputItem::Reasoning { ref summary, .. } => {
            for &SummaryPart::SummaryText { ref text } in summary {
               chars += text.len();
            }
         },
      }
   }
   for tool in &req.tools {
      chars += serde_json::to_string(tool).map_or(0, |text| text.len());
   }

   (chars as i64) / 4 + per_item_overhead + image_tokens + 30
}
