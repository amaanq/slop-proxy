use crate::codex::types::{ContentPart, InputItem, ResponsesRequest};

/// Rough chars/4 estimate.
pub fn estimate(req: &ResponsesRequest) -> i64 {
    let mut chars = req.instructions.len();
    let mut per_item_overhead = 0i64;
    let mut image_tokens = 0i64;

    for item in &req.input {
        per_item_overhead += 4;
        match item {
            InputItem::Message { content, .. } => {
                for part in content {
                    match part {
                        ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                            chars += text.len()
                        }
                        ContentPart::InputImage { .. } => image_tokens += 1000,
                        ContentPart::Other => {}
                    }
                }
            }
            InputItem::FunctionCall {
                name, arguments, ..
            } => chars += name.len() + arguments.len(),
            InputItem::FunctionCallOutput { output, .. }
            | InputItem::CustomToolCallOutput { output, .. } => chars += output.text().len(),
            InputItem::CustomToolCall { name, input, .. } => chars += name.len() + input.len(),
            InputItem::AdditionalTools { tools, .. } => {
                for tool in tools {
                    chars += serde_json::to_string(tool).map(|s| s.len()).unwrap_or(0);
                }
            }
            InputItem::Other => {}
            InputItem::Reasoning { summary, .. } => {
                for crate::codex::types::SummaryPart::SummaryText { text } in summary {
                    chars += text.len();
                }
            }
        }
    }
    for tool in &req.tools {
        chars += serde_json::to_string(tool).map(|s| s.len()).unwrap_or(0);
    }

    (chars as i64) / 4 + per_item_overhead + image_tokens + 30
}
