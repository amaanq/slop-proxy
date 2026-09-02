use serde::Deserialize;

use crate::codex::types::{TokenDetails, Usage};

/// The chat-completions usage block, which names the same quantities the
/// Responses API reports under different keys.
#[derive(Debug, Default, Deserialize)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub prompt_tokens_details: TokenDetails,
    #[serde(default)]
    pub completion_tokens_details: TokenDetails,
}

impl From<ChatUsage> for Usage {
    /// Google leaves thinking out of `completion_tokens` and reports it only
    /// in `total_tokens`.
    fn from(c: ChatUsage) -> Self {
        let billed_output = (c.total_tokens - c.prompt_tokens).max(c.completion_tokens);
        let mut output_details = c.completion_tokens_details;
        if output_details.reasoning_tokens == 0 {
            output_details.reasoning_tokens = billed_output - c.completion_tokens;
        }
        Usage {
            input_tokens: c.prompt_tokens,
            output_tokens: billed_output,
            input_tokens_details: c.prompt_tokens_details,
            output_tokens_details: output_details,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatEnvelope {
    pub usage: Option<ChatUsage>,
}

/// A chat `usage` block restated in the keys `Usage` deserializes from. The
/// bridge hands its frames to the Responses parser, which reads neither
/// `prompt_tokens` nor `completion_tokens` and would default every field to
/// zero.
pub fn as_responses(usage: &serde_json::Value) -> Option<serde_json::Value> {
    let chat: ChatUsage = serde_json::from_value(usage.clone()).ok()?;
    let u: Usage = chat.into();
    Some(serde_json::json!({
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        // Codex refuses a completed event without it.
        "total_tokens": u.input_tokens + u.output_tokens,
        "input_tokens_details": {"cached_tokens": u.input_tokens_details.cached_tokens},
        "output_tokens_details": {"reasoning_tokens": u.output_tokens_details.reasoning_tokens},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thinking_is_recovered_from_the_total() {
        let out = as_responses(&json!({
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 71
        }))
        .unwrap();
        assert_eq!(out["input_tokens"], 3);
        assert_eq!(out["total_tokens"], 71);
        assert_eq!(out["output_tokens"], 68);
        assert_eq!(out["output_tokens_details"]["reasoning_tokens"], 66);
    }

    #[test]
    fn cached_prompt_tokens_survive() {
        let out = as_responses(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 90}
        }))
        .unwrap();
        assert_eq!(out["input_tokens"], 100);
        assert_eq!(out["input_tokens_details"]["cached_tokens"], 90);
    }
}
