//! The chat-completions wire, request and response, both directions.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::codex::types::{TokenDetails, Usage};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

impl ChatRequest {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaFormat>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct JsonSchemaFormat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatMessage {
    #[serde(default = "default_role")]
    pub role: String,
    pub content: Option<ChatContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChatPart>>,
}

fn default_role() -> String {
    "user".into()
}

impl ChatMessage {
    pub fn text(&self) -> String {
        match &self.content {
            Some(ChatContent::Text(s)) => s.clone(),
            Some(ChatContent::Parts(parts)) => parts
                .iter()
                .filter_map(|p| match p {
                    ChatPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
            None => String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatPart {
    #[serde(alias = "input_text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(alias = "input_image", alias = "image")]
    ImageUrl {
        #[serde(alias = "image")]
        image_url: ImageRef,
    },
    InputAudio {
        input_audio: AudioData,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageRef {
    Url(String),
    Object { url: String },
}

impl ImageRef {
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) | Self::Object { url } => url,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioData {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: FunctionBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<ExtraContent>,
}

impl ChatToolCall {
    pub fn thought_signature(&self) -> Option<&str> {
        self.extra_content
            .as_ref()?
            .google
            .as_ref()?
            .thought_signature
            .as_deref()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FunctionBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtraContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google: Option<GoogleExtra>,
}

impl ExtraContent {
    pub fn with_signature(sig: &str) -> Self {
        Self {
            google: Some(GoogleExtra {
                thought_signature: Some(sig.to_string()),
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GoogleExtra {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatToolDef {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionDef>,
    #[serde(flatten)]
    pub flat: FunctionDef,
}

impl ChatToolDef {
    pub fn def(&self) -> &FunctionDef {
        self.function.as_ref().unwrap_or(&self.flat)
    }

    pub fn function(def: FunctionDef) -> Self {
        Self {
            kind: Some("function".into()),
            function: Some(def),
            flat: FunctionDef::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FunctionDef {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatToolChoice {
    Mode(String),
    Named {
        #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        function: Option<NamedFunction>,
    },
}

impl ChatToolChoice {
    pub fn function(name: String) -> Self {
        Self::Named {
            kind: Some("function".into()),
            function: Some(NamedFunction { name: Some(name) }),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NamedFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The chat-completions usage block, which names the same quantities the
/// Responses API reports under different keys.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatUsage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptTokensDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: i64,
}

impl From<ChatUsage> for Usage {
    /// Google leaves thinking out of `completion_tokens` and reports it only
    /// in `total_tokens`.
    fn from(c: ChatUsage) -> Self {
        let billed_output = (c.total_tokens - c.prompt_tokens).max(c.completion_tokens);
        let mut reasoning_tokens = c.completion_tokens_details.reasoning_tokens;
        if reasoning_tokens == 0 {
            reasoning_tokens = billed_output - c.completion_tokens;
        }
        Usage {
            input_tokens: c.prompt_tokens,
            output_tokens: billed_output,
            total_tokens: c.prompt_tokens + billed_output,
            input_tokens_details: TokenDetails {
                cached_tokens: c.prompt_tokens_details.cached_tokens,
                reasoning_tokens: 0,
            },
            output_tokens_details: TokenDetails {
                cached_tokens: 0,
                reasoning_tokens,
            },
        }
    }
}

impl From<&Usage> for ChatUsage {
    fn from(u: &Usage) -> Self {
        Self {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: u.input_tokens_details.cached_tokens,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: u.output_tokens_details.reasoning_tokens,
            },
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatEnvelope {
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatChoice {
    pub index: u64,
    pub message: ChatMessage,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<()>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkChoice {
    pub index: u64,
    pub delta: ChatDelta,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<()>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChatPart>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatError {
    pub error: ChatErrorBody,
}

impl ChatError {
    /// Anthropic, Google and Zen differ only in the sibling fields.
    pub fn reason(body: String) -> String {
        match serde_json::from_str::<Self>(&body) {
            Ok(env) if !env.error.message.is_empty() => env.error.message,
            _ => body,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatErrorBody {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub code: Option<ErrorCode>,
}

/// OpenAI sends a string, Google a number.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ErrorCode {
    Text(String),
    Number(i64),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn thinking_is_recovered_from_the_total() {
        let u: Usage = ChatUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 71,
            ..Default::default()
        }
        .into();
        assert_eq!(u.input_tokens, 3);
        assert_eq!(u.total_tokens, 71);
        assert_eq!(u.output_tokens, 68);
        assert_eq!(u.output_tokens_details.reasoning_tokens, 66);
    }

    #[test]
    fn cached_prompt_tokens_survive() {
        let u: Usage = ChatUsage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 90 },
            ..Default::default()
        }
        .into();
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.input_tokens_details.cached_tokens, 90);
    }

    #[test]
    fn usage_details_carry_only_their_own_key() {
        let v = serde_json::to_value(ChatUsage::from(&Usage {
            input_tokens: 5,
            output_tokens: 7,
            input_tokens_details: TokenDetails {
                cached_tokens: 2,
                reasoning_tokens: 0,
            },
            output_tokens_details: TokenDetails {
                cached_tokens: 0,
                reasoning_tokens: 3,
            },
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(v["prompt_tokens_details"], json!({"cached_tokens": 2}));
        assert_eq!(
            v["completion_tokens_details"],
            json!({"reasoning_tokens": 3})
        );
        assert_eq!(v["total_tokens"], 12);
    }

    #[test]
    fn an_error_frame_always_carries_a_code_key() {
        let v = serde_json::to_value(ChatError {
            error: ChatErrorBody {
                message: "m".into(),
                kind: Some("api_error".into()),
                code: None,
            },
        })
        .unwrap();
        assert!(v["error"].get("code").is_some());
        assert_eq!(v["error"]["code"], serde_json::Value::Null);
    }

    #[test]
    fn a_bare_image_part_type_is_accepted() {
        let part: ChatPart = serde_json::from_value(json!({
            "type": "image",
            "image_url": {"url": "u"},
        }))
        .unwrap();
        let ChatPart::ImageUrl { image_url } = part else {
            panic!("expected image part");
        };
        assert_eq!(image_url.url(), "u");
        let part: ChatPart =
            serde_json::from_value(json!({"type": "image_url", "image_url": "u"})).unwrap();
        let ChatPart::ImageUrl { image_url } = part else {
            panic!("expected image part");
        };
        assert_eq!(image_url.url(), "u");
    }

    #[test]
    fn an_unknown_finish_reason_still_finishes() {
        let choice: ChunkChoice =
            serde_json::from_value(json!({"delta": {}, "finish_reason": "weird"})).unwrap();
        assert_eq!(choice.finish_reason, Some(FinishReason::Other));
    }
}
