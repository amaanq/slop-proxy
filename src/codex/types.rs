use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub instructions: String,
    pub input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

impl ResponsesRequest {
    pub fn new(model: String, instructions: String) -> Self {
        Self {
            model,
            instructions,
            input: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            max_output_tokens: None,
            store: false,
            stream: true,
            include: vec!["reasoning.encrypted_content".into()],
            prompt_cache_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningConfig {
    pub effort: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: Vec<ContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        summary: Vec<SummaryPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SummaryPart {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub strict: bool,
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesEvent {
    #[serde(rename = "response.created")]
    Created { response: ResponseObj },
    #[serde(rename = "response.in_progress")]
    InProgress {},
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: OutputItem },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: OutputItem },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {},
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {},
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { delta: String },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {},
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {},
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {},
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta { delta: String },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {},
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta { delta: String },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {},
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { delta: String },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {},
    #[serde(rename = "response.completed")]
    Completed { response: ResponseObj },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseObj },
    #[serde(rename = "response.failed")]
    Failed { response: ResponseObj },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponseObj {
    pub id: Option<String>,
    pub usage: Option<Usage>,
    pub error: Option<UpstreamError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamError {
    pub code: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub input_tokens_details: TokenDetails,
    #[serde(default)]
    pub output_tokens_details: TokenDetails,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenDetails {
    #[serde(default)]
    pub cached_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        content: Option<Vec<Value>>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Option<Vec<SummaryPart>>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: Option<String>,
    },
    #[serde(other)]
    Other,
}
