use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::value::{RawValue, to_raw_value};
use serde_json::{Map, Value};

use crate::gemini::types::{
    ApiError, Blob, Candidate, Content, FileData, FinishReason, FunctionCall,
    FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration, FunctionResponse,
    GenerateContentRequest, GenerateContentResponse, GenerationConfig, Part, Tool, ToolConfig,
    UsageMetadata,
};
use crate::translate::chat::{
    self, ChatChoice, ChatChunk, ChatCompletion, ChatContent, ChatDelta, ChatMessage, ChatPart,
    ChatRequest, ChatToolCall, ChatToolChoice, ChatUsage, ChunkChoice, CompletionTokensDetails,
    ExtraContent, FunctionBody, ImageRef, PromptTokensDetails,
};

#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    #[error("invalid gemini model name")]
    InvalidModel,
    #[error("tool call is missing function name")]
    CallWithoutName,
    #[error("tool call arguments are not valid JSON: {0}")]
    CallArguments(serde_json::Error),
    #[error("tool message is missing tool_call_id")]
    ToolMessageWithoutCallId,
    #[error("tool message references unknown call {0}")]
    UnknownCall(String),
    #[error("unsupported message content type")]
    UnsupportedPart,
    #[error("invalid data URL")]
    InvalidDataUrl,
    #[error("native gemini only supports function tools")]
    NonFunctionTool,
    #[error("function tool is missing its declaration")]
    ToolWithoutDeclaration,
    #[error("invalid native gemini response: {0}")]
    Response(serde_json::Error),
}

#[derive(Debug)]
pub struct NativeRequest {
    pub model: String,
    pub streaming: bool,
    pub body: GenerateContentRequest,
}

pub fn request(req: &ChatRequest) -> Result<NativeRequest, NativeError> {
    let model = req.model.trim_start_matches("models/").to_owned();
    if model.is_empty()
        || !model
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(NativeError::InvalidModel);
    }

    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut call_names = BTreeMap::new();

    for message in &req.messages {
        let role = message.role.as_str();
        if matches!(role, "system" | "developer") {
            system_parts.extend(content_parts(message.content.as_ref())?);
            continue;
        }

        let mut parts = content_parts(message.content.as_ref())?;
        if role == "assistant"
            && let Some(tool_calls) = &message.tool_calls
        {
            for call in tool_calls {
                let name = call
                    .function
                    .name
                    .clone()
                    .ok_or(NativeError::CallWithoutName)?;
                if let Some(id) = &call.id {
                    call_names.insert(id.clone(), name.clone());
                }
                let args = match &call.function.arguments {
                    Some(raw) => serde_json::from_str(raw).map_err(NativeError::CallArguments)?,
                    None => Value::Object(Map::new()),
                };
                parts.push(Part {
                    function_call: Some(FunctionCall {
                        id: None,
                        name,
                        args: Some(args),
                    }),
                    // Gemini 3 rejects replayed history that lost this.
                    thought_signature: call.thought_signature().map(str::to_owned),
                    ..Part::default()
                });
            }
        }

        let native_role = if role == "assistant" { "model" } else { "user" };
        if role == "tool" {
            let id = message
                .tool_call_id
                .as_deref()
                .ok_or(NativeError::ToolMessageWithoutCallId)?;
            let name = call_names
                .get(id)
                .cloned()
                .or_else(|| message.name.clone())
                .ok_or_else(|| NativeError::UnknownCall(id.to_owned()))?;
            parts = vec![Part {
                function_response: Some(FunctionResponse {
                    name,
                    response: tool_response(message.content.as_ref()),
                }),
                ..Part::default()
            }];
        }
        if parts.is_empty() {
            parts.push(Part::text(""));
        }
        contents.push(Content {
            role: Some(native_role.into()),
            parts,
        });
    }

    Ok(NativeRequest {
        model,
        streaming: req.stream.unwrap_or(false),
        body: GenerateContentRequest {
            contents,
            system_instruction: (!system_parts.is_empty()).then_some(Content {
                role: None,
                parts: system_parts,
            }),
            generation_config: generation_config(req),
            tools: tools(req)?,
            tool_config: tool_config(req),
        },
    })
}

fn content_parts(content: Option<&ChatContent>) -> Result<Vec<Part>, NativeError> {
    match content {
        None => Ok(Vec::new()),
        Some(ChatContent::Text(text)) => Ok(vec![Part::text(text.clone())]),
        Some(ChatContent::Parts(items)) => items.iter().map(content_part).collect(),
    }
}

fn content_part(part: &ChatPart) -> Result<Part, NativeError> {
    match part {
        ChatPart::Text { text } => Ok(Part::text(text.clone())),
        ChatPart::ImageUrl { image_url } => media_part(image_url.url(), "image/*"),
        ChatPart::InputAudio { input_audio } => Ok(Part {
            inline_data: Some(Blob {
                mime_type: format!("audio/{}", input_audio.format.as_deref().unwrap_or("wav")),
                data: input_audio.data.clone(),
            }),
            ..Part::default()
        }),
        ChatPart::Other => Err(NativeError::UnsupportedPart),
    }
}

fn media_part(url: &str, fallback_mime: &str) -> Result<Part, NativeError> {
    if let Some(data) = url.strip_prefix("data:") {
        let (metadata, payload) = data.split_once(',').ok_or(NativeError::InvalidDataUrl)?;
        return Ok(Part {
            inline_data: Some(Blob {
                mime_type: metadata
                    .split(';')
                    .next()
                    .filter(|m| !m.is_empty())
                    .unwrap_or(fallback_mime)
                    .to_owned(),
                data: payload.to_owned(),
            }),
            ..Part::default()
        });
    }
    Ok(Part {
        file_data: Some(FileData {
            mime_type: fallback_mime.to_owned(),
            file_uri: url.to_owned(),
        }),
        ..Part::default()
    })
}

fn tool_response(content: Option<&ChatContent>) -> Box<RawValue> {
    #[derive(Serialize)]
    struct Wrapped {
        result: Box<RawValue>,
    }
    let text = match content {
        Some(ChatContent::Text(raw)) => raw.clone(),
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                ChatPart::Text { text } => Some(text.as_str()),
                ChatPart::ImageUrl { .. } | ChatPart::InputAudio { .. } | ChatPart::Other => None,
            })
            .collect(),
        None => "null".into(),
    };
    let wrap = |result| to_raw_value(&Wrapped { result }).expect("response serializes");
    match RawValue::from_string(text.clone()) {
        Ok(raw) if raw.get().trim_start().starts_with('{') => raw,
        Ok(raw) => wrap(raw),
        Err(_) => wrap(to_raw_value(&text).expect("string serializes")),
    }
}

fn generation_config(req: &ChatRequest) -> Option<GenerationConfig> {
    let (mime, schema) = match req.response_format.as_ref().map(|f| f.kind.as_str()) {
        Some("json_object") => (Some("application/json".to_owned()), None),
        Some("json_schema") => (
            Some("application/json".to_owned()),
            req.response_format
                .as_ref()
                .and_then(|f| f.json_schema.as_ref())
                .and_then(|s| s.schema.clone()),
        ),
        _ => (None, None),
    };
    let config = GenerationConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_completion_tokens.or(req.max_tokens),
        candidate_count: req.n,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        stop_sequences: req.stop.clone().map(chat::StopSequences::into_vec),
        response_mime_type: mime,
        response_json_schema: schema,
        thinking_config: None,
    };
    let empty = serde_json::to_value(&config)
        .ok()
        .and_then(|v| v.as_object().map(Map::is_empty))
        .unwrap_or(true);
    (!empty).then_some(config)
}

fn tools(req: &ChatRequest) -> Result<Option<Vec<Tool>>, NativeError> {
    let Some(tools) = &req.tools else {
        return Ok(None);
    };
    let mut declarations = Vec::new();
    for tool in tools {
        if tool.kind.as_deref() != Some("function") {
            return Err(NativeError::NonFunctionTool);
        }
        let def = tool.def();
        declarations.push(FunctionDeclaration {
            name: def
                .name
                .clone()
                .ok_or(NativeError::ToolWithoutDeclaration)?,
            description: def.description.clone(),
            parameters_json_schema: def.parameters.clone(),
        });
    }
    Ok((!declarations.is_empty()).then(|| {
        vec![Tool {
            function_declarations: declarations,
        }]
    }))
}

fn tool_config(req: &ChatRequest) -> Option<ToolConfig> {
    let config = match req.tool_choice.as_ref()? {
        ChatToolChoice::Mode(mode) if mode == "none" => FunctionCallingConfig {
            mode: FunctionCallingMode::None,
            allowed_function_names: None,
        },
        ChatToolChoice::Mode(mode) if mode == "required" => FunctionCallingConfig {
            mode: FunctionCallingMode::Any,
            allowed_function_names: None,
        },
        ChatToolChoice::Mode(_) => FunctionCallingConfig {
            mode: FunctionCallingMode::Auto,
            allowed_function_names: None,
        },
        ChatToolChoice::Named { function, .. } => FunctionCallingConfig {
            mode: FunctionCallingMode::Any,
            allowed_function_names: function
                .as_ref()
                .and_then(|f| f.name.clone())
                .map(|name| vec![name]),
        },
    };
    Some(ToolConfig {
        function_calling_config: config,
    })
}

pub fn response(body: &[u8], requested_model: &str) -> Result<ChatCompletion, NativeError> {
    let native: GenerateContentResponse =
        serde_json::from_slice(body).map_err(NativeError::Response)?;
    let choices = native
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let (index, message, finish_reason) = choice(candidate, index, false);
            ChatChoice {
                index,
                message,
                finish_reason,
                logprobs: None,
            }
        })
        .collect();
    Ok(ChatCompletion {
        id: native
            .response_id
            .map_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()), |id| format!("chatcmpl-{id}")),
        object: "chat.completion".into(),
        created: crate::clock::unix_now(),
        model: requested_model.to_owned(),
        choices,
        usage: native.usage_metadata.as_ref().map(chat_usage),
    })
}

/// A candidate's index, its message and its finish reason, shared by the
/// whole response and the streamed chunk, which differ only in what wraps
/// the message.
fn choice(
    candidate: &Candidate,
    fallback_index: usize,
    streaming: bool,
) -> (u64, ChatMessage, Option<chat::FinishReason>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut images = Vec::new();
    for part in candidate.content.iter().flat_map(|c| &c.parts) {
        if part.thought == Some(true) {
            continue;
        }
        if let Some(value) = &part.text {
            text.push_str(value);
        }
        if let Some(call) = &part.function_call {
            tool_calls.push(tool_call(
                call,
                part.thought_signature.as_deref(),
                tool_calls.len(),
                streaming,
            ));
        }
        if let Some(data) = &part.inline_data
            && !data.data.is_empty()
        {
            let mime = if data.mime_type.is_empty() {
                "application/octet-stream"
            } else {
                &data.mime_type
            };
            images.push(ChatPart::ImageUrl {
                image_url: ImageRef::Url(format!("data:{mime};base64,{}", data.data)),
            });
        }
    }
    let used_tools = !tool_calls.is_empty();
    let message = ChatMessage {
        role: "assistant".into(),
        content: (!text.is_empty()).then_some(ChatContent::Text(text)),
        tool_calls: used_tools.then_some(tool_calls),
        images: (!images.is_empty()).then_some(images),
        ..Default::default()
    };
    let mut reason = finish_reason(candidate.finish_reason.clone());
    if used_tools && reason == Some(chat::FinishReason::Stop) {
        reason = Some(chat::FinishReason::ToolCalls);
    }
    (
        candidate.index.unwrap_or(fallback_index as u64),
        message,
        reason,
    )
}

fn tool_call(
    call: &FunctionCall,
    signature: Option<&str>,
    index: usize,
    streaming: bool,
) -> ChatToolCall {
    let args = call
        .args
        .clone()
        .unwrap_or_else(|| Value::Object(Map::new()));
    ChatToolCall {
        id: Some(
            call.id
                .clone()
                .unwrap_or_else(|| format!("call_native_{index}")),
        ),
        index: streaming.then_some(index as u64),
        kind: Some("function".into()),
        function: FunctionBody {
            name: Some(call.name.clone()),
            arguments: Some(serde_json::to_string(&args).unwrap_or_default()),
        },
        extra_content: signature.map(ExtraContent::with_signature),
    }
}

fn finish_reason(reason: Option<FinishReason>) -> Option<chat::FinishReason> {
    match reason? {
        FinishReason::Unspecified => None,
        FinishReason::Stop => Some(chat::FinishReason::Stop),
        FinishReason::MaxTokens => Some(chat::FinishReason::Length),
        FinishReason::MalformedFunctionCall | FinishReason::UnexpectedToolCall => {
            Some(chat::FinishReason::Stop)
        }
        FinishReason::Other(_) => Some(chat::FinishReason::ContentFilter),
    }
}

pub fn chat_usage(usage: &UsageMetadata) -> ChatUsage {
    let prompt = usage.prompt_token_count;
    let completion = usage.candidates_token_count;
    let thoughts = usage.thoughts_token_count;
    ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: usage
            .total_token_count
            .unwrap_or(prompt + completion + thoughts),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: usage.cached_content_token_count,
        },
        completion_tokens_details: CompletionTokensDetails {
            reasoning_tokens: thoughts,
        },
    }
}

pub struct NativeStream {
    id: String,
    model: String,
    created: i64,
    frames: crate::gemini::sse::Frames,
    sent_roles: BTreeSet<u64>,
    done: bool,
}

impl NativeStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            model: model.to_owned(),
            created: crate::clock::unix_now(),
            frames: crate::gemini::sse::Frames::default(),
            sent_roles: BTreeSet::new(),
            done: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut output = Vec::new();
        for data in self.frames.feed(bytes) {
            if let Ok(event) = serde_json::from_slice::<GenerateContentResponse>(&data) {
                output.extend(self.event(&event));
            }
        }
        if !self.done
            && let Some(error) = self.frames.cutoff()
        {
            self.done = true;
            output.push(sse(&ErrorFrame { error: &error }));
            output.push(b"data: [DONE]\n\n".to_vec());
        }
        output
    }

    fn event(&mut self, event: &GenerateContentResponse) -> Vec<Vec<u8>> {
        if let Some(response_id) = &event.response_id {
            self.id = format!("chatcmpl-{response_id}");
        }
        if let Some(error) = &event.error {
            self.done = true;
            let frame = ErrorFrame { error };
            return vec![sse(&frame), b"data: [DONE]\n\n".to_vec()];
        }

        let mut choices = Vec::new();
        let mut finished = false;
        for (fallback_index, candidate) in event.candidates.iter().enumerate() {
            let (index, message, finish_reason) = choice(candidate, fallback_index, true);
            let delta = ChatDelta {
                role: self.sent_roles.insert(index).then(|| "assistant".into()),
                content: match message.content {
                    Some(ChatContent::Text(text)) => Some(text),
                    _ => None,
                },
                reasoning_content: None,
                tool_calls: message.tool_calls,
                images: message.images,
            };
            finished |= finish_reason.is_some();
            choices.push(ChunkChoice {
                index,
                delta,
                finish_reason,
                logprobs: None,
            });
        }

        let chunk = ChatChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".into(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage: event.usage_metadata.as_ref().map(chat_usage),
            error: None,
        };
        let mut output = vec![sse(&chunk)];
        if finished && !self.done {
            self.done = true;
            output.push(b"data: [DONE]\n\n".to_vec());
        }
        output
    }
}

/// Google's own error object, forwarded as the client would see it from the
/// OpenAI-compatible surface.
#[derive(Serialize)]
struct ErrorFrame<'a> {
    error: &'a ApiError,
}

fn sse(value: &impl Serialize) -> Vec<u8> {
    let mut out = b"data: ".to_vec();
    out.extend(serde_json::to_vec(value).unwrap_or_default());
    out.extend_from_slice(b"\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(v: Value) -> ChatRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn translates_fragmented_native_streams() {
        let mut stream = NativeStream::new("gemini-flash-latest");
        assert!(
            stream
                .feed(b"data: {\"responseId\":\"r2\",\"candidates\":[{\"content\":{")
                .is_empty()
        );
        let output = stream.feed(b"\"parts\":[{\"text\":\"OK\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n");
        assert_eq!(output.len(), 2);
        let text = String::from_utf8(output[0].clone()).unwrap();
        assert!(text.contains("chatcmpl-r2"));
        assert!(text.contains("\"content\":\"OK\""));
        assert_eq!(output[1], b"data: [DONE]\n\n");
    }

    #[test]
    fn a_data_url_without_a_payload_is_rejected() {
        let err = request(&chat(json!({
            "model": "gemini-3-flash-preview",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64"}}
                ]}
            ]
        })))
        .unwrap_err();
        assert!(
            err.to_string().contains("invalid data URL"),
            "unexpected error {err}"
        );
    }

    #[test]
    fn an_inline_part_without_data_yields_no_image() {
        let output = response(
            br#"{"candidates":[{"content":{"parts":[{"text":"x"},{"inlineData":{"mimeType":"image/png","data":""}}]},"finishReason":"STOP"}]}"#,
            "gemini-flash-latest",
        )
        .unwrap();
        let output = serde_json::to_value(&output).unwrap();
        assert!(output["choices"][0]["message"].get("images").is_none());
    }

    #[test]
    fn a_tool_only_message_keeps_content_null() {
        let output = response(
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"grep","args":{}}}]},"finishReason":"STOP"}]}"#,
            "gemini-flash-latest",
        )
        .unwrap();
        let output = serde_json::to_value(&output).unwrap();
        assert!(
            output["choices"][0]["message"]
                .get("content")
                .is_some_and(Value::is_null)
        );
        assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn a_native_error_frame_is_forwarded_verbatim() {
        let mut stream = NativeStream::new("m");
        let output = stream.feed(b"data: {\"error\":{\"code\":429,\"message\":\"slow down\",\"status\":\"RESOURCE_EXHAUSTED\",\"details\":[{\"@type\":\"x\"}]}}\n\n");
        assert_eq!(output.len(), 2);
        let first: Value = serde_json::from_slice(&output[0][6..]).unwrap();
        assert_eq!(first["error"]["code"], 429);
        assert_eq!(first["error"]["status"], "RESOURCE_EXHAUSTED");
        assert_eq!(first["error"]["details"][0]["@type"], "x");
        assert_eq!(output[1], b"data: [DONE]\n\n");
    }
}

#[cfg(test)]
mod signature_tests {
    use super::*;
    use serde_json::json;

    fn candidate(v: Value) -> Candidate {
        serde_json::from_value(v).unwrap()
    }

    /// The signature rides on the part, beside functionCall not inside it.
    #[test]
    fn a_replayed_tool_call_keeps_its_signature() {
        let candidate = candidate(json!({
            "content": {"parts": [{
                "functionCall": {"name": "shell", "args": {"cmd": "ls"}, "id": "call_1"},
                "thoughtSignature": "EtUBCtIB"
            }]}
        }));
        let (_, message, _) = choice(&candidate, 0, false);
        let call = serde_json::to_value(&message.tool_calls.unwrap()[0]).unwrap();
        assert_eq!(
            call["extra_content"]["google"]["thought_signature"],
            "EtUBCtIB"
        );

        let replay = serde_json::from_value::<ChatRequest>(json!({
            "model": "gemini-3-flash-preview",
            "messages": [
                {"role": "user", "content": "ls"},
                {"role": "assistant", "content": null, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
            ]
        }))
        .unwrap();
        let out = request(&replay).unwrap();
        assert_eq!(
            out.body.contents[1].parts[0].thought_signature.as_deref(),
            Some("EtUBCtIB")
        );
    }
}

#[cfg(test)]
mod cutoff_tests {
    use super::NativeStream;

    #[test]
    fn a_cutoff_ends_the_translated_stream_with_an_error_frame() {
        let mut stream = NativeStream::new("gemini-3.8-flash");
        let mut frames = Vec::new();
        for chunk in include_bytes!("fixtures/native_cutoff.sse").chunks(64) {
            frames.extend(stream.feed(chunk));
        }
        let tail: Vec<String> = frames
            .iter()
            .rev()
            .take(2)
            .map(|f| String::from_utf8_lossy(f).into_owned())
            .collect();
        assert_eq!(tail[0], "data: [DONE]\n\n");
        assert!(
            tail[1].contains("\"error\"") && tail[1].contains("UNAVAILABLE"),
            "{}",
            tail[1]
        );
    }
}
