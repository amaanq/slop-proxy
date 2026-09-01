use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

pub struct NativeRequest {
    pub model: String,
    pub streaming: bool,
    pub body: Value,
}

pub fn request(body: &Value) -> Result<NativeRequest, String> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?
        .trim_start_matches("models/")
        .to_string();
    if model.is_empty()
        || !model
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err("invalid gemini model name".into());
    }

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing messages".to_string())?;
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut call_names = BTreeMap::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "message is missing a role".to_string())?;
        if matches!(role, "system" | "developer") {
            system_parts.extend(content_parts(message.get("content"))?);
            continue;
        }

        let mut parts = content_parts(message.get("content"))?;
        if role == "assistant"
            && let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array)
        {
            for call in tool_calls {
                let function = call
                    .get("function")
                    .ok_or_else(|| "tool call is missing function".to_string())?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool call is missing function name".to_string())?;
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    call_names.insert(id.to_string(), name.to_string());
                }
                let args = function
                    .get("arguments")
                    .map(argument_value)
                    .transpose()?
                    .unwrap_or_else(|| json!({}));
                parts.push(json!({"functionCall": {"name": name, "args": args}}));
            }
        }

        let native_role = if role == "assistant" { "model" } else { "user" };
        if role == "tool" {
            let id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "tool message is missing tool_call_id".to_string())?;
            let name = call_names
                .get(id)
                .cloned()
                .or_else(|| {
                    message
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .ok_or_else(|| format!("tool message references unknown call {id}"))?;
            parts = vec![json!({
                "functionResponse": {
                    "name": name,
                    "response": tool_response(message.get("content"))
                }
            })];
        }
        if parts.is_empty() {
            parts.push(json!({"text": ""}));
        }
        contents.push(json!({"role": native_role, "parts": parts}));
    }

    let mut native = Map::new();
    native.insert("contents".into(), Value::Array(contents));
    if !system_parts.is_empty() {
        native.insert("systemInstruction".into(), json!({"parts": system_parts}));
    }
    if let Some(config) = generation_config(body) {
        native.insert("generationConfig".into(), config);
    }
    if let Some(tools) = tools(body)? {
        native.insert("tools".into(), tools);
    }
    if let Some(tool_config) = tool_config(body) {
        native.insert("toolConfig".into(), tool_config);
    }

    Ok(NativeRequest {
        model,
        streaming: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        body: Value::Object(native),
    })
}

fn content_parts(content: Option<&Value>) -> Result<Vec<Value>, String> {
    let Some(content) = content else {
        return Ok(Vec::new());
    };
    match content {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => Ok(vec![json!({"text": text})]),
        Value::Array(items) => items.iter().map(content_part).collect(),
        _ => Err("message content must be a string or array".into()),
    }
}

fn content_part(part: &Value) -> Result<Value, String> {
    let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
    match kind {
        "text" | "input_text" => part
            .get("text")
            .cloned()
            .map(|text| json!({"text": text}))
            .ok_or_else(|| "text content part is missing text".to_string()),
        "image_url" | "input_image" => {
            let image = part
                .get("image_url")
                .or_else(|| part.get("image"))
                .ok_or_else(|| "image content part is missing image_url".to_string())?;
            let url = image
                .as_str()
                .or_else(|| image.get("url").and_then(Value::as_str))
                .ok_or_else(|| "image_url is missing a URL".to_string())?;
            media_part(url, "image/*")
        }
        "input_audio" => {
            let audio = part
                .get("input_audio")
                .ok_or_else(|| "audio content part is missing input_audio".to_string())?;
            let data = audio
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| "input_audio is missing data".to_string())?;
            let format = audio.get("format").and_then(Value::as_str).unwrap_or("wav");
            Ok(json!({"inlineData": {"mimeType": format!("audio/{format}"), "data": data}}))
        }
        other => Err(format!("unsupported message content type {other}")),
    }
}

fn media_part(url: &str, fallback_mime: &str) -> Result<Value, String> {
    if let Some(data) = url.strip_prefix("data:") {
        let (metadata, payload) = data
            .split_once(',')
            .ok_or_else(|| "invalid data URL".to_string())?;
        let mime = metadata.split(';').next().unwrap_or(fallback_mime);
        return Ok(json!({"inlineData": {"mimeType": mime, "data": payload}}));
    }
    Ok(json!({"fileData": {"mimeType": fallback_mime, "fileUri": url}}))
}

fn argument_value(value: &Value) -> Result<Value, String> {
    match value {
        Value::String(raw) => serde_json::from_str(raw)
            .map_err(|e| format!("tool call arguments are not valid JSON: {e}")),
        value => Ok(value.clone()),
    }
}

fn tool_response(content: Option<&Value>) -> Value {
    let value = match content {
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or_else(|_| json!(raw)),
        Some(value) => value.clone(),
        None => Value::Null,
    };
    match value {
        Value::Object(_) => value,
        value => json!({"result": value}),
    }
}

fn generation_config(body: &Value) -> Option<Value> {
    let mut config = Map::new();
    for (source, target) in [
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("max_tokens", "maxOutputTokens"),
        ("max_completion_tokens", "maxOutputTokens"),
        ("n", "candidateCount"),
        ("presence_penalty", "presencePenalty"),
        ("frequency_penalty", "frequencyPenalty"),
        ("seed", "seed"),
    ] {
        if let Some(value) = body.get(source) {
            config.insert(target.into(), value.clone());
        }
    }
    if let Some(stop) = body.get("stop") {
        let sequences = match stop {
            Value::String(_) => Value::Array(vec![stop.clone()]),
            Value::Array(_) => stop.clone(),
            _ => Value::Null,
        };
        if !sequences.is_null() {
            config.insert("stopSequences".into(), sequences);
        }
    }
    if let Some(format) = body.get("response_format") {
        match format.get("type").and_then(Value::as_str) {
            Some("json_object") => {
                config.insert("responseMimeType".into(), json!("application/json"));
            }
            Some("json_schema") => {
                config.insert("responseMimeType".into(), json!("application/json"));
                if let Some(schema) = format.pointer("/json_schema/schema") {
                    config.insert("responseJsonSchema".into(), schema.clone());
                }
            }
            _ => {}
        }
    }
    (!config.is_empty()).then_some(Value::Object(config))
}

fn tools(body: &Value) -> Result<Option<Value>, String> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut declarations = Vec::new();
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err("native gemini only supports function tools".into());
        }
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| "function tool is missing its declaration".to_string())?;
        let mut declaration = function.clone();
        declaration.remove("strict");
        if let Some(parameters) = declaration.remove("parameters") {
            declaration.insert("parametersJsonSchema".into(), parameters);
        }
        declarations.push(Value::Object(declaration));
    }
    Ok((!declarations.is_empty()).then(|| json!([{"functionDeclarations": declarations}])))
}

fn tool_config(body: &Value) -> Option<Value> {
    let choice = body.get("tool_choice")?;
    let mut function = Map::new();
    match choice {
        Value::String(mode) if mode == "none" => {
            function.insert("mode".into(), json!("NONE"));
        }
        Value::String(mode) if mode == "required" => {
            function.insert("mode".into(), json!("ANY"));
        }
        Value::String(_) => {
            function.insert("mode".into(), json!("AUTO"));
        }
        Value::Object(obj) => {
            function.insert("mode".into(), json!("ANY"));
            if let Some(name) = obj.get("function").and_then(|value| value.get("name")) {
                function.insert("allowedFunctionNames".into(), json!([name]));
            }
        }
        _ => return None,
    }
    Some(json!({"functionCallingConfig": function}))
}

pub fn response(body: &[u8], requested_model: &str) -> Result<Vec<u8>, String> {
    let native: Value =
        serde_json::from_slice(body).map_err(|e| format!("invalid native gemini response: {e}"))?;
    let choices = native
        .get("candidates")
        .and_then(Value::as_array)
        .map(|candidates| {
            candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| response_choice(candidate, index))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let id = native
        .get("responseId")
        .and_then(Value::as_str)
        .map(|id| format!("chatcmpl-{id}"))
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));
    let mut out = json!({
        "id": id,
        "object": "chat.completion",
        "created": crate::clock::unix_now(),
        "model": requested_model,
        "choices": choices
    });
    if let Some(usage) = native.get("usageMetadata") {
        out["usage"] = usage_value(usage);
    }
    serde_json::to_vec(&out).map_err(|e| e.to_string())
}

fn response_choice(candidate: &Value, index: usize) -> Value {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut images = Vec::new();
    for part in candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(value) = part.get("text").and_then(Value::as_str) {
            text.push_str(value);
        }
        if let Some(call) = part.get("functionCall") {
            tool_calls.push(tool_call(call, tool_calls.len(), false));
        }
        if let Some(data) = part.get("inlineData") {
            let mime = data
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            if let Some(payload) = data.get("data").and_then(Value::as_str) {
                images.push(json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{mime};base64,{payload}")}
                }));
            }
        }
    }
    let mut message = Map::from_iter([
        ("role".into(), json!("assistant")),
        (
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        ),
    ]);
    let used_tools = !tool_calls.is_empty();
    if used_tools {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !images.is_empty() {
        message.insert("images".into(), Value::Array(images));
    }
    let mut reason = finish_reason(candidate.get("finishReason").and_then(Value::as_str));
    if used_tools && reason == "stop" {
        reason = json!("tool_calls");
    }
    json!({
        "index": candidate.get("index").and_then(Value::as_u64).unwrap_or(index as u64),
        "message": message,
        "finish_reason": reason,
        "logprobs": null
    })
}

fn tool_call(call: &Value, index: usize, streaming: bool) -> Value {
    let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_native_{index}"));
    let mut output = json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": serde_json::to_string(&args).unwrap_or_default()}
    });
    if streaming {
        output["index"] = json!(index);
    }
    output
}

fn finish_reason(reason: Option<&str>) -> Value {
    match reason {
        None | Some("FINISH_REASON_UNSPECIFIED") => Value::Null,
        Some("STOP") => json!("stop"),
        Some("MAX_TOKENS") => json!("length"),
        Some("MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL") => json!("stop"),
        Some(_) => json!("content_filter"),
    }
}

fn usage_value(usage: &Value) -> Value {
    let prompt = usage
        .get("promptTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let completion = usage
        .get("candidatesTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let thoughts = usage
        .get("thoughtsTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let total = usage
        .get("totalTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(prompt + completion + thoughts);
    let cached = usage
        .get("cachedContentTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": thoughts}
    })
}

pub struct NativeStream {
    id: String,
    model: String,
    created: i64,
    buf: Vec<u8>,
    sent_roles: BTreeSet<usize>,
    done: bool,
}

impl NativeStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            model: model.to_string(),
            created: crate::clock::unix_now(),
            buf: Vec::new(),
            sent_roles: BTreeSet::new(),
            done: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(bytes);
        let mut output = Vec::new();
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if let Ok(event) = serde_json::from_slice::<Value>(data) {
                output.extend(self.event(&event));
            }
        }
        output
    }

    fn event(&mut self, event: &Value) -> Vec<Vec<u8>> {
        if let Some(response_id) = event.get("responseId").and_then(Value::as_str) {
            self.id = format!("chatcmpl-{response_id}");
        }
        if let Some(error) = event.get("error") {
            self.done = true;
            return vec![sse(&json!({"error": error})), b"data: [DONE]\n\n".to_vec()];
        }

        let mut choices = Vec::new();
        let mut finished = false;
        for (fallback_index, candidate) in event
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let index = candidate
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(fallback_index as u64) as usize;
            let mut delta = Map::new();
            if self.sent_roles.insert(index) {
                delta.insert("role".into(), json!("assistant"));
            }
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            let mut images = Vec::new();
            for part in candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    continue;
                }
                if let Some(value) = part.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
                if let Some(call) = part.get("functionCall") {
                    tool_calls.push(tool_call(call, tool_calls.len(), true));
                }
                if let Some(data) = part.get("inlineData") {
                    let mime = data
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("application/octet-stream");
                    if let Some(payload) = data.get("data").and_then(Value::as_str) {
                        images.push(json!({
                            "type": "image_url",
                            "image_url": {"url": format!("data:{mime};base64,{payload}")}
                        }));
                    }
                }
            }
            if !text.is_empty() {
                delta.insert("content".into(), json!(text));
            }
            if !tool_calls.is_empty() {
                delta.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            if !images.is_empty() {
                delta.insert("images".into(), Value::Array(images));
            }
            let mut reason = finish_reason(candidate.get("finishReason").and_then(Value::as_str));
            if delta.contains_key("tool_calls") && reason == "stop" {
                reason = json!("tool_calls");
            }
            finished |= !reason.is_null();
            choices.push(json!({
                "index": index,
                "delta": delta,
                "finish_reason": reason,
                "logprobs": null
            }));
        }

        let mut chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices
        });
        if let Some(usage) = event.get("usageMetadata") {
            chunk["usage"] = usage_value(usage);
        }
        let mut output = vec![sse(&chunk)];
        if finished && !self.done {
            self.done = true;
            output.push(b"data: [DONE]\n\n".to_vec());
        }
        output
    }
}

fn sse(value: &Value) -> Vec<u8> {
    let mut out = b"data: ".to_vec();
    out.extend(serde_json::to_vec(value).unwrap_or_default());
    out.extend_from_slice(b"\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_messages_tools_and_generation_options() {
        let translated = request(&json!({
            "model": "gemini-3-flash-preview",
            "stream": true,
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "weather"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":20}"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "weather", "parameters": {"type": "object"}, "strict": true
            }}],
            "tool_choice": "required",
            "max_tokens": 100,
            "top_p": 0.8
        }))
        .unwrap();
        assert_eq!(translated.model, "gemini-3-flash-preview");
        assert!(translated.streaming);
        assert_eq!(
            translated.body["systemInstruction"]["parts"][0]["text"],
            "Be brief"
        );
        assert_eq!(
            translated.body["contents"][2]["parts"][0]["functionResponse"]["name"],
            "weather"
        );
        assert_eq!(translated.body["generationConfig"]["maxOutputTokens"], 100);
        assert_eq!(
            translated.body["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );
        assert_eq!(
            translated.body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"],
            "object"
        );
    }

    #[test]
    fn translates_native_output_and_usage() {
        let output = response(
            br#"{"responseId":"r1","candidates":[{"index":0,"content":{"parts":[{"text":"OK"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":3,"totalTokenCount":15,"cachedContentTokenCount":4}}"#,
            "gemini-flash-latest",
        )
        .unwrap();
        let output: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["id"], "chatcmpl-r1");
        assert_eq!(output["choices"][0]["message"]["content"], "OK");
        assert_eq!(output["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            output["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
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
}
