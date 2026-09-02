//! Claude Code only speaks the messages API and every Gemini surface speaks
//! chat completions.

use serde_json::{Map, Value, json};

use crate::codex::types::{ContentPart, InputItem, ResponsesRequest, ToolChoice};

pub fn to_chat(req: &ResponsesRequest) -> Value {
    let mut messages = vec![json!({"role": "system", "content": req.instructions})];
    for item in &req.input {
        match item {
            InputItem::Message { role, content } => {
                messages.push(json!({"role": chat_role(role), "content": parts(content)}));
            }
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            })),
            InputItem::FunctionCallOutput { call_id, output } => messages.push(json!({
                "role": "tool", "tool_call_id": call_id, "content": output,
            })),
            // Gemini rejects an unknown role rather than ignoring it.
            InputItem::Reasoning { .. } => {}
        }
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(req.model));
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(req.stream));
    if req.stream {
        // Without this the terminal chunk carries no usage and the request
        // bills as zero tokens.
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(max) = req.max_output_tokens {
        body.insert("max_tokens".into(), json!(max));
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".into(),
            json!(
                req.tools
                    .iter()
                    .map(|t| json!({"type": "function", "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }}))
                    .collect::<Vec<_>>()
            ),
        );
    }
    if let Some(choice) = &req.tool_choice {
        body.insert(
            "tool_choice".into(),
            match choice {
                ToolChoice::Mode(m) => json!(m),
                ToolChoice::Function { name, .. } => {
                    json!({"type": "function", "function": {"name": name}})
                }
            },
        );
    }
    Value::Object(body)
}

/// Chat completions has no `developer` role.
fn chat_role(role: &str) -> &str {
    match role {
        "developer" => "system",
        other => other,
    }
}

fn parts(content: &[ContentPart]) -> Value {
    json!(
        content
            .iter()
            .map(|p| match p {
                ContentPart::InputText { text } | ContentPart::OutputText { text } =>
                    json!({"type": "text", "text": text}),
                ContentPart::InputImage { image_url } =>
                    json!({"type": "image_url", "image_url": {"url": image_url}}),
            })
            .collect::<Vec<_>>()
    )
}

/// Tool calls arrive spread across chunks keyed by index, so a slot holds the
/// name until there is enough to open an item.
#[derive(Default)]
pub struct ChatToResponses {
    opened: bool,
    calls: Vec<OpenCall>,
    text: String,
    frames: usize,
    emitted: usize,
    first_frame: Option<String>,
}

/// A stream that produced nothing is the failure this path keeps hitting, and
/// the frame count separates nothing arriving from nothing being understood.
impl Drop for ChatToResponses {
    fn drop(&mut self) {
        if self.emitted == 0 {
            tracing::warn!(
                frames = self.frames,
                first = self.first_frame.as_deref().unwrap_or("<none>"),
                "gemini bridge produced no content"
            );
        }
    }
}

#[derive(Default)]
struct OpenCall {
    id: String,
    name: String,
    announced: bool,
}

impl ChatToResponses {
    pub fn feed(&mut self, chunk: &Value) -> Vec<Value> {
        self.frames += 1;
        if self.first_frame.is_none() {
            self.first_frame = Some(chunk.to_string().chars().take(400).collect());
        }
        let mut out = Vec::new();
        if !self.opened {
            self.opened = true;
            out.push(json!({"type": "response.created", "response": {"id": chunk.get("id")}}));
        }
        let delta = chunk.pointer("/choices/0/delta");
        if let Some(text) = delta.and_then(|d| d.get("content")).and_then(Value::as_str)
            && !text.is_empty()
        {
            self.text.push_str(text);
            out.push(json!({"type": "response.output_text.delta", "delta": text}));
        }
        if let Some(calls) = delta.and_then(|d| d.get("tool_calls")).and_then(Value::as_array) {
            for call in calls {
                out.extend(self.tool_call(call));
            }
        }
        if chunk
            .pointer("/choices/0/finish_reason")
            .is_some_and(|f| !f.is_null())
        {
            for call in &self.calls {
                if call.announced {
                    out.push(json!({"type": "response.output_item.done", "item": {
                        "type": "function_call", "call_id": call.id, "name": call.name,
                    }}));
                }
            }
            self.calls.clear();
            // The streaming translator reads text deltas, but `aggregate`
            // builds its blocks only from finished items, so a non-streaming
            // turn needs the whole message restated as one.
            if !self.text.is_empty() {
                out.push(json!({"type": "response.output_item.done", "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": std::mem::take(&mut self.text)}],
                }}));
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            out.push(json!({"type": "response.completed", "response": {"usage": usage}}));
        }
        // `created` and `completed` carry no answer, so they do not count as
        // content when deciding whether the bridge produced anything.
        self.emitted += out
            .iter()
            .filter(|e| {
                !matches!(
                    e["type"].as_str(),
                    Some("response.created") | Some("response.completed")
                )
            })
            .count();
        out
    }

    fn tool_call(&mut self, call: &Value) -> Vec<Value> {
        let idx = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        if self.calls.len() <= idx {
            self.calls.resize_with(idx + 1, OpenCall::default);
        }
        let mut out = Vec::new();
        let slot = &mut self.calls[idx];
        if let Some(id) = call.get("id").and_then(Value::as_str) {
            slot.id = id.to_string();
        }
        if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
            slot.name = name.to_string();
        }
        if !slot.announced && !slot.name.is_empty() {
            slot.announced = true;
            out.push(json!({"type": "response.output_item.added", "item": {
                "type": "function_call", "call_id": slot.id, "name": slot.name, "arguments": "",
            }}));
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str)
            && !args.is_empty()
        {
            out.push(json!({"type": "response.function_call_arguments.delta", "delta": args}));
        }
        out
    }
}

/// A referer-restricted key is served over the native surface, which answers
/// in Gemini's own frames, so that protocol is normalised before parsing.
pub fn event_stream(
    resp: reqwest::Response,
    protocol: crate::gemini::client::GeminiProtocol,
    model: &str,
) -> crate::codex::sse::EventStream {
    use crate::gemini::client::GeminiProtocol;
    use eventsource_stream::Eventsource;
    use futures_util::StreamExt;

    let mut native = (protocol == GeminiProtocol::Native)
        .then(|| crate::gemini::native::NativeStream::new(model));
    let chat_bytes = resp.bytes_stream().flat_map(move |item| {
        let frames = match (&mut native, item) {
            (Some(native), Ok(bytes)) => native.feed(&bytes),
            (None, Ok(bytes)) => vec![bytes.to_vec()],
            (_, Err(e)) => {
                tracing::warn!("gemini stream error: {e}");
                Vec::new()
            }
        };
        futures_util::stream::iter(frames.into_iter().map(Ok::<_, std::io::Error>))
    });

    let mut bridge = ChatToResponses::default();
    let stream = chat_bytes
        .eventsource()
        .flat_map(move |ev| {
            let events = match ev {
                Ok(ev) if ev.data != "[DONE]" => serde_json::from_str::<Value>(&ev.data)
                    .map(|chunk| bridge.feed(&chunk))
                    .unwrap_or_default(),
                Ok(_) => Vec::new(),
                Err(e) => {
                    tracing::warn!("gemini SSE parse error: {e}");
                    Vec::new()
                }
            };
            futures_util::stream::iter(events.into_iter().map(|e| {
                serde_json::from_value(e).unwrap_or(crate::codex::types::ResponsesEvent::Other)
            }))
        })
        .boxed();
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(b: &mut ChatToResponses, v: Value) -> Vec<String> {
        b.feed(&v)
            .iter()
            .map(|e| e["type"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn text_deltas_become_output_text() {
        let mut b = ChatToResponses::default();
        let first = feed(&mut b, json!({"choices": [{"delta": {"content": "hi"}}]}));
        assert_eq!(first, ["response.created", "response.output_text.delta"]);
        let next = feed(&mut b, json!({"choices": [{"delta": {"content": "!"}}]}));
        assert_eq!(next, ["response.output_text.delta"]);
    }

    #[test]
    fn a_tool_call_split_across_chunks_opens_once() {
        let mut b = ChatToResponses::default();
        b.feed(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": "{\"p"}}]}}]}));
        let more = feed(
            &mut b,
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "ath\":1}"}}]}}]}),
        );
        assert_eq!(more, ["response.function_call_arguments.delta"]);
        let done = feed(&mut b, json!({"choices": [{"finish_reason": "tool_calls"}]}));
        assert_eq!(done, ["response.output_item.done"]);
    }

    #[test]
    fn usage_closes_the_response() {
        let mut b = ChatToResponses::default();
        let out = feed(
            &mut b,
            json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1}}),
        );
        assert_eq!(out, ["response.created", "response.completed"]);
    }

    #[test]
    fn the_developer_role_survives_as_system() {
        let req = ResponsesRequest {
            input: vec![InputItem::Message {
                role: "developer".into(),
                content: vec![ContentPart::InputText { text: "ctx".into() }],
            }],
            ..ResponsesRequest::new("gemini-3.8-flash".into(), "sys".into())
        };
        let body = to_chat(&req);
        assert_eq!(body["messages"][1]["role"], "system");
        assert_eq!(body["messages"][1]["content"][0]["text"], "ctx");
    }
}

#[cfg(test)]
mod full_chain {
    use super::*;
    use crate::codex::types::ResponsesEvent;
    use crate::gemini::native::NativeStream;

    /// The whole native path offline: gemini frames in, Responses events out.
    #[test]
    fn a_captured_native_stream_yields_text() {
        let raw = include_bytes!("testdata/gemini_native.sse");
        let mut native = NativeStream::new("gemini-3.8-flash");
        let mut bridge = ChatToResponses::default();
        let mut kinds = Vec::new();
        let mut text = String::new();
        // The network delivers arbitrary chunks, not whole frames.
        let frames: Vec<Vec<u8>> = raw.chunks(64).flat_map(|c| native.feed(c)).collect();
        for frame in frames {
            let line = String::from_utf8_lossy(&frame);
            for data in line.lines().filter_map(|l| l.strip_prefix("data:")) {
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                for ev in bridge.feed(&chunk) {
                    let parsed: ResponsesEvent =
                        serde_json::from_value(ev.clone()).unwrap_or(ResponsesEvent::Other);
                    if let ResponsesEvent::OutputTextDelta { delta } = &parsed {
                        text.push_str(delta);
                    }
                    kinds.push(ev["type"].as_str().unwrap().to_string());
                }
            }
        }
        assert!(!text.is_empty(), "no text recovered, events were {kinds:?}");
    }
}

#[cfg(test)]
mod real_chunk {
    use super::*;
    use crate::codex::types::ResponsesEvent;

    #[test]
    fn a_real_gemini_chunk_survives_deserialisation() {
        let chunk: Value = serde_json::from_str(r#"{"choices":[{"delta":{"content":"OK.","role":"assistant"},"index":0}],"created":1788373398,"id":"x","model":"gemini-3.8-flash","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":71}}"#).unwrap();
        let raw = ChatToResponses::default().feed(&chunk);
        let kinds: Vec<_> = raw.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["response.created", "response.output_text.delta", "response.completed"]);
        let parsed: Vec<ResponsesEvent> = raw
            .into_iter()
            .map(|e| serde_json::from_value(e).unwrap_or(ResponsesEvent::Other))
            .collect();
        assert!(
            matches!(parsed[1], ResponsesEvent::OutputTextDelta { .. }),
            "text delta degraded to {:?}", parsed[1]
        );
    }
}

#[cfg(test)]
mod aggregate_path {
    use super::*;
    use crate::codex::types::{OutputContentPart, OutputItem, ResponsesEvent};

    #[test]
    fn a_finished_turn_restates_its_text_as_an_item() {
        let mut b = ChatToResponses::default();
        b.feed(&serde_json::json!({"choices": [{"delta": {"content": "ok"}}]}));
        let done = b.feed(&serde_json::json!({"choices": [{"finish_reason": "stop"}]}));
        let items: Vec<ResponsesEvent> = done
            .into_iter()
            .map(|e| serde_json::from_value(e).unwrap_or(ResponsesEvent::Other))
            .collect();
        let text = items.iter().find_map(|e| match e {
            ResponsesEvent::OutputItemDone {
                item: OutputItem::Message { content },
            } => content.as_ref().and_then(|c| c.first()).map(|p| match p {
                OutputContentPart::OutputText { text } => text.clone(),
                OutputContentPart::Other => String::new(),
            }),
            _ => None,
        });
        assert_eq!(text.as_deref(), Some("ok"));
    }
}
