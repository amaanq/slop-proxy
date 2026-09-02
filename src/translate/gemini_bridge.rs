//! Claude Code only speaks the messages API and every Gemini surface speaks
//! chat completions.

use serde_json::{Map, Value, json};


/// Takes the serialized request rather than the struct, so the responses
/// surface can hand over a body it received instead of one it built.
pub fn to_chat(req: &Value) -> Value {
    let mut messages = vec![json!({"role": "system", "content": req["instructions"]})];
    for item in req["input"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") => messages.push(json!({
                "role": chat_role(item["role"].as_str().unwrap_or("user")),
                "content": parts(&item["content"]),
            })),
            Some("function_call") => messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }],
            })),
            Some("function_call_output") => messages.push(json!({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item["output"],
            })),
            // Gemini rejects an unknown role rather than ignoring it.
            _ => {}
        }
    }

    let stream = req["stream"].as_bool().unwrap_or(true);
    let mut body = Map::new();
    body.insert("model".into(), req["model"].clone());
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(stream));
    if stream {
        // Without this the terminal chunk carries no usage and the request
        // bills as zero tokens.
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(max) = req["max_output_tokens"].as_u64() {
        body.insert("max_tokens".into(), json!(max));
    }
    let tools: Vec<Value> = req["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let name = t.get("name")?;
            Some(json!({"type": "function", "function": {
                "name": name,
                "description": t.get("description").cloned().unwrap_or(Value::Null),
                "parameters": t.get("parameters").cloned().unwrap_or(json!({})),
            }}))
        })
        .collect();
    if !tools.is_empty() {
        body.insert("tools".into(), json!(tools));
    }
    match &req["tool_choice"] {
        Value::String(mode) => {
            body.insert("tool_choice".into(), json!(mode));
        }
        Value::Object(o) if o.contains_key("name") => {
            body.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": o["name"]}}),
            );
        }
        _ => {}
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

fn parts(content: &Value) -> Value {
    json!(
        content
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| match p["type"].as_str() {
                Some("input_image") =>
                    json!({"type": "image_url", "image_url": {"url": p["image_url"]}}),
                _ => json!({"type": "text", "text": p["text"]}),
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
    msg_id: Option<String>,
    next_index: usize,
    finished: bool,
    usage: Option<Value>,
    completed: bool,
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
    item_id: String,
    name: String,
    arguments: String,
    index: usize,
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
            // Codex attaches a delta to an item by id, so text streamed before
            // the item is announced is parsed and then dropped.
            let id = self.msg_id.get_or_insert_with(|| {
                format!("msg_{}", uuid::Uuid::new_v4().simple())
            });
            if self.text.is_empty() {
                out.push(json!({"type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "id": id, "role": "assistant", "content": []}}));
                out.push(json!({"type": "response.content_part.added",
                    "item_id": id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": ""}}));
            }
            out.push(json!({"type": "response.output_text.delta",
                "item_id": id, "output_index": 0, "content_index": 0, "delta": text}));
            self.text.push_str(text);
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
            self.finished = true;
            for call in &self.calls {
                if call.announced {
                    // An item done without its arguments is a tool call the
                    // client cannot run.
                    out.push(json!({"type": "response.function_call_arguments.done",
                        "item_id": call.item_id, "output_index": call.index,
                        "arguments": call.arguments}));
                    out.push(json!({"type": "response.output_item.done",
                        "output_index": call.index,
                        "item": {
                            "type": "function_call", "id": call.item_id, "call_id": call.id,
                            "name": call.name, "arguments": call.arguments,
                            "status": "completed",
                        }}));
                }
            }
            self.calls.clear();
            // The streaming translator reads text deltas, but `aggregate`
            // builds its blocks only from finished items, so a non-streaming
            // turn needs the whole message restated as one.
            if !self.text.is_empty() {
                let id = self.msg_id.clone().unwrap_or_default();
                let text = std::mem::take(&mut self.text);
                out.push(json!({"type": "response.output_text.done",
                    "item_id": id, "output_index": 0, "content_index": 0, "text": text}));
                out.push(json!({"type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message", "id": id, "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text}],
                    }}));
            }
        }
        if let Some(usage) = chunk
            .get("usage")
            .filter(|u| !u.is_null())
            .and_then(crate::gemini::usage::as_responses)
        {
            self.usage = Some(usage);
        }
        // Gemini reports usage on a chunk of its own, often before the one
        // carrying finish_reason, so emitting on usage alone closed the
        // response before its content and then closed it twice.
        if self.finished
            && !self.completed
            && let Some(usage) = self.usage.take()
        {
            self.completed = true;
            out.push(json!({"type": "response.completed", "response": {
                "id": chunk.get("id"), "status": "completed", "usage": usage,
            }}));
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
            slot.item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
            slot.index = self.next_index;
            self.next_index += 1;
            let slot = &self.calls[idx];
            out.push(json!({"type": "response.output_item.added",
                "output_index": slot.index,
                "item": {
                    "type": "function_call", "id": slot.item_id, "call_id": slot.id,
                    "name": slot.name, "arguments": "",
                }}));
        }
        let slot = &mut self.calls[idx];
        if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str)
            && !args.is_empty()
        {
            slot.arguments.push_str(args);
            out.push(json!({"type": "response.function_call_arguments.delta",
                "item_id": slot.item_id, "output_index": slot.index, "delta": args}));
        }
        out
    }
}

/// A referer-restricted key is served over the native surface, which answers
/// in Gemini's own frames, so that protocol is normalised before parsing.
/// The bridge's frames as Responses-shaped JSON, before anything parses them
/// into `ResponsesEvent`. The responses surface relays frames rather than
/// events, so it needs them in this form.
pub fn value_stream(
    resp: reqwest::Response,
    protocol: crate::gemini::client::GeminiProtocol,
    model: &str,
) -> impl futures_util::Stream<Item = Value> + Send + use<> {
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
    chat_bytes.eventsource().flat_map(move |ev| {
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
        futures_util::stream::iter(events)
    })
}

pub fn event_stream(
    resp: reqwest::Response,
    protocol: crate::gemini::client::GeminiProtocol,
    model: &str,
) -> crate::codex::sse::EventStream {
    use futures_util::StreamExt;
    let stream = value_stream(resp, protocol, model)
        .map(|e| {
            serde_json::from_value(e).unwrap_or(crate::codex::types::ResponsesEvent::Other)
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
        assert_eq!(
            first,
            ["response.created", "response.output_item.added",
             "response.content_part.added", "response.output_text.delta"]
        );
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
        assert_eq!(
            done,
            ["response.function_call_arguments.done", "response.output_item.done"]
        );
        // An item done without its arguments is a call the client cannot run.
        let last = b.feed(&json!({"choices": [{"finish_reason": "tool_calls"}]}));
        assert!(last.is_empty(), "the turn closed twice: {last:?}");
    }

    #[test]
    fn a_finished_tool_call_carries_its_arguments() {
        let mut b = ChatToResponses::default();
        b.feed(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": "{\"p"}}]}}]}));
        b.feed(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "ath\":1}"}}]}}]}));
        let done = b.feed(&json!({"choices": [{"finish_reason": "tool_calls"}]}));
        let item = &done[1]["item"];
        assert_eq!(item["arguments"], "{\"path\":1}");
        assert_eq!(item["call_id"], "c1");
        assert_eq!(item["name"], "Read");
        assert!(item["id"].as_str().is_some_and(|s| s.starts_with("fc_")));
    }

    /// Gemini puts usage on a chunk of its own, often ahead of the one with
    /// finish_reason. Closing on usage alone ended the response before its
    /// content and then ended it a second time, which codex rejects.
    #[test]
    fn the_response_closes_once_and_after_its_content() {
        let mut b = ChatToResponses::default();
        assert_eq!(
            feed(&mut b, json!({"choices": [{"delta": {"content": "hi"}}]})),
            ["response.created", "response.output_item.added",
             "response.content_part.added", "response.output_text.delta"]
        );
        assert_eq!(
            feed(&mut b, json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1}})),
            Vec::<String>::new()
        );
        assert_eq!(
            feed(&mut b, json!({"choices": [{"delta": {}, "finish_reason": "stop"}]})),
            ["response.output_text.done", "response.output_item.done", "response.completed"]
        );
        assert_eq!(
            feed(&mut b, json!({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1}})),
            Vec::<String>::new()
        );
    }

    #[test]
    fn the_developer_role_survives_as_system() {
        let req = json!({
            "model": "gemini-3.8-flash", "instructions": "sys", "stream": true,
            "input": [{"type": "message", "role": "developer",
                       "content": [{"type": "input_text", "text": "ctx"}]}],
        });
        let body = to_chat(&req);
        assert_eq!(body["messages"][1]["role"], "system");
        assert_eq!(body["messages"][1]["content"][0]["text"], "ctx");
    }

    /// The responses surface forwards a body it received, so tool calls and
    /// their results have to survive as JSON rather than as typed items.
    #[test]
    fn a_tool_round_trip_survives_the_value_form() {
        let req = json!({
            "model": "gemini-3.8-flash", "instructions": "sys", "stream": true,
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "grep", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "hit"},
            ],
            "tools": [{"type": "function", "name": "grep", "description": "d",
                       "parameters": {"type": "object"}}],
            "tool_choice": "auto",
        });
        let body = to_chat(&req);
        assert_eq!(body["messages"][1]["tool_calls"][0]["function"]["name"], "grep");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["content"], "hit");
        assert_eq!(body["tools"][0]["function"]["name"], "grep");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["stream_options"]["include_usage"], true);
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
        assert_eq!(
            kinds,
            ["response.created", "response.output_item.added",
             "response.content_part.added", "response.output_text.delta"]
        );
        let parsed: Vec<ResponsesEvent> = raw
            .into_iter()
            .map(|e| serde_json::from_value(e).unwrap_or(ResponsesEvent::Other))
            .collect();
        assert!(
            parsed
                .iter()
                .any(|e| matches!(e, ResponsesEvent::OutputTextDelta { .. })),
            "no text delta survived: {parsed:?}"
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
