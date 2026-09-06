use std::collections::BTreeMap;

use futures_util::stream;
use serde_json::{Value, json};

use super::anthropic_stream::AnthropicStream;
use super::gemini_bridge::ChatToResponses;
use super::gemini_req::{custom_tools, to_chat};
use super::openai_stream::OpenAiStream;
use super::{Block, StopKind, UsageCapture, aggregate};
use crate::codex::types::{OutputItem, ResponsesEvent};
use crate::gemini::native::{NativeEvent, NativeStream, request};

fn interleaved() -> Vec<ResponsesEvent> {
   vec![
      json!({"type":"response.created","response":{"id":"r"}}),
      json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"a","name":"alpha"}}),
      json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"b","name":"beta"}}),
      json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}),
      json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"y\":"}),
      json!({"type":"response.output_text.delta","output_index":1,"delta":"hello"}),
      json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"2}"}),
      json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"1}"}),
      json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"a","name":"alpha","arguments":"{\"x\":1}"}}),
      json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","call_id":"b","name":"beta","arguments":"{\"y\":2}"}}),
      json!({"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":7}}}),
   ].into_iter().map(|value| serde_json::from_value(value).unwrap()).collect()
}

#[tokio::test]
async fn interleaved_calls_keep_their_arguments_in_every_renderer() {
   let capture = UsageCapture::default();
   let result = aggregate(Box::pin(stream::iter(interleaved())), &capture).await;
   let calls = result
      .blocks
      .iter()
      .filter_map(|block| match block {
         &Block::ToolCall {
            ref id,
            ref name,
            ref arguments,
         } => Some((id.as_str(), name.as_str(), arguments.as_str())),
         _ => None,
      })
      .collect::<Vec<_>>();
   assert_eq!(
      calls,
      [("a", "alpha", "{\"x\":1}"), ("b", "beta", "{\"y\":2}")]
   );
   assert_eq!(result.stop, StopKind::ToolUse);
   assert_eq!(result.usage.output_tokens, 7);

   let mut chat = OpenAiStream::new("m".into(), true, UsageCapture::default());
   let mut ids = BTreeMap::new();
   let mut arguments = BTreeMap::<u64, String>::new();
   let mut text = String::new();
   for frame in interleaved()
      .into_iter()
      .flat_map(|event| chat.handle(event))
   {
      let value: Value = serde_json::from_str(&frame).unwrap();
      let delta = &value["choices"][0]["delta"];
      if let Some(content) = delta["content"].as_str() {
         text.push_str(content);
      }
      if let Some(tool_calls) = delta["tool_calls"].as_array() {
         for call in tool_calls {
            let index = call["index"].as_u64().unwrap();
            if let Some(id) = call["id"].as_str() {
               ids.insert(index, id.to_owned());
            }
            if let Some(args) = call["function"]["arguments"].as_str() {
               arguments.entry(index).or_default().push_str(args);
            }
         }
      }
   }
   assert_eq!(text, "hello");
   assert_eq!(ids, BTreeMap::from([(0, "a".into()), (1, "b".into())]));
   assert_eq!(
      arguments,
      BTreeMap::from([(0, "{\"x\":1}".into()), (1, "{\"y\":2}".into())])
   );

   let mut messages = AnthropicStream::new("m".into(), 10, false, UsageCapture::default());
   let mut anthropic_arguments = BTreeMap::<u64, String>::new();
   let mut closed = Vec::new();
   for (kind, frame) in interleaved()
      .into_iter()
      .flat_map(|event| messages.handle(event))
   {
      let value: Value = serde_json::from_str(&frame).unwrap();
      if kind == "content_block_start" && value["content_block"]["type"] == "tool_use" {
         assert_eq!(value["content_block"]["input"], json!({}));
         assert_eq!(
            value["content_block"]["id"],
            ids[&value["index"].as_u64().unwrap()]
         );
      }
      if let Some(args) = value["delta"]["partial_json"].as_str() {
         anthropic_arguments
            .entry(value["index"].as_u64().unwrap())
            .or_default()
            .push_str(args);
      }
      if kind == "content_block_stop" {
         closed.push(value["index"].as_u64().unwrap());
      }
   }
   assert_eq!(
      anthropic_arguments,
      BTreeMap::from([(0, "{\"x\":1}".into()), (1, "{\"y\":2}".into())])
   );
   assert_eq!(closed, [0, 1, 2]);
}

#[test]
fn every_terminal_event_records_usage_and_status() {
   for kind in ["completed", "incomplete", "failed"] {
      let capture = UsageCapture::default();
      let event = serde_json::from_value(json!({
         "type": format!("response.{kind}"),
         "response": {"usage": {"input_tokens": 12_i32, "output_tokens": 5_i32, "input_tokens_details": {"cached_tokens": 2_i32}}}
      })).unwrap();
      capture.observe(&event);
      let snapshot = capture.snapshot();
      assert!(snapshot.completed);
      assert_eq!(snapshot.input_tokens, 10);
      assert_eq!(snapshot.cache_read_tokens, 2);
      assert_eq!(snapshot.output_tokens, 5);
      assert_eq!(snapshot.stop_reason.as_deref(), Some(kind));
      assert_eq!(snapshot.error_kind.is_some(), kind == "failed");
   }
}

#[test]
fn bridge_distinguishes_incomplete_responses_and_missing_terminals() {
   for (reason, expected) in [
      ("stop", "response.completed"),
      ("length", "response.incomplete"),
      ("content_filter", "response.incomplete"),
   ] {
      let mut bridge = ChatToResponses::default();
      bridge.feed(
         &serde_json::from_value(
            json!({"id":"r","choices":[{"delta":{"content":"partial"},"finish_reason":reason}]}),
         )
         .unwrap(),
      );
      let terminal = bridge.finalize();
      assert_eq!(terminal.len(), 1);
      assert_eq!(terminal[0].kind(), expected);
      assert_eq!(terminal[0].terminal().unwrap().1.id.as_deref(), Some("r"));
      assert!(bridge.finalize().is_empty());
   }
   let mut bridge = ChatToResponses::default();
   bridge.feed(
      &serde_json::from_value(
         json!({"choices":[{"delta":{"content":"partial"}}], "usage":{"prompt_tokens":5_i32}}),
      )
      .unwrap(),
   );
   let terminal = bridge.finalize();
   assert_eq!(terminal[0].kind(), "response.failed");
   assert_eq!(
      terminal[0]
         .terminal()
         .unwrap()
         .1
         .usage
         .as_ref()
         .unwrap()
         .input_tokens,
      5
   );
}

#[test]
fn upstream_call_ids_cannot_mix_signatures_between_turns() {
   let mut ids = Vec::new();
   for signature in ["first", "second"] {
      let mut bridge = ChatToResponses::default();
      let events = bridge.feed(
         &serde_json::from_value(json!({"choices":[{"delta":{"tool_calls":[{
            "id":"call_native_0", "index":0_i32, "function":{"name":"read","arguments":"{}"},
            "extra_content":{"google":{"thought_signature":signature}}
         }]}}]}))
         .unwrap(),
      );
      let call_id = events
         .iter()
         .find_map(|event| match event {
            &ResponsesEvent::OutputItemAdded {
               item: OutputItem::FunctionCall { ref call_id, .. },
               ..
            } => Some(call_id.clone()),
            _ => None,
         })
         .unwrap();
      ids.push(call_id);
   }
   assert_ne!(ids[0], ids[1]);
   for (call_id, signature) in ids.iter().zip(["first", "second"]) {
      let req = serde_json::from_value(json!({"model":"gemini-test","input":[{"type":"function_call","call_id":call_id,"name":"read","arguments":"{}"}]})).unwrap();
      let chat = to_chat(&req);
      assert_eq!(
         chat.messages[1].tool_calls.as_ref().unwrap()[0].thought_signature(),
         Some(signature)
      );
   }
}

#[test]
fn additional_tool_definitions_reach_gemini() {
   let req = serde_json::from_value(json!({"model":"gemini-test","input":[{
      "type":"additional_tools","role":"system","tools":[{"type":"custom","name":"shell"}]
   }]}))
   .unwrap();
   assert!(custom_tools(&req).contains("shell"));
   let chat = to_chat(&req);
   let tool = &chat.tools.as_ref().unwrap()[0];
   assert_eq!(tool.def().name.as_deref(), Some("shell"));
   let parameters: Value =
      serde_json::from_str(tool.def().parameters.as_ref().unwrap().get()).unwrap();
   assert_eq!(parameters["required"], json!(["input"]));
}

#[tokio::test]
async fn native_reasoning_and_tool_calls_survive_separate_frames() {
   let mut native = NativeStream::new("gemini-test");
   let mut bridge = ChatToResponses::default();
   let mut events = Vec::new();
   for value in [
      json!({"candidates":[{"content":{"parts":[{"thought":true,"text":"considering"},{"text":"answer"}]}}]}),
      json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"alpha","args":{"x":1_i32}}}]}}]}),
      json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"beta","args":{"y":2_i32}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10_i32,"candidatesTokenCount":2_i32}}),
   ] {
      for event in native.events(format!("data: {value}\n\n").as_bytes()) {
         if let NativeEvent::Chunk(chunk) = event {
            events.extend(bridge.feed(&chunk));
         }
      }
   }
   assert!(events.iter().any(|event| matches!(
      event,
      ResponsesEvent::OutputItemDone {
         item: OutputItem::Reasoning { .. },
         ..
      }
   )));
   let result = aggregate(Box::pin(stream::iter(events)), &UsageCapture::default()).await;
   assert!(matches!(&result.blocks[0], Block::Thinking { text, .. } if text == "considering"));
   assert!(matches!(&result.blocks[1], Block::Text { text } if text == "answer"));
   assert!(
      matches!(&result.blocks[2], Block::ToolCall { name, arguments, .. } if name == "alpha" && arguments == "{\"x\":1}")
   );
   assert!(
      matches!(&result.blocks[3], Block::ToolCall { name, arguments, .. } if name == "beta" && arguments == "{\"y\":2}")
   );
}

#[test]
fn reasoning_effort_reaches_the_native_generation_config() {
   for (model, expected) in [
      (
         "gemini-3.8-flash",
         json!({"thinkingLevel":"high","includeThoughts":true}),
      ),
      (
         "gemini-2.5-flash",
         json!({"thinkingBudget":24_576_i32,"includeThoughts":true}),
      ),
   ] {
      let req =
         serde_json::from_value(json!({"model":model,"messages":[],"reasoning_effort":"high"}))
            .unwrap();
      let native = request(&req).unwrap();
      let body = serde_json::to_value(&native.body).unwrap();
      assert_eq!(body["generationConfig"]["thinkingConfig"], expected);
   }
}

#[test]
fn unsupported_image_output_is_reported_instead_of_disappearing() {
   let mut bridge = ChatToResponses::default();
   let chunk = serde_json::from_value(json!({"choices":[{"delta":{"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}}]})).unwrap();
   let events = bridge.feed(&chunk);
   assert_eq!(events[0].kind(), "response.failed");
   assert_eq!(
      events[0]
         .terminal()
         .unwrap()
         .1
         .error
         .as_ref()
         .unwrap()
         .code
         .as_deref(),
      Some("unsupported_output")
   );
}
