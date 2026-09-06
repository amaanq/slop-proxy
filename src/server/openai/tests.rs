mod passthrough_tests {
   use super::super::*;

   #[test]
   fn the_service_tier_is_read_and_forwarded_unchanged() {
      let body = serde_json::json!({
          "model": "gpt-5.6-sol",
          "service_tier": "priority",
          "input": [],
      });
      let req: PassthroughRequest = serde_json::from_value(body).unwrap();
      assert_eq!(req.service_tier.as_deref(), Some("priority"));
      let out = serde_json::to_value(&req).unwrap();
      assert_eq!(out["service_tier"], "priority");
   }

   /// The typed request would serialise an item it does not know as
   /// `{"type":"Other"}` and drop a custom tool's grammar.
   #[test]
   fn what_the_proxy_has_no_opinion_on_goes_upstream_as_written() {
      let body = serde_json::json!({
          "model": "gpt-5.6-sol",
          "input": [{"type": "apply_patch_call", "call_id": "c1", "status": "completed"}],
          "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark"}}],
          "reasoning": {"summary": "auto"},
      });
      let req: PassthroughRequest = serde_json::from_value(body.clone()).unwrap();
      let out = serde_json::to_value(&req).unwrap();
      assert_eq!(out["input"], body["input"]);
      assert_eq!(out["tools"], body["tools"]);
      assert_eq!(out["reasoning"], body["reasoning"]);
   }

   fn cap(value: serde_json::Value) -> serde_json::Map<String, Value> {
      let mut rest = serde_json::Map::new();
      rest.insert("max_output_tokens".into(), value);
      rest
   }

   #[test]
   fn a_cap_below_the_upstream_floor_is_stripped_from_a_relayed_body() {
      let stripped = &mut cap(serde_json::json!(8_i64));
      drop_unusable_max_output_tokens(stripped, "u1");
      assert!(!stripped.contains_key("max_output_tokens"));

      let kept = &mut cap(serde_json::json!(4096_i64));
      drop_unusable_max_output_tokens(kept, "u1");
      assert_eq!(kept["max_output_tokens"], 4096_i64);

      let unparsed = &mut cap(serde_json::json!("auto"));
      drop_unusable_max_output_tokens(unparsed, "u1");
      assert_eq!(unparsed["max_output_tokens"], "auto");
   }
}

mod unpaired_tool_tests {
   use super::super::*;

   fn input(items: serde_json::Value) -> serde_json::Map<String, Value> {
      let mut rest = serde_json::Map::new();
      rest.insert("input".into(), items);
      rest
   }

   #[test]
   fn a_call_without_its_output_and_an_output_without_its_call_both_go() {
      let mut rest = input(serde_json::json!([
          {"type": "message", "role": "user", "content": "hi"},
          {"type": "function_call", "call_id": "kept", "name": "ls", "arguments": "{}"},
          {"type": "function_call_output", "call_id": "kept", "output": "ok"},
          {"type": "function_call", "call_id": "orphan-call", "name": "ls", "arguments": "{}"},
          {"type": "custom_tool_call_output", "call_id": "orphan-output", "output": "late"},
          {"type": "reasoning", "summary": []},
      ]));
      assert_eq!(drop_unpaired_tool_items(&mut rest), 2);
      let kinds: Vec<_> = rest["input"]
         .as_array()
         .unwrap()
         .iter()
         .map(|item| item["type"].as_str().unwrap())
         .collect();
      assert_eq!(
         kinds,
         [
            "message",
            "function_call",
            "function_call_output",
            "reasoning"
         ]
      );
   }

   #[test]
   fn codex_only_items_are_hoisted_rewritten_or_dropped() {
      let mut rest = input(serde_json::json!([
          {"type": "additional_tools", "role": "developer", "tools": [{"type": "function", "name": "spawn_agent"}, {"type": "function", "name": "wait_agent"}]},
          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
          {"type": "agent_message", "author": "main", "recipient": "worker", "content": [{"type": "input_text", "text": "check "}, {"type": "encrypted_content", "encrypted_content": "x"}, {"type": "input_text", "text": "the tree"}]},
          {"type": "local_shell_call", "call_id": "sh1", "action": {"type": "exec", "command": ["ls"]}},
          {"type": "function_call_output", "call_id": "sh1", "output": "files"},
          {"type": "context_compaction", "summary": "earlier"},
      ]));
      rest.insert(
         "tools".into(),
         serde_json::json!([{"type": "function", "name": "shell"}]),
      );
      let fixes = zen_input_fixups(&mut rest);
      assert_eq!(
         (
            fixes.hoisted,
            fixes.rewritten,
            fixes.dropped,
            fixes.unpaired
         ),
         (2, 1, 3, 1)
      );
      let names: Vec<_> = rest["tools"]
         .as_array()
         .unwrap()
         .iter()
         .map(|tool| tool["name"].as_str().unwrap())
         .collect();
      assert_eq!(names, ["shell", "spawn_agent", "wait_agent"]);
      assert_eq!(
         rest["input"],
         serde_json::json!([
             {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
             {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "check the tree"}]},
         ])
      );
   }

   #[test]
   fn the_rejected_index_is_read_from_zens_body() {
      let body = r#"{"error":{"param":"input[4]","type":"invalid_request_error"}}"#;
      assert_eq!(rejected_index(body), Some(4));
      assert_eq!(rejected_index(r#"{"error":{"param":"call_id"}}"#), None);
   }

   #[test]
   fn a_paired_history_is_untouched() {
      let items = serde_json::json!([
          {"type": "apply_patch_call", "call_id": "p1", "status": "completed"},
          {"type": "apply_patch_call_output", "call_id": "p1", "output": "done"},
      ]);
      let mut rest = input(items.clone());
      assert_eq!(drop_unpaired_tool_items(&mut rest), 0);
      assert_eq!(rest["input"], items);
   }
}

mod terminal_event_tests {
   use super::super::TerminalEvent;

   /// Parsed from the wire, not from a Value, because the bug this pins only
   /// shows on the streaming deserializer.
   #[test]
   fn the_final_response_survives_the_terminal_frame() {
      let frame = r#"{"type":"response.completed","sequence_number":9,"response":{"id":"resp_1","usage":{"input_tokens":1}}}"#;
      let event: TerminalEvent = serde_json::from_str(frame).unwrap();
      assert!(TerminalEvent::is_terminal(&event.kind));
      assert_eq!(
         event.response.unwrap().get(),
         r#"{"id":"resp_1","usage":{"input_tokens":1}}"#
      );
   }
}

mod rate_limit_header_tests {
   use super::super::*;
   use crate::pool::UsageWindow;

   fn window(name: &str, utilization: f64, resets_at: Option<i64>) -> UsageWindow {
      UsageWindow {
         name: name.into(),
         utilization,
         resets_at,
      }
   }

   #[test]
   fn the_shorter_window_is_primary() {
      let out = rate_limit_headers(&[
         window("7d", 0.42, Some(1000)),
         window("5h", 0.11, Some(500)),
      ]);
      let get = |key: &str| {
         out.iter()
            .find(|&&(ref name, _)| name == key)
            .map(|&(_, ref value)| value.as_str())
            .unwrap()
      };
      assert_eq!(get("x-codex-primary-window-minutes"), "300");
      assert_eq!(get("x-codex-primary-used-percent"), "11");
      assert_eq!(get("x-codex-secondary-window-minutes"), "10080");
      assert_eq!(get("x-codex-secondary-used-percent"), "42");
   }

   #[test]
   fn a_window_without_a_reset_still_reports() {
      let out = rate_limit_headers(&[window("7d", 0.8, None)]);
      assert!(
         out.iter()
            .any(|&(ref name, ref value)| name == "x-codex-primary-used-percent" && value == "80")
      );
      assert!(
         !out
            .iter()
            .any(|&(ref name, _)| name == "x-codex-primary-reset-at")
      );
   }
}
