use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
   pub slug: String,
   #[serde(default)]
   pub display_name: Option<String>,
   #[serde(default)]
   pub default_reasoning_level: Option<String>,
   #[serde(default)]
   pub supported_reasoning_levels: Vec<ReasoningLevel>,
   #[serde(default)]
   pub visibility: Option<String>,
   #[serde(default)]
   pub supported_in_api: Option<bool>,
   #[serde(default)]
   pub context_window: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningLevel {
   #[serde(default)]
   pub effort: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
   #[serde(default)]
   pub models: Vec<ModelInfo>,
}

impl ModelInfo {
   pub fn listed(&self) -> bool {
      !matches!(self.visibility.as_deref(), Some("none" | "hide"))
         && self.supported_in_api != Some(false)
   }
}

/// Codex's fallback for an unlisted slug lets code mode declare a `custom` tool zen refuses.
#[derive(Serialize)]
struct ZenEntry<'a> {
   slug: &'a str,
   display_name: &'a str,
   description: &'static str,
   tool_mode: &'static str,
   shell_type: &'static str,
   web_search_tool_type: &'static str,
   apply_patch_tool_type: Option<()>,
   use_responses_lite: bool,
   prefer_websockets: bool,
   supports_search_tool: bool,
   experimental_supported_tools: [(); 0],
   priority: i32,
   upgrade: Option<()>,
   availability_nux: Option<()>,
   comp_hash: Option<()>,
   default_reasoning_level: &'static str,
   supported_reasoning_levels: Vec<Value>,
}

const ZEN_EFFORTS: [&str; 4] = ["low", "medium", "high", "xhigh"];

/// Cloned from `template` so the fields codex requires track the backend.
pub fn with_zen_entries(raw: &str, template: &str, ids: &[String]) -> Option<String> {
   let mut catalog: Value = serde_json::from_str(raw).ok()?;
   let models = catalog.get_mut("models")?.as_array_mut()?;
   let present: Vec<String> = models
      .iter()
      .filter_map(|entry| entry.get("slug").and_then(Value::as_str).map(str::to_owned))
      .collect();
   let base = models
      .iter()
      .find(|entry| entry.get("slug").and_then(Value::as_str) == Some(template))
      .or_else(|| {
         models
            .iter()
            .find(|entry| entry.get("visibility").and_then(Value::as_str) == Some("list"))
      })?
      .as_object()?
      .clone();
   let levels: Vec<Value> = base
      .get("supported_reasoning_levels")
      .and_then(Value::as_array)
      .map(|levels| {
         levels
            .iter()
            .filter(|level| {
               level
                  .get("effort")
                  .and_then(Value::as_str)
                  .is_some_and(|effort| ZEN_EFFORTS.contains(&effort))
            })
            .cloned()
            .collect()
      })
      .unwrap_or_default();
   for id in ids {
      if present.contains(id) {
         continue;
      }
      let patch = serde_json::to_value(ZenEntry {
         slug: id,
         display_name: id,
         description: "Served by opencode zen",
         tool_mode: "direct",
         shell_type: "unified_exec",
         web_search_tool_type: "text",
         apply_patch_tool_type: None,
         use_responses_lite: false,
         prefer_websockets: false,
         supports_search_tool: false,
         experimental_supported_tools: [],
         priority: 99,
         upgrade: None,
         availability_nux: None,
         comp_hash: None,
         default_reasoning_level: "high",
         supported_reasoning_levels: levels.clone(),
      })
      .ok()?;
      let mut entry = base.clone();
      if let Value::Object(fields) = patch {
         entry.extend(fields);
      }
      models.push(Value::Object(entry));
   }
   serde_json::to_string(&catalog).ok()
}

#[cfg(test)]
mod zen_entry_tests {
   use super::with_zen_entries;

   const CATALOG: &str = r#"{"models":[
      {"slug":"gpt-5.6-sol","visibility":"list","tool_mode":"code_mode_only","apply_patch_tool_type":"freeform",
       "use_responses_lite":true,"prefer_websockets":true,"shell_type":"unified_exec","priority":1,
       "web_search_tool_type":"text_and_image",
       "supported_reasoning_levels":[{"effort":"low"},{"effort":"xhigh"},{"effort":"ultra"}],
       "model_messages":{"instructions_template":"be codex"}},
      {"slug":"muse-old","visibility":"list","tool_mode":"direct"}
   ]}"#;

   #[test]
   fn a_zen_model_inherits_the_template_with_function_tools_only() {
      let ids = [
         "muse-spark-1.3-contributor-free".to_owned(),
         "muse-old".to_owned(),
      ];
      let out = with_zen_entries(CATALOG, "gpt-5.6-sol", &ids).unwrap();
      let catalog: serde_json::Value = serde_json::from_str(&out).unwrap();
      let models = catalog["models"].as_array().unwrap();
      assert_eq!(models.len(), 3);
      let muse = &models[2];
      assert_eq!(muse["slug"], "muse-spark-1.3-contributor-free");
      assert_eq!(muse["tool_mode"], "direct");
      assert!(muse["apply_patch_tool_type"].is_null());
      assert_eq!(muse["use_responses_lite"], false);
      assert_eq!(muse["prefer_websockets"], false);
      assert_eq!(muse["web_search_tool_type"], "text");
      assert_eq!(muse["model_messages"]["instructions_template"], "be codex");
      let efforts: Vec<_> = muse["supported_reasoning_levels"]
         .as_array()
         .unwrap()
         .iter()
         .map(|level| level["effort"].as_str().unwrap())
         .collect();
      assert_eq!(efforts, ["low", "xhigh"]);
   }

   #[test]
   fn a_catalog_without_a_listed_entry_is_left_alone() {
      let raw = r#"{"models":[{"slug":"x","visibility":"hide"}]}"#;
      assert!(with_zen_entries(raw, "gpt-5.6-sol", &["muse".to_owned()]).is_none());
   }
}
