use serde::{Deserialize, Serialize};

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
