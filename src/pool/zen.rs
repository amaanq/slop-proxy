use std::mem;
use std::sync::LazyLock;

use axum::body::Bytes;
use serde_json::value::RawValue;

use crate::clock::unix_now_ms;
use crate::codex::types::ToolDef;
use crate::pool::{Backend, Pool, Relay, Route, Slot};
use crate::provider::Provider;
use crate::translate::chat::{ChatRequest, ChatToolDef, FunctionDef};
use crate::upstream::SendError;
use crate::zen::{ZenClient, ZenModel};

/// Zen over whatever credentials are stored, and over none at all when the
/// table is empty. The free models are served without a key today, so an
/// empty pool is a working pool rather than an error.
pub type ZenPool = Pool<ZenClient>;

const DECOY_HINT: &str = "Deprecated placeholder. Never call this tool.";

/// A tool the free tier insists on. Since 2026-10-13 zen also checks each
/// one's parameters against the CLI's own, so decoys carry them verbatim.
struct GateTool {
   name: &'static str,
   parameters: &'static str,
}

const SHELL: GateTool = GateTool {
   name: "bash",
   parameters: r#"{"properties":{"command":{"description":"The command to execute","type":"string"},"timeout":{"description":"Optional timeout in milliseconds","type":"integer"},"workdir":{"description":"The working directory to run the command in. Defaults to the current directory. Use this instead of 'cd' commands.","type":"string"}},"required":["command"],"type":"object"}"#,
};

const EDITOR: GateTool = GateTool {
   name: "edit",
   parameters: r#"{"properties":{"filePath":{"description":"The absolute path to the file to modify","type":"string"},"oldString":{"description":"The text to replace","type":"string"},"newString":{"description":"The text to replace it with (must be different from oldString)","type":"string"},"replaceAll":{"description":"Replace all occurrences of oldString (default false)","type":"boolean"}},"required":["filePath","oldString","newString"],"type":"object"}"#,
};

const READER: GateTool = GateTool {
   name: "read",
   parameters: r#"{"properties":{"filePath":{"description":"The absolute path to the file or directory to read","type":"string"},"offset":{"description":"The line number to start reading from (1-indexed)","type":"integer"},"limit":{"description":"The maximum number of lines to read (defaults to 2000)","type":"integer"}},"required":["filePath"],"type":"object"}"#,
};

impl GateTool {
   fn parameters(&self) -> Box<RawValue> {
      RawValue::from_string(self.parameters.to_owned()).expect("gate parameters are valid JSON")
   }
}

fn missing_gate_tools(declared: &[&str]) -> Vec<&'static GateTool> {
   let mut missing = Vec::new();
   if !declared.contains(&SHELL.name) {
      missing.push(&SHELL);
   }
   if !declared.iter().any(|&name| name == EDITOR.name || name == "write") {
      missing.push(&EDITOR);
   }
   if !declared.contains(&READER.name) {
      missing.push(&READER);
   }
   missing
}

/// `None` when the caller already satisfies the gate, so the usual request
/// reaches zen byte for byte. The CLI's SDK leaves input messages untagged,
/// and the gate now rejects an explicit `"type": "message"`.
pub fn satisfy_tool_gate(body: &Bytes) -> Option<Bytes> {
   let mut req: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(body).ok()?;
   let mut tools: Vec<serde_json::Value> = req
      .get_mut("tools")
      .and_then(serde_json::Value::as_array_mut)
      .map(mem::take)
      .unwrap_or_default();
   let declared: Vec<&str> = tools
      .iter()
      .filter_map(|tool| tool.get("name")?.as_str())
      .collect();
   let missing = missing_gate_tools(&declared);

   let mut untagged = 0_usize;
   let items = req
      .get_mut("input")
      .and_then(serde_json::Value::as_array_mut)
      .into_iter()
      .flatten()
      .filter_map(serde_json::Value::as_object_mut);
   for item in items {
      if item.get("type").and_then(serde_json::Value::as_str) == Some("message") {
         item.remove("type");
         untagged += 1;
      }
   }
   if missing.is_empty() && untagged == 0 {
      return None;
   }

   for tool in &missing {
      let decoy = ToolDef::function(
         tool.name.to_owned(),
         Some(DECOY_HINT.to_owned()),
         Some(tool.parameters()),
      );
      tools.push(serde_json::to_value(decoy).ok()?);
   }
   req.insert("tools".into(), serde_json::Value::Array(tools));
   tracing::debug!(
      added = missing.len(),
      untagged,
      "shaped zen request for the free-tier gate"
   );
   serde_json::to_vec(&req).ok().map(Bytes::from)
}

/// The same gate on the chat wire, where a tool's name sits under `function`
/// rather than at the top level.
pub fn satisfy_chat_tool_gate(req: &mut ChatRequest) {
   let tools = req.tools.get_or_insert_with(Vec::new);
   let declared: Vec<&str> = tools
      .iter()
      .filter_map(|tool| tool.def().name.as_deref())
      .collect();
   let missing = missing_gate_tools(&declared);
   if missing.is_empty() {
      return;
   }

   tools.extend(missing.iter().map(|tool| {
      ChatToolDef::function(FunctionDef {
         name: Some(tool.name.to_owned()),
         description: Some(DECOY_HINT.to_owned()),
         parameters: Some(tool.parameters()),
         strict: Some(false),
      })
   }));
   tracing::debug!(
      added = missing.len(),
      "padded zen chat tools past the free-tier gate"
   );
}

impl Backend for ZenClient {
   const PROVIDER: Provider = Provider::Zen;
   const ANONYMOUS: bool = true;
   type Request = Relay;
   type Response = reqwest::Response;

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      let session = session(route);
      Self::post(self, Some(token), &session, req.path, &req.body).await
   }

   async fn send_anonymous(
      &self,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      let session = session(route);
      Self::post(self, None, &session, req.path, &req.body).await
   }
}

const ALPHABET: &[u8; 62] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ID_TIME_MASK: i64 = 0xffff_ffff_ffff;
static SESSION_PREFIX: LazyLock<String> = LazyLock::new(session_prefix);

fn session(route: Route<'_>) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   if route.session_key.is_empty() {
      hasher.update(route.user.as_bytes());
      hasher.update(b"\0");
      hasher.update(route.model.as_bytes());
   } else {
      hasher.update(route.session_key.as_bytes());
   }
   let digest = hasher.finalize();
   let suffix = digest
      .iter()
      .take(14)
      .map(|byte| char::from(ALPHABET[usize::from(*byte) % ALPHABET.len()]))
      .collect::<String>();
   format!("ses_{}{suffix}", SESSION_PREFIX.as_str())
}

fn session_prefix() -> String {
   let timestamp = !(unix_now_ms().saturating_mul(0x1000) + 1) & ID_TIME_MASK;
   format!("{timestamp:012x}")
}

impl Pool<ZenClient> {
   pub async fn models(&self) -> Option<Vec<ZenModel>> {
      self.backend.models().await.ok()
   }
}
