use std::mem;
use std::sync::LazyLock;

use axum::body::Bytes;

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

/// Decoys that get a request past the free tier, which since 2026-10-13
/// also checks each one's parameters against the CLI's own.
static GATE_TOOLS: LazyLock<Vec<ToolDef>> = LazyLock::new(|| {
   serde_json::from_slice::<Vec<ToolDef>>(include_bytes!("zen_gate_tools.json"))
      .expect("zen_gate_tools.json is valid")
});

fn missing_gate_tools(declared: &[&str]) -> Vec<&'static ToolDef> {
   GATE_TOOLS
      .iter()
      .filter(|tool| !declared.contains(&tool.name.as_str()))
      .collect()
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
      tools.push(serde_json::to_value(tool).ok()?);
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
         name: Some(tool.name.clone()),
         description: tool.description.clone(),
         parameters: tool.parameters.clone(),
         strict: Some(tool.strict),
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
