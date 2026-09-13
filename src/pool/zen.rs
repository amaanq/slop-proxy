use std::mem;
use std::sync::LazyLock;

use axum::body::Bytes;
use rand::{Rng as _, thread_rng};

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::clock::unix_now_ms;
use crate::provider::Provider;
use crate::translate::anthropic_req::empty_schema;
use crate::translate::chat::{ChatError, ChatRequest, ChatToolDef, FunctionDef};
use crate::upstream::SendError;
use crate::zen::client::ZenClient;

/// Zen over whatever credentials are stored, and over none at all when the
/// table is empty. The free models are served without a key today, so an
/// empty pool is a working pool rather than an error.
pub type ZenPool = Pool<ZenClient>;

#[derive(Clone)]
pub struct Relay {
   pub path: &'static str,
   pub body: Bytes,
}

const GATE_SHELL: &str = "bash";
const GATE_READER: &str = "read";
const GATE_EDITORS: [&str; 2] = ["edit", "write"];
const DECOY_HINT: &str = "Deprecated placeholder. Never call this tool.";

fn missing_gate_tools(declared: &[&str]) -> Vec<&'static str> {
   let mut missing = Vec::new();
   if !declared.contains(&GATE_SHELL) {
      missing.push(GATE_SHELL);
   }
   if !declared.contains(&GATE_READER) {
      missing.push(GATE_READER);
   }
   if !GATE_EDITORS.iter().any(|name| declared.contains(name)) {
      missing.push(GATE_EDITORS[0]);
   }
   missing
}

/// `None` when the caller already satisfies the gate, so the usual request
/// reaches zen byte for byte.
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
   if missing.is_empty() {
      return None;
   }

   tools.extend(missing.iter().map(|name| {
      serde_json::json!({
         "type": "function",
         "name": name,
         "description": DECOY_HINT,
         "strict": false,
         "parameters": {"type": "object", "properties": {}, "additionalProperties": false},
      })
   }));
   req.insert("tools".into(), serde_json::Value::Array(tools));
   tracing::debug!(
      added = missing.len(),
      "padded zen tools past the free-tier gate"
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

   tools.extend(missing.iter().map(|name| {
      ChatToolDef::function(FunctionDef {
         name: Some((*name).to_owned()),
         description: Some(DECOY_HINT.to_owned()),
         parameters: Some(empty_schema()),
         strict: None,
      })
   }));
   tracing::debug!(
      added = missing.len(),
      "padded zen chat tools past the free-tier gate"
   );
}

impl Backend for ZenClient {
   const PROVIDER: Provider = Provider::Zen;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
   const ANONYMOUS: bool = true;
   type Request = Relay;
   type Response = reqwest::Response;

   fn reason(body: String) -> String {
      ChatError::reason(body)
   }

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
   let counter = thread_rng().gen_range(1..=0xfff_i64);
   let timestamp = !(unix_now_ms().saturating_mul(0x1000) + counter) & ID_TIME_MASK;
   format!("{timestamp:012x}")
}

impl Pool<ZenClient> {
   pub async fn models(&self) -> Vec<String> {
      self.backend.models().await.unwrap_or_default()
   }
}
