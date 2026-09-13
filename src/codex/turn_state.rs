use axum::http::HeaderMap;

pub const HEADER: &str = "x-codex-turn-state";

/// This will read this token's ciphertext block count as the backend's routing verdict,
/// 10 blocks for a personal account and 11 for one it believes is flagged.
#[derive(Debug, Clone)]
pub struct TurnState {
   pub token: String,
   pub blocks: usize,
}

impl TurnState {
   /// Fernet is 57 bytes of framing over 16-byte ciphertext blocks, so the
   /// token's length gives the block count without decoding the base64.
   pub fn parse(token: &str) -> Option<Self> {
      let core = token.trim_end_matches('=');
      let bytes = core.len() * 3 / 4;
      let framed = core.starts_with("gAAAA") && bytes >= 73 && (bytes - 57).is_multiple_of(16);
      framed.then(|| Self {
         token: token.to_owned(),
         blocks: (bytes - 57) / 16,
      })
   }

   pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
      Self::parse(headers.get(HEADER)?.to_str().ok()?)
   }

   /// A shorter ciphertext is cleaner, and an equal one refreshes a stale pin.
   pub const fn supersedes(&self, held: &Self) -> bool {
      self.blocks <= held.blocks
   }
}
