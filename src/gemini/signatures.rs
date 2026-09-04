//! Google rejects a replayed `functionCall` that lost its `thoughtSignature`.
//! See <https://ai.google.dev/gemini-api/docs/thought-signatures>.

use std::collections::HashMap;
use std::hash::{Hash as _, Hasher as _};
use std::sync::{LazyLock, Mutex};

use crate::gemini::types::{Content, FunctionCall, Part};

/// Two generations rather than one map, because clearing on overflow drops
/// every live conversation's signatures at the same instant and breaks all of
/// them at once.
struct Cache {
   current: HashMap<u64, String>,
   previous: HashMap<u64, String>,
}

const MAX_SIGNATURES: usize = 8192;

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| {
   Mutex::new(Cache {
      current: HashMap::new(),
      previous: HashMap::new(),
   })
});

impl Cache {
   fn put(&mut self, key: u64, signature: String) {
      if self.current.len() >= MAX_SIGNATURES {
         self.previous = std::mem::take(&mut self.current);
      }
      self.current.insert(key, signature);
   }

   fn get(&self, key: u64) -> Option<&String> {
      self.current.get(&key).or_else(|| self.previous.get(&key))
   }
}

pub fn put(key: u64, signature: &str) {
   CACHE.lock().unwrap().put(key, signature.to_owned());
}

pub fn get(key: u64) -> Option<String> {
   CACHE.lock().unwrap().get(key).cloned()
}

pub fn call_id_key(call_id: &str) -> u64 {
   let mut hasher = std::collections::hash_map::DefaultHasher::new();
   call_id.hash(&mut hasher);
   hasher.finish()
}

/// The native surface issues no call id, so a call is keyed by what it asks for.
fn part_key(call: &FunctionCall) -> u64 {
   let mut hash = hmac_sha256::Hash::new();
   hash.update(call.name.as_bytes());
   if let Some(args) = &call.args {
      hash.update(serde_json::to_string(args).unwrap_or_default().as_bytes());
   }
   u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap_or_default())
}

pub fn remember(parts: &[Part]) {
   for part in parts {
      if let (Some(call), Some(sig)) = (&part.function_call, &part.thought_signature) {
         put(part_key(call), sig);
      }
   }
}

/// True when one was put back, the only reason to re-encode the body.
pub fn restore(contents: &mut [Content]) -> bool {
   let mut patched = false;
   for part in contents.iter_mut().flat_map(|c| c.parts.iter_mut()) {
      if part.thought_signature.is_some() {
         continue;
      }
      let Some(sig) = part.function_call.as_ref().and_then(|c| get(part_key(c))) else {
         continue;
      };
      part.thought_signature = Some(sig);
      patched = true;
   }
   patched
}

#[cfg(test)]
mod tests {
   use super::*;

   fn call(name: &str) -> Part {
      Part {
         function_call: Some(FunctionCall {
            id: None,
            name: name.into(),
            args: Some(serde_json::json!({"path": "/tmp"})),
         }),
         ..Part::default()
      }
   }

   #[test]
   fn a_replayed_call_gets_its_signature_back() {
      let mut seen = call("list_dir");
      seen.thought_signature = Some("EtUBCtIB".into());
      remember(&[seen]);

      let mut contents = vec![Content {
         role: Some("model".into()),
         parts: vec![call("list_dir"), call("never_seen")],
      }];
      assert!(restore(&mut contents));
      assert_eq!(
         contents[0].parts[0].thought_signature.as_deref(),
         Some("EtUBCtIB")
      );
      assert_eq!(contents[0].parts[1].thought_signature, None);
   }

   #[test]
   fn an_overflow_keeps_the_generation_before_it() {
      let mut cache = Cache {
         current: HashMap::new(),
         previous: HashMap::new(),
      };
      cache.put(1, "first".into());
      for k in 0..MAX_SIGNATURES as u64 {
         cache.put(k + 100, "filler".into());
      }
      cache.put(999_999, "after".into());
      assert_eq!(cache.get(1).map(String::as_str), Some("first"));
   }
}
