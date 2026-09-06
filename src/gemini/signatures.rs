//! Google rejects a replayed `functionCall` that lost its `thoughtSignature`.
//! See <https://ai.google.dev/gemini-api/docs/thought-signatures>.

use std::collections::HashMap;
use std::mem;
use std::sync::{LazyLock, Mutex};

/// Two generations rather than one map, because clearing on overflow drops
/// every live conversation's signatures at the same instant and breaks all of
/// them at once.
struct Cache {
   current: HashMap<String, String>,
   previous: HashMap<String, String>,
}

const MAX_SIGNATURES: usize = 8192;

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| {
   Mutex::new(Cache {
      current: HashMap::new(),
      previous: HashMap::new(),
   })
});

impl Cache {
   fn put(&mut self, key: &str, signature: String) {
      if self.current.len() >= MAX_SIGNATURES {
         self.previous = mem::take(&mut self.current);
      }
      self.current.insert(key.to_owned(), signature);
   }

   fn get(&self, key: &str) -> Option<&String> {
      self.current.get(key).or_else(|| self.previous.get(key))
   }
}

pub fn put(key: &str, signature: &str) {
   CACHE.lock().unwrap().put(key, signature.to_owned());
}

pub fn get(key: &str) -> Option<String> {
   CACHE.lock().unwrap().get(key).cloned()
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn an_overflow_keeps_the_generation_before_it() {
      let mut cache = Cache {
         current: HashMap::new(),
         previous: HashMap::new(),
      };
      cache.put("first", "first".into());
      for key in 0..MAX_SIGNATURES as u64 {
         cache.put(&key.to_string(), "filler".into());
      }
      cache.put("after", "after".into());
      assert_eq!(cache.get("first").map(String::as_str), Some("first"));
   }
}
