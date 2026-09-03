//! Google rejects a replayed `functionCall` that lost its `thoughtSignature`.
//! See https://ai.google.dev/gemini-api/docs/thought-signatures.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::gemini::types::{Content, FunctionCall, Part};

static SIGNATURES: LazyLock<Mutex<HashMap<u64, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

const MAX_SIGNATURES: usize = 8192;

fn key(call: &FunctionCall) -> u64 {
    let mut hash = hmac_sha256::Hash::new();
    hash.update(call.name.as_bytes());
    if let Some(args) = &call.args {
        hash.update(serde_json::to_string(args).unwrap_or_default().as_bytes());
    }
    u64::from_be_bytes(hash.finalize()[..8].try_into().unwrap_or_default())
}

pub fn remember(parts: &[Part]) {
    let pairs: Vec<_> = parts
        .iter()
        .filter_map(|p| Some((key(p.function_call.as_ref()?), p.thought_signature.clone()?)))
        .collect();
    if pairs.is_empty() {
        return;
    }
    let mut map = SIGNATURES.lock().unwrap();
    if map.len().saturating_add(pairs.len()) > MAX_SIGNATURES {
        map.clear();
    }
    map.extend(pairs);
}

/// True when one was put back, the only reason to re-encode the body.
pub fn restore(contents: &mut [Content]) -> bool {
    let map = SIGNATURES.lock().unwrap();
    let mut patched = false;
    for part in contents.iter_mut().flat_map(|c| c.parts.iter_mut()) {
        if part.thought_signature.is_some() {
            continue;
        }
        let Some(call) = &part.function_call else {
            continue;
        };
        if let Some(sig) = map.get(&key(call)) {
            part.thought_signature = Some(sig.clone());
            patched = true;
        }
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
}
