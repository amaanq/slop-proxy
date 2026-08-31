use crate::config::{ModelsConfig, pattern_specificity};

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model: String,
    pub effort: Option<String>,
}

const EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max", "ultra"];

fn split_suffix(requested: &str) -> (&str, Option<String>) {
    match requested.rsplit_once(':') {
        Some((base, eff)) if EFFORTS.contains(&eff) => (base, Some(eff.to_string())),
        _ => (requested, None),
    }
}

/// The requested model is passed through to the backend as-is. No default
/// name mapping is applied; the only remapping is optional user-defined
/// aliases from config. An effort may be attached via a `model:effort` suffix.
pub fn resolve(cfg: &ModelsConfig, requested: &str) -> ResolvedModel {
    let (name, suffix_effort) = split_suffix(requested);

    let mut best = None;
    for (pattern, alias) in &cfg.aliases {
        let Some(specificity) = pattern_specificity(pattern, name) else {
            continue;
        };
        if best.map(|(s, _)| specificity > s).unwrap_or(true) {
            best = Some((specificity, alias));
        }
    }
    if let Some((_, alias)) = best {
        return ResolvedModel {
            model: alias.model.clone(),
            effort: suffix_effort.or_else(|| alias.effort.clone()),
        };
    }

    ResolvedModel {
        model: name.to_string(),
        effort: suffix_effort.or_else(|| cfg.default_effort.clone()),
    }
}

/// Codex models reject `minimal`.
pub fn clamp_effort(model: &str, effort: &str) -> String {
    if model.contains("codex") && effort == "minimal" {
        "low".into()
    } else {
        effort.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_and_suffix() {
        let cfg = ModelsConfig::default();
        let r = resolve(&cfg, "gpt-5.6-terra");
        assert_eq!(r.model, "gpt-5.6-terra");

        let r = resolve(&cfg, "gpt-5.6-sol:high");
        assert_eq!(r.model, "gpt-5.6-sol");
        assert_eq!(r.effort.as_deref(), Some("high"));

        let r = resolve(&cfg, "claude-sonnet-4");
        assert_eq!(r.model, "claude-sonnet-4");
    }

    #[test]
    fn optional_config_alias() {
        let mut cfg = ModelsConfig::default();
        cfg.aliases.insert(
            "claude-opus-*".into(),
            crate::config::ModelAlias {
                model: "gpt-5.6-sol".into(),
                effort: Some("high".into()),
            },
        );
        let r = resolve(&cfg, "claude-opus-4");
        assert_eq!(r.model, "gpt-5.6-sol");
        assert_eq!(r.effort.as_deref(), Some("high"));
    }

    #[test]
    fn clamp() {
        assert_eq!(clamp_effort("gpt-5.6-codex", "minimal"), "low");
        assert_eq!(clamp_effort("gpt-5.6-sol", "minimal"), "minimal");
    }
}
