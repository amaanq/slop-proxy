use crate::config::{ModelsConfig, pattern_specificity};

#[derive(Debug, Clone)]
pub struct ResolvedModel {
   pub model: String,
   pub effort: Option<String>,
}

const EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max", "ultra"];

fn split_suffix(requested: &str) -> (&str, Option<String>) {
   match requested.rsplit_once(':') {
      Some((base, eff)) if EFFORTS.contains(&eff) => (base, Some(eff.to_owned())),
      _ => (requested, None),
   }
}

pub fn suffix_effort(requested: &str) -> Option<String> {
   split_suffix(requested).1
}

/// The requested model is passed through to the backend as-is.
pub fn resolve(cfg: &ModelsConfig, requested: &str) -> ResolvedModel {
   let (name, suffix_effort) = split_suffix(requested);

   let mut best = None;
   for (pattern, alias) in &cfg.aliases {
      let Some(specificity) = pattern_specificity(pattern, name) else {
         continue;
      };
      if best.is_none_or(|(spec, _)| specificity > spec) {
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
      model: name.to_owned(),
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
   use crate::config::ModelAlias;

   #[test]
   fn passthrough_and_suffix() {
      let cfg = ModelsConfig::default();
      let res = resolve(&cfg, "gpt-5.6-terra");
      assert_eq!(res.model, "gpt-5.6-terra");

      let r_suffix = resolve(&cfg, "gpt-5.6-sol:high");
      assert_eq!(r_suffix.model, "gpt-5.6-sol");
      assert_eq!(r_suffix.effort.as_deref(), Some("high"));
   }

   #[test]
   fn optional_config_alias() {
      let mut cfg = ModelsConfig::default();
      cfg.aliases.insert(
         "claude-opus-*".into(),
         ModelAlias {
            model: "gpt-5.6-sol".into(),
            effort: Some("high".into()),
         },
      );
      let res = resolve(&cfg, "claude-opus-4");
      assert_eq!(res.model, "gpt-5.6-sol");
      assert_eq!(res.effort.as_deref(), Some("high"));
   }

   #[test]
   fn clamp() {
      assert_eq!(clamp_effort("gpt-5.6-codex", "minimal"), "low");
      assert_eq!(clamp_effort("gpt-5.6-sol", "minimal"), "minimal");
   }
}
