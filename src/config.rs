use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use eyre::{Result, WrapErr as _};

use crate::cli::Cli;
use crate::provider::Provider;

pub const DEFAULT_INSTRUCTIONS: &str = "You are Codex, based on GPT-5. You are running as a coding agent on a user's computer. Answer the user's requests directly and concisely.";

#[derive(Debug, Clone)]
pub struct Config {
   pub db_path: PathBuf,
   pub bind: String,
   /// Extra unauthenticated listener serving GET /metrics when set.
   pub metrics_bind: Option<String>,
   pub codex: CodexConfig,
   pub anthropic: AnthropicConfig,
   pub gemini: GeminiConfig,
   pub zen: ZenConfig,
   pub glm: GlmConfig,
   pub pricing: PricingConfig,
   pub models: ModelsConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct GeminiConfig {
   pub base_url: String,
   /// Sent on every upstream call. A key restricted to an HTTP origin needs
   /// `Referer` here, and some deployments key on `x-goog-api-client`.
   pub headers: BTreeMap<String, String>,
   pub soft_utilization_limit: f64,
   pub retry_budget_secs: u64,
}

impl Default for GeminiConfig {
   fn default() -> Self {
      Self {
         base_url: "https://generativelanguage.googleapis.com/v1beta/openai".into(),
         headers: BTreeMap::new(),
         soft_utilization_limit: 0.9,
         retry_budget_secs: 90,
      }
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct PricingConfig {
   /// `LiteLLM` lands price additions on a staging branch and promotes them to
   /// main on a release cut, so a day-0 model is priced there first.
   pub url: String,
}

impl Default for PricingConfig {
   fn default() -> Self {
      Self {
            url: "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json".into(),
        }
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct GlmConfig {
   pub base_url: String,
}

impl Default for GlmConfig {
   fn default() -> Self {
      Self {
         base_url: "https://api.z.ai/api/anthropic".into(),
      }
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ZenConfig {
   pub base_url: String,
   pub proxy_urls: Vec<String>,
   pub proxy_urls_file: Option<PathBuf>,
}

impl Default for ZenConfig {
   fn default() -> Self {
      Self {
         base_url: "https://opencode.ai/zen/v1".into(),
         proxy_urls: Vec::new(),
         proxy_urls_file: None,
      }
   }
}

impl ZenConfig {
   pub fn proxy_urls(&self) -> Result<Vec<String>> {
      let mut urls = self.proxy_urls.clone();
      if let Some(path) = self.proxy_urls_file.as_ref() {
         let contents = fs::read_to_string(path)
            .wrap_err_with(|| format!("reading zen proxy list {}", path.display()))?;
         urls.extend(
            contents
               .lines()
               .map(str::trim)
               .filter(|line| !line.is_empty() && !line.starts_with('#'))
               .map(str::to_owned),
         );
      }
      Ok(urls)
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct AnthropicConfig {
   pub base_url: String,
   /// Anthropic's subscription terms cover use through Claude Code, so a
   /// request that does not come from it is refused rather than served from
   /// someone's Max seat.
   pub require_claude_code: bool,
   /// Fraction of a rolling window past which an account is ranked behind
   /// its peers, so sessions migrate before the window rejects them.
   pub soft_utilization_limit: f64,
}

impl Default for AnthropicConfig {
   fn default() -> Self {
      Self {
         base_url: "https://api.anthropic.com".into(),
         require_claude_code: true,
         soft_utilization_limit: 0.9,
      }
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct CodexConfig {
   pub base_url: String,
   pub originator: String,
   pub user_agent: String,
   /// Sent as the `version` header and `client_version` query on backend calls.
   pub version: String,
   /// Base instructions sent in the `instructions` field of every request.
   pub instructions: Option<String>,
   pub instructions_file: Option<PathBuf>,
   pub forward_max_tokens: bool,
   /// Fraction of a rolling window past which an account is ranked behind
   /// its peers, so traffic moves before the window rejects it.
   pub soft_utilization_limit: f64,
}

impl Default for CodexConfig {
   fn default() -> Self {
      Self {
         base_url: "https://chatgpt.com/backend-api/codex".into(),
         originator: "codex_cli_rs".into(),
         user_agent: "codex_cli_rs/0.151.0 (Linux; x86_64)".into(),
         version: "0.151.0".into(),
         instructions: None,
         instructions_file: None,
         forward_max_tokens: true,
         soft_utilization_limit: 0.9,
      }
   }
}

impl CodexConfig {
   pub fn instructions(&self) -> String {
      if let Some(path) = self.instructions_file.as_ref()
         && let Ok(text) = fs::read_to_string(path)
      {
         return text;
      }
      self
         .instructions
         .clone()
         .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.into())
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
   pub default: String,
   pub default_effort: Option<String>,
   pub aliases: BTreeMap<String, ModelAlias>,
   pub known: Vec<String>,
   /// Model patterns relayed verbatim to the Anthropic backend instead of
   /// being translated for the codex one.
   pub anthropic_patterns: Vec<String>,
   /// Model patterns served by the Gemini backend.
   pub gemini_patterns: Vec<String>,
   /// Model patterns served by `OpenCode` Zen.
   pub zen_patterns: Vec<String>,
   /// Model patterns served by Z.ai's anthropic-compatible endpoint.
   pub glm_patterns: Vec<String>,
}

impl Default for ModelsConfig {
   fn default() -> Self {
      Self {
         default: String::new(),
         default_effort: None,
         aliases: BTreeMap::new(),
         known: Vec::new(),
         anthropic_patterns: vec!["claude-*".into()],
         gemini_patterns: vec!["gemini-*".into()],
         zen_patterns: Vec::new(),
         glm_patterns: vec!["glm-*".into()],
      }
   }
}

impl ModelsConfig {
   /// Which backend serves this model. The most specific pattern wins, and a
   /// tie goes to whichever backend is listed first here.
   pub fn route(&self, model: &str) -> Provider {
      self.matched(model).unwrap_or(Provider::OpenAi)
   }

   /// None when nothing claimed the name and codex took it as the default.
   pub fn matched(&self, model: &str) -> Option<Provider> {
      let mut best = Option::<(usize, Provider)>::None;
      for (provider, patterns) in self.sets() {
         let Some(score) = patterns
            .iter()
            .filter_map(|pattern| pattern_specificity(pattern, model))
            .max()
         else {
            continue;
         };
         if best.is_none_or(|(seen, _)| score > seen) {
            best = Some((score, provider));
         }
      }
      best.map(|(_, provider)| provider)
   }

   const fn sets(&self) -> [(Provider, &Vec<String>); 4] {
      [
         (Provider::Anthropic, &self.anthropic_patterns),
         (Provider::Gemini, &self.gemini_patterns),
         (Provider::Zen, &self.zen_patterns),
         (Provider::Glm, &self.glm_patterns),
      ]
   }

   /// The name meant when a backend prefix was dropped, `fable-5-1` for
   /// `claude-fable-5-1`.
   pub fn suggest(&self, model: &str) -> Option<String> {
      self
         .sets()
         .into_iter()
         .flat_map(|(_, patterns)| patterns)
         .filter_map(|pattern| pattern.strip_suffix('*'))
         .filter(|prefix| !model.starts_with(*prefix))
         .find_map(|prefix| {
            let kept = (1..prefix.len())
               .rev()
               .filter(|pos| prefix.is_char_boundary(*pos))
               .find(|pos| {
                  prefix
                     .get(*pos..)
                     .is_some_and(|text| model.starts_with(text))
               })?;
            Some(format!("{}{model}", prefix.get(..kept).unwrap_or("")))
         })
   }
}

/// Matches `model` against a literal pattern or a `prefix*` glob, returning
/// the match specificity (prefix length; `usize::MAX` for an exact match).
pub fn pattern_specificity(pattern: &str, model: &str) -> Option<usize> {
   match pattern.strip_suffix('*') {
      Some(prefix) if model.starts_with(prefix) => Some(prefix.len()),
      None if model == pattern => Some(usize::MAX),
      Some(_) | None => None,
   }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelAlias {
   pub model: String,
   #[serde(default)]
   pub effort: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct FileConfig {
   bind: Option<String>,
   metrics_bind: Option<String>,
   db: Option<PathBuf>,
   codex: Option<CodexConfig>,
   anthropic: Option<AnthropicConfig>,
   gemini: Option<GeminiConfig>,
   zen: Option<ZenConfig>,
   glm: Option<GlmConfig>,
   pricing: Option<PricingConfig>,
   models: Option<ModelsConfig>,
}

impl Config {
   pub fn load(args: &Cli) -> Result<Self> {
      let config_path = args
         .config
         .clone()
         .or_else(|| env::var("SLOP_CONFIG").ok().map(PathBuf::from))
         .unwrap_or_else(|| xdg_dir("XDG_CONFIG_HOME", ".config").join("slop-proxy/config.toml"));

      let file = if config_path.exists() {
         let raw = fs::read_to_string(&config_path)
            .wrap_err_with(|| format!("reading {}", config_path.display()))?;
         toml::from_str::<FileConfig>(&raw)
            .wrap_err_with(|| format!("parsing {}", config_path.display()))?
      } else {
         FileConfig::default()
      };

      let db_path = args
         .db
         .clone()
         .or_else(|| env::var("SLOP_DB").ok().map(PathBuf::from))
         .or(file.db)
         .unwrap_or_else(|| xdg_dir("XDG_DATA_HOME", ".local/share").join("slop-proxy/slop.db"));

      let bind = env::var("SLOP_BIND")
         .ok()
         .or(file.bind)
         .unwrap_or_else(|| "[::1]:8484".into());

      Ok(Self {
         db_path,
         bind,
         metrics_bind: file.metrics_bind,
         codex: file.codex.unwrap_or_default(),
         anthropic: file.anthropic.unwrap_or_default(),
         gemini: file.gemini.unwrap_or_default(),
         zen: file.zen.unwrap_or_default(),
         glm: file.glm.unwrap_or_default(),
         pricing: file.pricing.unwrap_or_default(),
         models: file.models.unwrap_or_default(),
      })
   }
}

#[cfg(test)]
impl Config {
   pub fn for_tests() -> Self {
      Self {
         db_path: PathBuf::new(),
         bind: String::new(),
         metrics_bind: None,
         codex: CodexConfig::default(),
         anthropic: AnthropicConfig::default(),
         gemini: GeminiConfig::default(),
         zen: ZenConfig::default(),
         glm: GlmConfig::default(),
         pricing: PricingConfig::default(),
         models: ModelsConfig::default(),
      }
   }
}

fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
   env::var(var)
      .ok()
      .filter(|value| !value.is_empty())
      .map_or_else(
         || {
            let home = env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(fallback)
         },
         PathBuf::from,
      )
}

#[cfg(test)]
mod route_tests {
   use super::*;
   use crate::translate::model_map;

   fn cfg() -> ModelsConfig {
      ModelsConfig {
         anthropic_patterns: vec!["claude-*".into()],
         gemini_patterns: vec!["gemini-*".into()],
         ..ModelsConfig::default()
      }
   }

   /// Claude models must never reach codex, so an alias that names no
   /// backend has to be claimed explicitly rather than falling through.
   #[test]
   fn a_bare_alias_falls_through_until_it_is_claimed() {
      assert_eq!(cfg().route("fable"), Provider::OpenAi);
      let claimed = ModelsConfig {
         anthropic_patterns: vec!["claude-*".into(), "fable*".into()],
         ..cfg()
      };
      assert_eq!(claimed.route("fable"), Provider::Anthropic);
   }

   #[test]
   fn the_longer_prefix_wins_over_a_broader_one() {
      let cfg = ModelsConfig {
         anthropic_patterns: vec!["gemini-*".into()],
         gemini_patterns: vec!["gemini-3-*".into()],
         ..cfg()
      };
      assert_eq!(cfg.route("gemini-3-pro"), Provider::Gemini);
      assert_eq!(cfg.route("gemini-2-flash"), Provider::Anthropic);
   }

   fn with_zen(zen: &[&str]) -> ModelsConfig {
      ModelsConfig {
         zen_patterns: zen.iter().map(ToString::to_string).collect(),
         ..cfg()
      }
   }

   /// The suffix is how a caller asks for an effort level, so it reaches
   /// `route()` attached to the model and must not decide the backend.
   #[test]
   fn an_effort_suffix_does_not_change_the_backend() {
      let cfg = with_zen(&["muse-spark-1.3-contributor-free"]);
      let resolve = |model: &str| model_map::resolve(&cfg, model).model;
      assert_eq!(
         cfg.route(&resolve("muse-spark-1.3-contributor-free:high")),
         Provider::Zen
      );
      assert_eq!(
         cfg.route(&resolve("gemini-3.8-flash:low")),
         Provider::Gemini
      );
   }

   /// Zen resells the other vendors under their own names, so a bare
   /// `claude-*` there would silently move subscription traffic off the Max
   /// seats. The longer prefix has to win for that split to be expressible.
   #[test]
   fn a_zen_pattern_only_takes_what_it_is_more_specific_about() {
      let cfg = with_zen(&["claude-haiku-*"]);
      assert_eq!(cfg.route("claude-haiku-4-5"), Provider::Zen);
      assert_eq!(cfg.route("claude-opus-5"), Provider::Anthropic);
      assert_eq!(
         with_zen(&["claude-*"]).route("claude-opus-5"),
         Provider::Anthropic
      );
   }
}

#[cfg(test)]
mod zen_tests {
   use super::*;

   #[test]
   fn proxy_urls_merge_inline_and_file_entries() {
      let path = env::temp_dir().join(format!("slop-proxies-{}", uuid::Uuid::new_v4()));
      fs::write(
         &path,
         " http://file-one.example:80\n\n# ignored\nhttp://file-two.example:80\n",
      )
      .unwrap();
      let config = ZenConfig {
         proxy_urls: vec!["http://inline.example:80".into()],
         proxy_urls_file: Some(path.clone()),
         ..ZenConfig::default()
      };

      assert_eq!(
         config.proxy_urls().unwrap(),
         [
            "http://inline.example:80",
            "http://file-one.example:80",
            "http://file-two.example:80",
         ]
      );
      fs::remove_file(path).unwrap();
   }
}

#[cfg(test)]
mod suggest_tests {
   use super::*;

   #[test]
   fn only_a_dropped_prefix_earns_a_suggestion() {
      let cfg = ModelsConfig {
         anthropic_patterns: vec!["claude-opus-*".into(), "claude-fable-*".into()],
         ..ModelsConfig::default()
      };
      assert_eq!(
         cfg.suggest("fable-5-1").as_deref(),
         Some("claude-fable-5-1")
      );
      assert_eq!(cfg.suggest("gpt-5.6-sol"), None);
   }
}
