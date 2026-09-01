use std::collections::BTreeMap;
use std::path::PathBuf;

use eyre::{Result, WrapErr};
use serde::Deserialize;

use crate::cli::Cli;

pub const DEFAULT_INSTRUCTIONS: &str = "You are Codex, based on GPT-5. You are running as a coding agent on a user's computer. Answer the user's requests directly and concisely.";

#[derive(Debug, Clone)]
pub struct Config {
    pub db_path: PathBuf,
    pub bind: String,
    /// Extra unauthenticated listener serving GET /metrics when set.
    pub metrics_bind: Option<String>,
    pub codex: CodexConfig,
    pub anthropic: AnthropicConfig,
    pub models: ModelsConfig,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
        if let Some(path) = &self.instructions_file
            && let Ok(s) = std::fs::read_to_string(path)
        {
            return s;
        }
        self.instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.into())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub default: String,
    pub default_effort: Option<String>,
    pub aliases: BTreeMap<String, ModelAlias>,
    pub known: Vec<String>,
    /// Model patterns relayed verbatim to the Anthropic backend instead of
    /// being translated for the codex one.
    pub anthropic_patterns: Vec<String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            default: String::new(),
            default_effort: None,
            aliases: BTreeMap::new(),
            known: Vec::new(),
            anthropic_patterns: vec!["claude-*".into()],
        }
    }
}

impl ModelsConfig {
    /// True when the requested model is relayed verbatim to the Anthropic
    /// backend instead of being translated for the codex one.
    pub fn routes_to_anthropic(&self, model: &str) -> bool {
        self.anthropic_patterns
            .iter()
            .any(|p| pattern_specificity(p, model).is_some())
    }
}

/// Matches `model` against a literal pattern or a `prefix*` glob, returning
/// the match specificity (prefix length; `usize::MAX` for an exact match).
pub fn pattern_specificity(pattern: &str, model: &str) -> Option<usize> {
    match pattern.strip_suffix('*') {
        Some(prefix) if model.starts_with(prefix) => Some(prefix.len()),
        Some(_) => None,
        None if model == pattern => Some(usize::MAX),
        None => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelAlias {
    pub model: String,
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    bind: Option<String>,
    metrics_bind: Option<String>,
    db: Option<PathBuf>,
    codex: Option<CodexConfig>,
    anthropic: Option<AnthropicConfig>,
    models: Option<ModelsConfig>,
}

impl Config {
    pub fn load(args: &Cli) -> Result<Self> {
        let config_path = args
            .config
            .clone()
            .or_else(|| std::env::var("SLOP_CONFIG").ok().map(PathBuf::from))
            .unwrap_or_else(|| {
                xdg_dir("XDG_CONFIG_HOME", ".config").join("slop-proxy/config.toml")
            });

        let file = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path)
                .wrap_err_with(|| format!("reading {}", config_path.display()))?;
            toml::from_str::<FileConfig>(&raw)
                .wrap_err_with(|| format!("parsing {}", config_path.display()))?
        } else {
            FileConfig::default()
        };

        let db_path = args
            .db
            .clone()
            .or_else(|| std::env::var("SLOP_DB").ok().map(PathBuf::from))
            .or(file.db)
            .unwrap_or_else(|| xdg_dir("XDG_DATA_HOME", ".local/share").join("slop-proxy/slop.db"));

        let bind = std::env::var("SLOP_BIND")
            .ok()
            .or(file.bind)
            .unwrap_or_else(|| "[::1]:8484".into());

        Ok(Self {
            db_path,
            bind,
            metrics_bind: file.metrics_bind,
            codex: file.codex.unwrap_or_default(),
            anthropic: file.anthropic.unwrap_or_default(),
            models: file.models.unwrap_or_default(),
        })
    }
}

fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(fallback)
        })
}
