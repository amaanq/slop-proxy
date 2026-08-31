use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::cli::Cli;

pub const DEFAULT_INSTRUCTIONS: &str = "You are Codex, based on GPT-5. You are running as a coding agent on a user's computer. Answer the user's requests directly and concisely.";

#[derive(Debug, Clone)]
pub struct Config {
    pub db_path: PathBuf,
    pub bind: String,
    pub codex: CodexConfig,
    pub models: ModelsConfig,
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
        }
    }
}

impl CodexConfig {
    pub fn instructions(&self) -> String {
        if let Some(path) = &self.instructions_file {
            if let Ok(s) = std::fs::read_to_string(path) {
                return s;
            }
        }
        self.instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.into())
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub default: String,
    pub default_effort: Option<String>,
    pub aliases: BTreeMap<String, ModelAlias>,
    pub known: Vec<String>,
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
    db: Option<PathBuf>,
    codex: Option<CodexConfig>,
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

        let file: FileConfig = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?;
            toml::from_str(&raw).with_context(|| format!("parsing {}", config_path.display()))?
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
            .unwrap_or_else(|| "127.0.0.1:8484".into());

        Ok(Self {
            db_path,
            bind,
            codex: file.codex.unwrap_or_default(),
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
