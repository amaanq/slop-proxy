//! GitHub Copilot over its OpenAI-compatible chat-completions surface.
//!
//! The account's long-lived credential is the GitHub OAuth token from the
//! device flow. It is stored as the refresh token and exchanged for a
//! short-lived Copilot token (`copilot_internal/v2/token`) whenever the
//! slot needs one, mirroring how the other OAuth pools refresh.

pub mod client;

pub const API_VERSION: &str = "2025-04-01";
pub const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.7";
pub const EDITOR_VERSION: &str = "vscode/1.105.0";
pub const USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
