//! GitHub Copilot over its OpenAI-compatible chat-completions surface.
//!
//! The account's long-lived credential is the GitHub OAuth token from the
//! device flow. It is stored as the refresh token and exchanged for a
//! short-lived Copilot token (`copilot_internal/v2/token`) whenever the
//! slot needs one, mirroring how the other OAuth pools refresh.

pub mod client;
