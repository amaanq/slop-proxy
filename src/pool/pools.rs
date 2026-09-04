use std::collections::BTreeSet;

use axum::body::Bytes;

use super::anthropic::AnthropicPool;
use super::codex::CodexPool;
use super::gemini::{Call, GeminiPool};
use super::glm::GlmPool;
use super::zen::ZenPool;
use super::{AccountSnapshot, Backend, PoolError, Route};
use crate::codex::sse::EventStream;
use crate::codex::types::ResponsesRequest;
use crate::db::Db;
use crate::gemini::client::{GeminiClient, GeminiProtocol};
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::gemini_bridge;

/// A backend's reply to a Responses request, before anything reads it.
pub enum Upstream {
    /// Responses SSE from codex or zen, relayable byte for byte.
    Responses(reqwest::Response),
    /// Gemini reached through the chat bridge, so the frames are chat
    /// completions or Google's own and need bridging back.
    Bridged {
        response: reqwest::Response,
        protocol: GeminiProtocol,
        custom: BTreeSet<String>,
    },
}

impl Upstream {
    /// The reply as Responses events, whichever dialect it arrived in.
    pub fn events(self, model: &str, capture: UsageCapture) -> EventStream {
        match self {
            Self::Responses(response) => crate::codex::sse::event_stream(response),
            Self::Bridged {
                response,
                protocol,
                custom,
            } => gemini_bridge::event_stream(response, protocol, model, custom, capture),
        }
    }
}

pub struct Dispatched {
    pub account_id: Option<i64>,
    pub upstream: Upstream,
}

pub struct Pools {
    pub codex: CodexPool,
    pub anthropic: AnthropicPool,
    pub gemini: GeminiPool,
    pub zen: ZenPool,
    pub glm: GlmPool,
}

impl Pools {
    pub async fn load(db: &Db, cfg: &crate::config::Config) -> eyre::Result<Self> {
        let codex = CodexPool::load(
            db.clone(),
            crate::codex::client::CodexClient::new(cfg.codex.clone()),
        )
        .await?;
        let anthropic = AnthropicPool::load(
            db.clone(),
            crate::anthropic::client::AnthropicClient::new(cfg.anthropic.clone()),
        )
        .await?;
        let gemini = GeminiPool::load(
            db.clone(),
            crate::gemini::client::GeminiClient::new(cfg.gemini.clone()),
        )
        .await?;
        let zen = ZenPool::load(
            db.clone(),
            crate::zen::client::ZenClient::new(cfg.zen.clone())?,
        )
        .await?;
        let glm = GlmPool::load(
            db.clone(),
            crate::glm::client::GlmClient::new(cfg.glm.clone()),
        )
        .await?;
        announce("codex", codex.len().await, Some("slop-proxy login"));
        announce(
            "anthropic",
            anthropic.len().await,
            Some("slop-proxy login --provider anthropic"),
        );
        announce("gemini", gemini.len().await, None);
        announce("zen", zen.len().await, None);
        announce("glm", glm.len().await, None);
        Ok(Self {
            codex,
            anthropic,
            gemini,
            zen,
            glm,
        })
    }

    pub async fn reload(&self) {
        if let Err(e) = self.codex.reload().await {
            tracing::warn!("reloading {} accounts: {e}", Provider::OpenAi);
        }
        if let Err(e) = self.anthropic.reload().await {
            tracing::warn!("reloading {} accounts: {e}", Provider::Anthropic);
        }
        if let Err(e) = self.gemini.reload().await {
            tracing::warn!("reloading {} accounts: {e}", Provider::Gemini);
        }
        if let Err(e) = self.zen.reload().await {
            tracing::warn!("reloading {} accounts: {e}", Provider::Zen);
        }
        if let Err(e) = self.glm.reload().await {
            tracing::warn!("reloading {} accounts: {e}", Provider::Glm);
        }
    }

    pub async fn poll_usage(&self) {
        self.codex.poll_usage().await;
        self.anthropic.poll_usage().await;
    }

    /// One Responses request to whichever backend serves the model. Codex and
    /// zen take the body as sent, gemini takes it through the chat bridge.
    pub async fn responses(
        &self,
        provider: Provider,
        route: Route<'_>,
        req: &ResponsesRequest,
    ) -> Result<Dispatched, PoolError> {
        let body = serde_json::to_vec(req)
            .map(Bytes::from)
            .map_err(|e| PoolError::Upstream(format!("serializing request: {e}")))?;
        self.responses_raw(provider, route, body, Some(req)).await
    }

    /// A caller already speaking Responses is forwarded byte for byte, since
    /// the typed request drops fields it has no opinion on (a custom tool's
    /// grammar, an item type it does not know). `typed` is the read-only view
    /// the bridge needs, absent when the body did not type.
    pub async fn responses_raw(
        &self,
        provider: Provider,
        route: Route<'_>,
        body: Bytes,
        typed: Option<&ResponsesRequest>,
    ) -> Result<Dispatched, PoolError> {
        let raw = |(account_id, response)| Dispatched {
            account_id,
            upstream: Upstream::Responses(response),
        };
        match provider {
            Provider::OpenAi => self.codex.execute(route, body).await.map(raw),
            Provider::Zen => self.zen.execute(route, body).await.map(raw),
            Provider::Gemini => {
                let Some(req) = typed else {
                    return Err(PoolError::BadRequest {
                        provider,
                        model: route.model.to_owned(),
                        body: "this request cannot be bridged to gemini; see the proxy log for the field that failed".into(),
                    });
                };
                let custom = gemini_bridge::custom_tools(req);
                let chat = gemini_bridge::to_chat(req);
                let (account_id, reply) = self
                    .gemini
                    .execute(route, Call::OpenAi(Box::new(chat)))
                    .await?;
                // Google answers a malformed request with a 400 body and no
                // frames, which read as an empty stream and billed as a
                // client disconnect.
                if !reply.response.status().is_success() {
                    let error_body = reply.response.text().await.unwrap_or_default();
                    return Err(PoolError::BadRequest {
                        provider,
                        model: route.model.to_owned(),
                        body: <GeminiClient as Backend>::reason(error_body),
                    });
                }
                Ok(Dispatched {
                    account_id,
                    upstream: Upstream::Bridged {
                        response: reply.response,
                        protocol: reply.protocol,
                        custom,
                    },
                })
            }
            Provider::Anthropic | Provider::Glm => Err(PoolError::BadRequest {
                provider,
                model: route.model.to_owned(),
                body: "not served over the responses api".into(),
            }),
        }
    }

    pub async fn snapshots(&self) -> Vec<AccountSnapshot> {
        let mut out = self.codex.snapshot().await;
        out.extend(self.anthropic.snapshot().await);
        out.extend(self.gemini.snapshot().await);
        out.extend(self.zen.snapshot().await);
        out.extend(self.glm.snapshot().await);
        out
    }
}

fn announce(name: &str, count: usize, login: Option<&str>) {
    match (count, login) {
        (0, Some(login)) => tracing::warn!("no {name} accounts in the database; run `{login}`"),
        (0, None) => {}
        (n, _) => tracing::info!("loaded {n} {name} account(s)"),
    }
}
