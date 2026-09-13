use std::collections::BTreeSet;

use axum::body::Bytes;
use reqwest::header::HeaderMap;

use super::anthropic::AnthropicPool;
use super::codex::CodexPool;
use super::deepseek::DeepSeekPool;
use super::experiential::ExperientialPool;
use super::gemini::{Call, GeminiPool};
use super::glm::GlmPool;
use super::zen::{Relay as ZenRelay, ZenPool, satisfy_chat_tool_gate, satisfy_tool_gate};
use super::{AccountSnapshot, Backend, PoolError, Route, Served};
use crate::anthropic::client::AnthropicClient;
use crate::codex::client::CodexClient;
use crate::codex::sse;
use crate::codex::sse::EventStream;
use crate::codex::types::ResponsesRequest;
use crate::config::{Config, ModelsConfig, ZenDialect};
use crate::db::Db;
use crate::deepseek::client::DeepSeekClient;
use crate::experiential::client::ExperientialClient;
use crate::gemini::client::GeminiClient;
use crate::glm::client::GlmClient;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::bridge;
use crate::translate::bridge::BridgeProtocol;
use crate::translate::chat_req::{custom_tools, to_chat};
use crate::zen::client::{ZenClient, egress_of};

/// A backend's reply to a Responses request, before anything reads it.
pub enum Upstream {
   /// Responses SSE from codex or zen, relayable byte for byte.
   Responses(reqwest::Response),
   /// Gemini or a chat-only zen model, so the frames are chat completions
   /// or Google's own and need bridging back.
   Bridged {
      response: reqwest::Response,
      protocol: BridgeProtocol,
      custom: BTreeSet<String>,
   },
}

impl Upstream {
   /// The reply as Responses events, whichever dialect it arrived in.
   pub fn events(self, model: &str, capture: UsageCapture) -> EventStream {
      match self {
         Self::Responses(response) => {
            if let Some(index) = egress_of(&response) {
               capture.note_egress(index);
            }
            sse::event_stream(response)
         },
         Self::Bridged {
            response,
            protocol,
            custom,
         } => {
            if let Some(index) = egress_of(&response) {
               capture.note_egress(index);
            }
            bridge::event_stream(response, protocol, model, custom, capture)
         },
      }
   }
}

pub struct Dispatched {
   pub account_id: Option<i64>,
   pub upstream: Upstream,
   pub attempts: u32,
}

pub struct Pools {
   pub codex: CodexPool,
   pub anthropic: AnthropicPool,
   pub gemini: GeminiPool,
   pub zen: ZenPool,
   pub glm: GlmPool,
   pub deepseek: DeepSeekPool,
   pub experiential: ExperientialPool,
}

impl Pools {
   pub async fn load(db: &Db, cfg: &Config) -> eyre::Result<Self> {
      let codex = CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone())).await?;
      let anthropic =
         AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone())).await?;
      let gemini = GeminiPool::load(db.clone(), GeminiClient::new(cfg.gemini.clone())?).await?;
      let zen = ZenPool::load(db.clone(), ZenClient::new(cfg.zen.clone())?).await?;
      let glm = GlmPool::load(db.clone(), GlmClient::new(cfg.glm.clone())?).await?;
      let deepseek =
         DeepSeekPool::load(db.clone(), DeepSeekClient::new(cfg.deepseek.clone())?).await?;
      let experiential = ExperientialPool::load(
         db.clone(),
         ExperientialClient::new(cfg.experiential.clone())?,
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
      announce("deepseek", deepseek.len().await, None);
      announce("experiential", experiential.len().await, None);
      Ok(Self {
         codex,
         anthropic,
         gemini,
         zen,
         glm,
         deepseek,
         experiential,
      })
   }

   pub async fn reload(&self) {
      let (codex, anthropic, gemini, zen, glm, deepseek, experiential) = tokio::join!(
         self.codex.reload(),
         self.anthropic.reload(),
         self.gemini.reload(),
         self.zen.reload(),
         self.glm.reload(),
         self.deepseek.reload(),
         self.experiential.reload()
      );
      for (provider, result) in [
         (Provider::OpenAi, codex),
         (Provider::Anthropic, anthropic),
         (Provider::Gemini, gemini),
         (Provider::Zen, zen),
         (Provider::Glm, glm),
         (Provider::DeepSeek, deepseek),
         (Provider::Experiential, experiential),
      ] {
         if let Err(err) = result {
            tracing::warn!("reloading {provider} accounts: {err}");
         }
      }
   }

   pub async fn poll_usage(&self) {
      self.codex.poll_usage().await;
      self.anthropic.poll_usage().await;
   }

   /// One Responses request to whichever backend serves the model. Codex and
   /// zen's responses models take the body as sent, the rest are bridged.
   pub async fn responses(
      &self,
      models: &ModelsConfig,
      provider: Provider,
      route: Route<'_>,
      req: &ResponsesRequest,
   ) -> Result<Dispatched, PoolError> {
      let body = serde_json::to_vec(req)
         .map(Bytes::from)
         .map_err(|err| PoolError::Upstream(format!("serializing request: {err}")))?;
      self
         .responses_raw(models, provider, route, body, Some(req), &HeaderMap::new())
         .await
   }

   /// A caller already speaking Responses is forwarded byte for byte, since
   /// the typed request drops fields it has no opinion on (a custom tool's
   /// grammar, an item type it does not know). `typed` is the read-only view
   /// the bridge needs, absent when the body did not type.
   pub async fn responses_raw(
      &self,
      models: &ModelsConfig,
      provider: Provider,
      route: Route<'_>,
      body: Bytes,
      typed: Option<&ResponsesRequest>,
      headers: &HeaderMap,
   ) -> Result<Dispatched, PoolError> {
      let raw = |served: Served<reqwest::Response>| Dispatched {
         account_id: served.account_id,
         upstream: Upstream::Responses(served.response),
         attempts: served.attempts,
      };
      let unbridgeable = |backend: &str| PoolError::BadRequest {
         provider,
         model: route.model.to_owned(),
         body: format!(
            "this request cannot be bridged to {backend}; see the proxy log for the field that failed"
         ),
      };
      match provider {
         Provider::OpenAi => self.codex.post(route, body, headers.clone()).await.map(raw),
         Provider::Zen if models.zen_dialect(route.model) == ZenDialect::Chat => {
            let Some(req) = typed else {
               return Err(unbridgeable("zen"));
            };
            let custom = custom_tools(req);
            let mut chat = to_chat(req);
            satisfy_chat_tool_gate(&mut chat);
            let bridged = serde_json::to_vec(&chat)
               .map(Bytes::from)
               .map_err(|err| PoolError::Upstream(format!("serializing request: {err}")))?;
            let served = self
               .zen
               .execute(
                  route,
                  ZenRelay {
                     path: "/chat/completions",
                     body: bridged,
                  },
               )
               .await?;
            Ok(Dispatched {
               account_id: served.account_id,
               attempts: served.attempts,
               upstream: Upstream::Bridged {
                  response: served.response,
                  protocol: BridgeProtocol::Chat,
                  custom,
               },
            })
         },
         Provider::Zen => self
            .zen
            .execute(
               route,
               ZenRelay {
                  path: "/responses",
                  body: satisfy_tool_gate(&body).unwrap_or(body),
               },
            )
            .await
            .map(raw),
         Provider::Gemini => {
            let Some(req) = typed else {
               return Err(unbridgeable("gemini"));
            };
            let custom = custom_tools(req);
            let chat = to_chat(req);
            let served = self
               .gemini
               .execute(route, Call::OpenAi(Box::new(chat)))
               .await?;
            // Google answers a malformed request with a 400 body and no
            // frames, which read as an empty stream and billed as a
            // client disconnect.
            if !served.response.response.status().is_success() {
               let error_body = served.response.response.text().await.unwrap_or_default();
               return Err(PoolError::BadRequest {
                  provider,
                  model: route.model.to_owned(),
                  body: <GeminiClient as Backend>::reason(error_body),
               });
            }
            Ok(Dispatched {
               account_id: served.account_id,
               attempts: served.attempts,
               upstream: Upstream::Bridged {
                  response: served.response.response,
                  protocol: served.response.protocol,
                  custom,
               },
            })
         },
         Provider::Anthropic | Provider::Glm | Provider::DeepSeek | Provider::Experiential => {
            Err(PoolError::BadRequest {
               provider,
               model: route.model.to_owned(),
               body: "not served over the responses api".into(),
            })
         },
      }
   }

   pub async fn snapshots(&self) -> Vec<AccountSnapshot> {
      let mut out = self.codex.snapshot().await;
      out.extend(self.anthropic.snapshot().await);
      out.extend(self.gemini.snapshot().await);
      out.extend(self.zen.snapshot().await);
      out.extend(self.glm.snapshot().await);
      out.extend(self.deepseek.snapshot().await);
      out.extend(self.experiential.snapshot().await);
      out
   }
}

fn announce(name: &str, count: usize, login: Option<&str>) {
   match (count, login) {
      (0, Some(login)) => tracing::warn!("no {name} accounts in the database; run `{login}`"),
      (0, None) => {},
      (count, _) => tracing::info!("loaded {count} {name} account(s)"),
   }
}
