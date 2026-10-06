use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use axum::body::Bytes;
use futures_util::future::join_all;
use reqwest::header::HeaderMap;
use serde::Serialize;

use crate::anthropic::{AnthropicClient, Model};
use crate::codex::client::CodexClient;
use crate::codex::sse;
use crate::codex::sse::EventStream;
use crate::codex::types::ResponsesRequest;
use crate::config::{Config, ModelsConfig, ZenDialect};
use crate::copilot::CopilotClient;
use crate::db::Db;
use crate::deepseek::DeepSeekClient;
use crate::egress::egress_of;
use crate::experiential::ExperientialClient;
use crate::gemini::client::GeminiClient;
use crate::gemini::types::ListedModel;
use crate::glm::GlmClient;
use crate::pool::anthropic::AnthropicPool;
use crate::pool::codex::CodexPool;
use crate::pool::copilot::CopilotPool;
use crate::pool::deepseek::DeepSeekPool;
use crate::pool::experiential::ExperientialPool;
use crate::pool::gemini::{Call, GeminiPool};
use crate::pool::glm::GlmPool;
use crate::pool::zen::{ZenPool, satisfy_chat_tool_gate, satisfy_tool_gate};
use crate::pool::{AccountSnapshot, Backend, PoolError, Relay, Route, Served, Slots};
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::bridge;
use crate::translate::bridge::BridgeProtocol;
use crate::translate::chat_req::{custom_tools, to_chat};
use crate::zen::{ZenClient, ZenModel};

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
   pub copilot: CopilotPool,
   catalogs: RwLock<Arc<Catalogs>>,
}

/// The upstream catalogs `/models` replies from. Each upstream takes up to
/// seconds to list, so a caller reads the last refresh instead of waiting
/// on all of them.
#[derive(Default)]
pub struct Catalogs {
   pub anthropic: Arc<[Model]>,
   pub gemini: Arc<[ListedModel]>,
   pub zen: Arc<[ZenModel]>,
   pub glm: Arc<[Model]>,
   pub deepseek: Arc<[String]>,
   pub copilot: Arc<[String]>,
}

impl Pools {
   pub async fn load(db: &Db, cfg: &Config) -> eyre::Result<Self> {
      let codex = CodexPool::load(db.clone(), CodexClient::new(cfg.codex.clone())).await?;
      let anthropic =
         AnthropicPool::load(db.clone(), AnthropicClient::new(cfg.anthropic.clone())?).await?;
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
      let copilot = CopilotPool::load(db.clone(), CopilotClient::new(cfg.copilot.clone())).await?;
      let pools = Self {
         codex,
         anthropic,
         gemini,
         zen,
         glm,
         deepseek,
         experiential,
         copilot,
         catalogs: RwLock::default(),
      };
      for slots in pools.slots() {
         announce(slots.provider(), slots.len().await);
      }
      Ok(pools)
   }

   const fn slots(&self) -> [&Slots; 8] {
      [
         &self.codex.slots,
         &self.anthropic.slots,
         &self.gemini.slots,
         &self.zen.slots,
         &self.glm.slots,
         &self.deepseek.slots,
         &self.experiential.slots,
         &self.copilot.slots,
      ]
   }

   pub async fn reload(&self) {
      let results = join_all(self.slots().map(Slots::reload)).await;
      for (slots, result) in self.slots().into_iter().zip(results) {
         if let Err(err) = result {
            tracing::warn!("reloading {} accounts: {err}", slots.provider());
         }
      }
   }

   pub fn refresh_egresses(&self, cfg: &Config) {
      let egresses = [
         (&cfg.anthropic.egress, self.anthropic.backend.proxied()),
         (&cfg.gemini.egress, Some(self.gemini.backend.egresses())),
         (&cfg.zen.egress, Some(self.zen.backend.egresses())),
         (&cfg.glm.egress, Some(self.glm.backend.egresses())),
         (&cfg.deepseek.egress, Some(self.deepseek.backend.egresses())),
         (
            &cfg.experiential.egress,
            Some(self.experiential.backend.egresses()),
         ),
      ];
      for (egress_cfg, target) in egresses {
         let Some(target) = target.filter(|_| egress_cfg.proxy_urls_file.is_some()) else {
            continue;
         };
         match egress_cfg.urls() {
            Ok(urls) if urls.is_empty() => {
               tracing::warn!("proxy list is empty, keeping the last one");
            },
            Ok(urls) => {
               if let Err(error) = target.replace(&urls) {
                  tracing::warn!("{error:#}");
               }
            },
            Err(error) => tracing::warn!("{error:#}"),
         }
      }
   }

   pub fn catalogs(&self) -> Arc<Catalogs> {
      Arc::clone(&self.catalogs.read().expect("catalogs lock poisoned"))
   }

   /// An upstream that fails to list keeps its previous catalog, so one
   /// flaky provider does not vanish from every client's model picker.
   pub async fn refresh_catalogs(&self) {
      fn keep<T>(fresh: Option<Vec<T>>, old: &Arc<[T]>) -> Arc<[T]> {
         fresh.map_or_else(|| Arc::clone(old), Arc::from)
      }

      let (anthropic, gemini, zen, glm, deepseek, copilot) = tokio::join!(
         self.anthropic.catalog(),
         self.gemini.models(),
         self.zen.models(),
         self.glm.models(),
         self.deepseek.models(),
         self.copilot.models(),
      );
      let previous = self.catalogs();
      let next = Catalogs {
         anthropic: keep(anthropic, &previous.anthropic),
         gemini: keep(gemini, &previous.gemini),
         zen: keep(zen, &previous.zen),
         glm: keep(glm, &previous.glm),
         deepseek: keep(deepseek, &previous.deepseek),
         copilot: keep(copilot, &previous.copilot),
      };
      *self.catalogs.write().expect("catalogs lock poisoned") = Arc::new(next);
   }

   pub async fn poll_usage(&self) {
      self.codex.poll_usage().await;
      self.anthropic.poll_usage().await;
      self.copilot.poll_usage().await;
      self.glm.poll_usage().await;
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
      let body = to_bytes(req)?;
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
      let chatted = |backend: &str| {
         let req = typed.ok_or_else(|| unbridgeable(backend))?;
         Ok::<_, PoolError>((custom_tools(req), to_chat(req)))
      };
      let bridged = |account_id, attempts, response, protocol, custom| Dispatched {
         account_id,
         attempts,
         upstream: Upstream::Bridged {
            response,
            protocol,
            custom,
         },
      };
      match provider {
         Provider::OpenAi => self.codex.post(route, body, headers.clone()).await.map(raw),
         Provider::Zen if models.zen_dialect(route.model) == ZenDialect::Chat => {
            let (custom, mut chat) = chatted("zen")?;
            satisfy_chat_tool_gate(&mut chat);
            let served = self
               .zen
               .execute(
                  route,
                  Relay {
                     path: "/chat/completions",
                     body: to_bytes(&chat)?,
                  },
               )
               .await?;
            Ok(bridged(
               served.account_id,
               served.attempts,
               served.response,
               BridgeProtocol::Chat,
               custom,
            ))
         },
         Provider::Zen => self
            .zen
            .execute(
               route,
               Relay {
                  path: "/responses",
                  body: satisfy_tool_gate(&body).unwrap_or(body),
               },
            )
            .await
            .map(raw),
         Provider::Gemini => {
            let (custom, chat) = chatted("gemini")?;
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
            Ok(bridged(
               served.account_id,
               served.attempts,
               served.response.response,
               served.response.protocol,
               custom,
            ))
         },
         Provider::Anthropic
         | Provider::Glm
         | Provider::DeepSeek
         | Provider::Experiential
         | Provider::Copilot => Err(PoolError::BadRequest {
            provider,
            model: route.model.to_owned(),
            body: "not served over the responses api".into(),
         }),
      }
   }

   pub async fn snapshots(&self) -> Vec<AccountSnapshot> {
      join_all(self.slots().map(Slots::snapshot))
         .await
         .into_iter()
         .flatten()
         .collect()
   }
}

fn to_bytes(value: &impl Serialize) -> Result<Bytes, PoolError> {
   serde_json::to_vec(value)
      .map(Bytes::from)
      .map_err(|err| PoolError::Upstream(format!("serializing request: {err}")))
}

fn announce(provider: Provider, count: usize) {
   let login = match provider {
      Provider::OpenAi => Some("slop-proxy login"),
      Provider::Anthropic => Some("slop-proxy login --provider anthropic"),
      Provider::Copilot => Some("slop-proxy login --provider copilot"),
      Provider::Gemini
      | Provider::Zen
      | Provider::Glm
      | Provider::DeepSeek
      | Provider::Experiential => None,
   };
   match (count, login) {
      (0, Some(login)) => tracing::warn!("no {provider} accounts in the database; run `{login}`"),
      (0, None) => {},
      (count, _) => tracing::info!("loaded {count} {provider} account(s)"),
   }
}
