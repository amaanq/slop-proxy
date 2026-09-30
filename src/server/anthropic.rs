use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Event;
use axum::response::{IntoResponse as _, Response};
use axum::{Extension, Json};

use crate::codex::types::ResponsesEvent;
use crate::config::ZenDialect;
use crate::provider::Provider;
use crate::server::auth::AuthInfo;
use crate::server::error::{Dialect, body_at, translation_error};
use crate::server::facts::RequestFacts;
use crate::server::pipeline::{self, Reply};
use crate::server::relay;
use crate::server::{AppState, cache_key, log_rejected};
use crate::translate::anthropic_req::{self, AnthropicRequest};
use crate::translate::anthropic_stream::{AnthropicStream, render_aggregated};
use crate::translate::{Aggregated, UsageCapture, count_tokens};

const DIALECT: Dialect = Dialect::Anthropic;

pub async fn messages(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let started = Instant::now();
   let peek = relay::Peek::from_slice(&body, &state.cfg.models);
   // An effort suffix is part of what the caller typed, not part of the model
   // name a pattern matches, so routing the raw string sent muse:high to
   // codex and burned the pool on a model it cannot serve.
   let provider = match pipeline::admit(
      &state,
      &auth,
      DIALECT,
      "messages",
      &peek.model,
      &peek.upstream_model,
   ) {
      Ok(provider) => provider,
      Err(response) => return *response,
   };
   match provider {
      // Z.ai speaks this dialect, so the body it needs is the one that
      // arrived and the reply needs no translating back.
      Provider::Anthropic | Provider::Glm | Provider::DeepSeek | Provider::Experiential => {
         return relay::messages(state, auth, headers, body, peek, provider).await;
      },
      Provider::Zen
         if state.cfg.models.zen_dialect(&peek.upstream_model) == ZenDialect::Messages =>
      {
         return relay::messages(state, auth, headers, body, peek, provider).await;
      },
      Provider::Gemini | Provider::Zen | Provider::OpenAi | Provider::Copilot => {},
   }
   let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
      Ok(req) => req,
      Err(err) => {
         tracing::warn!(near = %body_at(&body, &err), "anthropic request did not parse");
         log_rejected(&state, &auth, "messages", &peek.model);
         return translation_error(DIALECT, &format!("invalid request: {err}"));
      },
   };
   let mut upstream = anthropic_req::to_responses(&req, &state.cfg, provider);
   upstream.prompt_cache_key = Some(cache_key(&auth.user, &upstream));
   let est_input = count_tokens::estimate(&upstream);

   let record = pipeline::record(
      &auth,
      "messages",
      provider,
      req.model.clone(),
      upstream.model.clone(),
      RequestFacts::from_anthropic(&req, &headers),
   );
   let emit_thinking = req.thinking_enabled();
   let model = req.model.clone();
   let stream = move |capture: UsageCapture| {
      let mut translator = AnthropicStream::new(model, est_input, emit_thinking, capture);
      move |event: Option<ResponsesEvent>| {
         let frames = match event {
            Some(event) => translator.handle(event),
            None => translator.finalize(),
         };
         frames
            .into_iter()
            .map(|(name, data)| Event::default().event(name).data(data))
            .collect()
      }
   };
   let reply = Reply {
      dialect: DIALECT,
      stream: req.stream.unwrap_or(false).then_some(stream),
      render: |agg: &Aggregated| render_aggregated(agg, &req.model, emit_thinking),
   };
   pipeline::serve_translated(state, &auth, record, provider, &upstream, started, reply).await
}

pub async fn count_tokens(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   #[derive(serde::Serialize)]
   struct TokenCount {
      input_tokens: i64,
   }

   let peek = relay::Peek::from_slice(&body, &state.cfg.models);
   let provider = match pipeline::admit(
      &state,
      &auth,
      DIALECT,
      "count_tokens",
      &peek.model,
      &peek.upstream_model,
   ) {
      Ok(provider) => provider,
      Err(response) => return *response,
   };
   match provider {
      Provider::Anthropic => {
         return relay::count_tokens(state, auth, headers, body, peek).await;
      },
      Provider::Gemini
      | Provider::Zen
      | Provider::Glm
      | Provider::DeepSeek
      | Provider::OpenAi
      | Provider::Copilot
      | Provider::Experiential => {},
   }
   let req = match serde_json::from_slice::<AnthropicRequest>(&body) {
      Ok(req) => req,
      Err(err) => return translation_error(DIALECT, &format!("invalid request: {err}")),
   };
   Json(TokenCount {
      input_tokens: count_tokens::estimate(&anthropic_req::to_responses(
         &req, &state.cfg, provider,
      )),
   })
   .into_response()
}
