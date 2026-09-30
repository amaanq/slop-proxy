use std::time::Instant;

use axum::body::Bytes;
use axum::response::Response;

use crate::pool::copilot::Call;
use crate::provider::Provider;
use crate::server::auth::AuthInfo;
use crate::server::chat::{ChatUsageScan, first_turn_key, force_usage};
use crate::server::error::{Dialect, error_response};
use crate::server::facts::RequestFacts;
use crate::server::pipeline;
use crate::server::{AppState, log_rejected};
use crate::translate::chat::{ChatContent, ChatPart, ChatRequest};

const DIALECT: Dialect = Dialect::OpenAi;

pub async fn chat_completions(
   state: AppState,
   auth: AuthInfo,
   mut body: ChatRequest,
   model: String,
   facts: RequestFacts,
) -> Response {
   let started = Instant::now();
   let streaming = body.stream.unwrap_or(false);
   body.stream_options = None;
   force_usage(&mut body, streaming);

   let mut record = pipeline::record(
      &auth,
      "chat",
      Provider::Copilot,
      model.clone(),
      body.model.clone(),
      facts,
   );
   record.session_key = first_turn_key(&auth.user, body.messages.first());
   record.effort = body.reasoning_effort.clone().unwrap_or_default();

   let agent = body
      .messages
      .iter()
      .any(|msg| matches!(msg.role.as_str(), "assistant" | "tool"));
   let vision = body.messages.iter().any(|msg| {
      matches!(msg.content, Some(ChatContent::Parts(ref parts))
         if parts.iter().any(|part| matches!(*part, ChatPart::ImageUrl { .. })))
   });
   let encoded = match serde_json::to_vec(&body) {
      Ok(bytes) => Bytes::from(bytes),
      Err(err) => {
         log_rejected(&state, &auth, "chat", &model);
         return error_response(DIALECT, 400, "invalid_request_error", &err.to_string());
      },
   };
   let served = state
      .pools
      .copilot
      .execute(
         auth.route(&record.session_key, &record.upstream_model),
         Call {
            body: encoded,
            agent,
            vision,
         },
      )
      .await;
   pipeline::forward(
      state,
      record,
      served,
      Vec::new(),
      started,
      streaming,
      |capture| ChatUsageScan::new(capture, Provider::Copilot),
   )
   .await
}
