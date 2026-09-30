use std::time::Instant;

use axum::body::Bytes;
use axum::response::Response;

use super::AppState;
use super::auth::AuthInfo;
use super::error::{Dialect, error_response};
use super::gemini::{relay_chat_body, relay_chat_stream, session_key, upstream_rejected};
use super::pipeline::{dispatch_failed, read_body};
use super::relay::forwarded_response;
use crate::pool::Route;
use crate::pool::copilot::Call;
use crate::provider::Provider;
use crate::translate::chat::{ChatContent, ChatPart, ChatRequest, StreamOptions};

const DIALECT: Dialect = Dialect::OpenAi;

pub async fn chat_completions(
   state: AppState,
   auth: AuthInfo,
   mut body: ChatRequest,
   model: String,
   facts: super::facts::RequestFacts,
) -> Response {
   let started = Instant::now();
   let streaming = body.stream.unwrap_or(false);
   // Without this the terminal chunk carries no usage and the request bills
   // as zero tokens.
   body.stream_options = streaming.then_some(StreamOptions {
      include_usage: true,
   });

   let mut record = super::pipeline::record(
      &auth,
      "chat",
      Provider::Copilot,
      model.clone(),
      body.model.clone(),
      facts,
   );
   record.session_key = session_key(&auth.user, &body);
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
         super::log_rejected(&state, &auth, "chat", &model);
         return error_response(DIALECT, 400, "invalid_request_error", &err.to_string());
      },
   };
   let served = match state
      .pools
      .copilot
      .execute(
         Route {
            session_key: &record.session_key,
            model: &record.upstream_model,
            service_tier: None,
            user: &auth.user,
            pinned_account: auth.limits.pinned_account,
            prefer_trusted: false,
            reserved_only: auth.limits.reserved_only,
         },
         Call {
            body: encoded,
            agent,
            vision,
         },
      )
      .await
   {
      Ok(served) => served,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   let resp = served.response;
   record.account_id = served.account_id;
   record.attempts = i64::from(served.attempts);
   record.status = i64::from(resp.status().as_u16());

   let builder = forwarded_response(&resp);
   if !resp.status().is_success() {
      return upstream_rejected(&state, record, builder, resp, started, Provider::Copilot).await;
   }
   if streaming {
      return relay_chat_stream(state, record, builder, resp, started, Provider::Copilot);
   }
   match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => relay_chat_body(&state, record, builder, bytes, started),
      Err(resp) => resp,
   }
}
