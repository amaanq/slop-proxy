//! GitHub Copilot over its OpenAI-compatible chat-completions surface.
//!
//! The body is relayed rather than translated and only usage is read back
//! out, the same shape as the Gemini chat path.

use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::response::Response;

use super::AppState;
use super::auth::AuthInfo;
use super::error::{Dialect, error_response};
use super::gemini::{relay_chat_stream, session_key, upstream_rejected};
use super::pipeline::{apply_snapshot, dispatch_failed, read_body};
use super::relay::forwarded_response;
use crate::pool::Route;
use crate::pool::copilot::Call;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::chat::{
   ChatCompletion, ChatEnvelope, ChatRequest, FinishReason, StreamOptions,
};

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
   if streaming {
      body.stream_options = Some(StreamOptions {
         include_usage: true,
      });
   } else {
      body.stream = Some(false);
      body.stream_options = None;
   }

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

   let session_key = record.session_key.clone();
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
            session_key: &session_key,
            model: &record.upstream_model,
            service_tier: None,
            user: &auth.user,
            pinned_account: auth.limits.pinned_account,
            prefer_trusted: false,
            reserved_only: auth.limits.reserved_only,
         },
         Call { body: encoded },
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

   let bytes = match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   let capture = UsageCapture::default();
   if let Ok(completion) = serde_json::from_slice::<ChatCompletion>(&bytes) {
      if let Some(usage) = completion.usage {
         capture.record(&usage.into());
      }
      if let Some(reason) = completion
         .choices
         .iter()
         .find_map(|choice| choice.finish_reason)
      {
         let reason = match reason {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other => "other",
         };
         capture.note_stop_reason(reason);
      }
   } else if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
      && let Some(usage) = env.usage
   {
      capture.record(&usage.into());
   }
   apply_snapshot(&mut record, &capture.snapshot(), started);
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   record.response_bytes = bytes.len() as i64;
   super::log_usage(&state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}
