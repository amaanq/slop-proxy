use axum::body::{Body, Bytes};
use axum::http::response::Builder;
use axum::response::Response;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::error::{Dialect, error_response};
use super::pipeline::{apply_snapshot, relayed};
use super::{AppState, LogGuard};
use crate::db::usage::UsageRecord;
use crate::gemini::sse::Frames;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::chat::{
   ChatChunk, ChatEnvelope, ChatError, ChatErrorBody, ChatRequest, ErrorCode, FinishReason,
   StreamOptions,
};

const DIALECT: Dialect = Dialect::OpenAi;

pub const fn force_usage(body: &mut ChatRequest, streaming: bool) {
   // Without this the terminal chunk carries no usage and the request bills
   // as zero tokens.
   if streaming {
      body.stream_options = Some(StreamOptions {
         include_usage: true,
      });
   }
}

/// Meters usage out of a finished chat completion and hands it back as is.
pub fn relay_chat_body(
   state: &AppState,
   mut record: UsageRecord,
   builder: Builder,
   bytes: Bytes,
   started: Instant,
) -> Response {
   if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
      && let Some(usage) = env.usage
   {
      let capture = UsageCapture::default();
      capture.record(&usage.into());
      apply_snapshot(&mut record, &capture.snapshot(), started);
   }
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   record.response_bytes = bytes.len() as i64;
   super::log_usage(state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}

/// Relays a chat-completions stream byte for byte and meters usage out of its
/// frames. Shared with the Copilot handler, which speaks the same dialect.
pub fn relay_chat_stream(
   state: AppState,
   record: UsageRecord,
   builder: Builder,
   resp: reqwest::Response,
   started: Instant,
   provider: Provider,
) -> Response {
   let capture = UsageCapture::default();
   let scan = Arc::new(Mutex::new(ChatUsageScan::new(capture.clone(), provider)));
   let each = {
      let scan = Arc::clone(&scan);
      move |bytes: Bytes| {
         scan.lock().unwrap().feed(&bytes);
         bytes
      }
   };
   relayed(
      builder,
      resp,
      LogGuard::new(state, capture.clone(), record, started),
      capture,
      DIALECT,
      each,
      move || {
         let cutoff = scan.lock().unwrap().frames.cutoff();
         let frame = match cutoff {
            Some(error) => {
               let err = ChatError {
                  error: ChatErrorBody {
                     message: error.message.unwrap_or_default(),
                     kind: Some("server_error".into()),
                     code: error.status.map(ErrorCode::Text),
                  },
               };
               format!(
                  "data: {}\n\ndata: [DONE]\n\n",
                  serde_json::to_string(&err).unwrap_or_default()
               )
            },
            None => String::new(),
         };
         Bytes::from(frame)
      },
   )
}

/// A non-2xx carries no SSE frames, so the usage scanner would log a phantom
/// `client_disconnect` and drop the body.
pub async fn upstream_rejected(
   state: &AppState,
   mut record: UsageRecord,
   builder: Builder,
   resp: reqwest::Response,
   started: Instant,
   provider: Provider,
) -> Response {
   let bytes = resp.bytes().await.unwrap_or_default();
   tracing::warn!(
       user = %record.user,
       model = %record.requested_model,
       dialect = record.dialect,
       status = record.status,
       body = %String::from_utf8_lossy(&bytes).chars().take(2000).collect::<String>(),
       "{provider} rejected the request"
   );
   record.error_kind = Some("upstream_rejected".into());
   record.response_bytes = bytes.len() as i64;
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   super::log_usage(state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}

/// Pins a conversation to one account.
pub fn session_key(user: &str, body: &ChatRequest) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   hasher.update(user.as_bytes());
   if let Some(first) = body.messages.first() {
      hasher.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
   }
   data_encoding::HEXLOWER.encode(&hasher.finalize())
}

/// Reads usage out of the `data:` frames of a chat stream. Only the terminal
/// frame carries it, so every frame is tried and the last one wins.
pub struct ChatUsageScan {
   capture: UsageCapture,
   frames: Frames,
   cut: bool,
   provider: Provider,
}

impl ChatUsageScan {
   pub fn new(capture: UsageCapture, provider: Provider) -> Self {
      Self {
         capture,
         frames: Frames::default(),
         cut: false,
         provider,
      }
   }

   pub fn feed(&mut self, bytes: &[u8]) {
      for data in self.frames.feed(bytes) {
         if data == b"[DONE]" {
            continue;
         }
         if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&data)
            && let Some(usage) = env.usage
         {
            self.capture.record(&usage.into());
         }
         if let Ok(chunk) = serde_json::from_slice::<ChatChunk>(&data)
            && let Some(reason) = chunk.choices.iter().find_map(|choice| choice.finish_reason)
         {
            let reason = match reason {
               FinishReason::Stop => "stop",
               FinishReason::Length => "length",
               FinishReason::ToolCalls => "tool_calls",
               FinishReason::ContentFilter => "content_filter",
               FinishReason::Other => "other",
            };
            self.capture.note_stop_reason(reason);
         }
         if !self.cut
            && let Ok(error) = serde_json::from_slice::<ChatError>(&data)
            && !error.error.message.is_empty()
         {
            self.cut = true;
            let code = match error.error.code.as_ref() {
               Some(&ErrorCode::Text(ref text)) => text.clone(),
               _ => "error".to_owned(),
            };
            tracing::warn!(
                status = %code,
                "{} gave up mid-stream after its 200: {}",
                self.provider,
                error.error.message
            );
            self.capture.note_cutoff(&code);
         }
      }
      if !self.cut
         && let Some(error) = self.frames.cutoff()
      {
         self.cut = true;
         let status = error.status.clone().unwrap_or_else(|| "cutoff".into());
         tracing::warn!(
             code = error.code.unwrap_or(0),
             status = %status,
             "{} gave up mid-stream after its 200: {}",
             self.provider,
             error.message.as_deref().unwrap_or("")
         );
         self.capture.note_cutoff(&status);
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn the_terminal_frame_supplies_usage() {
      let capture = UsageCapture::default();
      let mut scan = ChatUsageScan::new(capture.clone(), Provider::Gemini);
      scan.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
      scan.feed(
         b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":7,\
              \"total_tokens\":107,\"prompt_tokens_details\":{\"cached_tokens\":40}}}\n\n",
      );
      scan.feed(b"data: [DONE]\n\n");
      let snap = capture.snapshot();
      // Cached tokens come out of prompt_tokens so the two bill separately.
      assert_eq!(snap.input_tokens, 60);
      assert_eq!(snap.cache_read_tokens, 40);
      assert_eq!(snap.output_tokens, 7);
   }

   #[test]
   fn thinking_left_out_of_completion_tokens_is_recovered() {
      let capture = UsageCapture::default();
      let mut scan = ChatUsageScan::new(capture.clone(), Provider::Gemini);
      scan.feed(
         b"data: {\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":10,\
              \"total_tokens\":309}}\n",
      );
      let snap = capture.snapshot();
      assert_eq!(snap.input_tokens, 13);
      assert_eq!(snap.output_tokens, 296);
      assert_eq!(snap.reasoning_tokens, 286);
   }

   #[test]
   fn a_frame_split_across_chunks_still_parses() {
      let capture = UsageCapture::default();
      let mut scan = ChatUsageScan::new(capture.clone(), Provider::Gemini);
      scan.feed(b"data: {\"usage\":{\"prompt_tokens\":10,");
      scan.feed(b"\"completion_tokens\":2,\"total_tokens\":12}}\n");
      let snap = capture.snapshot();
      assert_eq!(snap.input_tokens, 10);
      assert_eq!(snap.output_tokens, 2);
   }
}
