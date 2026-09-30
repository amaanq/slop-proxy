use axum::body::Bytes;
use serde::Serialize;

use crate::gemini::sse::Frames;
use crate::provider::Provider;
use crate::server::error::Dialect;
use crate::server::pipeline::Scan;
use crate::translate::UsageCapture;
use crate::translate::chat::{
   ChatChunk, ChatEnvelope, ChatError, ChatErrorBody, ChatRequest, ErrorCode, FinishReason,
   StreamOptions,
};

pub const fn force_usage(body: &mut ChatRequest, streaming: bool) {
   // Without this the terminal chunk carries no usage and the request bills
   // as zero tokens.
   if streaming {
      body.stream_options = Some(StreamOptions {
         include_usage: true,
      });
   }
}

/// Pins a conversation to one account.
pub fn first_turn_key<T>(user: &str, first: Option<&T>) -> String
where
   T: Serialize,
{
   let mut hasher = hmac_sha256::Hash::new();
   hasher.update(user.as_bytes());
   if let Some(first) = first {
      hasher.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
   }
   data_encoding::HEXLOWER.encode(&hasher.finalize())
}

pub fn note_cutoff(frames: &Frames, cut: &mut bool, capture: &UsageCapture, provider: Provider) {
   if *cut {
      return;
   }
   let Some(error) = frames.cutoff() else {
      return;
   };
   *cut = true;
   let status = error.status.clone().unwrap_or_else(|| "cutoff".into());
   tracing::warn!(
       code = error.code.unwrap_or(0),
       status = %status,
       "{provider} gave up mid-stream after its 200: {}",
       error.message.as_deref().unwrap_or("")
   );
   capture.note_cutoff(&status);
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
      note_cutoff(&self.frames, &mut self.cut, &self.capture, self.provider);
   }
}

impl Scan for ChatUsageScan {
   const DIALECT: Dialect = Dialect::OpenAi;
   const REJECTED: &'static str = "upstream_rejected";

   fn chunk(&mut self, bytes: Bytes) -> Bytes {
      self.feed(&bytes);
      bytes
   }

   fn body(&mut self, bytes: Bytes) -> Result<Bytes, String> {
      if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
         && let Some(usage) = env.usage
      {
         self.capture.record(&usage.into());
      }
      Ok(bytes)
   }

   fn tail(&mut self) -> Bytes {
      let Some(error) = self.frames.cutoff() else {
         return Bytes::new();
      };
      let err = ChatError {
         error: ChatErrorBody {
            message: error.message.unwrap_or_default(),
            kind: Some("server_error".into()),
            code: error.status.map(ErrorCode::Text),
         },
      };
      Bytes::from(format!(
         "data: {}\n\ndata: [DONE]\n\n",
         serde_json::to_string(&err).unwrap_or_default()
      ))
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
