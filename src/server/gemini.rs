use axum::body::{Body, Bytes};
use axum::extract::{Path, RawQuery, State};
use axum::http::HeaderMap;
use axum::http::response::Builder;
use axum::response::Response;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::auth::AuthInfo;
use super::error::{Dialect, error_response};
use super::pipeline::{apply_snapshot, dispatch_failed, read_body, relayed};
use super::relay::forwarded_response;
use super::{AppState, LogGuard, log_error};
use crate::codex::types::Usage;
use crate::db::usage::UsageRecord;
use crate::gemini::client::GeminiProtocol;
use crate::gemini::native::{NativeStream, chat_usage, response};
use crate::gemini::signatures;
use crate::gemini::sse::Frames;
use crate::gemini::types::{GenerateContentRequest, GenerateContentResponse};
use crate::pool::Route;
use crate::pool::gemini::Call;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::chat::{
   ChatChunk, ChatEnvelope, ChatError, ChatErrorBody, ChatRequest, ErrorCode, StreamOptions,
};
use crate::translate::gemini_bridge;

const DIALECT: Dialect = Dialect::OpenAi;

/// Google's OpenAI-compatible surface speaks the dialect the caller already
/// sent, so the body is relayed rather than translated and only usage is read
/// back out.
pub async fn chat_completions(
   state: AppState,
   auth: AuthInfo,
   mut body: ChatRequest,
   model: String,
   facts: super::facts::RequestFacts,
) -> Response {
   let started = Instant::now();
   let streaming = body.stream.unwrap_or(false);
   if let Some(effort) = body.reasoning_effort.as_ref() {
      body.reasoning_effort = Some(gemini_bridge::gemini_effort(effort).to_owned());
   }
   // Without this the terminal chunk carries no usage and the request bills
   // as zero tokens.
   if streaming {
      body.stream_options = Some(StreamOptions {
         include_usage: true,
      });
   }

   let mut record = super::pipeline::record(
      &auth,
      "chat",
      Provider::Gemini,
      model.clone(),
      model.clone(),
      facts,
   );
   record.session_key = session_key(&auth.user, &body);

   let session_key = record.session_key.clone();
   let (account_id, upstream) = match state
      .pools
      .gemini
      .execute(
         Route {
            session_key: &session_key,
            model: &model,
            user: &auth.user,
            prefer_trusted: false,
         },
         Call::OpenAi(Box::new(body)),
      )
      .await
   {
      Ok(res) => res,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   let protocol = upstream.protocol;
   let resp = upstream.response;
   record.account_id = account_id;
   record.status = i64::from(resp.status().as_u16());

   let builder = forwarded_response(&resp);
   let ok = resp.status().is_success();
   if !ok {
      return upstream_rejected(&state, record, builder, resp, started).await;
   }
   if streaming && protocol == GeminiProtocol::Native {
      let capture = UsageCapture::default();
      let mut native = NativeStream::new(&model);
      let mut scan = ChatUsageScan::new(capture.clone());
      return relayed(
         builder,
         resp,
         LogGuard::new(state, capture.clone(), record, started),
         capture,
         DIALECT,
         move |bytes| {
            let frames = native.feed(&bytes);
            for frame in &frames {
               scan.feed(frame);
            }
            Bytes::from(frames.concat())
         },
         Bytes::new,
      );
   }
   if streaming && protocol == GeminiProtocol::OpenAi {
      let capture = UsageCapture::default();
      let scan = Arc::new(Mutex::new(ChatUsageScan::new(capture.clone())));
      let each = {
         let scan = Arc::clone(&scan);
         move |bytes: Bytes| {
            scan.lock().unwrap().feed(&bytes);
            bytes
         }
      };
      return relayed(
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
      );
   }

   let bytes = match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   let bytes = if protocol == GeminiProtocol::Native {
      match response(&bytes, &model)
         .map_err(|err| err.to_string())
         .and_then(|env| serde_json::to_vec(&env).map_err(|err| err.to_string()))
      {
         Ok(payload) => Bytes::from(payload),
         Err(error) => {
            log_error(&state, record, 502, "upstream_decode");
            return error_response(DIALECT, 502, "api_error", &error);
         },
      }
   } else {
      bytes
   };
   if let Ok(env) = serde_json::from_slice::<ChatEnvelope>(&bytes)
      && let Some(usage) = env.usage
   {
      let capture = UsageCapture::default();
      capture.record(&usage.into());
      apply_snapshot(&mut record, &capture.snapshot());
   }
   super::log_usage(&state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}

/// A non-2xx carries no SSE frames, so the usage scanner logged a phantom
/// `client_disconnect` and dropped the body.
async fn upstream_rejected(
   state: &AppState,
   mut record: UsageRecord,
   builder: Builder,
   resp: reqwest::Response,
   started: Instant,
) -> Response {
   let bytes = resp.bytes().await.unwrap_or_default();
   tracing::warn!(
       user = %record.user,
       model = %record.requested_model,
       dialect = record.dialect,
       status = record.status,
       body = %String::from_utf8_lossy(&bytes).chars().take(2000).collect::<String>(),
       "gemini rejected the request"
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
fn session_key(user: &str, body: &ChatRequest) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   hasher.update(user.as_bytes());
   if let Some(first) = body.messages.first() {
      hasher.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
   }
   data_encoding::HEXLOWER.encode(&hasher.finalize())
}

/// Reads usage out of the `data:` frames of a chat stream. Only the terminal
/// frame carries it, so every frame is tried and the last one wins.
struct ChatUsageScan {
   capture: UsageCapture,
   frames: Frames,
   cut: bool,
}

impl ChatUsageScan {
   fn new(capture: UsageCapture) -> Self {
      Self {
         capture,
         frames: Frames::default(),
         cut: false,
      }
   }

   fn feed(&mut self, bytes: &[u8]) {
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
            && let Ok(serde_json::Value::String(reason)) = serde_json::to_value(reason)
         {
            self.capture.note_stop_reason(&reason);
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
                "gemini gave up mid-stream after its 200: {}",
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
             "gemini gave up mid-stream after its 200: {}",
             error.message.as_deref().unwrap_or("")
         );
         self.capture.note_cutoff(&status);
      }
   }
}

/// The native surface Gemini CLI speaks. Nothing is translated in either
/// direction, so the reply is byte-identical to Google's and only usage is
/// read out of it on the way past.
pub async fn native(
   State(state): State<AppState>,
   axum::Extension(auth): axum::Extension<AuthInfo>,
   Path(spec): Path<String>,
   RawQuery(query): RawQuery,
   headers: HeaderMap,
   body: Bytes,
) -> Response {
   let Some((model, action)) = spec.split_once(':') else {
      return error_response(
         DIALECT,
         404,
         "invalid_request_error",
         "expected /v1beta/models/{{model}}:{{generateContent|streamGenerateContent}}",
      );
   };
   if !matches!(action, "generateContent" | "streamGenerateContent") {
      return error_response(
         DIALECT,
         404,
         "invalid_request_error",
         "unsupported action on the native surface",
      );
   }
   if state.cfg.models.route(model) != Provider::Gemini {
      return error_response(
         DIALECT,
         400,
         "invalid_request_error",
         "this model is not served by the gemini backend",
      );
   }

   let started = Instant::now();
   if !auth.may_use(Provider::Gemini) {
      return super::error::out_of_scope(DIALECT, Provider::Gemini);
   }
   let streaming = action == "streamGenerateContent";
   let parsed = match serde_json::from_slice::<GenerateContentRequest>(&body) {
      Ok(req) => req,
      Err(err) => {
         return error_response(
            DIALECT,
            400,
            "invalid_request_error",
            &format!("invalid request: {err}"),
         );
      },
   };
   let mut request = parsed;
   let body = if signatures::restore(&mut request.contents) {
      serde_json::to_vec(&request).map_or(body, Bytes::from)
   } else {
      body
   };
   let key = native_session_key(&auth.user, &request);
   let facts = super::facts::RequestFacts::from_native(&request, &headers);
   let mut record = super::pipeline::record(
      &auth,
      "native",
      Provider::Gemini,
      model.to_owned(),
      model.to_owned(),
      facts,
   );
   record.session_key = key.clone();
   let call = Call::Native {
      model: model.to_owned(),
      action: action.to_owned(),
      query,
      body,
   };
   let (account_id, upstream) = match state
      .pools
      .gemini
      .execute(
         Route {
            session_key: &key,
            model,
            user: &auth.user,
            prefer_trusted: false,
         },
         call,
      )
      .await
   {
      Ok(res) => res,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   let resp = upstream.response;
   record.account_id = account_id;
   record.status = i64::from(resp.status().as_u16());
   let ok = resp.status().is_success();
   let builder = forwarded_response(&resp);
   if !ok {
      return upstream_rejected(&state, record, builder, resp, started).await;
   }

   if streaming {
      let capture = UsageCapture::default();
      let mut scan = NativeUsageScan::new(capture.clone());
      return relayed(
         builder,
         resp,
         LogGuard::new(state, capture.clone(), record, started),
         capture,
         DIALECT,
         move |bytes| {
            scan.feed(&bytes);
            bytes
         },
         Bytes::new,
      );
   }

   let bytes = match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   if let Ok(value) = serde_json::from_slice::<GenerateContentResponse>(&bytes) {
      for content in value
         .candidates
         .iter()
         .filter_map(|candidate| candidate.content.as_ref())
      {
         signatures::remember(&content.parts);
      }
      if let Some(reason) = finish_reason(&value) {
         record.stop_reason = reason;
      }
      if let Some(usage) = value.usage_metadata.as_ref() {
         let capture = UsageCapture::default();
         capture.record(&chat_usage(usage).into());
         apply_snapshot(&mut record, &capture.snapshot());
      }
   }
   super::log_usage(&state, record);
   builder
      .body(Body::from(bytes))
      .unwrap_or_else(|err| error_response(DIALECT, 502, "api_error", &err.to_string()))
}

fn finish_reason(chunk: &GenerateContentResponse) -> Option<String> {
   Some(chunk.candidates.first()?.finish_reason.as_ref()?.label())
}

/// The native request nests its first turn under `contents`, where the chat
/// dialect uses `messages`.
fn native_session_key(user: &str, body: &GenerateContentRequest) -> String {
   let mut hasher = hmac_sha256::Hash::new();
   hasher.update(user.as_bytes());
   if let Some(first) = body.contents.first() {
      hasher.update(serde_json::to_string(first).unwrap_or_default().as_bytes());
   }
   data_encoding::HEXLOWER.encode(&hasher.finalize())
}

/// Reads `usageMetadata` out of a native SSE stream. Only the terminal chunk
/// carries totals, so every frame is tried and the last one wins.
struct NativeUsageScan {
   capture: UsageCapture,
   frames: Frames,
   cut: bool,
   seen_finish: bool,
}

impl NativeUsageScan {
   fn new(capture: UsageCapture) -> Self {
      Self {
         capture,
         frames: Frames::default(),
         cut: false,
         seen_finish: false,
      }
   }

   fn feed(&mut self, bytes: &[u8]) {
      for data in self.frames.feed(bytes) {
         let Ok(value) = serde_json::from_slice::<GenerateContentResponse>(&data) else {
            continue;
         };
         for content in value
            .candidates
            .iter()
            .filter_map(|candidate| candidate.content.as_ref())
         {
            signatures::remember(&content.parts);
         }
         if let Some(reason) = finish_reason(&value) {
            self.seen_finish = true;
            self.capture.note_stop_reason(&reason);
         }
         if let Some(usage) = value.usage_metadata.as_ref() {
            let usage: Usage = chat_usage(usage).into();
            if self.seen_finish {
               self.capture.record(&usage);
            } else {
               self.capture.record_partial(&usage);
            }
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
             "gemini gave up mid-stream after its 200: {}",
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
      let mut scan = ChatUsageScan::new(capture.clone());
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
      let mut scan = ChatUsageScan::new(capture.clone());
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
      let mut scan = ChatUsageScan::new(capture.clone());
      scan.feed(b"data: {\"usage\":{\"prompt_tokens\":10,");
      scan.feed(b"\"completion_tokens\":2,\"total_tokens\":12}}\n");
      let snap = capture.snapshot();
      assert_eq!(snap.input_tokens, 10);
      assert_eq!(snap.output_tokens, 2);
   }
}
