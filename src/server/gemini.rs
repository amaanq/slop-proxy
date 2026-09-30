use axum::body::{Body, Bytes};
use axum::extract::{Path, RawQuery, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};
use std::time::Instant;

use super::auth::AuthInfo;
use super::chat::{
   ChatUsageScan, force_usage, relay_chat_body, relay_chat_stream, session_key, upstream_rejected,
};
use super::error::{Dialect, error_response};
use super::pipeline::{apply_snapshot, dispatch_failed, read_body, relayed};
use super::relay::forwarded_response;
use super::{AppState, LogGuard, log_error};
use crate::codex::types::Usage;
use crate::gemini::native::{NativeStream, chat_usage, response};
use crate::gemini::sse::Frames;
use crate::gemini::types::{GenerateContentRequest, GenerateContentResponse};
use crate::pool::Route;
use crate::pool::gemini::Call;
use crate::provider::Provider;
use crate::translate::UsageCapture;
use crate::translate::bridge::BridgeProtocol;
use crate::translate::chat::ChatRequest;
use crate::translate::chat_req;
use crate::translate::model_map::resolve;

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
      body.reasoning_effort = Some(chat_req::clamped_effort(effort).to_owned());
   }
   force_usage(&mut body, streaming);

   let mut record = super::pipeline::record(
      &auth,
      "chat",
      Provider::Gemini,
      model.clone(),
      body.model.clone(),
      facts,
   );
   record.session_key = session_key(&auth.user, &body);
   record.effort = body.reasoning_effort.clone().unwrap_or_default();

   let session_key = record.session_key.clone();
   let served = match state
      .pools
      .gemini
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
         Call::OpenAi(Box::new(body)),
      )
      .await
   {
      Ok(served) => served,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   let protocol = served.response.protocol;
   let resp = served.response.response;
   record.account_id = served.account_id;
   record.attempts = i64::from(served.attempts);
   record.status = i64::from(resp.status().as_u16());

   let builder = forwarded_response(&resp);
   let ok = resp.status().is_success();
   if !ok {
      return upstream_rejected(&state, record, builder, resp, started, Provider::Gemini).await;
   }
   if streaming && protocol == BridgeProtocol::GeminiNative {
      let capture = UsageCapture::default();
      let mut native = NativeStream::new(&model);
      let mut scan = ChatUsageScan::new(capture.clone(), Provider::Gemini);
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
   if streaming && protocol == BridgeProtocol::Chat {
      return relay_chat_stream(state, record, builder, resp, started, Provider::Gemini);
   }

   let bytes = match read_body(&state, &record, DIALECT, resp).await {
      Ok(bytes) => bytes,
      Err(resp) => return resp,
   };
   let bytes = if protocol == BridgeProtocol::GeminiNative {
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
   relay_chat_body(&state, record, builder, bytes, started)
}

/// A catalog for a client pinned to the `/v1beta` base URL. Google's own
/// `ListModels` keys each entry by `name`, which a discovering harness reading
/// `data[].id` cannot see, so this answers in the same shape as `/v1/models`
/// narrowed to the Gemini pool.
pub async fn models(State(state): State<AppState>) -> Response {
   axum::Json(super::openai::ModelList {
      object: "list",
      data: super::openai::gemini_entries(&state).await,
   })
   .into_response()
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
   let Some((raw_model, action)) = spec.rsplit_once(':') else {
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
   let resolved = resolve(&state.cfg.models, raw_model);
   if state.cfg.models.blocked(&resolved.model) {
      return super::error::blocked_model(DIALECT, &resolved.model);
   }
   if state.cfg.models.route(&resolved.model) != Provider::Gemini {
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
   let request = parsed;
   let key = native_session_key(&auth.user, &request);
   let facts = super::facts::RequestFacts::from_native(&request, &headers);
   let mut record = super::pipeline::record(
      &auth,
      "native",
      Provider::Gemini,
      raw_model.to_owned(),
      resolved.model.clone(),
      facts,
   );
   record.session_key = key.clone();
   let call = Call::Native {
      model: resolved.model.clone(),
      action: action.to_owned(),
      query,
      body,
   };
   let served = match state
      .pools
      .gemini
      .execute(
         Route {
            session_key: &key,
            model: &resolved.model,
            service_tier: None,
            user: &auth.user,
            pinned_account: auth.limits.pinned_account,
            prefer_trusted: false,
            reserved_only: auth.limits.reserved_only,
         },
         call,
      )
      .await
   {
      Ok(served) => served,
      Err(err) => return dispatch_failed(&state, record, DIALECT, err),
   };
   let resp = served.response.response;
   record.account_id = served.account_id;
   record.attempts = i64::from(served.attempts);
   record.status = i64::from(resp.status().as_u16());
   let ok = resp.status().is_success();
   let builder = forwarded_response(&resp);
   if !ok {
      return upstream_rejected(&state, record, builder, resp, started, Provider::Gemini).await;
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
      if let Some(reason) = finish_reason(&value) {
         record.stop_reason = reason;
      }
      if let Some(usage) = value.usage_metadata.as_ref() {
         let capture = UsageCapture::default();
         capture.record(&chat_usage(usage).into());
         apply_snapshot(&mut record, &capture.snapshot(), started);
      }
   }
   record.duration_ms = Some(started.elapsed().as_millis() as i64);
   record.response_bytes = bytes.len() as i64;
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
