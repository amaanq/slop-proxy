use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse as _, Response};
use std::time::Instant;

use crate::codex::types::Usage;
use crate::gemini::client::GeminiResponse;
use crate::gemini::native::{NativeStream, chat_usage, response};
use crate::gemini::sse::Frames;
use crate::gemini::types::{GenerateContentRequest, GenerateContentResponse};
use crate::pool::gemini::Call;
use crate::pool::{PoolError, Served};
use crate::provider::Provider;
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::chat::{ChatUsageScan, first_turn_key, force_usage, note_cutoff};
use crate::server::error::{Dialect, error_response};
use crate::server::facts::RequestFacts;
use crate::server::openai::{ModelList, gemini_entries};
use crate::server::pipeline::{self, Scan};
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
   facts: RequestFacts,
) -> Response {
   let started = Instant::now();
   let streaming = body.stream.unwrap_or(false);
   if let Some(effort) = body.reasoning_effort.as_ref() {
      body.reasoning_effort = Some(chat_req::clamped_effort(effort).to_owned());
   }
   force_usage(&mut body, streaming);

   let mut record = pipeline::record(
      &auth,
      "chat",
      Provider::Gemini,
      model.clone(),
      body.model.clone(),
      facts,
   );
   record.session_key = first_turn_key(&auth.user, body.messages.first());
   record.effort = body.reasoning_effort.clone().unwrap_or_default();

   let served = state
      .pools
      .gemini
      .execute(
         auth.route(&record.session_key, &record.upstream_model),
         Call::OpenAi(Box::new(body)),
      )
      .await;
   let native = served
      .as_ref()
      .is_ok_and(|served| served.response.protocol == BridgeProtocol::GeminiNative);
   pipeline::forward(
      state,
      record,
      bare(served),
      Vec::new(),
      started,
      streaming,
      |capture| ChatScan {
         chat: ChatUsageScan::new(capture, Provider::Gemini),
         native: native.then(|| NativeStream::new(&model)),
         model,
      },
   )
   .await
}

fn bare(
   served: Result<Served<GeminiResponse>, PoolError>,
) -> Result<Served<reqwest::Response>, PoolError> {
   served.map(|served| Served {
      account_id: served.account_id,
      response: served.response.response,
      attempts: served.attempts,
   })
}

/// A chat reply, rewritten from Google's own frames when the account answered
/// natively.
struct ChatScan {
   chat: ChatUsageScan,
   native: Option<NativeStream>,
   model: String,
}

impl Scan for ChatScan {
   const DIALECT: Dialect = DIALECT;
   const REJECTED: &'static str = "upstream_rejected";

   fn chunk(&mut self, bytes: Bytes) -> Bytes {
      let Some(native) = self.native.as_mut() else {
         return self.chat.chunk(bytes);
      };
      let frames = native.feed(&bytes);
      for frame in &frames {
         self.chat.feed(frame);
      }
      Bytes::from(frames.concat())
   }

   fn body(&mut self, bytes: Bytes) -> Result<Bytes, String> {
      if self.native.is_none() {
         return self.chat.body(bytes);
      }
      let payload = response(&bytes, &self.model)
         .map_err(|err| err.to_string())
         .and_then(|env| serde_json::to_vec(&env).map_err(|err| err.to_string()))?;
      self.chat.body(Bytes::from(payload))
   }

   fn tail(&mut self) -> Bytes {
      self.chat.tail()
   }
}

/// A catalog for a client pinned to the `/v1beta` base URL. Google's own
/// `ListModels` keys each entry by `name`, which a discovering harness reading
/// `data[].id` cannot see, so this answers in the same shape as `/v1/models`
/// narrowed to the Gemini pool.
pub async fn models(State(state): State<AppState>) -> Response {
   axum::Json(ModelList {
      object: "list",
      data: gemini_entries(&state).await,
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
         StatusCode::NOT_FOUND,
         "invalid_request_error",
         "expected /v1beta/models/{{model}}:{{generateContent|streamGenerateContent}}",
      );
   };
   if !matches!(action, "generateContent" | "streamGenerateContent") {
      return error_response(
         DIALECT,
         StatusCode::NOT_FOUND,
         "invalid_request_error",
         "unsupported action on the native surface",
      );
   }
   let resolved = resolve(&state.cfg.models, raw_model);
   match pipeline::admit(&state, &auth, DIALECT, "native", raw_model, &resolved.model) {
      Ok(Provider::Gemini) => {},
      Ok(_) => {
         return error_response(
            DIALECT,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "this model is not served by the gemini backend",
         );
      },
      Err(response) => return *response,
   }

   let started = Instant::now();
   let streaming = action == "streamGenerateContent";
   let request = match serde_json::from_slice::<GenerateContentRequest>(&body) {
      Ok(req) => req,
      Err(err) => {
         return error_response(
            DIALECT,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("invalid request: {err}"),
         );
      },
   };
   let mut record = pipeline::record(
      &auth,
      "native",
      Provider::Gemini,
      raw_model.to_owned(),
      resolved.model.clone(),
      RequestFacts::from_native(&request, &headers),
   );
   record.session_key = first_turn_key(&auth.user, request.contents.first());
   let call = Call::Native {
      model: resolved.model.clone(),
      action: action.to_owned(),
      query,
      body,
   };
   let served = state
      .pools
      .gemini
      .execute(auth.route(&record.session_key, &resolved.model), call)
      .await;
   pipeline::forward(
      state,
      record,
      bare(served),
      Vec::new(),
      started,
      streaming,
      |capture| NativeUsageScan {
         capture,
         frames: Frames::default(),
         cut: false,
         seen_finish: false,
      },
   )
   .await
}

fn finish_reason(chunk: &GenerateContentResponse) -> Option<String> {
   Some(chunk.candidates.first()?.finish_reason.as_ref()?.label())
}

/// Reads `usageMetadata` out of a native SSE stream. Only the terminal chunk
/// carries totals, so every frame is tried and the last one wins.
struct NativeUsageScan {
   capture: UsageCapture,
   frames: Frames,
   cut: bool,
   seen_finish: bool,
}

impl Scan for NativeUsageScan {
   const DIALECT: Dialect = DIALECT;
   const REJECTED: &'static str = "upstream_rejected";

   fn chunk(&mut self, bytes: Bytes) -> Bytes {
      for data in self.frames.feed(&bytes) {
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
      note_cutoff(&self.frames, &mut self.cut, &self.capture, Provider::Gemini);
      bytes
   }

   fn body(&mut self, bytes: Bytes) -> Result<Bytes, String> {
      if let Ok(value) = serde_json::from_slice::<GenerateContentResponse>(&bytes) {
         if let Some(reason) = finish_reason(&value) {
            self.capture.note_stop_reason(&reason);
         }
         if let Some(usage) = value.usage_metadata.as_ref() {
            self.capture.record(&chat_usage(usage).into());
         }
      }
      Ok(bytes)
   }
}
