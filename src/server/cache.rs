use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse as _, Response};
use axum::{Extension, Json};
use serde::Serialize;

use crate::clock;
use crate::db::usage::SessionCache;
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::error::{Dialect, error_response};

const LOOKBACK_SECS: i64 = 24 * 3600;

#[derive(Serialize)]
struct CacheStatus {
   #[serde(flatten)]
   session: SessionCache,
   idle_secs: i64,
   expires_in_secs: Option<i64>,
}

pub async fn status(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   Path(session_id): Path<String>,
) -> Response {
   let now = clock::unix_now();
   let found = state
      .db
      .session_cache(auth.user, session_id, now - LOOKBACK_SECS)
      .await;
   let session = match found {
      Ok(Some(session)) => session,
      Ok(None) => {
         return error_response(
            Dialect::OpenAi,
            StatusCode::NOT_FOUND,
            "not_found_error",
            "no recent requests for this session",
         );
      },
      Err(err) => {
         tracing::error!("reading session cache: {err}");
         return error_response(
            Dialect::OpenAi,
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "internal error",
         );
      },
   };

   let idle_secs = (now - session.last.finished_at).max(0);
   let expires_in_secs = session.last.ttl_secs.map(|ttl| (ttl - idle_secs).max(0));
   Json(CacheStatus {
      session,
      idle_secs,
      expires_in_secs,
   })
   .into_response()
}
