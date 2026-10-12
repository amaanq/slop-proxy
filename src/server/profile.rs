//! Codex's usage limits, profile and token history, answered from the
//! proxy's own usage log for the token's owner.
use axum::extract::{Query, State};
use axum::response::{IntoResponse as _, Response};
use axum::routing::get;
use axum::{Extension, Json, Router};
use jiff::ToSpan as _;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use jiff::{Timestamp, Zoned};
use serde::{Deserialize, Serialize};

use crate::clock;
use crate::db::usage::DayUsage;
use crate::pool::{UsageWindow, window_seconds};
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::clientcfg::PLAN;
use crate::server::error::BackendError;

/// The history the desktop shows when it names no range.
const DEFAULT_HISTORY_DAYS: i64 = 30;
const MAX_HISTORY_DAYS: i64 = 366;

fn today() -> Date {
   Zoned::new(Timestamp::now(), TimeZone::UTC).date()
}

pub fn routes() -> Router<AppState> {
   Router::new()
      .route("/backend-api/wham/usage", get(limits))
      .route("/backend-api/wham/profiles/me", get(stats))
      .route("/backend-api/profiles/me", get(identity))
      .route("/backend-api/profiles/me/page", get(page))
      .route("/backend-api/profiles/{username}/page", get(page))
      .route(
         "/backend-api/wham/usage/daily-token-usage-breakdown",
         get(history),
      )
      .route(
         "/backend-api/wham/usage/daily-workspace-user-token-usage-breakdown",
         get(history),
      )
}

#[derive(Serialize)]
struct RateLimits {
   plan_type: &'static str,
   rate_limit: RateLimit,
   credits: Option<()>,
   additional_rate_limits: [(); 0],
   rate_limit_reset_credits: ResetCredits,
}

#[derive(Serialize)]
struct RateLimit {
   allowed: bool,
   limit_reached: bool,
   primary_window: Option<Window>,
   secondary_window: Option<Window>,
}

#[derive(Serialize)]
struct Window {
   used_percent: i64,
   limit_window_seconds: i64,
   reset_after_seconds: i64,
   reset_at: i64,
}

impl Window {
   fn new(window: &UsageWindow, now: i64) -> Self {
      let reset_at = window.resets_at.unwrap_or(now);
      Self {
         used_percent: (window.utilization * 100.0_f64)
            .round()
            .clamp(0.0_f64, 100.0_f64) as i64,
         limit_window_seconds: window_seconds(&window.name).unwrap_or(0),
         reset_after_seconds: (reset_at - now).max(0),
         reset_at,
      }
   }
}

#[derive(Serialize)]
struct ResetCredits {
   available_count: i64,
}

/// The pool's limits averaged over the accounts that serve the token,
/// shortest window first as codex lays them out.
async fn limits(State(state): State<AppState>, Extension(auth): Extension<AuthInfo>) -> Response {
   let mut windows = state
      .pools
      .codex
      .pool_windows(&auth.user, auth.limits.pinned_account, None)
      .await;
   windows.sort_by_key(|window| window_seconds(&window.name).unwrap_or(i64::MAX));
   let reached = windows.iter().any(|window| window.utilization >= 1.0_f64);
   let now = clock::unix_now();
   let mut windows = windows.iter().map(|window| Window::new(window, now));
   Json(RateLimits {
      plan_type: PLAN,
      rate_limit: RateLimit {
         allowed: !reached,
         limit_reached: reached,
         primary_window: windows.next(),
         secondary_window: windows.next(),
      },
      credits: None,
      additional_rate_limits: [],
      rate_limit_reset_credits: ResetCredits { available_count: 0 },
   })
   .into_response()
}

/// A user's tokens per active day, and what they add up to.
struct Activity {
   days: Vec<(Date, i64)>,
   totals: Totals,
   streaks: Streaks,
}

#[derive(Clone, Copy, Serialize)]
struct Totals {
   lifetime_tokens: i64,
   peak_daily_tokens: i64,
   /// A proxy request is not a whole codex turn, so the longest is unknown.
   longest_running_turn_sec: Option<()>,
}

#[derive(Clone, Copy, Serialize)]
struct Streaks {
   current_streak_days: i64,
   longest_streak_days: i64,
}

impl Activity {
   /// `usage` is in day order, as [`crate::db::Db::daily_usage`] returns it.
   fn new(usage: &[DayUsage], today: Date) -> Self {
      let days = usage
         .chunk_by(|left, right| left.day == right.day)
         .map(|models| (models[0].day, models.iter().map(DayUsage::total).sum()))
         .collect::<Vec<_>>();
      let mut streak = 0;
      let mut longest_streak_days = 0;
      let mut previous = None::<Date>;
      for &(day, _) in &days {
         let consecutive = previous.and_then(|last| last.tomorrow().ok()) == Some(day);
         streak = if consecutive { streak + 1 } else { 1 };
         longest_streak_days = longest_streak_days.max(streak);
         previous = Some(day);
      }
      // A streak lives on until a whole day passes without use.
      let alive = previous.is_some() && previous >= today.yesterday().ok();
      let tokens = days.iter().map(|&(_, tokens)| tokens);
      Self {
         totals: Totals {
            lifetime_tokens: tokens.clone().sum(),
            peak_daily_tokens: tokens.max().unwrap_or(0),
            longest_running_turn_sec: None,
         },
         streaks: Streaks {
            current_streak_days: if alive { streak } else { 0 },
            longest_streak_days,
         },
         days,
      }
   }

   fn daily(&self) -> Vec<Bucket> {
      self.days.iter().copied().map(Bucket::from).collect()
   }

   /// Codex's activity calendar starts its weeks on Sunday.
   fn weekly(&self) -> Vec<Bucket> {
      let mut weeks = Vec::<Bucket>::new();
      for &(day, tokens) in &self.days {
         let back = i64::from(day.weekday().to_sunday_zero_offset());
         let sunday = day
            .checked_sub(back.days())
            .expect("usage days sit far from the calendar's limits");
         match weeks.last_mut() {
            Some(week) if week.start_date == sunday => week.tokens += tokens,
            _ => weeks.push(Bucket::from((sunday, tokens))),
         }
      }
      weeks
   }

   fn cumulative(&self) -> Vec<Bucket> {
      self
         .days
         .iter()
         .scan(0, |total, &(day, tokens)| {
            *total += tokens;
            Some(Bucket::from((day, *total)))
         })
         .collect()
   }
}

#[derive(Serialize)]
struct Bucket {
   start_date: Date,
   tokens: i64,
   chat_turns: Option<()>,
}

impl From<(Date, i64)> for Bucket {
   fn from((start_date, tokens): (Date, i64)) -> Self {
      Self {
         start_date,
         tokens,
         chat_turns: None,
      }
   }
}

/// Every profile is the token owner's own and private.
#[derive(Serialize)]
struct Details<'a> {
   display_name: &'a str,
   username: &'a str,
   profile_picture_url: Option<()>,
   description: Option<()>,
   photo_frame_style: &'static str,
   profile_visibility: &'static str,
}

impl<'a> From<&'a AuthInfo> for Details<'a> {
   fn from(auth: &'a AuthInfo) -> Self {
      Self {
         display_name: &auth.user,
         username: &auth.user,
         profile_picture_url: None,
         description: None,
         photo_frame_style: "circle",
         profile_visibility: "private",
      }
   }
}

#[derive(Serialize)]
struct Stats<'a> {
   stats: StatsBody,
   profile: Details<'a>,
   metadata: StatsMetadata,
}

#[derive(Serialize)]
struct StatsBody {
   #[serde(flatten)]
   totals: Totals,
   #[serde(flatten)]
   streaks: Streaks,
   daily_usage_buckets: Vec<Bucket>,
}

#[derive(Serialize)]
struct StatsMetadata {
   stats_error: Option<()>,
}

#[derive(Serialize)]
struct Identity<'a> {
   profile_details: Details<'a>,
}

#[derive(Serialize)]
struct Page<'a> {
   is_self: bool,
   can_edit: bool,
   has_public_personal_profile: bool,
   profile_details: Details<'a>,
   page_version: i64,
   page: PageBody,
}

#[derive(Serialize)]
struct PageBody {
   workspace_id: Option<()>,
   workspace_name: Option<()>,
   visibility: Visibility,
   display_settings: DisplaySettings,
   stats: PageStats,
   activity_graph: ActivityGraph,
   insights: Option<()>,
   top_plugins: [(); 0],
   pins: Pins,
}

#[derive(Serialize)]
struct Visibility {
   value: &'static str,
   can_edit: bool,
}

/// The proxy has usage and activity to show, but no insights or plugins.
#[derive(Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "the page's section toggles")]
struct DisplaySettings {
   show_usage_stats_section: bool,
   show_activity_graph_section: bool,
   show_insights_section: bool,
   show_top_plugins_section: bool,
}

#[derive(Serialize)]
struct PageStats {
   #[serde(flatten)]
   streaks: Streaks,
   agentic: Totals,
}

#[derive(Serialize)]
struct ActivityGraph {
   daily_usage_buckets: Vec<Bucket>,
   weekly_usage_buckets: Vec<Bucket>,
   cumulative_daily_usage_buckets: Vec<Bucket>,
}

#[derive(Serialize)]
struct Pins {
   items: Vec<()>,
}

async fn activity(state: &AppState, auth: &AuthInfo) -> Result<Activity, BackendError> {
   let usage = state.db.daily_usage(auth.user.clone(), None).await?;
   Ok(Activity::new(&usage, today()))
}

async fn stats(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
) -> Result<Response, BackendError> {
   let activity = activity(&state, &auth).await?;
   let body = Stats {
      stats: StatsBody {
         totals: activity.totals,
         streaks: activity.streaks,
         daily_usage_buckets: activity.daily(),
      },
      profile: Details::from(&auth),
      metadata: StatsMetadata { stats_error: None },
   };
   Ok(Json(body).into_response())
}

async fn identity(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(Identity {
      profile_details: Details::from(&auth),
   })
   .into_response()
}

/// Also answers a profile linked by username, since the only one a token
/// can see is its owner's.
async fn page(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
) -> Result<Response, BackendError> {
   let activity = activity(&state, &auth).await?;
   let body = Page {
      is_self: true,
      can_edit: false,
      has_public_personal_profile: false,
      profile_details: Details::from(&auth),
      page_version: 0,
      page: PageBody {
         workspace_id: None,
         workspace_name: None,
         visibility: Visibility {
            value: "private",
            can_edit: false,
         },
         display_settings: DisplaySettings {
            show_usage_stats_section: true,
            show_activity_graph_section: true,
            show_insights_section: false,
            show_top_plugins_section: false,
         },
         stats: PageStats {
            streaks: activity.streaks,
            agentic: activity.totals,
         },
         activity_graph: ActivityGraph {
            daily_usage_buckets: activity.daily(),
            weekly_usage_buckets: activity.weekly(),
            cumulative_daily_usage_buckets: activity.cumulative(),
         },
         insights: None,
         top_plugins: [],
         pins: Pins { items: Vec::new() },
      },
   };
   Ok(Json(body).into_response())
}

#[derive(Deserialize)]
struct HistoryQuery {
   start_date: Option<Date>,
   end_date: Option<Date>,
}

#[derive(Serialize)]
struct History {
   units: &'static str,
   data: Vec<HistoryDay>,
   data_freshness_ts: i64,
}

#[derive(Serialize)]
struct HistoryDay {
   date: Date,
   models: Vec<ModelTokens>,
   product_surface_usage_values: SurfaceUsage,
}

#[derive(Serialize)]
struct ModelTokens {
   model: String,
   /// The history calls its total credits, even counting tokens.
   credits: i64,
   uncached_text_input_tokens: i64,
   cached_text_input_tokens: i64,
   text_output_tokens: i64,
   text_total_tokens: i64,
}

impl From<&DayUsage> for ModelTokens {
   fn from(usage: &DayUsage) -> Self {
      Self {
         model: usage.model.clone(),
         credits: usage.total(),
         uncached_text_input_tokens: usage.input_tokens,
         cached_text_input_tokens: usage.cached_tokens,
         text_output_tokens: usage.output_tokens,
         text_total_tokens: usage.total(),
      }
   }
}

#[derive(Serialize)]
struct SurfaceUsage {
   codex: i64,
}

/// Tokens per day and model between two inclusive dates.
async fn history(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   Query(query): Query<HistoryQuery>,
) -> Result<Json<History>, BackendError> {
   const BACKWARDS: BackendError =
      BackendError::Invalid("the range runs backwards or spans more than 366 days");
   let end = query.end_date.unwrap_or_else(today);
   let start = match query.start_date {
      Some(start) => start,
      None => end
         .checked_sub((DEFAULT_HISTORY_DAYS - 1).days())
         .map_err(|_| BACKWARDS)?,
   };
   let limit = start
      .checked_add(MAX_HISTORY_DAYS.days())
      .map_err(|_| BACKWARDS)?;
   if end < start || end >= limit {
      return Err(BACKWARDS);
   }

   let usage = state
      .db
      .daily_usage(auth.user.clone(), Some(start..=end))
      .await?;
   let data = usage
      .chunk_by(|left, right| left.day == right.day)
      .map(|models| HistoryDay {
         date: models[0].day,
         models: models.iter().map(ModelTokens::from).collect(),
         product_surface_usage_values: SurfaceUsage {
            codex: models.iter().map(DayUsage::total).sum(),
         },
      })
      .collect();
   Ok(Json(History {
      units: "tokens",
      data,
      data_freshness_ts: clock::unix_now(),
   }))
}
