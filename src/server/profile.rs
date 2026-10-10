//! Codex's usage limits, profile and token history, answered from the
//! proxy's own usage log for the token's owner.
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse as _, Response};
use axum::routing::get;
use axum::{Extension, Json, Router};
use serde::{Deserialize, Serialize};

use crate::clock;
use crate::db::usage::DayUsage;
use crate::pool::{UsageWindow, window_seconds};
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::clientcfg::PLAN;
use crate::server::error::{Dialect, error_response};

/// The history the desktop shows when it names no range.
const DEFAULT_HISTORY_DAYS: i64 = 30;
const MAX_HISTORY_DAYS: i64 = 366;

pub fn routes() -> Router<AppState> {
   Router::new()
      .route("/backend-api/wham/usage", get(limits))
      .route("/backend-api/wham/profiles/me", get(stats))
      .route("/backend-api/profiles/me", get(identity))
      .route("/backend-api/profiles/me/page", get(own_page))
      .route("/backend-api/profiles/{username}/page", get(page))
      .route(
         "/backend-api/wham/usage/daily-token-usage-breakdown",
         get(history),
      )
      .route(
         "/backend-api/wham/usage/daily-workspace-user-token-usage-breakdown",
         get(history),
      )
      .route(
         "/backend-api/wham/analytics/daily-code-review-metrics",
         get(code_review_history),
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
#[derive(Debug, PartialEq, Eq)]
struct Activity {
   days: Vec<(i64, i64)>,
   lifetime_tokens: i64,
   peak_daily_tokens: i64,
   current_streak_days: i64,
   longest_streak_days: i64,
}

impl Activity {
   /// `usage` is in day order, as [`crate::db::Db::daily_usage`] returns it.
   fn new(usage: &[DayUsage], today: i64) -> Self {
      let days: Vec<(i64, i64)> = usage
         .chunk_by(|left, right| left.day == right.day)
         .map(|models| (models[0].day, models.iter().map(DayUsage::total).sum()))
         .collect();
      let mut streak = 0;
      let mut longest_streak_days = 0;
      let mut previous = None;
      for &(day, _) in &days {
         streak = if previous == Some(day - 1) {
            streak + 1
         } else {
            1
         };
         longest_streak_days = longest_streak_days.max(streak);
         previous = Some(day);
      }
      // A streak lives on until a whole day passes without use.
      let current_streak_days = if previous.is_some_and(|day| day >= today - 1) {
         streak
      } else {
         0
      };
      Self {
         lifetime_tokens: days.iter().map(|&(_, tokens)| tokens).sum(),
         peak_daily_tokens: days.iter().map(|&(_, tokens)| tokens).max().unwrap_or(0),
         current_streak_days,
         longest_streak_days,
         days,
      }
   }

   fn daily(&self) -> Vec<Bucket> {
      self
         .days
         .iter()
         .map(|&(day, tokens)| Bucket::new(day, tokens))
         .collect()
   }

   /// Codex's activity calendar starts its weeks on Sunday, and the epoch
   /// fell on a Thursday.
   fn weekly(&self) -> Vec<Bucket> {
      let mut weeks: Vec<(i64, i64)> = Vec::new();
      for &(day, tokens) in &self.days {
         let sunday = day - (day + 4).rem_euclid(7);
         match weeks.last_mut() {
            Some(&mut (week, ref mut total)) if week == sunday => *total += tokens,
            _ => weeks.push((sunday, tokens)),
         }
      }
      weeks
         .into_iter()
         .map(|(week, tokens)| Bucket::new(week, tokens))
         .collect()
   }

   fn cumulative(&self) -> Vec<Bucket> {
      let mut total = 0;
      self
         .days
         .iter()
         .map(|&(day, tokens)| {
            total += tokens;
            Bucket::new(day, total)
         })
         .collect()
   }
}

#[derive(Serialize)]
struct Bucket {
   start_date: String,
   tokens: i64,
   chat_turns: Option<()>,
}

impl Bucket {
   fn new(day: i64, tokens: i64) -> Self {
      Self {
         start_date: clock::date(day),
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

impl<'a> Details<'a> {
   const fn new(auth: &'a AuthInfo) -> Self {
      Self {
         display_name: auth.user.as_str(),
         username: auth.user.as_str(),
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
   lifetime_tokens: i64,
   peak_daily_tokens: i64,
   /// A proxy request is not a whole codex turn, so the longest is unknown.
   longest_running_turn_sec: Option<()>,
   current_streak_days: i64,
   longest_streak_days: i64,
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
   current_streak_days: i64,
   longest_streak_days: i64,
   agentic: AgenticStats,
}

#[derive(Serialize)]
struct AgenticStats {
   lifetime_tokens: i64,
   peak_daily_tokens: i64,
   longest_running_turn_sec: Option<()>,
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

async fn activity(state: &AppState, auth: &AuthInfo) -> eyre::Result<Activity> {
   let usage = state.db.daily_usage(auth.user.clone(), 0, i64::MAX).await?;
   Ok(Activity::new(&usage, clock::unix_now().div_euclid(86400)))
}

fn unreadable(err: &eyre::Report) -> Response {
   tracing::error!("reading daily usage: {err}");
   error_response(
      Dialect::OpenAi,
      StatusCode::INTERNAL_SERVER_ERROR,
      "api_error",
      "internal error",
   )
}

async fn stats(State(state): State<AppState>, Extension(auth): Extension<AuthInfo>) -> Response {
   let activity = match activity(&state, &auth).await {
      Ok(activity) => activity,
      Err(err) => return unreadable(&err),
   };
   Json(Stats {
      stats: StatsBody {
         lifetime_tokens: activity.lifetime_tokens,
         peak_daily_tokens: activity.peak_daily_tokens,
         longest_running_turn_sec: None,
         current_streak_days: activity.current_streak_days,
         longest_streak_days: activity.longest_streak_days,
         daily_usage_buckets: activity.daily(),
      },
      profile: Details::new(&auth),
      metadata: StatsMetadata { stats_error: None },
   })
   .into_response()
}

async fn identity(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(Identity {
      profile_details: Details::new(&auth),
   })
   .into_response()
}

async fn own_page(State(state): State<AppState>, Extension(auth): Extension<AuthInfo>) -> Response {
   let activity = match activity(&state, &auth).await {
      Ok(activity) => activity,
      Err(err) => return unreadable(&err),
   };
   Json(Page {
      is_self: true,
      can_edit: false,
      has_public_personal_profile: false,
      profile_details: Details::new(&auth),
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
            current_streak_days: activity.current_streak_days,
            longest_streak_days: activity.longest_streak_days,
            agentic: AgenticStats {
               lifetime_tokens: activity.lifetime_tokens,
               peak_daily_tokens: activity.peak_daily_tokens,
               longest_running_turn_sec: None,
            },
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
   })
   .into_response()
}

/// The desktop links a profile by username, and the only one a token can
/// see is its owner's.
async fn page(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   Path(username): Path<String>,
) -> Response {
   if username != auth.user {
      return error_response(
         Dialect::OpenAi,
         StatusCode::NOT_FOUND,
         "not_found_error",
         "no such profile",
      );
   }
   own_page(State(state), Extension(auth)).await
}

#[derive(Deserialize)]
struct HistoryQuery {
   start_date: Option<String>,
   end_date: Option<String>,
}

#[derive(Serialize)]
struct History {
   units: &'static str,
   data: Vec<HistoryDay>,
   data_freshness_ts: i64,
}

#[derive(Serialize)]
struct HistoryDay {
   date: String,
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

impl ModelTokens {
   fn new(usage: &DayUsage) -> Self {
      let total = usage.total();
      Self {
         model: usage.model.clone(),
         credits: total,
         uncached_text_input_tokens: usage.input_tokens,
         cached_text_input_tokens: usage.cached_tokens,
         text_output_tokens: usage.output_tokens,
         text_total_tokens: total,
      }
   }
}

#[derive(Serialize)]
struct SurfaceUsage {
   codex: i64,
}

/// Tokens per day and model between two inclusive `YYYY-MM-DD` dates.
async fn history(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   Query(query): Query<HistoryQuery>,
) -> Response {
   let now = clock::unix_now();
   let today = now.div_euclid(86400);
   let parse = |date: Option<&str>| date.map_or(Some(None), |date| clock::day(date).map(Some));
   let (Some(end), Some(start)) = (
      parse(query.end_date.as_deref()),
      parse(query.start_date.as_deref()),
   ) else {
      return history_error("dates are YYYY-MM-DD");
   };
   let end = end.unwrap_or(today);
   let start = start.unwrap_or(end - (DEFAULT_HISTORY_DAYS - 1));
   if end < start || end - start >= MAX_HISTORY_DAYS {
      return history_error("the range runs backwards or spans more than 366 days");
   }

   let usage = match state
      .db
      .daily_usage(auth.user.clone(), start * 86400, (end + 1) * 86400)
      .await
   {
      Ok(usage) => usage,
      Err(err) => return unreadable(&err),
   };
   let data = usage
      .chunk_by(|left, right| left.day == right.day)
      .map(|models| HistoryDay {
         date: clock::date(models[0].day),
         models: models.iter().map(ModelTokens::new).collect(),
         product_surface_usage_values: SurfaceUsage {
            codex: models.iter().map(DayUsage::total).sum(),
         },
      })
      .collect();
   Json(History {
      units: "tokens",
      data,
      data_freshness_ts: now,
   })
   .into_response()
}

fn history_error(message: &str) -> Response {
   error_response(
      Dialect::OpenAi,
      StatusCode::BAD_REQUEST,
      "invalid_request_error",
      message,
   )
}

/// Code review is not something the proxy records.
async fn code_review_history() -> Response {
   Json(History {
      units: "tokens",
      data: Vec::new(),
      data_freshness_ts: clock::unix_now(),
   })
   .into_response()
}

#[cfg(test)]
mod tests {
   use super::*;

   fn usage(day: i64, model: &str, input_tokens: i64) -> DayUsage {
      DayUsage {
         day,
         model: model.into(),
         input_tokens,
         cached_tokens: 0,
         output_tokens: 0,
      }
   }

   #[test]
   fn activity_sums_days_and_keeps_a_streak_until_a_day_is_missed() {
      let rows = [
         usage(10, "a", 5),
         usage(12, "a", 1),
         usage(12, "b", 2),
         usage(13, "a", 4),
         usage(14, "a", 1),
      ];
      let activity = Activity::new(&rows, 15);
      assert_eq!(activity.days, [(10, 5), (12, 3), (13, 4), (14, 1)]);
      assert_eq!(activity.lifetime_tokens, 13);
      assert_eq!(activity.peak_daily_tokens, 5);
      assert_eq!(activity.longest_streak_days, 3);
      assert_eq!(activity.current_streak_days, 3);
      assert_eq!(Activity::new(&rows, 16).current_streak_days, 0);
      assert_eq!(Activity::new(&[], 16).peak_daily_tokens, 0);
   }

   #[test]
   fn weeks_start_on_sunday() {
      // 2026-10-03 is a Saturday, 2026-10-04 the Sunday after it.
      let saturday = clock::day("2026-10-03").unwrap();
      let rows = [usage(saturday, "a", 1), usage(saturday + 1, "a", 2)];
      let weeks = Activity::new(&rows, saturday + 1).weekly();
      assert_eq!(weeks[0].start_date, "2026-09-27");
      assert_eq!(weeks[1].start_date, "2026-10-04");
      assert_eq!(weeks[1].tokens, 2);
   }
}
