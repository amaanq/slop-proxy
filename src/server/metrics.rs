use std::array::from_fn;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Display;
use std::fmt::Write as _;
use std::hash::Hash;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse as _;
use axum::response::Response;
use axum::routing::get;
use tokio::net::TcpListener;
use tokio::time::{self, MissedTickBehavior};

use crate::clock;
use crate::db::Db;
use crate::db::usage::{
   ErrorRow, InsightRow, MetricsDelta, MetricsRow, SessionRow, ToolRow, USAGE_DIMENSIONS,
   cache_hit_ratio, rewrites,
};
use crate::pool::{AccountSnapshot, UsageWindow};
use crate::provider::Provider;
use crate::server::AppState;

/// The usage history behind every counter, rendered off the scrape path.
/// A scrape that overruns its timeout blanks every series, live quota
/// gauges included.
#[derive(Clone)]
pub struct Metrics {
   state: AppState,
   history: Arc<RwLock<Arc<str>>>,
}

impl Metrics {
   pub fn new(state: AppState) -> Self {
      Self {
         state,
         history: Arc::new(RwLock::new(Arc::from(""))),
      }
   }

   pub async fn refresh(&self, totals: &mut Totals) {
      if let Err(err) = totals.update(&self.state.db).await {
         tracing::error!("reading usage metrics: {err}");
         return;
      }
      let text = totals.render();
      *self.history.write().expect("metrics lock poisoned") = Arc::from(text);
   }
}

pub fn spawn(state: AppState, listener: TcpListener) {
   let metrics = Metrics::new(state);

   let refresher = metrics.clone();
   tokio::spawn(async move {
      let mut totals = Totals::default();
      let mut tick = time::interval(Duration::from_mins(1));
      tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
      loop {
         tick.tick().await;
         refresher.refresh(&mut totals).await;
      }
   });

   tokio::spawn(async move {
      let app = Router::new()
         .route("/metrics", get(serve))
         .with_state(metrics);
      if let Err(err) = axum::serve(listener, app).await {
         tracing::error!("metrics listener failed: {err}");
      }
   });
}

pub async fn serve(State(metrics): State<Metrics>) -> Response {
   let accounts = metrics.state.pools.snapshots().await;
   let history = Arc::clone(&metrics.history.read().expect("metrics lock poisoned"));

   let mut text = String::with_capacity(4096 + history.len());
   render_accounts(&mut text, &accounts, clock::unix_now());
   text.push_str(&history);
   ([(CONTENT_TYPE, "text/plain; version=0.0.4")], text).into_response()
}

/// Rows of one family summed per label set, in first-seen order.
struct Family<K, R> {
   index: HashMap<K, usize>,
   rows: Vec<R>,
}

impl<K, R> Default for Family<K, R> {
   fn default() -> Self {
      Self {
         index: HashMap::new(),
         rows: Vec::new(),
      }
   }
}

impl<K: Hash + Eq, R: AddAssign> Family<K, R> {
   fn fold(&mut self, rows: Vec<R>, key: impl Fn(&R) -> K) {
      for row in rows {
         match self.index.entry(key(&row)) {
            Entry::Occupied(slot) => self.rows[*slot.get()] += row,
            Entry::Vacant(slot) => {
               slot.insert(self.rows.len());
               self.rows.push(row);
            },
         }
      }
   }
}

#[derive(Default)]
struct Session {
   accounts: BTreeSet<i64>,
   deepest: i64,
   tokens: i64,
}

/// The whole log folded one delta at a time, so each refresh reads only
/// what was logged since the last.
#[derive(Default)]
pub struct Totals {
   rewrites: u64,
   accounts: String,
   last_id: i64,
   usage: Family<[String; USAGE_DIMENSIONS.len()], MetricsRow>,
   errors: Family<[String; 3], ErrorRow>,
   tools: Family<[String; 2], ToolRow>,
   insights: Family<[String; 3], InsightRow>,
   sessions: BTreeMap<(String, String), Session>,
}

impl Totals {
   async fn update(&mut self, db: &Db) -> misstep::Result<()> {
      let rewrites = rewrites();
      if rewrites != self.rewrites {
         *self = Self {
            rewrites,
            ..Self::default()
         };
      }
      let mut delta = db.metrics_since(self.last_id).await?;
      if self.last_id > 0 && delta.accounts != self.accounts {
         *self = Self {
            rewrites,
            ..Self::default()
         };
         delta = db.metrics_since(0).await?;
      }
      self.fold(delta);
      Ok(())
   }

   fn fold(&mut self, delta: MetricsDelta) {
      self.accounts = delta.accounts;
      self.last_id = delta.last_id;
      self.usage.fold(delta.usage, |row| row.dimensions.clone());
      self.errors.fold(delta.errors, |row| {
         [row.user.clone(), row.provider.clone(), row.kind.clone()]
      });
      self
         .tools
         .fold(delta.tools, |row| [row.user.clone(), row.tool.clone()]);
      self.insights.fold(delta.insights, |row| {
         [row.user.clone(), row.account.clone(), row.stop_reason.clone()]
      });
      for part in delta.sessions {
         let session = self.sessions.entry((part.user, part.session_key)).or_default();
         session.accounts.extend(part.account_id);
         session.deepest = session.deepest.max(part.deepest);
         session.tokens += part.tokens;
      }
   }

   fn session_rows(&self) -> Vec<SessionRow> {
      let mut users = BTreeMap::<&str, SessionRow>::new();
      for (&(ref user, _), session) in &self.sessions {
         let row = users.entry(user).or_insert_with(|| SessionRow {
            user: user.clone(),
            sessions: 0,
            deepest: 0,
            switches: 0,
            tokens_max: 0,
         });
         row.sessions += 1;
         row.deepest = row.deepest.max(session.deepest);
         row.switches += i64::try_from(session.accounts.len().saturating_sub(1)).unwrap_or(i64::MAX);
         row.tokens_max = row.tokens_max.max(session.tokens);
      }
      users.into_values().collect()
   }

   fn render(&self) -> String {
      let mut text = String::with_capacity(4096);
      let out = &mut text;
      render_usage(out, &self.usage.rows);
      counter(out, "slop_errors_total", "Failed requests by cause");
      for row in &self.errors.rows {
         let labels = [
            ("user", row.user.as_str()),
            ("provider", &row.provider),
            ("kind", &row.kind),
         ];
         sample(out, "slop_errors_total", &labels, row.count);
      }
      families(out, "counter", &TOOL_COUNTERS, &self.tools.rows, |row| {
         [("user", row.user.as_str()), ("tool", &row.tool)]
      });
      families(out, "counter", &INSIGHTS, &self.insights.rows, |row| {
         [
            ("user", row.user.as_str()),
            ("account", &row.account),
            ("stop_reason", &row.stop_reason),
         ]
      });
      families(out, "gauge", &SESSION_GAUGES, &self.session_rows(), |row| {
         [("user", row.user.as_str())]
      });
      text
   }
}


type Column<R, V> = (&'static str, &'static str, fn(&R) -> V);

fn families<R, V, const N: usize>(
   out: &mut String,
   kind: &str,
   table: &[Column<R, V>],
   rows: &[R],
   labels: fn(&R) -> [(&'static str, &str); N],
) where
   V: Display,
{
   for &(name, help, get) in table {
      family(out, name, kind, help);
      for row in rows {
         sample(out, name, &labels(row), get(row));
      }
   }
}

fn render_accounts(out: &mut String, accounts: &[AccountSnapshot], now: i64) {
   // An info metric keeps slow-moving identity off the sampled series while
   // still letting a dashboard group or weight by it, e.g.
   //   slop_account_utilization_ratio * on(account) group_left(plan) slop_account_info
   gauge(
      out,
      "slop_account_info",
      "Constant 1, carrying the account's plan and trusted access as labels",
   );
   for snap in accounts {
      let plan = snap.plan.as_deref().unwrap_or("unknown");
      let labels = [("plan", plan), ("trusted", bool_label(snap.trusted))];
      account(out, "slop_account_info", snap, &labels, 1.0_f64);
   }
   for (name, help, get) in ACCOUNT_GAUGES {
      gauge(out, name, help);
      for snap in accounts {
         if let Some(value) = get(snap, now) {
            account(out, name, snap, &[], value);
         }
      }
   }
   for (name, help, get) in WINDOW_GAUGES {
      gauge(out, name, help);
      for snap in accounts {
         let Some(usage) = snap.usage.as_ref() else {
            continue;
         };
         for window in &usage.windows {
            if let Some(value) = get(window, now) {
               account(out, name, snap, &[("window", &window.name)], value);
            }
         }
      }
   }
   gauge(
      out,
      "slop_account_model_utilization_ratio",
      "Fraction of a model's own sub-limit consumed. Measured against that \
         model's allowance rather than the account's, so it is not comparable \
         to slop_account_utilization_ratio",
   );
   gauge(
      out,
      "slop_account_model_window_reset_seconds",
      "Seconds until the model's own limit window rolls over",
   );
   for snap in accounts {
      let Some(usage) = snap.usage.as_ref() else {
         continue;
      };
      for window in &usage.model_windows {
         let labels = [
            ("window", window.window.as_str()),
            ("model", window.model.as_str()),
            ("active", window.is_active.map_or("unknown", bool_label)),
         ];
         account(
            out,
            "slop_account_model_utilization_ratio",
            snap,
            &labels,
            window.utilization,
         );
         if let Some(reset) = window.resets_at {
            account(
               out,
               "slop_account_model_window_reset_seconds",
               snap,
               &labels,
               (reset - now).max(0),
            );
         }
      }
   }
}

/// Measured weekly throughput puts a 20x account at 1.7-2x a 5x one, not the
/// 4x the plan names imply.
fn plan_capacity(snap: &AccountSnapshot) -> f64 {
   if let Some(capacity) = snap.usage.as_ref().and_then(|usage| usage.capacity) {
      return capacity;
   }
   match (snap.provider, snap.plan.as_deref()) {
      (Provider::Anthropic, Some("default_claude_max_5x")) => 0.5,
      _ => 1.0,
   }
}

fn render_usage(out: &mut String, rows: &[MetricsRow]) {
   families(out, "counter", &USAGE_COUNTERS, rows, usage_labels);
   counter(out, "slop_tokens_total", "Tokens by kind");
   for (kind, get) in TOKEN_KINDS {
      for row in rows {
         let mut labels = usage_labels(row).to_vec();
         labels.push(("kind", kind));
         sample(out, "slop_tokens_total", &labels, get(row));
      }
   }

   gauge(
      out,
      "slop_cache_hit_ratio",
      "Share of all recorded prompt tokens served from cache per label set",
   );
   for row in rows {
      sample(
         out,
         "slop_cache_hit_ratio",
         &usage_labels(row),
         cache_hit_ratio(
            row.input_tokens,
            row.cache_read_tokens,
            row.cache_write_tokens,
         ),
      );
   }
}

fn usage_labels(row: &MetricsRow) -> [(&'static str, &str); USAGE_DIMENSIONS.len()] {
   from_fn(|index| (USAGE_DIMENSIONS[index].0, row.dimensions[index].as_str()))
}

type AccountGauge = fn(&AccountSnapshot, i64) -> Option<f64>;
const ACCOUNT_GAUGES: [(&str, &str, AccountGauge); 6] = [
   (
      "slop_account_capacity",
      "Share of the provider's largest plan this account is worth, for \
         weighting a pooled average. Emitted only where quota is reported, so \
         a provider that reports none stays out of the average entirely",
      |snap, _| snap.usage.as_ref().map(|_| plan_capacity(snap)),
   ),
   (
      "slop_account_status",
      "Account state: 0 active, 1 cooling down, 2 disabled",
      |snap, _| Some(f64::from(snap.status)),
   ),
   (
      "slop_account_cooldown_seconds",
      "Seconds until the account is retried, 0 when active",
      |snap, _| Some(snap.cooldown_seconds as f64),
   ),
   (
      "slop_account_consecutive_fails",
      "Consecutive upstream failures feeding the backoff",
      |snap, _| Some(f64::from(snap.consecutive_fails)),
   ),
   (
      "slop_account_usage_age_seconds",
      "Age of the quota sample. Codex only reports quota on a served \
         response, so an idle account's figures stop advancing",
      |snap, now| {
         snap
            .usage
            .as_ref()
            .map(|usage| (now - usage.observed_at).max(0) as f64)
      },
   ),
   (
      "slop_account_locked",
      "1 when the provider has locked the account out until a window resets",
      |snap, _| snap.usage.as_ref().map(|usage| f64::from(usage.locked)),
   ),
];

type WindowGauge = fn(&UsageWindow, i64) -> Option<f64>;
const WINDOW_GAUGES: [(&str, &str, WindowGauge); 2] = [
   (
      "slop_account_utilization_ratio",
      "Fraction of the account's rolling limit window consumed",
      |window, _| Some(window.utilization),
   ),
   (
      "slop_account_window_reset_seconds",
      "Seconds until the account's limit window rolls over",
      |window, now| window.resets_at.map(|reset| (reset - now).max(0) as f64),
   ),
];

type UsageCounter = fn(&MetricsRow) -> f64;
const USAGE_COUNTERS: [(&str, &str, UsageCounter); 5] = [
   ("slop_requests_total", "Requests served", |row| {
      row.requests as f64
   }),
   ("slop_request_errors_total", "Requests that failed", |row| {
      row.errors as f64
   }),
   (
      "slop_request_duration_seconds_sum",
      "Total time served, divide by slop_requests_total for the mean",
      |row| row.duration_ms as f64 / 1000.0_f64,
   ),
   (
      "slop_cost_usd_total",
      "What the traffic actually cost",
      |row| row.cost_usd,
   ),
   (
      "slop_list_cost_usd_total",
      "What the same traffic would have cost at the model's list rate, so a \
         free tier is worth something rather than reading as zero spend",
      |row| row.list_cost_usd,
   ),
];

type ToolCounter = fn(&ToolRow) -> i64;
const TOOL_COUNTERS: [(&str, &str, ToolCounter); 2] = [
   (
      "slop_tool_turns_total",
      "Turns that used each tool. Names only, never their arguments. A turn \
         calling one tool repeatedly counts once",
      |row| row.count,
   ),
   (
      "slop_tool_errors_total",
      "Failed turns that used each tool. Names only, never their arguments or results",
      |row| row.errors,
   ),
];

type TokenGetter = fn(&MetricsRow) -> i64;
const TOKEN_KINDS: [(&str, TokenGetter); 5] = [
   ("input", |row| row.input_tokens),
   ("output", |row| row.output_tokens),
   ("cache_read", |row| row.cache_read_tokens),
   ("cache_write", |row| row.cache_write_tokens),
   ("reasoning", |row| row.reasoning_tokens),
];

type InsightGetter = fn(&InsightRow) -> i64;
const INSIGHTS: [(&str, &str, InsightGetter); 10] = [
   (
      "slop_stop_reason_total",
      "Requests by how the turn ended",
      |row| row.requests,
   ),
   (
      "slop_request_bytes_total",
      "Request bodies as the handler parsed them",
      |row| row.request_bytes,
   ),
   (
      "slop_response_bytes_total",
      "Response bytes relayed",
      |row| row.response_bytes,
   ),
   (
      "slop_turns_total",
      "Messages carried in, summed. Divide by requests for conversation depth",
      |row| row.turns,
   ),
   ("slop_images_total", "Image parts carried in", |row| {
      row.images
   }),
   (
      "slop_thinking_budget_total",
      "Thinking tokens asked for, against reasoning tokens actually spent",
      |row| row.thinking_budget,
   ),
   (
      "slop_tools_declared_total",
      "Tools offered to the model, summed. Every one costs prompt on every turn",
      |row| row.tools_declared,
   ),
   (
      "slop_attempts_total",
      "Upstream account claims, summed. Divide by slop_stop_reason_total for attempts per request",
      |row| row.attempts,
   ),
   (
      "slop_ttft_ms_total",
      "Time to first byte, summed. Divide by slop_ttft_samples_total",
      |row| row.ttft_ms,
   ),
   (
      "slop_ttft_samples_total",
      "Requests that produced a first byte",
      |row| row.ttft_samples,
   ),
];

type SessionGauge = fn(&SessionRow) -> i64;
const SESSION_GAUGES: [(&str, &str, SessionGauge); 4] = [
   (
      "slop_sessions",
      "Distinct conversations seen. Divide slop_requests_total by it for turns per session",
      |row| row.sessions,
   ),
   (
      "slop_session_account_switches",
      "Times a conversation moved to a different account. Each move re-bills \
         the whole prefix upstream, so this is what a cache hit rate collapse \
         looks like before the cost shows up",
      |row| row.switches,
   ),
   (
      "slop_session_tokens_max",
      "Tokens the largest single conversation has billed over its life, \
         counting every turn it ever resent rather than what fits in the \
         context now. Reasoning is excluded, being a subset of output",
      |row| row.tokens_max,
   ),
   (
      "slop_session_deepest_turn",
      "Longest conversation seen, in messages carried",
      |row| row.deepest,
   ),
];

fn gauge(out: &mut String, name: &str, help: &str) {
   family(out, name, "gauge", help);
}

fn counter(out: &mut String, name: &str, help: &str) {
   family(out, name, "counter", help);
}

fn family(out: &mut String, name: &str, kind: &str, help: &str) {
   let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

fn account(
   out: &mut String,
   name: &str,
   snap: &AccountSnapshot,
   extra: &[(&str, &str)],
   value: impl Display,
) {
   let mut labels = vec![
      ("provider", snap.provider.as_str()),
      ("account", snap.display.as_str()),
   ];
   labels.extend_from_slice(extra);
   sample(out, name, &labels, value);
}

fn sample(out: &mut String, name: &str, labels: &[(&str, &str)], value: impl Display) {
   out.push_str(name);
   out.push('{');
   for (index, &(key, text)) in labels.iter().enumerate() {
      if index > 0 {
         out.push(',');
      }
      let _ = write!(out, "{key}={}", label(text));
   }
   let _ = writeln!(out, "}} {value}");
}

const fn bool_label(value: bool) -> &'static str {
   if value { "true" } else { "false" }
}

/// Prometheus drops a label whose value is empty, so a query grouping on one
/// gets a different field set per series. Grafana's `organize` transformation
/// is single-frame, so the merge it depends on then silently does nothing and
/// the table renders raw `Value #A` columns.
fn label(text: &str) -> String {
   let text = if text.is_empty() { "unset" } else { text };
   format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::pool::{AccountUsage, ModelWindow};

   #[test]
   fn model_quota_metrics_keep_activity_and_matching_resets() {
      let now = 1_800_000_000_i64;
      let snapshot = AccountSnapshot {
         provider: Provider::Anthropic,
         display: "claude".into(),
         plan: None,
         trusted: false,
         status: 0,
         cooldown_seconds: 0,
         consecutive_fails: 0,
         usage: Some(AccountUsage {
            windows: vec![UsageWindow {
               name: "5h".into(),
               utilization: 1.0,
               resets_at: Some(now + 180),
            }],
            model_windows: vec![
               ModelWindow {
                  model: "fable".into(),
                  window: "7d".into(),
                  utilization: 0.25,
                  is_active: Some(true),
                  resets_at: Some(now + 86400),
               },
               ModelWindow {
                  model: "dormant".into(),
                  window: "7d".into(),
                  utilization: 1.0,
                  is_active: Some(false),
                  resets_at: Some(now - 1),
               },
               ModelWindow {
                  model: "unknown".into(),
                  window: "7d".into(),
                  utilization: 0.12,
                  is_active: None,
                  resets_at: None,
               },
            ],
            ..AccountUsage::default()
         }),
      };
      let mut out = String::new();
      render_accounts(&mut out, &[snapshot], now);
      for (model, active, utilization, reset) in [
         ("fable", "true", "0.25", Some("86400")),
         ("dormant", "false", "1", Some("0")),
         ("unknown", "unknown", "0.12", None),
      ] {
         let labels = format!(
            "{{provider=\"anthropic\",account=\"claude\",window=\"7d\",model=\"{model}\",active=\"{active}\"}}"
         );
         assert!(out.contains(&format!(
            "slop_account_model_utilization_ratio{labels} {utilization}\n"
         )));
         let metric = format!("slop_account_model_window_reset_seconds{labels}");
         if let Some(reset) = reset {
            assert!(out.contains(&format!("{metric} {reset}\n")));
         } else {
            assert!(!out.contains(&metric));
         }
      }
      assert!(!out.contains(
         "slop_account_utilization_ratio{provider=\"anthropic\",account=\"claude\",window=\"7d\"}"
      ));
      assert!(out.contains("slop_account_window_reset_seconds{provider=\"anthropic\",account=\"claude\",window=\"5h\"} 180\n"));
   }
}
