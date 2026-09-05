use std::fmt::Display;
use std::fmt::Write as _;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse as _;
use axum::response::Response;

use super::AppState;
use crate::clock;
use crate::db::usage::{ErrorRow, InsightRow, MetricsRow, SessionRow, ToolRow};
use crate::pool::{AccountSnapshot, UsageWindow};
use crate::provider::Provider;

pub async fn metrics(State(state): State<AppState>) -> Response {
   let accounts = state.pools.snapshots().await;

   let mut out = String::with_capacity(4096);
   render_accounts(&mut out, &accounts, clock::unix_now());
   match state.db.usage_metrics().await {
      Ok(rows) => render_usage(&mut out, &rows),
      Err(err) => tracing::error!("reading usage metrics: {err}"),
   }
   match state.db.error_metrics().await {
      Ok(rows) => render_errors(&mut out, &rows),
      Err(err) => tracing::error!("reading error metrics: {err}"),
   }
   match state.db.tool_metrics().await {
      Ok(rows) => render_tools(&mut out, &rows),
      Err(err) => tracing::error!("reading tool metrics: {err}"),
   }
   match state.db.insight_metrics().await {
      Ok(rows) => render_insights(&mut out, &rows),
      Err(err) => tracing::error!("reading insight metrics: {err}"),
   }
   match state.db.session_metrics().await {
      Ok(rows) => render_sessions(&mut out, &rows),
      Err(err) => tracing::error!("reading session metrics: {err}"),
   }

   ([(CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
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
   match (snap.provider, snap.plan.as_deref()) {
      (Provider::Anthropic, Some("default_claude_max_5x")) => 0.5,
      _ => 1.0,
   }
}

fn render_usage(out: &mut String, rows: &[MetricsRow]) {
   for (name, help, get) in USAGE_COUNTERS {
      counter(out, name, help);
      for row in rows {
         sample(out, name, &usage_labels(row), get(row));
      }
   }
   counter(out, "slop_tokens_total", "Tokens by kind");
   for (kind, get) in TOKEN_KINDS {
      for row in rows {
         let labels = [
            ("user", row.user.as_str()),
            ("account", &row.account),
            ("provider", &row.provider),
            ("requested_model", &row.requested_model),
            ("model", &row.model),
            ("effort", &row.effort),
            ("dialect", &row.dialect),
            ("kind", kind),
         ];
         sample(out, "slop_tokens_total", &labels, get(row));
      }
   }
}

fn usage_labels(row: &MetricsRow) -> [(&'static str, &str); 8] {
   [
      ("user", &row.user),
      ("account", &row.account),
      ("provider", &row.provider),
      ("requested_model", &row.requested_model),
      ("model", &row.model),
      ("effort", &row.effort),
      ("service_tier", &row.service_tier),
      ("dialect", &row.dialect),
   ]
}

fn render_errors(out: &mut String, rows: &[ErrorRow]) {
   counter(out, "slop_errors_total", "Failed requests by cause");
   for row in rows {
      let labels = [
         ("user", row.user.as_str()),
         ("provider", &row.provider),
         ("kind", &row.kind),
      ];
      sample(out, "slop_errors_total", &labels, row.count);
   }
}

fn render_tools(out: &mut String, rows: &[ToolRow]) {
   counter(
      out,
      "slop_tool_turns_total",
      "Turns that used each tool. Names only, never their arguments. A turn \
         calling one tool repeatedly counts once",
   );
   for row in rows {
      let labels = [("user", row.user.as_str()), ("tool", &row.tool)];
      sample(out, "slop_tool_turns_total", &labels, row.count);
   }
}

fn render_insights(out: &mut String, rows: &[InsightRow]) {
   for (name, help, get) in INSIGHTS {
      counter(out, name, help);
      for row in rows {
         let labels = [
            ("user", row.user.as_str()),
            ("account", &row.account),
            ("stop_reason", &row.stop_reason),
         ];
         sample(out, name, &labels, get(row));
      }
   }
}

fn render_sessions(out: &mut String, rows: &[SessionRow]) {
   for (name, help, get) in SESSION_GAUGES {
      gauge(out, name, help);
      for row in rows {
         sample(out, name, &[("user", &row.user)], get(row));
      }
   }
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

type TokenGetter = fn(&MetricsRow) -> i64;
const TOKEN_KINDS: [(&str, TokenGetter); 5] = [
   ("input", |row| row.input_tokens),
   ("output", |row| row.output_tokens),
   ("cache_read", |row| row.cache_read_tokens),
   ("cache_write", |row| row.cache_write_tokens),
   ("reasoning", |row| row.reasoning_tokens),
];

type InsightGetter = fn(&InsightRow) -> i64;
const INSIGHTS: [(&str, &str, InsightGetter); 9] = [
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
   let text = if text.is_empty() { "none" } else { text };
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
