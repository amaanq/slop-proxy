use std::fmt::{Display, Write};

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};

use super::AppState;
use crate::db::usage::{ErrorRow, InsightRow, MetricsRow, SessionRow, ToolRow};
use crate::pool::{AccountSnapshot, UsageWindow};
use crate::provider::Provider;

pub async fn metrics(State(state): State<AppState>) -> Response {
    let accounts = state.pools.snapshots().await;

    let mut out = String::with_capacity(4096);
    render_accounts(&mut out, &accounts);
    match state.db.usage_metrics().await {
        Ok(rows) => render_usage(&mut out, &rows),
        Err(e) => tracing::error!("reading usage metrics: {e}"),
    }
    match state.db.error_metrics().await {
        Ok(rows) => render_errors(&mut out, &rows),
        Err(e) => tracing::error!("reading error metrics: {e}"),
    }
    match state.db.tool_metrics().await {
        Ok(rows) => render_tools(&mut out, &rows),
        Err(e) => tracing::error!("reading tool metrics: {e}"),
    }
    match state.db.insight_metrics().await {
        Ok(rows) => render_insights(&mut out, &rows),
        Err(e) => tracing::error!("reading insight metrics: {e}"),
    }
    match state.db.session_metrics().await {
        Ok(rows) => render_sessions(&mut out, &rows),
        Err(e) => tracing::error!("reading session metrics: {e}"),
    }

    ([(CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
}

fn render_accounts(out: &mut String, accounts: &[AccountSnapshot]) {
    // An info metric keeps slow-moving identity off the sampled series while
    // still letting a dashboard group or weight by it, e.g.
    //   slop_account_utilization_ratio * on(account) group_left(plan) slop_account_info
    gauge(
        out,
        "slop_account_info",
        "Constant 1, carrying the account's plan and trusted access as labels",
    );
    for a in accounts {
        let plan = a.plan.as_deref().unwrap_or("unknown");
        let l = [("plan", plan), ("trusted", bool_label(a.trusted))];
        account(out, "slop_account_info", a, &l, 1.0);
    }
    let now = crate::clock::unix_now();
    for (name, help, get) in ACCOUNT_GAUGES {
        gauge(out, name, help);
        for a in accounts {
            if let Some(v) = get(a, now) {
                account(out, name, a, &[], v);
            }
        }
    }
    for (name, help, get) in WINDOW_GAUGES {
        gauge(out, name, help);
        for a in accounts {
            let Some(usage) = &a.usage else { continue };
            for w in &usage.windows {
                if let Some(v) = get(w, now) {
                    account(out, name, a, &[("window", &w.name)], v);
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
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        for w in &usage.model_windows {
            let l = [("window", w.window.as_str()), ("model", w.model.as_str())];
            account(
                out,
                "slop_account_model_utilization_ratio",
                a,
                &l,
                w.utilization,
            );
        }
    }
}

/// Measured weekly throughput puts a 20x account at 1.7-2x a 5x one, not the
/// 4x the plan names imply.
fn plan_capacity(a: &AccountSnapshot) -> f64 {
    match (a.provider, a.plan.as_deref()) {
        (Provider::Anthropic, Some("default_claude_max_5x")) => 0.5,
        _ => 1.0,
    }
}

fn render_usage(out: &mut String, rows: &[MetricsRow]) {
    for (name, help, get) in USAGE_COUNTERS {
        counter(out, name, help);
        for r in rows {
            sample(out, name, &usage_labels(r), get(r));
        }
    }
    counter(out, "slop_tokens_total", "Tokens by kind");
    for (kind, get) in TOKEN_KINDS {
        for r in rows {
            let l = [
                ("user", r.user.as_str()),
                ("account", &r.account),
                ("provider", &r.provider),
                ("requested_model", &r.requested_model),
                ("model", &r.model),
                ("effort", &r.effort),
                ("dialect", &r.dialect),
                ("kind", kind),
            ];
            sample(out, "slop_tokens_total", &l, get(r));
        }
    }
}

fn usage_labels(r: &MetricsRow) -> [(&'static str, &str); 8] {
    [
        ("user", &r.user),
        ("account", &r.account),
        ("provider", &r.provider),
        ("requested_model", &r.requested_model),
        ("model", &r.model),
        ("effort", &r.effort),
        ("service_tier", &r.service_tier),
        ("dialect", &r.dialect),
    ]
}

fn render_errors(out: &mut String, rows: &[ErrorRow]) {
    counter(out, "slop_errors_total", "Failed requests by cause");
    for r in rows {
        let l = [
            ("user", r.user.as_str()),
            ("provider", &r.provider),
            ("kind", &r.kind),
        ];
        sample(out, "slop_errors_total", &l, r.count);
    }
}

fn render_tools(out: &mut String, rows: &[ToolRow]) {
    counter(
        out,
        "slop_tool_turns_total",
        "Turns that used each tool. Names only, never their arguments. A turn \
         calling one tool repeatedly counts once",
    );
    for r in rows {
        let l = [("user", r.user.as_str()), ("tool", &r.tool)];
        sample(out, "slop_tool_turns_total", &l, r.count);
    }
}

fn render_insights(out: &mut String, rows: &[InsightRow]) {
    for (name, help, get) in INSIGHTS {
        counter(out, name, help);
        for r in rows {
            let l = [
                ("user", r.user.as_str()),
                ("account", &r.account),
                ("stop_reason", &r.stop_reason),
            ];
            sample(out, name, &l, get(r));
        }
    }
}

fn render_sessions(out: &mut String, rows: &[SessionRow]) {
    for (name, help, get) in SESSION_GAUGES {
        gauge(out, name, help);
        for r in rows {
            sample(out, name, &[("user", &r.user)], get(r));
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
        |a, _| a.usage.as_ref().map(|_| plan_capacity(a)),
    ),
    (
        "slop_account_status",
        "Account state: 0 active, 1 cooling down, 2 disabled",
        |a, _| Some(a.status as f64),
    ),
    (
        "slop_account_cooldown_seconds",
        "Seconds until the account is retried, 0 when active",
        |a, _| Some(a.cooldown_seconds as f64),
    ),
    (
        "slop_account_consecutive_fails",
        "Consecutive upstream failures feeding the backoff",
        |a, _| Some(a.consecutive_fails as f64),
    ),
    (
        "slop_account_usage_age_seconds",
        "Age of the quota sample. Codex only reports quota on a served \
         response, so an idle account's figures stop advancing",
        |a, now| {
            a.usage
                .as_ref()
                .map(|u| (now - u.observed_at).max(0) as f64)
        },
    ),
    (
        "slop_account_locked",
        "1 when the provider has locked the account out until a window resets",
        |a, _| a.usage.as_ref().map(|u| f64::from(u.locked)),
    ),
];

type WindowGauge = fn(&UsageWindow, i64) -> Option<f64>;
const WINDOW_GAUGES: [(&str, &str, WindowGauge); 2] = [
    (
        "slop_account_utilization_ratio",
        "Fraction of the account's rolling limit window consumed",
        |w, _| Some(w.utilization),
    ),
    (
        "slop_account_window_reset_seconds",
        "Seconds until the account's limit window rolls over",
        |w, now| w.resets_at.map(|t| (t - now).max(0) as f64),
    ),
];

type UsageCounter = fn(&MetricsRow) -> f64;
const USAGE_COUNTERS: [(&str, &str, UsageCounter); 5] = [
    ("slop_requests_total", "Requests served", |r| {
        r.requests as f64
    }),
    ("slop_request_errors_total", "Requests that failed", |r| {
        r.errors as f64
    }),
    (
        "slop_request_duration_seconds_sum",
        "Total time served, divide by slop_requests_total for the mean",
        |r| r.duration_ms as f64 / 1000.0,
    ),
    (
        "slop_cost_usd_total",
        "What the traffic actually cost",
        |r| r.cost_usd,
    ),
    (
        "slop_list_cost_usd_total",
        "What the same traffic would have cost at the model's list rate, so a \
         free tier is worth something rather than reading as zero spend",
        |r| r.list_cost_usd,
    ),
];

type TokenGetter = fn(&MetricsRow) -> i64;
const TOKEN_KINDS: [(&str, TokenGetter); 5] = [
    ("input", |r| r.input_tokens),
    ("output", |r| r.output_tokens),
    ("cache_read", |r| r.cache_read_tokens),
    ("cache_write", |r| r.cache_write_tokens),
    ("reasoning", |r| r.reasoning_tokens),
];

type InsightGetter = fn(&InsightRow) -> i64;
const INSIGHTS: [(&str, &str, InsightGetter); 9] = [
    (
        "slop_stop_reason_total",
        "Requests by how the turn ended",
        |r| r.requests,
    ),
    (
        "slop_request_bytes_total",
        "Request bodies as the handler parsed them",
        |r| r.request_bytes,
    ),
    ("slop_response_bytes_total", "Response bytes relayed", |r| {
        r.response_bytes
    }),
    (
        "slop_turns_total",
        "Messages carried in, summed. Divide by requests for conversation depth",
        |r| r.turns,
    ),
    ("slop_images_total", "Image parts carried in", |r| r.images),
    (
        "slop_thinking_budget_total",
        "Thinking tokens asked for, against reasoning tokens actually spent",
        |r| r.thinking_budget,
    ),
    (
        "slop_tools_declared_total",
        "Tools offered to the model, summed. Every one costs prompt on every turn",
        |r| r.tools_declared,
    ),
    (
        "slop_ttft_ms_total",
        "Time to first byte, summed. Divide by slop_ttft_samples_total",
        |r| r.ttft_ms,
    ),
    (
        "slop_ttft_samples_total",
        "Requests that produced a first byte",
        |r| r.ttft_samples,
    ),
];

type SessionGauge = fn(&SessionRow) -> i64;
const SESSION_GAUGES: [(&str, &str, SessionGauge); 4] = [
    (
        "slop_sessions",
        "Distinct conversations seen. Divide slop_requests_total by it for turns per session",
        |r| r.sessions,
    ),
    (
        "slop_session_account_switches",
        "Times a conversation moved to a different account. Each move re-bills \
         the whole prefix upstream, so this is what a cache hit rate collapse \
         looks like before the cost shows up",
        |r| r.switches,
    ),
    (
        "slop_session_tokens_max",
        "Tokens the largest single conversation has billed over its life, \
         counting every turn it ever resent rather than what fits in the \
         context now. Reasoning is excluded, being a subset of output",
        |r| r.tokens_max,
    ),
    (
        "slop_session_deepest_turn",
        "Longest conversation seen, in messages carried",
        |r| r.deepest,
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
    a: &AccountSnapshot,
    extra: &[(&str, &str)],
    value: impl Display,
) {
    let mut labels = vec![
        ("provider", a.provider.as_str()),
        ("account", a.display.as_str()),
    ];
    labels.extend_from_slice(extra);
    sample(out, name, &labels, value);
}

fn sample(out: &mut String, name: &str, labels: &[(&str, &str)], value: impl Display) {
    out.push_str(name);
    out.push('{');
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{k}={}", label(v));
    }
    let _ = writeln!(out, "}} {value}");
}

fn bool_label(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

/// Prometheus drops a label whose value is empty, so a query grouping on one
/// gets a different field set per series. Grafana's `organize` transformation
/// is single-frame, so the merge it depends on then silently does nothing and
/// the table renders raw `Value #A` columns.
fn label(s: &str) -> String {
    let s = if s.is_empty() { "none" } else { s };
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
