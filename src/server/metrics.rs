use std::fmt::Write;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};

use super::AppState;
use crate::pool::AccountSnapshot;
use crate::provider::Provider;

pub async fn metrics(State(state): State<AppState>) -> Response {
    let mut accounts = state.codex.snapshot().await;
    accounts.extend(state.anthropic.snapshot().await);
    accounts.extend(state.gemini.snapshot().await);

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
    gauge_header(
        out,
        "slop_account_info",
        "Constant 1, carrying the account's plan and trusted access as labels",
    );
    for a in accounts {
        let plan = a.plan.as_deref().unwrap_or("unknown");
        line(
            out,
            "slop_account_info",
            a,
            &[("plan", plan), ("trusted", bool_label(a.trusted))],
            1.0,
        );
    }
    gauge_header(
        out,
        "slop_account_capacity",
        "Share of the provider's largest plan this account is worth, for \
         weighting a pooled average. Emitted only where quota is reported, so \
         a provider that reports none stays out of the average entirely",
    );
    for a in accounts {
        if a.usage.is_none() {
            continue;
        }
        line(out, "slop_account_capacity", a, &[], plan_capacity(a));
    }
    gauge_header(
        out,
        "slop_account_status",
        "Account state: 0 active, 1 cooling down, 2 disabled",
    );
    for a in accounts {
        line(out, "slop_account_status", a, &[], a.status as f64);
    }
    gauge_header(
        out,
        "slop_account_cooldown_seconds",
        "Seconds until the account is retried, 0 when active",
    );
    for a in accounts {
        line(
            out,
            "slop_account_cooldown_seconds",
            a,
            &[],
            a.cooldown_seconds as f64,
        );
    }
    gauge_header(
        out,
        "slop_account_consecutive_fails",
        "Consecutive upstream failures feeding the backoff",
    );
    for a in accounts {
        line(
            out,
            "slop_account_consecutive_fails",
            a,
            &[],
            a.consecutive_fails as f64,
        );
    }
    gauge_header(
        out,
        "slop_account_utilization_ratio",
        "Fraction of the account's rolling limit window consumed",
    );
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        for w in &usage.windows {
            line(
                out,
                "slop_account_utilization_ratio",
                a,
                &[("window", &w.name)],
                w.utilization,
            );
        }
    }
    gauge_header(
        out,
        "slop_account_model_utilization_ratio",
        "Fraction of a model's own sub-limit consumed. Measured against that \
         model's allowance rather than the account's, so it is not comparable \
         to slop_account_utilization_ratio",
    );
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        for w in &usage.model_windows {
            line(
                out,
                "slop_account_model_utilization_ratio",
                a,
                &[("window", &w.window), ("model", &w.model)],
                w.utilization,
            );
        }
    }
    gauge_header(
        out,
        "slop_account_window_reset_seconds",
        "Seconds until the account's limit window rolls over",
    );
    let now = crate::clock::unix_now();
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        for w in &usage.windows {
            let Some(resets_at) = w.resets_at else {
                continue;
            };
            line(
                out,
                "slop_account_window_reset_seconds",
                a,
                &[("window", &w.name)],
                (resets_at - now).max(0) as f64,
            );
        }
    }
    gauge_header(
        out,
        "slop_account_usage_age_seconds",
        "Age of the quota sample. Codex only reports quota on a served \
         response, so an idle account's figures stop advancing",
    );
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        line(
            out,
            "slop_account_usage_age_seconds",
            a,
            &[],
            (now - usage.observed_at).max(0) as f64,
        );
    }
    gauge_header(
        out,
        "slop_account_locked",
        "1 when the provider has locked the account out until a window resets",
    );
    for a in accounts {
        let Some(usage) = &a.usage else { continue };
        line(out, "slop_account_locked", a, &[], f64::from(usage.locked));
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

fn render_usage(out: &mut String, rows: &[crate::db::usage::MetricsRow]) {
    counter_header(out, "slop_requests_total", "Requests served");
    for r in rows {
        usage_line(out, "slop_requests_total", r, r.requests as f64);
    }
    counter_header(out, "slop_request_errors_total", "Requests that failed");
    for r in rows {
        usage_line(out, "slop_request_errors_total", r, r.errors as f64);
    }
    counter_header(
        out,
        "slop_request_duration_seconds_sum",
        "Total time served, divide by slop_requests_total for the mean",
    );
    for r in rows {
        usage_line(
            out,
            "slop_request_duration_seconds_sum",
            r,
            r.duration_ms as f64 / 1000.0,
        );
    }
    counter_header(
        out,
        "slop_cost_usd_total",
        "What the same traffic would have cost at the provider's list rate",
    );
    for r in rows {
        usage_line(out, "slop_cost_usd_total", r, r.cost_usd);
    }
    counter_header(out, "slop_tokens_total", "Tokens by kind");
    for (kind, get) in TOKEN_KINDS {
        for r in rows {
            let _ = writeln!(
                out,
                "slop_tokens_total{{user={},account={},provider={},requested_model={},model={},effort={},dialect={},kind=\"{kind}\"}} {}",
                quote(&r.user),
                quote(&r.account),
                quote(&r.provider),
                quote(&r.requested_model),
                quote(&r.model),
                quote(&r.effort),
                quote(&r.dialect),
                get(r),
            );
        }
    }
}

fn render_errors(out: &mut String, rows: &[crate::db::usage::ErrorRow]) {
    counter_header(out, "slop_errors_total", "Failed requests by cause");
    for r in rows {
        let _ = writeln!(
            out,
            "slop_errors_total{{user={},provider={},kind={}}} {}",
            quote(&r.user),
            quote(&r.provider),
            quote(&r.kind),
            r.count,
        );
    }
}

fn render_tools(out: &mut String, rows: &[crate::db::usage::ToolRow]) {
    counter_header(
        out,
        "slop_tool_turns_total",
        "Turns that used each tool. Names only, never their arguments. A turn \
         calling one tool repeatedly counts once",
    );
    for r in rows {
        let _ = writeln!(
            out,
            "slop_tool_turns_total{{user={},tool={}}} {}",
            quote(&r.user),
            quote(&r.tool),
            r.count,
        );
    }
}

fn render_insights(out: &mut String, rows: &[crate::db::usage::InsightRow]) {
    let line = |out: &mut String, name: &str, r: &crate::db::usage::InsightRow, v: i64| {
        let _ = writeln!(
            out,
            "{name}{{user={},account={},stop_reason={}}} {v}",
            quote(&r.user),
            quote(&r.account),
            quote(&r.stop_reason),
        );
    };
    for (name, help, get) in INSIGHTS {
        counter_header(out, name, help);
        for r in rows {
            line(out, name, r, get(r));
        }
    }
}

fn render_sessions(out: &mut String, rows: &[crate::db::usage::SessionRow]) {
    gauge_header(
        out,
        "slop_sessions",
        "Distinct conversations seen. Divide slop_requests_total by it for turns per session",
    );
    for r in rows {
        let _ = writeln!(out, "slop_sessions{{user={}}} {}", quote(&r.user), r.sessions);
    }
    gauge_header(
        out,
        "slop_session_account_switches",
        "Times a conversation moved to a different account. Each move re-bills \
         the whole prefix upstream, so this is what a cache hit rate collapse \
         looks like before the cost shows up",
    );
    for r in rows {
        let _ = writeln!(
            out,
            "slop_session_account_switches{{user={}}} {}",
            quote(&r.user),
            r.switches
        );
    }
    gauge_header(
        out,
        "slop_session_deepest_turn",
        "Longest conversation seen, in messages carried",
    );
    for r in rows {
        let _ = writeln!(
            out,
            "slop_session_deepest_turn{{user={}}} {}",
            quote(&r.user),
            r.deepest
        );
    }
}

type InsightGetter = fn(&crate::db::usage::InsightRow) -> i64;
const INSIGHTS: [(&str, &str, InsightGetter); 9] = [
    ("slop_stop_reason_total", "Requests by how the turn ended", |r| r.requests),
    ("slop_request_bytes_total", "Request bodies as the handler parsed them", |r| r.request_bytes),
    ("slop_response_bytes_total", "Response bytes relayed", |r| r.response_bytes),
    ("slop_turns_total", "Messages carried in, summed. Divide by requests for conversation depth", |r| r.turns),
    ("slop_images_total", "Image parts carried in", |r| r.images),
    ("slop_thinking_budget_total", "Thinking tokens asked for, against reasoning tokens actually spent", |r| r.thinking_budget),
    ("slop_tools_declared_total", "Tools offered to the model, summed. Every one costs prompt on every turn", |r| r.tools_declared),
    ("slop_ttft_ms_total", "Time to first byte, summed. Divide by slop_ttft_samples_total", |r| r.ttft_ms),
    ("slop_ttft_samples_total", "Requests that produced a first byte", |r| r.ttft_samples),
];

type TokenGetter = fn(&crate::db::usage::MetricsRow) -> i64;
const TOKEN_KINDS: [(&str, TokenGetter); 5] = [
    ("input", |r| r.input_tokens),
    ("output", |r| r.output_tokens),
    ("cache_read", |r| r.cache_read_tokens),
    ("cache_write", |r| r.cache_write_tokens),
    ("reasoning", |r| r.reasoning_tokens),
];

fn gauge_header(out: &mut String, name: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge");
}

fn counter_header(out: &mut String, name: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter");
}

fn line(out: &mut String, name: &str, a: &AccountSnapshot, extra: &[(&str, &str)], value: f64) {
    let mut labels = format!("provider=\"{}\",account={}", a.provider, quote(&a.display));
    for (k, v) in extra {
        let _ = write!(labels, ",{k}={}", quote(v));
    }
    let _ = writeln!(out, "{name}{{{labels}}} {value}");
}

fn usage_line(out: &mut String, name: &str, r: &crate::db::usage::MetricsRow, value: f64) {
    let _ = writeln!(
        out,
        "{name}{{user={},account={},provider={},requested_model={},model={},effort={},service_tier={},dialect={}}} {value}",
        quote(&r.user),
        quote(&r.account),
        quote(&r.provider),
        quote(&r.requested_model),
        quote(&r.model),
        quote(&r.effort),
        quote(&r.service_tier),
        quote(&r.dialect),
    );
}

fn bool_label(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
