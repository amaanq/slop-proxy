use std::fmt::Write;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};

use super::AppState;
use crate::pool::AccountSnapshot;

pub async fn metrics(State(state): State<AppState>) -> Response {
    let mut accounts = state.codex.snapshot().await;
    accounts.extend(state.anthropic.snapshot().await);

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
        "{name}{{user={},account={},provider={},requested_model={},model={},effort={},dialect={}}} {value}",
        quote(&r.user),
        quote(&r.account),
        quote(&r.provider),
        quote(&r.requested_model),
        quote(&r.model),
        quote(&r.effort),
        quote(&r.dialect),
    );
}

fn bool_label(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
