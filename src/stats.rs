use anyhow::{bail, Result};
use serde_json::json;

use crate::db::usage::{UsageAgg, UsageDim};
use crate::db::Db;

pub async fn run(db: &Db, since: Option<String>, until: Option<String>) -> Result<()> {
    let now = chrono::Utc::now().timestamp();
    let since_ts = match &since {
        Some(s) => parse_time(s, now)?,
        None => 0,
    };
    let until_ts = match &until {
        Some(s) => parse_time(s, now)?,
        None => now + 1,
    };

    let totals = db.usage_totals(since_ts, until_ts).await?;
    let by_user = db.usage_by(UsageDim::User, since_ts, until_ts).await?;
    let by_account = db.usage_by(UsageDim::Account, since_ts, until_ts).await?;
    let by_model = db.usage_by(UsageDim::Model, since_ts, until_ts).await?;

    let fmt = |a: &UsageAgg| {
        json!({
            "requests": a.requests,
            "errors": a.errors,
            "input_tokens": a.input_tokens,
            "output_tokens": a.output_tokens,
            "cache_read_tokens": a.cache_read_tokens,
            "reasoning_tokens": a.reasoning_tokens,
        })
    };
    let keyed = |rows: &[UsageAgg]| {
        rows.iter()
            .map(|a| {
                let mut v = fmt(a);
                v["name"] = json!(a.key);
                v
            })
            .collect::<Vec<_>>()
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "since": chrono::DateTime::from_timestamp(since_ts, 0).map(|d| d.to_rfc3339()),
            "until": chrono::DateTime::from_timestamp(until_ts, 0).map(|d| d.to_rfc3339()),
            "totals": fmt(&totals),
            "by_user": keyed(&by_user),
            "by_account": keyed(&by_account),
            "by_model": keyed(&by_model),
        }))?
    );
    Ok(())
}

fn parse_time(s: &str, now: i64) -> Result<i64> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp());
    }
    if let Some(rest) = s.strip_suffix(['h', 'd', 'm', 'w'])
        && let Ok(n) = rest.parse::<i64>() {
            let secs = match s.chars().last().unwrap() {
                'm' => n * 60,
                'h' => n * 3600,
                'd' => n * 86400,
                'w' => n * 7 * 86400,
                _ => unreachable!(),
            };
            return Ok(now - secs);
        }
    bail!("cannot parse time {s:?}; use RFC3339 or 30m/24h/7d/2w");
}
