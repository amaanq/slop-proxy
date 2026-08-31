use eyre::Result;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::Serialize;

use super::Db;
use super::tokens::TokenLimits;

#[derive(Debug, Clone, Default)]
pub struct UsageRecord {
    pub meter_id: Option<i64>,
    pub token_id: Option<i64>,
    pub user: String,
    pub account_id: Option<i64>,
    pub dialect: &'static str,
    pub requested_model: String,
    pub upstream_model: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub reasoning_tokens: i64,
    pub status: i64,
    pub error_kind: Option<String>,
    pub duration_ms: Option<i64>,
}

#[derive(Debug, Serialize, Default)]
pub struct UsageAgg {
    pub key: String,
    pub requests: i64,
    pub errors: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Debug)]
pub enum AdmissionError {
    RequestLimit { retry_after: i64 },
    TokenLimit { retry_after: i64 },
}

#[derive(Debug)]
pub struct Admission {
    pub meter_id: i64,
    pub request_limit: Option<i64>,
    pub requests_remaining: Option<i64>,
    pub token_limit: Option<i64>,
    pub tokens_remaining: Option<i64>,
    pub reset_after: i64,
    pub slowdown_ms: i64,
}

#[derive(Debug, Serialize)]
pub struct TokenMeter {
    pub id: i64,
    pub user: String,
    pub prefix: String,
    pub window_seconds: i64,
    pub request_limit: Option<i64>,
    pub requests: i64,
    pub requests_remaining: Option<i64>,
    pub token_limit: Option<i64>,
    pub tokens: i64,
    pub tokens_remaining: Option<i64>,
    pub slowdown_ms: i64,
    pub reset_after_seconds: i64,
}

impl Db {
    pub async fn log_usage(&self, r: &UsageRecord) -> Result<()> {
        let mut conn = self.0.lock().await;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO usage_log (token_id, user, account_id, dialect, requested_model, upstream_model,
               input_tokens, output_tokens, cache_read_tokens, reasoning_tokens, status, error_kind, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                r.token_id,
                r.user,
                r.account_id,
                r.dialect,
                r.requested_model,
                r.upstream_model,
                r.input_tokens,
                r.output_tokens,
                r.cache_read_tokens,
                r.reasoning_tokens,
                r.status,
                r.error_kind,
                r.duration_ms,
            ],
        )?;
        if let Some(meter_id) = r.meter_id {
            tx.execute(
                "UPDATE api_meter SET input_tokens = ?2, output_tokens = ?3 WHERE id = ?1",
                params![meter_id, r.input_tokens, r.output_tokens],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Admission is persisted before upstream dispatch inside an IMMEDIATE
    /// transaction, so concurrent requests cannot race past the request
    /// limit. Token counts settle later via log_usage, which means a request
    /// can overshoot the token limit once and the window absorbs it.
    pub async fn admit_token(
        &self,
        token_id: i64,
        limits: &TokenLimits,
    ) -> Result<std::result::Result<Admission, AdmissionError>> {
        let now = crate::clock::unix_now_ms();
        let window_ms = limits.window_seconds.saturating_mul(1000);
        let since = now.saturating_sub(window_ms);
        let mut conn = self.0.lock().await;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (requests, tokens, oldest_request, oldest_tokens): (
            i64,
            i64,
            Option<i64>,
            Option<i64>,
        ) = tx.query_row(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens + output_tokens), 0), MIN(ts_ms),
                    MIN(CASE WHEN input_tokens + output_tokens > 0 THEN ts_ms END)
             FROM api_meter WHERE token_id = ?1 AND ts_ms > ?2",
            params![token_id, since],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;

        if limits.requests.is_some_and(|limit| requests >= limit) {
            let retry_after = retry_after(oldest_request, window_ms, now);
            return Ok(Err(AdmissionError::RequestLimit { retry_after }));
        }
        if limits.tokens.is_some_and(|limit| tokens >= limit) {
            let retry_after = retry_after(oldest_tokens.or(oldest_request), window_ms, now);
            return Ok(Err(AdmissionError::TokenLimit { retry_after }));
        }

        tx.execute(
            "INSERT INTO api_meter (token_id, ts_ms) VALUES (?1, ?2)",
            params![token_id, now],
        )?;
        let meter_id = tx.last_insert_rowid();
        tx.commit()?;

        Ok(Ok(Admission {
            meter_id,
            request_limit: limits.requests,
            requests_remaining: limits.requests.map(|limit| (limit - requests - 1).max(0)),
            token_limit: limits.tokens,
            tokens_remaining: limits.tokens.map(|limit| (limit - tokens).max(0)),
            reset_after: retry_after(oldest_request.or(Some(now)), window_ms, now),
            slowdown_ms: limits.slowdown_ms,
        }))
    }

    pub async fn token_meter(&self, key: &str) -> Result<Option<TokenMeter>> {
        let now = crate::clock::unix_now_ms();
        let id = key.parse::<i64>().unwrap_or(-1);
        let conn = self.0.lock().await;
        let token = conn
            .query_row(
                "SELECT id, user, token_prefix, request_limit, token_limit, window_seconds, slowdown_ms
                 FROM api_tokens WHERE id = ?1 OR token_prefix = ?2 ORDER BY id LIMIT 1",
                params![id, key],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<i64>>(3)?,
                        r.get::<_, Option<i64>>(4)?,
                        r.get::<_, i64>(5)?,
                        r.get::<_, i64>(6)?,
                    ))
                },
            )
            .optional()?;
        let Some((token_id, user, prefix, request_limit, token_limit, window_seconds, slowdown_ms)) =
            token
        else {
            return Ok(None);
        };
        let since = now.saturating_sub(window_seconds.saturating_mul(1000));
        let (requests, tokens, oldest): (i64, i64, Option<i64>) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens + output_tokens), 0), MIN(ts_ms)
             FROM api_meter WHERE token_id = ?1 AND ts_ms > ?2",
            params![token_id, since],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        Ok(Some(TokenMeter {
            id: token_id,
            user,
            prefix,
            window_seconds,
            request_limit,
            requests,
            requests_remaining: request_limit.map(|limit| (limit - requests).max(0)),
            token_limit,
            tokens,
            tokens_remaining: token_limit.map(|limit| (limit - tokens).max(0)),
            slowdown_ms,
            reset_after_seconds: retry_after(oldest, window_seconds.saturating_mul(1000), now),
        }))
    }

    pub async fn usage_totals(&self, since: i64, until: i64) -> Result<UsageAgg> {
        let conn = self.0.lock().await;
        let agg = conn.query_row(
            "SELECT COUNT(*), SUM(status >= 400), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(reasoning_tokens),0)
             FROM usage_log WHERE ts >= ?1 AND ts < ?2",
            params![since, until],
            |r| {
                Ok(UsageAgg {
                    key: "total".into(),
                    requests: r.get(0)?,
                    errors: r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    input_tokens: r.get(2)?,
                    output_tokens: r.get(3)?,
                    cache_read_tokens: r.get(4)?,
                    reasoning_tokens: r.get(5)?,
                })
            },
        )?;
        Ok(agg)
    }

    pub async fn usage_by(&self, dim: UsageDim, since: i64, until: i64) -> Result<Vec<UsageAgg>> {
        let key_expr = match dim {
            UsageDim::User => "u.user",
            UsageDim::Account => {
                "COALESCE((SELECT COALESCE(a.label, a.email, 'account#' || a.id) FROM accounts a WHERE a.id = u.account_id), 'none')"
            }
            UsageDim::Model => "u.upstream_model",
        };
        let conn = self.0.lock().await;
        let sql = format!(
            "SELECT {key_expr} AS k, COUNT(*), SUM(u.status >= 400), COALESCE(SUM(u.input_tokens),0),
                    COALESCE(SUM(u.output_tokens),0), COALESCE(SUM(u.cache_read_tokens),0), COALESCE(SUM(u.reasoning_tokens),0)
             FROM usage_log u WHERE u.ts >= ?1 AND u.ts < ?2
             GROUP BY k ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![since, until], |r| {
            Ok(UsageAgg {
                key: r.get(0)?,
                requests: r.get(1)?,
                errors: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                input_tokens: r.get(3)?,
                output_tokens: r.get(4)?,
                cache_read_tokens: r.get(5)?,
                reasoning_tokens: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}

fn retry_after(oldest: Option<i64>, window_ms: i64, now: i64) -> i64 {
    oldest
        .map(|ts| ((ts.saturating_add(window_ms).saturating_sub(now) + 999) / 1000).max(1))
        .unwrap_or(1)
}

#[derive(Clone, Copy)]
pub enum UsageDim {
    User,
    Account,
    Model,
}

#[derive(Debug)]
pub struct MetricsRow {
    pub user: String,
    pub model: String,
    pub dialect: String,
    pub requests: i64,
    pub errors: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub reasoning_tokens: i64,
}

impl Db {
    /// Whole-table sums per (user, model, dialect). The log is append-only,
    /// so these are monotonic and safe to expose as Prometheus counters.
    pub async fn usage_metrics(&self) -> Result<Vec<MetricsRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT user, upstream_model, dialect, COUNT(*), SUM(status >= 400),
                    COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(reasoning_tokens),0)
             FROM usage_log GROUP BY user, upstream_model, dialect",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(MetricsRow {
                user: r.get(0)?,
                model: r.get(1)?,
                dialect: r.get(2)?,
                requests: r.get(3)?,
                errors: r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                input_tokens: r.get(5)?,
                output_tokens: r.get(6)?,
                cache_read_tokens: r.get(7)?,
                reasoning_tokens: r.get(8)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}
