use eyre::Result;
use rusqlite::params;
use serde::Serialize;

use super::Db;

#[derive(Debug, Clone, Default)]
pub struct UsageRecord {
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

impl Db {
    pub async fn log_usage(&self, r: &UsageRecord) -> Result<()> {
        let conn = self.0.lock().await;
        conn.execute(
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
        Ok(())
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
