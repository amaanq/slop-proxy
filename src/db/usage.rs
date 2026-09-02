use eyre::Result;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::Serialize;

use super::Db;
use super::tokens::TokenLimits;
use crate::provider::Provider;

#[derive(Debug, Clone, Default)]
pub struct UsageRecord {
    pub meter_id: Option<i64>,
    pub token_id: Option<i64>,
    pub user: String,
    pub account_id: Option<i64>,
    /// The backend the request was routed to. Kept beside the account rather
    /// than derived from it, because a keyless backend has no account and a
    /// request that never reached one still knows where it was headed.
    pub provider: Option<Provider>,
    pub dialect: &'static str,
    pub requested_model: String,
    pub upstream_model: String,
    pub effort: String,
    pub service_tier: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub reasoning_tokens: i64,
    pub cost_usd: f64,
    pub status: i64,
    pub error_kind: Option<String>,
    pub duration_ms: Option<i64>,
    pub session_key: String,
    pub turn_index: i64,
    pub tools_declared: i64,
    pub tools_called: String,
    pub thinking_budget: i64,
    pub image_count: i64,
    pub request_bytes: i64,
    pub response_bytes: i64,
    pub ttft_ms: Option<i64>,
    pub stop_reason: String,
}

#[derive(Debug, Serialize, Default)]
pub struct UsageAgg {
    pub key: String,
    pub requests: i64,
    pub errors: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
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

#[derive(Debug)]
pub struct UnpricedRow {
    pub id: i64,
    pub model: String,
    pub tokens: crate::pricing::Tokens,
}

#[derive(Debug, Clone)]
pub struct ErrorRow {
    pub user: String,
    pub provider: String,
    pub kind: String,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct ToolRow {
    pub user: String,
    pub tool: String,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct InsightRow {
    pub user: String,
    pub account: String,
    pub stop_reason: String,
    pub requests: i64,
    pub request_bytes: i64,
    pub response_bytes: i64,
    pub turns: i64,
    pub images: i64,
    pub thinking_budget: i64,
    pub tools_declared: i64,
    pub ttft_ms: i64,
    pub ttft_samples: i64,
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub user: String,
    pub sessions: i64,
    pub deepest: i64,
    pub switches: i64,
    pub tokens_max: i64,
}

impl Db {
    pub async fn log_usage(&self, r: &UsageRecord) -> Result<()> {
        let mut conn = self.0.lock().await;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO usage_log (token_id, user, account_id, provider, dialect, requested_model, upstream_model, effort, service_tier,
               input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_usd, status, error_kind, duration_ms,
               session_key, turn_index, tools_declared, tools_called, thinking_budget, image_count, request_bytes, response_bytes, ttft_ms, stop_reason)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
                     ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28)",
            params![
                r.token_id,
                r.user,
                r.account_id,
                r.provider.map(|p| p.as_str()),
                r.dialect,
                r.requested_model,
                r.upstream_model,
                r.effort,
                r.service_tier,
                r.input_tokens,
                r.output_tokens,
                r.cache_read_tokens,
                r.cache_write_tokens,
                r.reasoning_tokens,
                r.cost_usd,
                r.status,
                r.error_kind,
                r.duration_ms,
                r.session_key,
                r.turn_index,
                r.tools_declared,
                r.tools_called,
                r.thinking_budget,
                r.image_count,
                r.request_bytes,
                r.response_bytes,
                r.ttft_ms,
                r.stop_reason,
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
            "DELETE FROM api_meter WHERE token_id = ?1 AND ts_ms <= ?2",
            params![token_id, since],
        )?;
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
                    COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0),
                    COALESCE(SUM(reasoning_tokens),0)
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
                    cache_write_tokens: r.get(5)?,
                    reasoning_tokens: r.get(6)?,
                })
            },
        )?;
        Ok(agg)
    }

    pub async fn usage_by(&self, dim: UsageDim, since: i64, until: i64) -> Result<Vec<UsageAgg>> {
        let key_expr = match dim {
            UsageDim::User => "u.user",
            UsageDim::Account => {
                "COALESCE((SELECT COALESCE(a.label, a.email, 'account#' || a.id) FROM accounts a WHERE a.id = u.account_id), NULLIF(u.provider, ''), 'none')"
            }
            UsageDim::Model => "u.upstream_model",
        };
        let conn = self.0.lock().await;
        let sql = format!(
            "SELECT {key_expr} AS k, COUNT(*), SUM(u.status >= 400 OR u.error_kind IS NOT NULL), COALESCE(SUM(u.input_tokens),0),
                    COALESCE(SUM(u.output_tokens),0), COALESCE(SUM(u.cache_read_tokens),0),
                    COALESCE(SUM(u.cache_write_tokens),0), COALESCE(SUM(u.reasoning_tokens),0)
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
                cache_write_tokens: r.get(6)?,
                reasoning_tokens: r.get(7)?,
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
    pub account: String,
    pub provider: String,
    pub requested_model: String,
    pub model: String,
    pub effort: String,
    pub service_tier: String,
    pub dialect: String,
    pub requests: i64,
    pub errors: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_write_tokens: i64,
    pub reasoning_tokens: i64,
    pub cost_usd: f64,
    pub duration_ms: i64,
}

impl Db {
    /// Whole-table sums per (user, model, dialect). The log is append-only,
    /// so these are monotonic and safe to expose as Prometheus counters.
    pub async fn usage_metrics(&self) -> Result<Vec<MetricsRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT u.user,
                    COALESCE((SELECT COALESCE(a.label, a.email, 'account#' || a.id)
                              FROM accounts a WHERE a.id = u.account_id),
                             CASE WHEN u.provider <> '' AND NOT EXISTS
                                    (SELECT 1 FROM accounts a2 WHERE a2.provider = u.provider)
                                  THEN u.provider END,
                             'none') AS account,
                    COALESCE(NULLIF(u.provider, ''),
                             (SELECT a.provider FROM accounts a WHERE a.id = u.account_id),
                             'none') AS provider,
                    u.requested_model, u.upstream_model, u.effort, u.service_tier, u.dialect, COUNT(*),
                    SUM(u.status >= 400 OR u.error_kind IS NOT NULL),
                    COALESCE(SUM(u.input_tokens),0), COALESCE(SUM(u.output_tokens),0),
                    COALESCE(SUM(u.cache_read_tokens),0), COALESCE(SUM(u.cache_write_tokens),0),
                    COALESCE(SUM(u.reasoning_tokens),0), COALESCE(SUM(u.cost_usd),0),
                    COALESCE(SUM(u.duration_ms),0)
             FROM usage_log u
             GROUP BY u.user, account, provider, u.requested_model, u.upstream_model,
                      u.effort, u.service_tier, u.dialect",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(MetricsRow {
                user: r.get(0)?,
                account: r.get(1)?,
                provider: r.get(2)?,
                requested_model: r.get(3)?,
                model: r.get(4)?,
                effort: r.get(5)?,
                service_tier: r.get(6)?,
                dialect: r.get(7)?,
                requests: r.get(8)?,
                errors: r.get::<_, Option<i64>>(9)?.unwrap_or(0),
                input_tokens: r.get(10)?,
                output_tokens: r.get(11)?,
                cache_read_tokens: r.get(12)?,
                cache_write_tokens: r.get(13)?,
                reasoning_tokens: r.get(14)?,
                cost_usd: r.get(15)?,
                duration_ms: r.get(16)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Rows that carry tokens but no cost, which is every row written before
    /// a price table was available.
    pub async fn unpriced_usage(&self) -> Result<Vec<UnpricedRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, upstream_model, input_tokens, output_tokens,
                    cache_read_tokens, cache_write_tokens
             FROM usage_log
             WHERE cost_usd = 0
               AND input_tokens + output_tokens + cache_read_tokens + cache_write_tokens > 0",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(UnpricedRow {
                id: r.get(0)?,
                model: r.get(1)?,
                tokens: crate::pricing::Tokens {
                    input: r.get(2)?,
                    output: r.get(3)?,
                    cache_read: r.get(4)?,
                    cache_write: r.get(5)?,
                },
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn price_usage(&self, priced: &[(i64, f64)]) -> Result<()> {
        let mut conn = self.0.lock().await;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare("UPDATE usage_log SET cost_usd = ?2 WHERE id = ?1")?;
            for (id, cost) in priced {
                stmt.execute(params![id, cost])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Failures grouped by what went wrong. Kept off MetricsRow because
    /// error_kind would otherwise split every token counter by it too.
    /// `tools_called` holds a comma-joined list, so the split has to happen in
    /// SQL to yield one row per tool.
    pub async fn tool_metrics(&self) -> Result<Vec<ToolRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "WITH RECURSIVE split(user, tool, rest) AS (
               SELECT user, '', tools_called || ',' FROM usage_log WHERE tools_called <> ''
               UNION ALL
               SELECT user, substr(rest, 1, instr(rest, ',') - 1), substr(rest, instr(rest, ',') + 1)
               FROM split WHERE rest <> ''
             )
             SELECT user, tool, COUNT(*) FROM split WHERE tool <> '' GROUP BY user, tool",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ToolRow {
                user: r.get(0)?,
                tool: r.get(1)?,
                count: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn insight_metrics(&self) -> Result<Vec<InsightRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT u.user,
                    COALESCE((SELECT COALESCE(a.label, a.email, 'account#' || a.id)
                              FROM accounts a WHERE a.id = u.account_id),
                             CASE WHEN u.provider <> '' AND NOT EXISTS
                                    (SELECT 1 FROM accounts a2 WHERE a2.provider = u.provider)
                                  THEN u.provider END,
                             'none') AS account,
                    COALESCE(NULLIF(u.stop_reason, ''), u.error_kind,
                             CASE WHEN u.status >= 400
                                  THEN 'http_' || (u.status / 100) || 'xx' END,
                             'unrecorded') AS stop_reason,
                    COUNT(*),
                    COALESCE(SUM(u.request_bytes),0), COALESCE(SUM(u.response_bytes),0),
                    COALESCE(SUM(u.turn_index),0), COALESCE(SUM(u.image_count),0),
                    COALESCE(SUM(u.thinking_budget),0), COALESCE(SUM(u.tools_declared),0),
                    COALESCE(SUM(u.ttft_ms),0), SUM(u.ttft_ms IS NOT NULL)
             FROM usage_log u
             GROUP BY u.user, account, stop_reason",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(InsightRow {
                user: r.get(0)?,
                account: r.get(1)?,
                stop_reason: r.get(2)?,
                requests: r.get(3)?,
                request_bytes: r.get(4)?,
                response_bytes: r.get(5)?,
                turns: r.get(6)?,
                images: r.get(7)?,
                thinking_budget: r.get(8)?,
                tools_declared: r.get(9)?,
                ttft_ms: r.get(10)?,
                ttft_samples: r.get::<_, Option<i64>>(11)?.unwrap_or(0),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Grouped by user alone, because a session spans models and accounts and
    /// counting it once per group would multiply it.
    pub async fn session_metrics(&self) -> Result<Vec<SessionRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT user, COUNT(*), COALESCE(MAX(deepest),0), COALESCE(SUM(accounts - 1),0),
                    COALESCE(MAX(tokens),0)
             FROM (SELECT user, COUNT(DISTINCT account_id) AS accounts,
                          MAX(turn_index) AS deepest,
                          SUM(input_tokens + output_tokens
                              + cache_read_tokens + cache_write_tokens) AS tokens
                   FROM usage_log
                   WHERE session_key <> '' AND account_id IS NOT NULL
                   GROUP BY user, session_key)
             GROUP BY user",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SessionRow {
                user: r.get(0)?,
                sessions: r.get(1)?,
                deepest: r.get(2)?,
                switches: r.get(3)?,
                tokens_max: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn error_metrics(&self) -> Result<Vec<ErrorRow>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT u.user,
                    COALESCE(NULLIF(u.provider, ''),
                             (SELECT a.provider FROM accounts a WHERE a.id = u.account_id),
                             'none') AS provider,
                    COALESCE(u.error_kind, 'http_' || (u.status / 100) || 'xx') AS kind,
                    COUNT(*)
             FROM usage_log u
             WHERE u.status >= 400 OR u.error_kind IS NOT NULL
             GROUP BY u.user, provider, kind",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ErrorRow {
                user: r.get(0)?,
                provider: r.get(1)?,
                kind: r.get(2)?,
                count: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}
