use std::result::Result as StdResult;

use eyre::Result;
use rusqlite::{OptionalExtension as _, Row, TransactionBehavior, params};
use serde::Serialize;

use crate::clock;
use crate::db::Db;
use crate::db::tokens::TokenLimits;
use crate::pricing::Tokens;
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
   /// The same tokens at list price, so a free tier can be valued.
   pub list_cost_usd: f64,
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
   pub attempts: i64,
   pub turn_state_blocks: Option<i64>,
   pub cache_ttl_secs: Option<i64>,
}

#[derive(Debug, Serialize, Default)]
pub struct UsageAgg {
   #[serde(rename = "name", skip_serializing_if = "String::is_empty")]
   pub key: String,
   pub requests: i64,
   pub errors: i64,
   pub input_tokens: i64,
   pub output_tokens: i64,
   pub cache_read_tokens: i64,
   pub cache_write_tokens: i64,
   pub cache_hit_ratio: f64,
   pub reasoning_tokens: i64,
}

fn agg_from_row(key: String, row: &Row, base: usize) -> rusqlite::Result<UsageAgg> {
   let input_tokens = row.get(base + 2)?;
   let cache_read_tokens = row.get(base + 4)?;
   let cache_write_tokens = row.get(base + 5)?;
   Ok(UsageAgg {
      key,
      requests: row.get(base)?,
      errors: row.get::<_, Option<i64>>(base + 1)?.unwrap_or(0),
      input_tokens,
      output_tokens: row.get(base + 3)?,
      cache_read_tokens,
      cache_write_tokens,
      cache_hit_ratio: cache_hit_ratio(input_tokens, cache_read_tokens, cache_write_tokens),
      reasoning_tokens: row.get(base + 6)?,
   })
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
   pub tokens: Tokens,
}

#[derive(Debug, Clone)]
pub struct ErrorRow {
   pub user: String,
   pub provider: String,
   pub kind: String,
   pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct CacheTurn {
   pub finished_at: i64,
   pub input_tokens: i64,
   pub cache_read_tokens: i64,
   pub cache_write_tokens: i64,
   pub hit_ratio: f64,
   pub ttl_secs: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct SessionCache {
   pub session_key: String,
   pub requests: i64,
   pub hit_ratio: f64,
   pub last: CacheTurn,
   pub account_switched: bool,
}

#[derive(Debug, Clone)]
pub struct ToolRow {
   pub user: String,
   pub tool: String,
   pub count: i64,
   pub errors: i64,
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
   pub attempts: i64,
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
   pub fn enqueue_usage(&self, record: UsageRecord) -> Result<()> {
      self
         .writer
         .0
         .send(Box::new(move |conn| {
            if let Err(err) = insert_usage(conn, &record) {
               tracing::error!("writing usage log failed {err}");
            }
         }))
         .map_err(|_| eyre::eyre!("database worker stopped"))
   }

   /// Admission is persisted before upstream dispatch inside an IMMEDIATE
   /// transaction, so concurrent requests cannot race past the request
   /// limit. Token counts settle later via `log_usage`, which means a request
   /// can overshoot the token limit once and the window absorbs it.
   pub async fn admit_token(
      &self,
      token_id: i64,
      limits: &TokenLimits,
   ) -> Result<StdResult<Admission, AdmissionError>> {
      let now = clock::unix_now_ms();
      let window_ms = limits.window_seconds.saturating_mul(1000);
      let since = now.saturating_sub(window_ms);
      let limits = limits.clone();
      self
         .call(move |conn| {
            let txn = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let (requests, tokens, oldest_request, oldest_tokens): (
               i64,
               i64,
               Option<i64>,
               Option<i64>,
            ) = txn.query_row(
               "SELECT COUNT(*), COALESCE(SUM(input_tokens + output_tokens), 0), MIN(ts_ms),
                    MIN(CASE WHEN input_tokens + output_tokens > 0 THEN ts_ms END)
             FROM api_meter WHERE token_id = ?1 AND ts_ms > ?2",
               params![token_id, since],
               |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;

            if limits.requests.is_some_and(|limit| requests >= limit) {
               let retry_after = retry_after(oldest_request, window_ms, now);
               return Ok(Err(AdmissionError::RequestLimit { retry_after }));
            }
            if limits.tokens.is_some_and(|limit| tokens >= limit) {
               let retry_after = retry_after(oldest_tokens.or(oldest_request), window_ms, now);
               return Ok(Err(AdmissionError::TokenLimit { retry_after }));
            }

            txn.execute(
               "DELETE FROM api_meter WHERE token_id = ?1 AND ts_ms <= ?2",
               params![token_id, since],
            )?;
            txn.execute(
               "INSERT INTO api_meter (token_id, ts_ms) VALUES (?1, ?2)",
               params![token_id, now],
            )?;
            let meter_id = txn.last_insert_rowid();
            txn.commit()?;

            Ok(Ok(Admission {
               meter_id,
               request_limit: limits.requests,
               requests_remaining: limits.requests.map(|limit| (limit - requests - 1).max(0)),
               token_limit: limits.tokens,
               tokens_remaining: limits.tokens.map(|limit| (limit - tokens).max(0)),
               reset_after: retry_after(oldest_request.or(Some(now)), window_ms, now),
               slowdown_ms: limits.slowdown_ms,
            }))
         })
         .await
   }

   pub async fn token_meter(&self, key: &str) -> Result<Option<TokenMeter>> {
      let now = clock::unix_now_ms();
      let Some(token) = self.find_token(key).await? else {
         return Ok(None);
      };
      let limits = token.limits;
      let since = now.saturating_sub(limits.window_seconds.saturating_mul(1000));
      let (requests, tokens, oldest): (i64, i64, Option<i64>) = self
         .call(move |conn| {
            Ok(conn.query_row(
               "SELECT COUNT(*), COALESCE(SUM(input_tokens + output_tokens), 0), MIN(ts_ms)
             FROM api_meter WHERE token_id = ?1 AND ts_ms > ?2",
               params![token.id, since],
               |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?)
         })
         .await?;
      Ok(Some(TokenMeter {
         id: token.id,
         user: token.user,
         prefix: token.token_prefix,
         window_seconds: limits.window_seconds,
         request_limit: limits.requests,
         requests,
         requests_remaining: limits.requests.map(|limit| (limit - requests).max(0)),
         token_limit: limits.tokens,
         tokens,
         tokens_remaining: limits.tokens.map(|limit| (limit - tokens).max(0)),
         slowdown_ms: limits.slowdown_ms,
         reset_after_seconds: retry_after(oldest, limits.window_seconds.saturating_mul(1000), now),
      }))
   }

   pub async fn usage_totals(&self, since: i64, until: i64) -> Result<UsageAgg> {
      self
         .reports
         .call(move |conn| {
            Ok(conn.query_row(
               "SELECT COUNT(*), SUM(status >= 400), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0),
                    COALESCE(SUM(reasoning_tokens),0)
             FROM usage_log WHERE ts >= ?1 AND ts < ?2",
               params![since, until],
               |row| agg_from_row(String::new(), row, 0),
            )?)
         })
         .await
   }

   pub async fn usage_by(&self, dim: UsageDim, since: i64, until: i64) -> Result<Vec<UsageAgg>> {
      let key_expr = match dim {
         UsageDim::User => "u.user",
         UsageDim::Account => ACCOUNT_LABEL,
         UsageDim::Model => "u.upstream_model",
      };
      self
         .reports
         .rows(
            format!(
               "SELECT {key_expr} AS k, COUNT(*), SUM(u.status >= 400 OR u.error_kind IS NOT NULL), COALESCE(SUM(u.input_tokens),0),
                    COALESCE(SUM(u.output_tokens),0), COALESCE(SUM(u.cache_read_tokens),0),
                    COALESCE(SUM(u.cache_write_tokens),0), COALESCE(SUM(u.reasoning_tokens),0)
             FROM usage_log u WHERE u.ts >= ?1 AND u.ts < ?2
             GROUP BY k ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC"
            ),
            (since, until),
            |row| agg_from_row(row.get(0)?, row, 1),
         )
         .await
   }
}

fn retry_after(oldest: Option<i64>, window_ms: i64, now: i64) -> i64 {
   oldest.map_or(1, |timestamp| {
      ((timestamp.saturating_add(window_ms).saturating_sub(now) + 999) / 1000).max(1)
   })
}

#[derive(Clone, Copy)]
pub enum UsageDim {
   User,
   Account,
   Model,
}

pub fn cache_hit_ratio(input_tokens: i64, cache_read_tokens: i64, cache_write_tokens: i64) -> f64 {
   let cached = cache_read_tokens.max(0) as f64;
   let prompt = input_tokens.max(0) as f64 + cached + cache_write_tokens.max(0) as f64;
   if prompt <= 0.0_f64 {
      return 0.0;
   }

   cached / prompt
}

/// Rows that carried a prompt, which leaves out `count_tokens` and refusals.
const SESSION_TURNS: &str = "FROM usage_log
   WHERE session_key = ?1 AND user = ?2 AND status < 400
     AND input_tokens + cache_read_tokens + cache_write_tokens > 0";

const ACCOUNT_LABEL: &str = "COALESCE((SELECT COALESCE(a.label, a.email, 'account#' || a.id)
                 FROM accounts a WHERE a.id = u.account_id),
                CASE WHEN u.provider <> '' AND NOT EXISTS
                       (SELECT 1 FROM accounts a2 WHERE a2.provider = u.provider)
                     THEN u.provider END,
                'none')";

const PROVIDER_LABEL: &str = "COALESCE(NULLIF(u.provider, ''),
                (SELECT a.provider FROM accounts a WHERE a.id = u.account_id),
                'none')";

/// Every column `usage_metrics` groups by, against its exported label. A
/// column missing here duplicates a label set, and Prometheus keeps whichever
/// the scrape emitted first.
pub const USAGE_DIMENSIONS: [(&str, &str); 8] = [
   ("user", "u.user"),
   ("account", ACCOUNT_LABEL),
   ("provider", PROVIDER_LABEL),
   ("requested_model", "u.requested_model"),
   ("model", "u.upstream_model"),
   ("effort", "u.effort"),
   ("service_tier", "u.service_tier"),
   ("dialect", "u.dialect"),
];

#[derive(Debug)]
pub struct MetricsRow {
   pub dimensions: [String; USAGE_DIMENSIONS.len()],
   pub requests: i64,
   pub errors: i64,
   pub input_tokens: i64,
   pub output_tokens: i64,
   pub cache_read_tokens: i64,
   pub cache_write_tokens: i64,
   pub reasoning_tokens: i64,
   pub cost_usd: f64,
   pub list_cost_usd: f64,
   pub duration_ms: i64,
}

impl Db {
   /// Whole-table sums per [`USAGE_DIMENSIONS`]. The log is append-only, so
   /// these are monotonic and safe to expose as Prometheus counters.
   pub async fn usage_metrics(&self) -> Result<Vec<MetricsRow>> {
      let selected = USAGE_DIMENSIONS
         .into_iter()
         .map(|(label, expr)| format!("{expr} AS {label}"))
         .collect::<Vec<_>>()
         .join(", ");
      let grouped = USAGE_DIMENSIONS
         .into_iter()
         .map(|(label, _)| label)
         .collect::<Vec<_>>()
         .join(", ");
      let after = USAGE_DIMENSIONS.len();
      self
         .reports
         .rows(
            format!(
               "SELECT {selected}, COUNT(*),
                    SUM(u.status >= 400 OR u.error_kind IS NOT NULL),
                    COALESCE(SUM(u.input_tokens),0), COALESCE(SUM(u.output_tokens),0),
                    COALESCE(SUM(u.cache_read_tokens),0), COALESCE(SUM(u.cache_write_tokens),0),
                    COALESCE(SUM(u.reasoning_tokens),0), COALESCE(SUM(u.cost_usd),0),
                    COALESCE(SUM(u.list_cost_usd),0),
                    COALESCE(SUM(u.duration_ms),0)
             FROM usage_log u
             GROUP BY {grouped}",
            ),
            [],
            move |row| {
               let mut dimensions = USAGE_DIMENSIONS.map(|_| String::new());
               for (index, value) in dimensions.iter_mut().enumerate() {
                  *value = row.get(index)?;
               }
               Ok(MetricsRow {
                  dimensions,
                  requests: row.get(after)?,
                  errors: row.get::<_, Option<i64>>(after + 1)?.unwrap_or(0),
                  input_tokens: row.get(after + 2)?,
                  output_tokens: row.get(after + 3)?,
                  cache_read_tokens: row.get(after + 4)?,
                  cache_write_tokens: row.get(after + 5)?,
                  reasoning_tokens: row.get(after + 6)?,
                  cost_usd: row.get(after + 7)?,
                  list_cost_usd: row.get(after + 8)?,
                  duration_ms: row.get(after + 9)?,
               })
            },
         )
         .await
   }

   /// Rows that carry tokens but no cost, which is every row written before
   /// a price table was available.
   pub async fn unpriced_usage(&self) -> Result<Vec<UnpricedRow>> {
      self
         .reports
         .rows(
            "SELECT id, upstream_model, input_tokens, output_tokens,
                    cache_read_tokens, cache_write_tokens
             FROM usage_log
             WHERE (cost_usd = 0 OR list_cost_usd = 0)
               AND input_tokens + output_tokens + cache_read_tokens + cache_write_tokens > 0",
            [],
            |row| {
               Ok(UnpricedRow {
                  id: row.get(0)?,
                  model: row.get(1)?,
                  tokens: Tokens {
                     input: row.get(2)?,
                     output: row.get(3)?,
                     cache_read: row.get(4)?,
                     cache_write: row.get(5)?,
                  },
               })
            },
         )
         .await
   }

   pub async fn price_usage(&self, priced: &[(i64, f64, f64)]) -> Result<()> {
      let priced = priced.to_vec();
      self
         .call(move |conn| {
            let txn = conn.transaction()?;
            {
               let mut stmt = txn.prepare(
                  "UPDATE usage_log SET cost_usd = ?2, list_cost_usd = ?3 WHERE id = ?1",
               )?;
               for &(id, cost, list_cost) in &priced {
                  stmt.execute(params![id, cost, list_cost])?;
               }
            }
            txn.commit()?;
            Ok(())
         })
         .await
   }

   /// Failures grouped by what went wrong. Kept off `MetricsRow` because
   /// `error_kind` would otherwise split every token counter by it too.
   /// `tools_called` holds a comma-joined list, so the split has to happen in
   /// SQL to yield one row per tool.
   pub async fn tool_metrics(&self) -> Result<Vec<ToolRow>> {
      self
         .reports
         .rows(
            "WITH RECURSIVE split(user, failed, tool, rest) AS (
               SELECT user, (status >= 400 OR error_kind IS NOT NULL), '', tools_called || ','
               FROM usage_log WHERE tools_called <> ''
               UNION ALL
               SELECT user, failed, substr(rest, 1, instr(rest, ',') - 1), substr(rest, instr(rest, ',') + 1)
               FROM split WHERE rest <> ''
             )
             SELECT user, tool, COUNT(*), SUM(failed)
             FROM split WHERE tool <> '' GROUP BY user, tool",
            [],
            |row| {
               Ok(ToolRow {
                  user: row.get(0)?,
                  tool: row.get(1)?,
                  count: row.get(2)?,
                  errors: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
               })
            },
         )
         .await
   }

   pub async fn insight_metrics(&self) -> Result<Vec<InsightRow>> {
      self
         .reports
         .rows(
            format!(
               "SELECT u.user,
                    {ACCOUNT_LABEL} AS account,
                    COALESCE(NULLIF(u.stop_reason, ''), u.error_kind,
                             CASE WHEN u.status >= 400
                                  THEN 'http_' || (u.status / 100) || 'xx' END,
                             'unrecorded') AS stop_reason,
                    COUNT(*),
                    COALESCE(SUM(u.request_bytes),0), COALESCE(SUM(u.response_bytes),0),
                    COALESCE(SUM(u.turn_index),0), COALESCE(SUM(u.image_count),0),
                    COALESCE(SUM(u.thinking_budget),0), COALESCE(SUM(u.tools_declared),0),
                    COALESCE(SUM(u.ttft_ms),0), SUM(u.ttft_ms IS NOT NULL),
                    COALESCE(SUM(u.attempts),0)
             FROM usage_log u
             GROUP BY u.user, account, stop_reason"
            ),
            [],
            |row| {
               Ok(InsightRow {
                  user: row.get(0)?,
                  account: row.get(1)?,
                  stop_reason: row.get(2)?,
                  requests: row.get(3)?,
                  request_bytes: row.get(4)?,
                  response_bytes: row.get(5)?,
                  turns: row.get(6)?,
                  images: row.get(7)?,
                  thinking_budget: row.get(8)?,
                  tools_declared: row.get(9)?,
                  ttft_ms: row.get(10)?,
                  ttft_samples: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
                  attempts: row.get(12)?,
               })
            },
         )
         .await
   }

   /// Grouped by user alone, because a session spans models and accounts and
   /// counting it once per group would multiply it.
   pub async fn session_metrics(&self) -> Result<Vec<SessionRow>> {
      self
         .reports
         .rows(
            "SELECT user, COUNT(*), COALESCE(MAX(deepest),0),
                    COALESCE(SUM(MAX(accounts - 1, 0)),0),
                    COALESCE(MAX(tokens),0)
             FROM (SELECT user, COUNT(DISTINCT account_id) AS accounts,
                          MAX(turn_index) AS deepest,
                          SUM(input_tokens + output_tokens
                              + cache_read_tokens + cache_write_tokens) AS tokens
                   FROM usage_log
                   WHERE session_key <> '' AND (account_id IS NOT NULL OR provider <> '')
                   GROUP BY user, session_key)
             GROUP BY user",
            [],
            |row| {
               Ok(SessionRow {
                  user: row.get(0)?,
                  sessions: row.get(1)?,
                  deepest: row.get(2)?,
                  switches: row.get(3)?,
                  tokens_max: row.get(4)?,
               })
            },
         )
         .await
   }

   /// Claude Code keys a session by its JSON `metadata.user_id`, so the id
   /// its statusline knows is matched as the `session_id` field inside it.
   pub async fn session_cache(
      &self,
      user: String,
      session_id: String,
      since: i64,
   ) -> Result<Option<SessionCache>> {
      self
         .reports
         .call(move |conn| {
            let field = format!("\"session_id\":\"{session_id}\"");
            let Some(key) = conn
               .query_row(
                  "SELECT session_key FROM usage_log
                   WHERE user = ?1 AND ts >= ?2 AND status < 400
                     AND (session_key = ?3 OR instr(session_key, ?4) > 0)
                   ORDER BY ts DESC, id DESC LIMIT 1",
                  params![user, since, session_id, field],
                  |row| row.get::<_, String>(0),
               )
               .optional()?
            else {
               return Ok(None);
            };

            let mut recent = conn
               .prepare(&format!(
                  "SELECT ts, account_id, input_tokens, cache_read_tokens, cache_write_tokens,
                          cache_ttl_secs
                   {SESSION_TURNS} ORDER BY ts DESC, id DESC LIMIT 2"
               ))?
               .query_map(params![key, user], |row| {
                  let input_tokens = row.get(2)?;
                  let cache_read_tokens = row.get(3)?;
                  let cache_write_tokens = row.get(4)?;
                  let turn = CacheTurn {
                     finished_at: row.get(0)?,
                     input_tokens,
                     cache_read_tokens,
                     cache_write_tokens,
                     hit_ratio: cache_hit_ratio(
                        input_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                     ),
                     ttl_secs: row.get(5)?,
                  };
                  Ok((row.get::<_, Option<i64>>(1)?, turn))
               })?
               .collect::<rusqlite::Result<Vec<_>>>()?
               .into_iter();
            let Some((account, last)) = recent.next() else {
               return Ok(None);
            };
            let previous = recent.next().and_then(|(previous, _)| previous);

            let (requests, input, read, write) = conn.query_row(
               &format!(
                  "SELECT COUNT(*), COALESCE(SUM(input_tokens),0),
                          COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0)
                   {SESSION_TURNS}"
               ),
               params![key, user],
               |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )?;
            Ok(Some(SessionCache {
               session_key: key,
               requests,
               hit_ratio: cache_hit_ratio(input, read, write),
               last,
               account_switched: matches!(
                  (account, previous),
                  (Some(now), Some(before)) if now != before
               ),
            }))
         })
         .await
   }

   pub async fn error_metrics(&self) -> Result<Vec<ErrorRow>> {
      self
         .reports
         .rows(
            format!(
               "SELECT u.user,
                    {PROVIDER_LABEL} AS provider,
                    COALESCE(u.error_kind, 'http_' || (u.status / 100) || 'xx') AS kind,
                    COUNT(*)
             FROM usage_log u
             WHERE u.status >= 400 OR u.error_kind IS NOT NULL
             GROUP BY u.user, provider, kind"
            ),
            [],
            |row| {
               Ok(ErrorRow {
                  user: row.get(0)?,
                  provider: row.get(1)?,
                  kind: row.get(2)?,
                  count: row.get(3)?,
               })
            },
         )
         .await
   }
}

fn insert_usage(conn: &mut rusqlite::Connection, record: &UsageRecord) -> Result<()> {
   let txn = conn.transaction()?;
   txn.execute(
            "INSERT INTO usage_log (token_id, user, account_id, provider, dialect, requested_model, upstream_model, effort, service_tier,
               input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_usd, list_cost_usd, status, error_kind, duration_ms,
               session_key, turn_index, tools_declared, tools_called, thinking_budget, image_count, request_bytes, response_bytes, ttft_ms, stop_reason, attempts, turn_state_blocks, cache_ttl_secs)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
                     ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32)",
            params![
                record.token_id,
                record.user,
                record.account_id,
                record.provider.map(Provider::as_str).unwrap_or_default(),
                record.dialect,
                record.requested_model,
                record.upstream_model,
                record.effort,
                record.service_tier,
                record.input_tokens,
                record.output_tokens,
                record.cache_read_tokens,
                record.cache_write_tokens,
                record.reasoning_tokens,
                record.cost_usd,
                record.list_cost_usd,
                record.status,
                record.error_kind,
                record.duration_ms,
                record.session_key,
                record.turn_index,
                record.tools_declared,
                record.tools_called,
                record.thinking_budget,
                record.image_count,
                record.request_bytes,
                record.response_bytes,
                record.ttft_ms,
                record.stop_reason,
                record.attempts,
                record.turn_state_blocks,
                record.cache_ttl_secs,
            ],
        )?;
   if let Some(meter_id) = record.meter_id {
      txn.execute(
         "UPDATE api_meter SET input_tokens = ?2, output_tokens = ?3 WHERE id = ?1",
         params![meter_id, record.input_tokens, record.output_tokens],
      )?;
   }
   txn.commit()?;
   Ok(())
}
