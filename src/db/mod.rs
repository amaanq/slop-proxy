pub mod accounts;
pub mod tokens;
pub mod usage;

use std::fs;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use eyre::{Result, WrapErr as _};
use rusqlite::Connection;
use tokio::sync::oneshot;

#[derive(Clone)]
pub struct Db {
   writer: Worker,
   reports: Worker,
}

type Job = Box<dyn FnOnce(&mut Connection) + Send>;

#[derive(Clone)]
struct Worker(mpsc::Sender<Job>);

impl Worker {
   fn start(mut conn: Connection) -> Result<Self> {
      let (sender, receiver) = mpsc::channel::<Job>();
      thread::Builder::new()
         .name("sqlite".into())
         .spawn(move || {
            for job in receiver {
               job(&mut conn);
            }
         })?;
      Ok(Self(sender))
   }

   async fn call<T>(
      &self,
      query: impl FnOnce(&mut Connection) -> Result<T> + Send + 'static,
   ) -> Result<T>
   where
      T: Send + 'static,
   {
      let (sender, receiver) = oneshot::channel();
      self
         .0
         .send(Box::new(move |conn| {
            let _ = sender.send(query(conn));
         }))
         .map_err(|_| eyre::eyre!("database worker stopped"))?;
      receiver.await.wrap_err("database worker stopped")?
   }
}

const SCHEMA: &str = include_str!("schema.sql");

impl Db {
   pub fn open(path: &Path) -> Result<Self> {
      if let Some(dir) = path.parent() {
         fs::create_dir_all(dir).wrap_err_with(|| format!("creating {}", dir.display()))?;
      }
      let conn = Connection::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
      conn.pragma_update(None, "journal_mode", "WAL")?;
      conn.pragma_update(None, "busy_timeout", 5000_i64)?;
      conn.pragma_update(None, "foreign_keys", "ON")?;

      conn.execute_batch(SCHEMA).wrap_err("creating schema")?;
      // A column added to schema.sql never reaches an existing table.
      for &(table, column, ddl) in ADDED_COLUMNS {
         add_column(&conn, table, column, ddl)?;
      }
      conn
         .execute_batch(LATE_INDEXES)
         .wrap_err("creating indexes")?;

      let reports = Connection::open(path)?;
      reports.pragma_update(None, "busy_timeout", 5000_i64)?;
      reports.pragma_update(None, "query_only", true)?;
      Ok(Self {
         writer: Worker::start(conn)?,
         reports: Worker::start(reports)?,
      })
   }

   pub(crate) async fn call<T>(
      &self,
      query: impl FnOnce(&mut Connection) -> Result<T> + Send + 'static,
   ) -> Result<T>
   where
      T: Send + 'static,
   {
      self.writer.call(query).await
   }

   pub async fn flush(&self) -> Result<()> {
      self.call(|_| Ok(())).await
   }
}

const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
   ("accounts", "http_referer", "TEXT"),
   ("accounts", "auth_mode", "TEXT NOT NULL DEFAULT 'oauth'"),
   ("accounts", "allowed_users", "TEXT NOT NULL DEFAULT ''"),
   ("usage_log", "provider", "TEXT NOT NULL DEFAULT ''"),
   ("usage_log", "list_cost_usd", "REAL NOT NULL DEFAULT 0"),
   ("usage_log", "service_tier", "TEXT NOT NULL DEFAULT ''"),
   ("usage_log", "session_key", "TEXT NOT NULL DEFAULT ''"),
   ("usage_log", "turn_index", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "tools_declared", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "tools_called", "TEXT NOT NULL DEFAULT ''"),
   ("usage_log", "thinking_budget", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "image_count", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "request_bytes", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "response_bytes", "INTEGER NOT NULL DEFAULT 0"),
   ("usage_log", "ttft_ms", "INTEGER"),
   ("usage_log", "stop_reason", "TEXT NOT NULL DEFAULT ''"),
   (
      "api_tokens",
      "allowed_providers",
      "TEXT NOT NULL DEFAULT ''",
   ),
   ("api_tokens", "pinned_account", "INTEGER"),
];

/// Indexes over columns `ADDED_COLUMNS` introduces, so they are built after
/// the ALTERs rather than failing on a database that predates them.
const LATE_INDEXES: &str =
   "CREATE INDEX IF NOT EXISTS idx_usage_session_ts ON usage_log(session_key, ts);";

fn add_column(conn: &Connection, table: &str, column: &str, ddl: &str) -> Result<()> {
   let present = conn
      .prepare(&format!("PRAGMA table_info({table})"))?
      .query_map([], |row| row.get::<_, String>(1))?
      .collect::<rusqlite::Result<Vec<_>>>()?
      .iter()
      .any(|name| name == column);
   if !present {
      conn
         .execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {ddl}"),
            [],
         )
         .wrap_err_with(|| format!("adding {table}.{column}"))?;
   }
   Ok(())
}

#[cfg(test)]
mod tests {
   use super::*;
   use crate::db::usage::UsageRecord;
   use std::env;
   use std::time::Duration;
   use tokio::time::timeout;

   fn database() -> Db {
      Db::open(&env::temp_dir().join(format!("slop-db-{}.db", uuid::Uuid::new_v4()))).unwrap()
   }

   #[tokio::test]
   async fn reporting_does_not_block_admission_or_the_runtime() {
      let db = database();
      db.create_token("alice", "secret", "test").await.unwrap();
      let (started, ready) = oneshot::channel();
      let (release, blocked) = mpsc::channel();
      let reports = db.reports.clone();
      let report = tokio::spawn(async move {
         reports
            .call(move |_| {
               started.send(()).unwrap();
               blocked.recv().unwrap();
               Ok(())
            })
            .await
      });
      ready.await.unwrap();
      let auth = timeout(Duration::from_secs(1), db.auth_token("secret")).await;
      release.send(()).unwrap();
      report.await.unwrap().unwrap();
      assert_eq!(auth.unwrap().unwrap().unwrap().user, "alice");
   }

   #[tokio::test]
   async fn flush_waits_for_queued_usage() {
      let db = database();
      for output_tokens in [3, 5] {
         db.enqueue_usage(UsageRecord {
            user: "alice".into(),
            output_tokens,
            status: 200,
            ..Default::default()
         })
         .unwrap();
      }
      db.flush().await.unwrap();
      let rows = db.usage_totals(0, i64::MAX).await.unwrap();
      assert_eq!(rows.requests, 2);
      assert_eq!(rows.output_tokens, 8);
   }
}
