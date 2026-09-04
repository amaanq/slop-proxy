pub mod accounts;
pub mod tokens;
pub mod usage;

use std::path::Path;
use std::sync::Arc;

use eyre::{Result, WrapErr as _};
use rusqlite::Connection;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Db(pub Arc<Mutex<Connection>>);

const SCHEMA: &str = include_str!("schema.sql");

impl Db {
   pub fn open(path: &Path) -> Result<Self> {
      if let Some(dir) = path.parent() {
         std::fs::create_dir_all(dir).wrap_err_with(|| format!("creating {}", dir.display()))?;
      }
      let conn = Connection::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
      conn.pragma_update(None, "journal_mode", "WAL")?;
      conn.pragma_update(None, "busy_timeout", 5000)?;
      conn.pragma_update(None, "foreign_keys", "ON")?;

      conn.execute_batch(SCHEMA).wrap_err("creating schema")?;
      // A column added to schema.sql never reaches an existing table.
      for (table, column, ddl) in ADDED_COLUMNS {
         add_column(&conn, table, column, ddl)?;
      }
      conn
         .execute_batch(LATE_INDEXES)
         .wrap_err("creating indexes")?;

      Ok(Self(Arc::new(Mutex::new(conn))))
   }
}

const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
   ("accounts", "http_referer", "TEXT"),
   ("accounts", "auth_mode", "TEXT NOT NULL DEFAULT 'oauth'"),
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
