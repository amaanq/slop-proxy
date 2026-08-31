pub mod accounts;
pub mod tokens;
pub mod usage;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use rusqlite::Connection;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Db(pub(crate) Arc<Mutex<Connection>>);

const MIGRATIONS: &[&str] = &[
    r#"
CREATE TABLE accounts (
  id INTEGER PRIMARY KEY,
  chatgpt_account_id TEXT NOT NULL UNIQUE,
  email TEXT, label TEXT, plan_type TEXT,
  access_token TEXT NOT NULL,
  refresh_token TEXT NOT NULL,
  id_token TEXT,
  access_expires_at INTEGER,
  last_refresh_at INTEGER,
  status TEXT NOT NULL DEFAULT 'active',
  cooldown_until INTEGER,
  disabled_reason TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE TABLE api_tokens (
  id INTEGER PRIMARY KEY,
  user TEXT NOT NULL,
  token_hash BLOB NOT NULL UNIQUE,
  token_prefix TEXT NOT NULL,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  revoked_at INTEGER
);
CREATE TABLE usage_log (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL DEFAULT (unixepoch()),
  token_id INTEGER REFERENCES api_tokens(id),
  user TEXT NOT NULL,
  account_id INTEGER REFERENCES accounts(id),
  dialect TEXT NOT NULL,
  requested_model TEXT NOT NULL,
  upstream_model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens INTEGER NOT NULL DEFAULT 0,
  status INTEGER NOT NULL,
  error_kind TEXT,
  duration_ms INTEGER
);
CREATE INDEX idx_usage_ts ON usage_log(ts);
CREATE INDEX idx_usage_user_ts ON usage_log(user, ts);
CREATE INDEX idx_usage_account_ts ON usage_log(account_id, ts);
"#,
    r#"
ALTER TABLE accounts RENAME COLUMN chatgpt_account_id TO provider_account_id;
ALTER TABLE accounts ADD COLUMN provider TEXT NOT NULL DEFAULT 'codex';
"#,
];

impl Db {
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        let version = conn.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))?;
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(version as usize) {
            conn.execute_batch(sql)
                .with_context(|| format!("applying migration {}", i + 1))?;
            conn.pragma_update(None, "user_version", (i + 1) as i64)?;
        }

        Ok(Self(Arc::new(Mutex::new(conn))))
    }
}
