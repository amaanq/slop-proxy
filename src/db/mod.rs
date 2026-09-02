pub mod accounts;
pub mod tokens;
pub mod usage;

use std::path::Path;
use std::sync::Arc;

use eyre::{Result, WrapErr};
use rusqlite::Connection;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Db(pub(crate) Arc<Mutex<Connection>>);

const SCHEMA: &str = include_str!("schema.sql");

impl Db {
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).wrap_err_with(|| format!("creating {}", dir.display()))?;
        }
        let conn =
            Connection::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch(SCHEMA).wrap_err("creating schema")?;
        // A column added to schema.sql never reaches an existing table.
        for (table, column, ddl) in ADDED_COLUMNS {
            add_column(&conn, table, column, ddl)?;
        }

        Ok(Self(Arc::new(Mutex::new(conn))))
    }
}

const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    ("accounts", "http_referer", "TEXT"),
    ("accounts", "auth_mode", "TEXT NOT NULL DEFAULT 'oauth'"),
    ("usage_log", "service_tier", "TEXT NOT NULL DEFAULT ''"),
];

fn add_column(conn: &Connection, table: &str, column: &str, ddl: &str) -> Result<()> {
    let present = conn
        .prepare(&format!("PRAGMA table_info({table})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .iter()
        .any(|name| name == column);
    if !present {
        conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {ddl}"),
            [],
        )
        .wrap_err_with(|| format!("adding {table}.{column}"))?;
    }
    Ok(())
}
