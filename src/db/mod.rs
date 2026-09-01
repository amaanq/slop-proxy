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
        let conn = Connection::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch(SCHEMA).wrap_err("creating schema")?;
        let has_http_referer = conn
            .prepare("PRAGMA table_info(accounts)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .iter()
            .any(|name| name == "http_referer");
        if !has_http_referer {
            conn.execute("ALTER TABLE accounts ADD COLUMN http_referer TEXT", [])?;
        }

        Ok(Self(Arc::new(Mutex::new(conn))))
    }
}
