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

const SCHEMA: &str = include_str!("schema.sql");

impl Db {
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch(SCHEMA).context("creating schema")?;

        Ok(Self(Arc::new(Mutex::new(conn))))
    }
}
