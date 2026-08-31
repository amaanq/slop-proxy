use eyre::Result;
use rand::RngCore;
use rusqlite::params;

use super::Db;

#[derive(Debug, Clone)]
pub struct ApiToken {
    pub id: i64,
    pub user: String,
    pub token_prefix: String,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
}

pub fn generate() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let raw = format!("sp-{}", data_encoding::BASE64URL_NOPAD.encode(&bytes));
    let prefix = raw.chars().take(12).collect();
    (raw, prefix)
}

pub fn hash(raw: &str) -> Vec<u8> {
    hmac_sha256::Hash::hash(raw.as_bytes()).to_vec()
}

impl Db {
    pub async fn create_token(&self, user: &str, raw: &str, prefix: &str) -> Result<i64> {
        let conn = self.0.lock().await;
        conn.execute(
            "INSERT INTO api_tokens (user, token_hash, token_prefix) VALUES (?1, ?2, ?3)",
            params![user, hash(raw), prefix],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub async fn list_tokens(&self) -> Result<Vec<ApiToken>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, user, token_prefix, created_at, revoked_at FROM api_tokens ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ApiToken {
                id: r.get(0)?,
                user: r.get(1)?,
                token_prefix: r.get(2)?,
                created_at: r.get(3)?,
                revoked_at: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn revoke_token(&self, key: &str) -> Result<usize> {
        let conn = self.0.lock().await;
        let id = key.parse::<i64>().unwrap_or(-1);
        Ok(conn.execute(
            "UPDATE api_tokens SET revoked_at = unixepoch()
             WHERE revoked_at IS NULL AND (id = ?1 OR token_prefix = ?2)",
            params![id, key],
        )?)
    }

    pub async fn auth_token(&self, raw: &str) -> Result<Option<(i64, String)>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, user FROM api_tokens WHERE token_hash = ?1 AND revoked_at IS NULL",
        )?;
        let mut rows = stmt.query_map(params![hash(raw)], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.next().transpose()?)
    }
}
