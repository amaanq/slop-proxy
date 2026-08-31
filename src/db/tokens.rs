use anyhow::Result;
use base64::Engine;
use rand::RngCore;
use rusqlite::params;
use sha2::{Digest, Sha256};

use super::Db;

#[derive(Debug, Clone)]
pub struct ApiToken {
    pub id: i64,
    pub user: String,
    pub token_prefix: String,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
    pub limits: TokenLimits,
}

#[derive(Debug, Clone, Default)]
pub struct TokenLimits {
    pub requests: Option<i64>,
    pub tokens: Option<i64>,
    pub window_seconds: i64,
    pub slowdown_ms: i64,
}

#[derive(Debug, Clone)]
pub struct AuthenticatedToken {
    pub id: i64,
    pub user: String,
    pub limits: TokenLimits,
}

pub fn generate() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let raw = format!(
        "sp-{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    );
    let prefix = raw.chars().take(12).collect();
    (raw, prefix)
}

pub fn hash(raw: &str) -> Vec<u8> {
    Sha256::digest(raw.as_bytes()).to_vec()
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
            "SELECT id, user, token_prefix, created_at, revoked_at,
                    request_limit, token_limit, window_seconds, slowdown_ms
             FROM api_tokens ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ApiToken {
                id: r.get(0)?,
                user: r.get(1)?,
                token_prefix: r.get(2)?,
                created_at: r.get(3)?,
                revoked_at: r.get(4)?,
                limits: TokenLimits {
                    requests: r.get(5)?,
                    tokens: r.get(6)?,
                    window_seconds: r.get(7)?,
                    slowdown_ms: r.get(8)?,
                },
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn revoke_token(&self, key: &str) -> Result<usize> {
        let conn = self.0.lock().await;
        let id: i64 = key.parse().unwrap_or(-1);
        Ok(conn.execute(
            "UPDATE api_tokens SET revoked_at = unixepoch()
             WHERE revoked_at IS NULL AND (id = ?1 OR token_prefix = ?2)",
            params![id, key],
        )?)
    }

    pub async fn set_token_limits(&self, key: &str, limits: &TokenLimits) -> Result<usize> {
        let conn = self.0.lock().await;
        let id: i64 = key.parse().unwrap_or(-1);
        Ok(conn.execute(
            "UPDATE api_tokens
             SET request_limit = ?3, token_limit = ?4, window_seconds = ?5, slowdown_ms = ?6
             WHERE id = ?1 OR token_prefix = ?2",
            params![
                id,
                key,
                limits.requests,
                limits.tokens,
                limits.window_seconds,
                limits.slowdown_ms,
            ],
        )?)
    }

    pub async fn auth_token(&self, raw: &str) -> Result<Option<AuthenticatedToken>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, user, request_limit, token_limit, window_seconds, slowdown_ms
             FROM api_tokens WHERE token_hash = ?1 AND revoked_at IS NULL",
        )?;
        let mut rows = stmt.query_map(params![hash(raw)], |r| {
            Ok(AuthenticatedToken {
                id: r.get(0)?,
                user: r.get(1)?,
                limits: TokenLimits {
                    requests: r.get(2)?,
                    tokens: r.get(3)?,
                    window_seconds: r.get(4)?,
                    slowdown_ms: r.get(5)?,
                },
            })
        })?;
        Ok(rows.next().transpose()?)
    }
}
