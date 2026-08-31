use anyhow::Result;
use rusqlite::{Row, params};

use super::Db;
use crate::oauth::TokenSet;
use crate::provider::Provider;

#[derive(Debug, Clone)]
pub struct Account {
    pub id: i64,
    pub provider: Provider,
    pub provider_account_id: String,
    pub email: Option<String>,
    pub label: Option<String>,
    pub plan_type: Option<String>,
    pub access_token: String,
    pub refresh_token: String,
    pub access_expires_at: Option<i64>,
    pub status: String,
    pub cooldown_until: Option<i64>,
    pub disabled_reason: Option<String>,
}

fn from_row(row: &Row) -> rusqlite::Result<Account> {
    Ok(Account {
        id: row.get("id")?,
        provider: row.get("provider")?,
        provider_account_id: row.get("provider_account_id")?,
        email: row.get("email")?,
        label: row.get("label")?,
        plan_type: row.get("plan_type")?,
        access_token: row.get("access_token")?,
        refresh_token: row.get("refresh_token")?,
        access_expires_at: row.get("access_expires_at")?,
        status: row.get("status")?,
        cooldown_until: row.get("cooldown_until")?,
        disabled_reason: row.get("disabled_reason")?,
    })
}

const COLS: &str = "id, provider, provider_account_id, email, label, plan_type, access_token, refresh_token, access_expires_at, status, cooldown_until, disabled_reason";

impl Db {
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_account(
        &self,
        provider: Provider,
        provider_account_id: &str,
        email: Option<&str>,
        label: Option<&str>,
        plan_type: Option<&str>,
        tokens: &TokenSet,
    ) -> Result<i64> {
        let conn = self.0.lock().await;
        conn.execute(
            "INSERT INTO accounts (provider, provider_account_id, email, label, plan_type, access_token, refresh_token, id_token, access_expires_at, last_refresh_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, unixepoch())
             ON CONFLICT(provider_account_id) DO UPDATE SET
               provider = excluded.provider,
               email = excluded.email,
               label = COALESCE(excluded.label, label),
               plan_type = excluded.plan_type,
               access_token = excluded.access_token,
               refresh_token = excluded.refresh_token,
               id_token = excluded.id_token,
               access_expires_at = excluded.access_expires_at,
               last_refresh_at = unixepoch(),
               status = 'active',
               cooldown_until = NULL,
               disabled_reason = NULL,
               updated_at = unixepoch()",
            params![
                provider,
                provider_account_id,
                email,
                label,
                plan_type,
                tokens.access_token,
                tokens.refresh_token,
                tokens.id_token,
                tokens.expires_at,
            ],
        )?;
        let id = conn.query_row(
            "SELECT id FROM accounts WHERE provider_account_id = ?1",
            [provider_account_id],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    pub async fn list_accounts(&self) -> Result<Vec<Account>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(&format!("SELECT {COLS} FROM accounts ORDER BY id"))?;
        let rows = stmt.query_map([], from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub async fn find_account(&self, key: &str) -> Result<Option<Account>> {
        let conn = self.0.lock().await;
        let mut stmt = conn.prepare(&format!(
            "SELECT {COLS} FROM accounts WHERE id = ?1 OR email = ?2 OR label = ?2"
        ))?;
        let id = key.parse::<i64>().unwrap_or(-1);
        let mut rows = stmt.query_map(params![id, key], from_row)?;
        Ok(rows.next().transpose()?)
    }

    pub async fn remove_account(&self, key: &str) -> Result<usize> {
        let conn = self.0.lock().await;
        let id = key.parse::<i64>().unwrap_or(-1);
        Ok(conn.execute(
            "DELETE FROM accounts WHERE id = ?1 OR email = ?2 OR label = ?2",
            params![id, key],
        )?)
    }

    pub async fn update_account_tokens(&self, id: i64, tokens: &TokenSet) -> Result<()> {
        let conn = self.0.lock().await;
        conn.execute(
            "UPDATE accounts SET access_token = ?2, refresh_token = ?3,
               id_token = COALESCE(?4, id_token), access_expires_at = ?5,
               last_refresh_at = unixepoch(), updated_at = unixepoch()
             WHERE id = ?1",
            params![
                id,
                tokens.access_token,
                tokens.refresh_token,
                tokens.id_token,
                tokens.expires_at
            ],
        )?;
        Ok(())
    }

    pub async fn set_account_status(
        &self,
        id: i64,
        status: &str,
        cooldown_until: Option<i64>,
        disabled_reason: Option<&str>,
    ) -> Result<()> {
        let conn = self.0.lock().await;
        conn.execute(
            "UPDATE accounts SET status = ?2, cooldown_until = ?3, disabled_reason = ?4, updated_at = unixepoch()
             WHERE id = ?1",
            params![id, status, cooldown_until, disabled_reason],
        )?;
        Ok(())
    }
}
