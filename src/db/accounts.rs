use eyre::Result;
use rusqlite::{Row, params};

use super::Db;
use crate::oauth::TokenSet;
use crate::provider::{AuthMode, Provider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountStatus {
    Active,
    Cooldown,
    Disabled,
}

impl AccountStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Cooldown => "cooldown",
            Self::Disabled => "disabled",
        }
    }
}

impl std::str::FromStr for AccountStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "cooldown" => Ok(Self::Cooldown),
            "disabled" => Ok(Self::Disabled),
            other => Err(format!("unknown account status {other:?}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Account {
    pub id: i64,
    pub provider: Provider,
    pub provider_account_id: String,
    pub trusted: bool,
    pub auth_mode: AuthMode,
    pub email: Option<String>,
    pub label: Option<String>,
    pub plan_type: Option<String>,
    pub access_token: String,
    pub refresh_token: String,
    pub http_referer: Option<String>,
    pub access_expires_at: Option<i64>,
    pub status: AccountStatus,
    pub cooldown_until: Option<i64>,
    pub disabled_reason: Option<String>,
}

fn from_row(row: &Row) -> rusqlite::Result<Account> {
    Ok(Account {
        id: row.get("id")?,
        provider: row.get("provider")?,
        provider_account_id: row.get("provider_account_id")?,
        trusted: row.get("trusted")?,
        auth_mode: row.get("auth_mode")?,
        email: row.get("email")?,
        label: row.get("label")?,
        plan_type: row.get("plan_type")?,
        access_token: row.get("access_token")?,
        refresh_token: row.get("refresh_token")?,
        http_referer: row.get("http_referer")?,
        access_expires_at: row.get("access_expires_at")?,
        status: {
            let raw: String = row.get("status")?;
            raw.parse()
                .map_err(|e: String| rusqlite::types::FromSqlError::Other(e.into()))?
        },
        cooldown_until: row.get("cooldown_until")?,
        disabled_reason: row.get("disabled_reason")?,
    })
}

const COLS: &str = "id, provider, provider_account_id, trusted, auth_mode, email, label, plan_type, access_token, refresh_token, http_referer, access_expires_at, status, cooldown_until, disabled_reason";

pub struct NewAccount<'a> {
    pub provider: Provider,
    pub id: &'a str,
    pub email: Option<&'a str>,
    pub label: Option<&'a str>,
    pub plan: Option<&'a str>,
    pub tokens: &'a TokenSet,
    pub auth_mode: AuthMode,
}

impl Db {
    pub async fn upsert_account(&self, account: NewAccount<'_>) -> Result<i64> {
        let NewAccount {
            provider,
            id: provider_account_id,
            email,
            label,
            plan: plan_type,
            tokens,
            auth_mode,
        } = account;
        let conn = self.0.lock().await;
        conn.execute(
            "INSERT INTO accounts (provider, provider_account_id, email, label, plan_type, access_token, refresh_token, access_expires_at, last_refresh_at, auth_mode)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, unixepoch(), ?9)
             ON CONFLICT(provider, provider_account_id) DO UPDATE SET
               auth_mode = excluded.auth_mode,
               email = excluded.email,
               label = COALESCE(excluded.label, label),
               plan_type = excluded.plan_type,
               access_token = excluded.access_token,
               refresh_token = excluded.refresh_token,
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
                tokens.expires_at,
                auth_mode,
            ],
        )?;
        let id = conn.query_row(
            "SELECT id FROM accounts WHERE provider = ?1 AND provider_account_id = ?2",
            params![provider, provider_account_id],
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
               access_expires_at = ?4,
               last_refresh_at = unixepoch(), updated_at = unixepoch()
             WHERE id = ?1",
            params![
                id,
                tokens.access_token,
                tokens.refresh_token,
                tokens.expires_at
            ],
        )?;
        Ok(())
    }

    pub async fn set_account_trusted(&self, key: &str, trusted: bool) -> Result<usize> {
        let conn = self.0.lock().await;
        let id = key.parse::<i64>().unwrap_or(-1);
        Ok(conn.execute(
            "UPDATE accounts SET trusted = ?3, updated_at = unixepoch()
             WHERE id = ?1 OR email = ?2 OR label = ?2",
            params![id, key, trusted],
        )?)
    }

    pub async fn set_account_http_referer(&self, id: i64, referer: Option<&str>) -> Result<()> {
        let conn = self.0.lock().await;
        conn.execute(
            "UPDATE accounts SET http_referer = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![id, referer],
        )?;
        Ok(())
    }

    pub async fn set_account_status(
        &self,
        id: i64,
        status: AccountStatus,
        cooldown_until: Option<i64>,
        disabled_reason: Option<&str>,
    ) -> Result<()> {
        let conn = self.0.lock().await;
        conn.execute(
            "UPDATE accounts SET status = ?2, cooldown_until = ?3, disabled_reason = ?4, updated_at = unixepoch()
             WHERE id = ?1",
            params![id, status.as_str(), cooldown_until, disabled_reason],
        )?;
        Ok(())
    }
}
