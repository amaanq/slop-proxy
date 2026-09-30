use eyre::Result;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, Value, ValueRef};
use rusqlite::{Row, params};

use crate::db::Db;
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

impl FromSql for AccountStatus {
   fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
      let text = value.as_str()?;
      [Self::Active, Self::Cooldown, Self::Disabled]
         .into_iter()
         .find(|status| status.as_str() == text)
         .ok_or_else(|| FromSqlError::Other(format!("unknown account status {text:?}").into()))
   }
}

#[derive(Debug, Clone)]
pub struct Account {
   pub id: i64,
   pub provider: Provider,
   pub provider_account_id: String,
   pub trusted: bool,
   /// Only served to tokens with `reserved_only`; never to the shared pool.
   pub reserved: bool,
   pub egress: bool,
   pub allowed_users: Vec<String>,
   pub auth_mode: AuthMode,
   pub email: Option<String>,
   pub label: Option<String>,
   pub plan_type: Option<String>,
   pub access_token: String,
   pub refresh_token: String,
   pub http_referer: Option<String>,
   pub turn_state: Option<String>,
   pub access_expires_at: Option<i64>,
   pub status: AccountStatus,
   pub cooldown_until: Option<i64>,
   pub disabled_reason: Option<String>,
}

fn parse_users(raw: &str) -> Vec<String> {
   raw.split(',')
      .map(str::trim)
      .filter(|user| !user.is_empty())
      .map(str::to_owned)
      .collect()
}

fn from_row(row: &Row) -> rusqlite::Result<Account> {
   Ok(Account {
      id: row.get("id")?,
      provider: row.get("provider")?,
      provider_account_id: row.get("provider_account_id")?,
      trusted: row.get("trusted")?,
      reserved: row.get("reserved")?,
      egress: row.get("egress")?,
      allowed_users: parse_users(&row.get::<_, String>("allowed_users")?),
      auth_mode: row.get("auth_mode")?,
      email: row.get("email")?,
      label: row.get("label")?,
      plan_type: row.get("plan_type")?,
      access_token: row.get("access_token")?,
      refresh_token: row.get("refresh_token")?,
      http_referer: row.get("http_referer")?,
      turn_state: row.get("turn_state")?,
      access_expires_at: row.get("access_expires_at")?,
      status: row.get("status")?,
      cooldown_until: row.get("cooldown_until")?,
      disabled_reason: row.get("disabled_reason")?,
   })
}

const COLS: &str = "id, provider, provider_account_id, trusted, reserved, egress, auth_mode, email, label, plan_type, access_token, refresh_token, http_referer, turn_state, access_expires_at, status, cooldown_until, disabled_reason, allowed_users";

pub enum AccountField {
   Trusted(bool),
   Egress(bool),
   Reserved(bool),
   AllowedUsers(String),
   HttpReferer(Option<String>),
   TurnState(String),
}

impl AccountField {
   fn column(self) -> (&'static str, Value) {
      match self {
         Self::Trusted(enabled) => ("trusted", enabled.into()),
         Self::Egress(enabled) => ("egress", enabled.into()),
         Self::Reserved(enabled) => ("reserved", enabled.into()),
         Self::AllowedUsers(users) => ("allowed_users", users.into()),
         Self::HttpReferer(referer) => ("http_referer", referer.into()),
         Self::TurnState(token) => ("turn_state", token.into()),
      }
   }
}

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
      let provider_account_id = provider_account_id.to_owned();
      let email = email.map(str::to_owned);
      let label = label.map(str::to_owned);
      let plan_type = plan_type.map(str::to_owned);
      let tokens = tokens.clone();
      self.call(move |conn| {
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
         |row| row.get(0),
      )?;
      Ok(id)
      }).await
   }

   pub async fn list_accounts(&self) -> Result<Vec<Account>> {
      self
         .writer
         .rows(
            format!("SELECT {COLS} FROM accounts ORDER BY id"),
            [],
            from_row,
         )
         .await
   }

   pub async fn find_account(&self, key: &str) -> Result<Option<Account>> {
      let key = key.to_owned();
      self
         .call(move |conn| {
            let mut stmt = conn.prepare(&format!(
               "SELECT {COLS} FROM accounts WHERE id = ?1 OR email = ?2 OR label = ?2"
            ))?;
            let id = key.parse::<i64>().unwrap_or(-1);
            let mut rows = stmt.query_map(params![id, key], from_row)?;
            Ok(rows.next().transpose()?)
         })
         .await
   }

   pub async fn remove_account(&self, id: i64) -> Result<usize> {
      self
         .call(move |conn| {
            let txn = conn.transaction()?;
            txn.execute(
               "UPDATE usage_log SET account_id = NULL WHERE account_id = ?1",
               params![id],
            )?;
            let removed = txn.execute("DELETE FROM accounts WHERE id = ?1", params![id])?;
            txn.commit()?;
            Ok(removed)
         })
         .await
   }

   pub async fn update_account_tokens(&self, id: i64, tokens: &TokenSet) -> Result<()> {
      let tokens = tokens.clone();
      self
         .call(move |conn| {
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
         })
         .await
   }

   pub async fn set_account(&self, id: i64, field: AccountField) -> Result<usize> {
      let (column, value) = field.column();
      self
         .call(move |conn| {
            Ok(conn.execute(
               &format!(
                  "UPDATE accounts SET {column} = ?2, updated_at = unixepoch() WHERE id = ?1"
               ),
               params![id, value],
            )?)
         })
         .await
   }

   pub async fn set_account_status(
      &self,
      id: i64,
      status: AccountStatus,
      cooldown_until: Option<i64>,
      disabled_reason: Option<&str>,
   ) -> Result<()> {
      let disabled_reason = disabled_reason.map(str::to_owned);
      self.call(move |conn| {
      conn.execute(
            "UPDATE accounts SET status = ?2, cooldown_until = ?3, disabled_reason = ?4, updated_at = unixepoch()
             WHERE id = ?1",
            params![id, status.as_str(), cooldown_until, disabled_reason],
        )?;
      Ok(())
      }).await
   }

   pub async fn cas_account_cooldown(&self, id: i64, until: i64) -> Result<bool> {
      self.call(move |conn| {
         let rows = conn.execute(
            "UPDATE accounts SET status = 'cooldown', cooldown_until = ?2, disabled_reason = NULL, updated_at = unixepoch()
             WHERE id = ?1 AND status != 'disabled' AND (status != 'cooldown' OR cooldown_until IS NULL OR cooldown_until < ?2)",
            params![id, until],
         )?;
         Ok(rows > 0)
      }).await
   }

   pub async fn clear_account_cooldown(&self, id: i64, expected_until: i64) -> Result<bool> {
      self
         .call(move |conn| {
            Ok(conn.execute(
         "UPDATE accounts SET status = 'active', cooldown_until = NULL, updated_at = unixepoch()
          WHERE id = ?1 AND status = 'cooldown' AND cooldown_until = ?2",
         params![id, expected_until],
         )? > 0)
         })
         .await
   }
}
