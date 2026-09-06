use eyre::Result;
use rand::RngCore as _;
use rusqlite::params;

use super::Db;
use crate::provider::Provider;

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
   pub prefer_trusted: bool,
   /// The one account this token may be served by. `None` leaves it free to
   /// use any account the pool offers.
   pub pinned_account: Option<i64>,
   /// Providers this token may reach. Empty means every one, so a token
   /// created before the column existed keeps its old reach.
   pub providers: Vec<Provider>,
}

impl TokenLimits {
   pub fn may_use(&self, provider: Provider) -> bool {
      self.providers.is_empty() || self.providers.contains(&provider)
   }

   fn encode(&self) -> String {
      self
         .providers
         .iter()
         .map(|provider| provider.as_str())
         .collect::<Vec<_>>()
         .join(",")
   }

   fn decode(raw: &str) -> Vec<Provider> {
      raw.split(',').filter_map(Provider::from_str).collect()
   }
}

#[derive(Debug, Clone)]
pub struct AuthenticatedToken {
   pub id: i64,
   pub user: String,
   pub limits: TokenLimits,
}

pub fn generate() -> (String, String) {
   let mut bytes = [0_u8; 32];
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
      let user = user.to_owned();
      let token_hash = hash(raw);
      let prefix = prefix.to_owned();
      self
         .call(move |conn| {
            conn.execute(
               "INSERT INTO api_tokens (user, token_hash, token_prefix) VALUES (?1, ?2, ?3)",
               params![user, token_hash, prefix],
            )?;
            Ok(conn.last_insert_rowid())
         })
         .await
   }

   pub async fn list_tokens(&self) -> Result<Vec<ApiToken>> {
      self
         .call(move |conn| {
            let mut stmt = conn.prepare(
               "SELECT id, user, token_prefix, created_at, revoked_at,
                    request_limit, token_limit, window_seconds, slowdown_ms, prefer_trusted,
                    pinned_account, allowed_providers
             FROM api_tokens ORDER BY id",
            )?;
            let rows = stmt.query_map([], |row| {
               Ok(ApiToken {
                  id: row.get(0)?,
                  user: row.get(1)?,
                  token_prefix: row.get(2)?,
                  created_at: row.get(3)?,
                  revoked_at: row.get(4)?,
                  limits: TokenLimits {
                     requests: row.get(5)?,
                     tokens: row.get(6)?,
                     window_seconds: row.get(7)?,
                     slowdown_ms: row.get(8)?,
                     prefer_trusted: row.get(9)?,
                     pinned_account: row.get(10)?,
                     providers: TokenLimits::decode(&row.get::<_, String>(11)?),
                  },
               })
            })?;
            Ok(rows.collect::<rusqlite::Result<_>>()?)
         })
         .await
   }

   pub async fn revoke_token(&self, key: &str) -> Result<usize> {
      let key = key.to_owned();
      self
         .call(move |conn| {
            let id = key.parse::<i64>().unwrap_or(-1);
            Ok(conn.execute(
               "UPDATE api_tokens SET revoked_at = unixepoch()
             WHERE revoked_at IS NULL AND (id = ?1 OR token_prefix = ?2)",
               params![id, key],
            )?)
         })
         .await
   }

   pub async fn set_token_limits(&self, key: &str, limits: &TokenLimits) -> Result<usize> {
      let key = key.to_owned();
      let limits = limits.clone();
      self
         .call(move |conn| {
            let id = key.parse::<i64>().unwrap_or(-1);
            Ok(conn.execute(
               "UPDATE api_tokens
             SET request_limit = ?3, token_limit = ?4, window_seconds = ?5, slowdown_ms = ?6,
                 prefer_trusted = ?7, pinned_account = ?8, allowed_providers = ?9
             WHERE id = ?1 OR token_prefix = ?2",
               params![
                  id,
                  key,
                  limits.requests,
                  limits.tokens,
                  limits.window_seconds,
                  limits.slowdown_ms,
                  limits.prefer_trusted,
                  limits.pinned_account,
                  limits.encode(),
               ],
            )?)
         })
         .await
   }

   pub async fn auth_token(&self, raw: &str) -> Result<Option<AuthenticatedToken>> {
      let token_hash = hash(raw);
      self
         .call(move |conn| {
            let mut stmt = conn.prepare(
               "SELECT id, user, request_limit, token_limit, window_seconds, slowdown_ms,
                    prefer_trusted, pinned_account, allowed_providers
             FROM api_tokens WHERE token_hash = ?1 AND revoked_at IS NULL",
            )?;
            let mut rows = stmt.query_map(params![token_hash], |row| {
               Ok(AuthenticatedToken {
                  id: row.get(0)?,
                  user: row.get(1)?,
                  limits: TokenLimits {
                     requests: row.get(2)?,
                     tokens: row.get(3)?,
                     window_seconds: row.get(4)?,
                     slowdown_ms: row.get(5)?,
                     prefer_trusted: row.get(6)?,
                     pinned_account: row.get(7)?,
                     providers: TokenLimits::decode(&row.get::<_, String>(8)?),
                  },
               })
            })?;
            Ok(rows.next().transpose()?)
         })
         .await
   }
}

#[cfg(test)]
mod tests {
   use super::TokenLimits;
   use crate::provider::Provider;

   #[test]
   fn an_unscoped_token_reaches_every_backend() {
      let limits = TokenLimits::default();
      assert!(limits.may_use(Provider::Gemini) && limits.may_use(Provider::Anthropic));
   }

   #[test]
   fn a_scoped_token_reaches_only_its_own() {
      let limits = TokenLimits {
         providers: vec![Provider::Gemini],
         ..TokenLimits::default()
      };
      assert!(limits.may_use(Provider::Gemini));
      assert!(!limits.may_use(Provider::Anthropic) && !limits.may_use(Provider::OpenAi));
   }
}
