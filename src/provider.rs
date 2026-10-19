use std::fmt;
use std::str::FromStr;

use pound::ArgValue as _;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, pound::ValueEnum)]
pub enum Provider {
   #[pound(name = "openai")]
   OpenAi,
   Anthropic,
   Gemini,
   Zen,
   Glm,
   #[pound(name = "deepseek")]
   DeepSeek,
   Experiential,
   Copilot,
   #[pound(name = "openrouter")]
   OpenRouter,
}

impl FromStr for Provider {
   type Err = String;

   fn from_str(text: &str) -> Result<Self, Self::Err> {
      let text = text.trim();
      Self::ALL
         .iter()
         .copied()
         .find(|provider| provider.as_str() == text)
         .ok_or_else(|| format!("unknown provider {text:?}"))
   }
}

/// How an account proves itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthMode {
   #[default]
   OAuth,
   ApiKey,
}

impl AuthMode {
   pub const fn as_str(self) -> &'static str {
      match self {
         Self::OAuth => "oauth",
         Self::ApiKey => "api_key",
      }
   }

   /// An API key carries no expiry and nothing to exchange, so the refresh
   /// path never applies to it.
   pub const fn refreshable(self) -> bool {
      matches!(self, Self::OAuth)
   }
}

impl fmt::Display for AuthMode {
   fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
      formatter.write_str(self.as_str())
   }
}

impl FromSql for AuthMode {
   fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
      let text = value.as_str()?;
      [Self::OAuth, Self::ApiKey]
         .into_iter()
         .find(|mode| mode.as_str() == text)
         .ok_or_else(|| FromSqlError::Other(format!("unknown auth mode {text:?}").into()))
   }
}

impl ToSql for AuthMode {
   fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
      Ok(self.as_str().into())
   }
}

impl fmt::Display for Provider {
   fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
      formatter.write_str(self.as_str())
   }
}

impl FromSql for Provider {
   fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
      value
         .as_str()?
         .parse()
         .map_err(|err: String| FromSqlError::Other(err.into()))
   }
}

impl ToSql for Provider {
   fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
      Ok(self.as_str().into())
   }
}
