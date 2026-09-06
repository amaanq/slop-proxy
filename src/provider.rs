use std::fmt;

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
   OpenAi,
   Anthropic,
   Gemini,
   Zen,
   Glm,
   Experiential,
}

/// `ValueEnum` would derive `open-ai` from the variant name, which is a
/// spelling no config pattern, database row or `--providers` list accepts.
impl pound::FromArg for Provider {
   const POSSIBLE: Option<&'static [&'static str]> =
      Some(&["openai", "anthropic", "gemini", "zen", "glm"]);

   fn from_arg(text: &str) -> Result<Self, pound::ValueError> {
      Self::from_str(text).ok_or_else(|| pound::ValueError::new(text, "unrecognised provider"))
   }
}

impl Provider {
   pub fn from_str(text: &str) -> Option<Self> {
      match text.trim() {
         "openai" => Some(Self::OpenAi),
         "anthropic" => Some(Self::Anthropic),
         "gemini" => Some(Self::Gemini),
         "zen" => Some(Self::Zen),
         "glm" => Some(Self::Glm),
         "experiential" => Some(Self::Experiential),
         _ => None,
      }
   }

   pub const fn as_str(self) -> &'static str {
      match self {
         Self::OpenAi => "openai",
         Self::Anthropic => "anthropic",
         Self::Gemini => "gemini",
         Self::Zen => "zen",
         Self::Glm => "glm",
         Self::Experiential => "experiential",
      }
   }
}

/// How an account proves itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, pound::ValueEnum)]
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
      match value.as_str()? {
         "oauth" => Ok(Self::OAuth),
         "api_key" => Ok(Self::ApiKey),
         other => Err(FromSqlError::Other(
            format!("unknown auth mode {other:?}").into(),
         )),
      }
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
      match value.as_str()? {
         // "codex" is the name this column used before the rename, and a
         // db written by the older binary outlives the deploy.
         "openai" | "codex" => Ok(Self::OpenAi),
         "anthropic" => Ok(Self::Anthropic),
         "gemini" => Ok(Self::Gemini),
         "zen" => Ok(Self::Zen),
         "glm" => Ok(Self::Glm),
         "experiential" => Ok(Self::Experiential),
         other => Err(FromSqlError::Other(
            format!("unknown provider {other:?}").into(),
         )),
      }
   }
}

impl ToSql for Provider {
   fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
      Ok(self.as_str().into())
   }
}
