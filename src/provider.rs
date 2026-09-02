use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, pound::ValueEnum)]
pub enum Provider {
    OpenAi,
    Anthropic,
    Gemini,
}

impl Provider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "openai" => Some(Provider::OpenAi),
            "anthropic" => Some(Provider::Anthropic),
            "gemini" => Some(Provider::Gemini),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Gemini => "gemini",
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
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMode::OAuth => "oauth",
            AuthMode::ApiKey => "api_key",
        }
    }

    /// An API key carries no expiry and nothing to exchange, so the refresh
    /// path never applies to it.
    pub fn refreshable(self) -> bool {
        matches!(self, AuthMode::OAuth)
    }
}

impl std::fmt::Display for AuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromSql for AuthMode {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_str()? {
            "oauth" => Ok(AuthMode::OAuth),
            "api_key" => Ok(AuthMode::ApiKey),
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

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromSql for Provider {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_str()? {
            // "codex" is the name this column used before the rename, and a
            // db written by the older binary outlives the deploy.
            "openai" | "codex" => Ok(Provider::OpenAi),
            "anthropic" => Ok(Provider::Anthropic),
            "gemini" => Ok(Provider::Gemini),
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
