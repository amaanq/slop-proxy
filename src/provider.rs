use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, pound::ValueEnum)]
pub enum Provider {
    OpenAi,
    Anthropic,
    Gemini,
    Zen,
    Glm,
}

impl Provider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "gemini" => Some(Self::Gemini),
            "zen" => Some(Self::Zen),
            "glm" => Some(Self::Glm),
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

impl std::fmt::Display for AuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
            "openai" | "codex" => Ok(Self::OpenAi),
            "anthropic" => Ok(Self::Anthropic),
            "gemini" => Ok(Self::Gemini),
            "zen" => Ok(Self::Zen),
            "glm" => Ok(Self::Glm),
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
