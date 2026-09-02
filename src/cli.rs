use std::path::PathBuf;

use eyre::{Result, bail, eyre};
use pound::Parse;

use crate::config::Config;
use crate::db::Db;
use crate::provider::{AuthMode, Provider};

/// Anthropic/OpenAI API proxy backed by Codex subscription accounts
#[derive(Parse)]
#[pound(name = "slop-proxy")]
pub struct Cli {
    /// Path to the sqlite database
    #[pound(long, global)]
    pub db: Option<PathBuf>,

    /// Path to config.toml
    #[pound(long, global)]
    pub config: Option<PathBuf>,

    #[pound(short, long, global)]
    pub verbose: bool,

    #[pound(subcommand)]
    pub command: Command,
}

#[derive(Parse)]
pub enum Command {
    /// Log in to a subscription account and store it
    Login {
        /// Human-readable label for the account
        #[pound(long)]
        label: Option<String>,

        /// Which backend to log in to
        #[pound(long, default = "openai")]
        provider: Provider,
    },
    /// Manage stored accounts
    Accounts {
        #[pound(subcommand)]
        command: AccountsCommand,
    },
    /// Manage issued API tokens
    Token {
        #[pound(subcommand)]
        command: TokenCommand,
    },
    /// Run the API server
    Serve {
        /// Listen address
        #[pound(long)]
        bind: Option<String>,
    },
    /// Show usage statistics as JSON
    Stats {
        /// Window start: 24h, 7d, 30m, or RFC3339
        #[pound(long)]
        since: Option<String>,
        /// Window end (RFC3339)
        #[pound(long)]
        until: Option<String>,
    },
    /// List the models available from the codex backend, as JSON
    Models,
    /// Debug helpers
    #[pound(hidden)]
    Debug {
        #[pound(subcommand)]
        command: DebugCommand,
    },
}

#[derive(Parse)]
pub enum AccountsCommand {
    /// List stored accounts
    List,
    /// Store an account that authenticates with a long-lived API key
    AddKey {
        #[pound(long)]
        provider: Provider,
        #[pound(long)]
        key: String,
        #[pound(long)]
        label: Option<String>,
        #[pound(long)]
        referer: Option<String>,
    },
    /// Remove an account by id or email
    Remove { account: String },
    /// Mark an account as trusted, or clear the flag with --off
    Trust {
        account: String,
        #[pound(long)]
        off: bool,
    },
}

#[derive(Parse)]
pub enum TokenCommand {
    /// Issue a new API token for a user
    Create {
        #[pound(long)]
        user: String,
        /// Maximum requests in each rolling window
        #[pound(long)]
        requests: Option<i64>,
        /// Maximum input plus output tokens in each rolling window
        #[pound(long)]
        tokens: Option<i64>,
        #[pound(long, default = "3600")]
        window_seconds: i64,
        /// Delay every admitted request by this many milliseconds
        #[pound(long, default = "0")]
        slowdown_ms: i64,
        /// Serve this token from trusted accounts when any are available
        #[pound(long)]
        prefer_trusted: bool,
        /// Providers this token may reach, comma separated. Empty allows all.
        #[pound(long)]
        providers: Option<String>,
    },
    /// List issued tokens
    List,
    /// Revoke a token by id or prefix
    Revoke { token: String },
    /// Replace limits for a token id or prefix; omitted limits are unlimited
    Limits {
        token: String,
        #[pound(long)]
        requests: Option<i64>,
        #[pound(long)]
        tokens: Option<i64>,
        #[pound(long, default = "3600")]
        window_seconds: i64,
        #[pound(long, default = "0")]
        slowdown_ms: i64,
        #[pound(long)]
        prefer_trusted: bool,
        /// Providers this token may reach, comma separated. Empty allows all.
        #[pound(long)]
        providers: Option<String>,
    },
    /// Show metered usage for a token's current rolling window
    Usage { token: String },
}

#[derive(Parse)]
pub enum DebugCommand {
    /// Send a raw request upstream and dump the SSE events
    Ping {
        #[pound(long)]
        model: Option<String>,
        #[pound(long, default = "Say the word: pong")]
        prompt: String,
    },
    /// Force a token refresh for an account
    Refresh { account: String },
    /// Dump the raw models endpoint response from the codex backend
    Models,
}

pub async fn run(args: Cli, cfg: Config) -> Result<()> {
    let db = Db::open(&cfg.db_path).await?;

    match args.command {
        Command::Login { label, provider } => match provider {
            Provider::OpenAi => crate::oauth::login(&db, label).await,
            Provider::Anthropic => crate::oauth::anthropic::login(&db, label).await,
            Provider::Gemini => Err(eyre::eyre!(
                "google has no device-code flow here, use `accounts add-key --provider gemini`"
            )),
        },
        Command::Accounts { command } => match command {
            AccountsCommand::List => accounts_list(&db).await,
            AccountsCommand::AddKey {
                provider,
                key,
                label,
                referer,
            } => accounts_add_key(&db, provider, &key, label.as_deref(), referer.as_deref()).await,
            AccountsCommand::Remove { account } => accounts_remove(&db, &account).await,
            AccountsCommand::Trust { account, off } => accounts_trust(&db, &account, !off).await,
        },
        Command::Token { command } => match command {
            TokenCommand::Create {
                user,
                requests,
                tokens,
                window_seconds,
                slowdown_ms,
                prefer_trusted,
                providers,
            } => {
                let limits = token_limits(
                    requests,
                    tokens,
                    window_seconds,
                    slowdown_ms,
                    prefer_trusted,
                    providers,
                )?;
                token_create(&db, &user, &limits).await
            }
            TokenCommand::List => token_list(&db).await,
            TokenCommand::Revoke { token } => token_revoke(&db, &token).await,
            TokenCommand::Limits {
                token,
                requests,
                tokens,
                window_seconds,
                slowdown_ms,
                prefer_trusted,
                providers,
            } => {
                let limits = token_limits(
                    requests,
                    tokens,
                    window_seconds,
                    slowdown_ms,
                    prefer_trusted,
                    providers,
                )?;
                token_set_limits(&db, &token, &limits).await
            }
            TokenCommand::Usage { token } => token_usage(&db, &token).await,
        },
        Command::Serve { bind } => {
            let bind = bind.unwrap_or_else(|| cfg.bind.clone());
            crate::server::serve(db, cfg, &bind).await
        }
        Command::Stats { since, until } => crate::stats::run(&db, since, until).await,
        Command::Models => models(&db, &cfg).await,
        Command::Debug { command } => match command {
            DebugCommand::Ping { model, prompt } => {
                crate::codex::debug_ping(&db, &cfg, model, prompt).await
            }
            DebugCommand::Refresh { account } => debug_refresh(&db, &account).await,
            DebugCommand::Models => crate::codex::debug_models(&db, &cfg).await,
        },
    }
}

/// A key is its own identity here. Google exposes nothing to call for an
/// account id, and hashing the key keeps re-adding the same one an update
/// rather than a duplicate slot.
async fn accounts_add_key(
    db: &Db,
    provider: Provider,
    key: &str,
    label: Option<&str>,
    referer: Option<&str>,
) -> Result<()> {
    if referer.is_some() && provider != Provider::Gemini {
        bail!("--referer is only supported for gemini keys");
    }
    let mut h = hmac_sha256::Hash::new();
    h.update(key.as_bytes());
    let account_id = data_encoding::HEXLOWER.encode(&h.finalize()[..8]);
    let tokens = crate::oauth::TokenSet {
        access_token: key.to_string(),
        refresh_token: String::new(),
        id_token: None,
        expires_at: None,
    };
    let id = db
        .upsert_account(
            provider,
            &account_id,
            None,
            label,
            None,
            &tokens,
            AuthMode::ApiKey,
        )
        .await?;
    if let Some(referer) = referer {
        let referer = (!referer.is_empty()).then_some(referer);
        db.set_account_http_referer(id, referer).await?;
    }
    println!("stored {provider} account {id} ({account_id})");
    Ok(())
}

async fn accounts_list(db: &Db) -> Result<()> {
    // A struct keeps field order; serde_json alphabetizes json! maps.
    #[derive(serde::Serialize)]
    struct AccountRow<'a> {
        id: i64,
        provider: &'a str,
        trusted: bool,
        email: Option<&'a str>,
        plan_type: Option<&'a str>,
        status: &'a str,
        label: Option<&'a str>,
        cooldown_seconds_left: Option<i64>,
        disabled_reason: Option<&'a str>,
        http_referer: Option<&'a str>,
    }

    let accounts = db.list_accounts().await?;
    let now = crate::clock::unix_now();
    let rows = accounts
        .iter()
        .map(|a| AccountRow {
            id: a.id,
            provider: a.provider.as_str(),
            trusted: a.trusted,
            email: a.email.as_deref(),
            plan_type: a.plan_type.as_deref(),
            status: &a.status,
            label: a.label.as_deref(),
            cooldown_seconds_left: (a.status == "cooldown")
                .then(|| a.cooldown_until.map(|c| (c - now).max(0)))
                .flatten(),
            disabled_reason: a.disabled_reason.as_deref(),
            http_referer: a.http_referer.as_deref(),
        })
        .collect::<Vec<AccountRow>>();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

async fn accounts_trust(db: &Db, account: &str, trusted: bool) -> Result<()> {
    if db.set_account_trusted(account, trusted).await? == 0 {
        bail!("no account matched {account:?}");
    }
    println!(
        "account {account} is now {}",
        if trusted { "trusted" } else { "untrusted" }
    );
    Ok(())
}

async fn accounts_remove(db: &Db, account: &str) -> Result<()> {
    let n = db.remove_account(account).await?;
    if n == 0 {
        bail!("no account matched {account:?}");
    }
    println!("removed {n} account(s)");
    Ok(())
}

async fn token_create(db: &Db, user: &str, limits: &crate::db::tokens::TokenLimits) -> Result<()> {
    let (raw, prefix) = crate::db::tokens::generate();
    let id = db.create_token(user, &raw, &prefix).await?;
    db.set_token_limits(&id.to_string(), limits).await?;
    println!("token for {user}: {raw}");
    println!("(shown once; only a hash is stored)");
    Ok(())
}

fn token_limits(
    requests: Option<i64>,
    tokens: Option<i64>,
    window_seconds: i64,
    slowdown_ms: i64,
    prefer_trusted: bool,
    providers: Option<String>,
) -> Result<crate::db::tokens::TokenLimits> {
    if requests.is_some_and(|v| v <= 0) {
        bail!("--requests must be greater than zero");
    }
    if tokens.is_some_and(|v| v <= 0) {
        bail!("--tokens must be greater than zero");
    }
    if window_seconds <= 0 {
        bail!("--window-seconds must be greater than zero");
    }
    if slowdown_ms < 0 {
        bail!("--slowdown-ms cannot be negative");
    }
    let providers = providers
        .filter(|p| !p.trim().is_empty())
        .map(|raw| {
            raw.split(',')
                .map(|p| {
                    crate::provider::Provider::from_str(p)
                        .ok_or_else(|| eyre::eyre!("unknown provider: {p}"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(crate::db::tokens::TokenLimits {
        requests,
        tokens,
        window_seconds,
        slowdown_ms,
        prefer_trusted,
        providers,
    })
}

async fn token_set_limits(
    db: &Db,
    token: &str,
    limits: &crate::db::tokens::TokenLimits,
) -> Result<()> {
    if db.set_token_limits(token, limits).await? == 0 {
        bail!("no token matched {token:?}");
    }
    println!("updated limits for {token}");
    Ok(())
}

async fn token_usage(db: &Db, token: &str) -> Result<()> {
    let usage = db
        .token_meter(token)
        .await?
        .ok_or_else(|| eyre!("no token matched {token:?}"))?;
    println!("{}", serde_json::to_string_pretty(&usage)?);
    Ok(())
}

async fn token_list(db: &Db) -> Result<()> {
    #[derive(serde::Serialize)]
    struct TokenRow<'a> {
        id: i64,
        user: &'a str,
        prefix: &'a str,
        created_at: i64,
        revoked: bool,
        revoked_at: Option<i64>,
        request_limit: Option<i64>,
        token_limit: Option<i64>,
        window_seconds: i64,
        slowdown_ms: i64,
    }

    let tokens = db.list_tokens().await?;
    let rows = tokens
        .iter()
        .map(|t| TokenRow {
            id: t.id,
            user: &t.user,
            prefix: &t.token_prefix,
            created_at: t.created_at,
            revoked: t.revoked_at.is_some(),
            revoked_at: t.revoked_at,
            request_limit: t.limits.requests,
            token_limit: t.limits.tokens,
            window_seconds: t.limits.window_seconds,
            slowdown_ms: t.limits.slowdown_ms,
        })
        .collect::<Vec<TokenRow>>();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

async fn token_revoke(db: &Db, token: &str) -> Result<()> {
    let n = db.revoke_token(token).await?;
    if n == 0 {
        bail!("no active token matched {token:?}");
    }
    println!("revoked {n} token(s)");
    Ok(())
}

async fn models(db: &Db, cfg: &Config) -> Result<()> {
    let client = crate::codex::client::CodexClient::new(cfg.codex.clone());
    let pool = crate::pool::codex::CodexPool::load(db.clone(), client).await?;

    let models = match pool.list_models().await {
        Ok(models) => models,
        Err(e) => {
            eprintln!("could not fetch models from the codex backend: {e}");
            Vec::new()
        }
    };

    #[derive(serde::Serialize)]
    struct ModelRow<'a> {
        #[serde(flatten)]
        info: &'a crate::codex::models::ModelInfo,
        listed: bool,
    }

    let arr = models
        .iter()
        .map(|m| ModelRow {
            info: m,
            listed: m.listed(),
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string_pretty(&arr)?);
    Ok(())
}

async fn debug_refresh(db: &Db, account: &str) -> Result<()> {
    let acc = db
        .find_account(account)
        .await?
        .ok_or_else(|| eyre!("no account matched"))?;
    let tokens = crate::oauth::refresh::refresh(&acc.refresh_token).await?;
    db.update_account_tokens(acc.id, &tokens).await?;
    println!(
        "refreshed account {} ({})",
        acc.id,
        acc.email.as_deref().unwrap_or("-")
    );
    Ok(())
}
