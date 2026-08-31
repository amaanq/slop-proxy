use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use crate::config::Config;
use crate::db::Db;

#[derive(Parser)]
#[command(
    name = "slop-proxy",
    about = "Anthropic/OpenAI API proxy backed by Codex subscription accounts"
)]
pub struct Cli {
    /// Path to the sqlite database
    #[arg(long, global = true)]
    pub db: Option<PathBuf>,

    /// Path to config.toml
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Log in to a Codex (ChatGPT) account and store it
    Login {
        /// Human-readable label for the account
        #[arg(long)]
        label: Option<String>,
    },
    /// Manage stored Codex accounts
    Accounts {
        #[command(subcommand)]
        command: AccountsCommand,
    },
    /// Manage issued API tokens
    Token {
        #[command(subcommand)]
        command: TokenCommand,
    },
    /// Run the API server
    Serve {
        /// Listen address
        #[arg(long)]
        bind: Option<String>,
    },
    /// Show usage statistics as JSON
    Stats {
        /// Accepted for compatibility; output is always JSON
        #[arg(long, hide = true)]
        json: bool,
        /// Window start: 24h, 7d, 30m, or RFC3339
        #[arg(long)]
        since: Option<String>,
        /// Window end (RFC3339)
        #[arg(long)]
        until: Option<String>,
    },
    /// List the models available from the codex backend, as JSON
    Models,
    /// Debug helpers
    #[command(hide = true)]
    Debug {
        #[command(subcommand)]
        command: DebugCommand,
    },
}

#[derive(Subcommand)]
pub enum AccountsCommand {
    /// List stored accounts
    List,
    /// Remove an account by id or email
    Remove { account: String },
}

#[derive(Subcommand)]
pub enum TokenCommand {
    /// Issue a new API token for a user
    Create {
        #[arg(long)]
        user: String,
        /// Maximum requests in each rolling window
        #[arg(long)]
        requests: Option<i64>,
        /// Maximum input plus output tokens in each rolling window
        #[arg(long)]
        tokens: Option<i64>,
        #[arg(long, default_value_t = 3600)]
        window_seconds: i64,
        /// Delay every admitted request by this many milliseconds
        #[arg(long, default_value_t = 0)]
        slowdown_ms: i64,
    },
    /// List issued tokens
    List,
    /// Revoke a token by id or prefix
    Revoke { token: String },
    /// Replace limits for a token id or prefix; omitted limits are unlimited
    Limits {
        token: String,
        #[arg(long)]
        requests: Option<i64>,
        #[arg(long)]
        tokens: Option<i64>,
        #[arg(long, default_value_t = 3600)]
        window_seconds: i64,
        #[arg(long, default_value_t = 0)]
        slowdown_ms: i64,
    },
    /// Show metered usage for a token's current rolling window
    Usage { token: String },
}

#[derive(Subcommand)]
pub enum DebugCommand {
    /// Send a raw request upstream and dump the SSE events
    Ping {
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "Say the word: pong")]
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
        Command::Login { label } => crate::oauth::login(&db, label).await,
        Command::Accounts { command } => match command {
            AccountsCommand::List => accounts_list(&db).await,
            AccountsCommand::Remove { account } => accounts_remove(&db, &account).await,
        },
        Command::Token { command } => match command {
            TokenCommand::Create {
                user,
                requests,
                tokens,
                window_seconds,
                slowdown_ms,
            } => {
                let limits = token_limits(requests, tokens, window_seconds, slowdown_ms)?;
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
            } => {
                let limits = token_limits(requests, tokens, window_seconds, slowdown_ms)?;
                token_set_limits(&db, &token, &limits).await
            }
            TokenCommand::Usage { token } => token_usage(&db, &token).await,
        },
        Command::Serve { bind } => {
            let bind = bind.unwrap_or_else(|| cfg.bind.clone());
            crate::server::serve(db, cfg, &bind).await
        }
        Command::Stats {
            json: _,
            since,
            until,
        } => crate::stats::run(&db, since, until).await,
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

async fn accounts_list(db: &Db) -> Result<()> {
    // A struct keeps field order; serde_json alphabetizes json! maps.
    #[derive(serde::Serialize)]
    struct AccountRow<'a> {
        id: i64,
        email: Option<&'a str>,
        plan_type: Option<&'a str>,
        status: &'a str,
        label: Option<&'a str>,
        cooldown_seconds_left: Option<i64>,
        disabled_reason: Option<&'a str>,
    }

    let accounts = db.list_accounts().await?;
    let now = chrono::Utc::now().timestamp();
    let rows: Vec<AccountRow> = accounts
        .iter()
        .map(|a| AccountRow {
            id: a.id,
            email: a.email.as_deref(),
            plan_type: a.plan_type.as_deref(),
            status: &a.status,
            label: a.label.as_deref(),
            cooldown_seconds_left: (a.status == "cooldown")
                .then(|| a.cooldown_until.map(|c| (c - now).max(0)))
                .flatten(),
            disabled_reason: a.disabled_reason.as_deref(),
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&rows)?);
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
    let rows: Vec<TokenRow> = tokens
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
        .collect();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

fn token_limits(
    requests: Option<i64>,
    tokens: Option<i64>,
    window_seconds: i64,
    slowdown_ms: i64,
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
    Ok(crate::db::tokens::TokenLimits {
        requests,
        tokens,
        window_seconds,
        slowdown_ms,
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
        .with_context(|| format!("no token matched {token:?}"))?;
    println!("{}", serde_json::to_string_pretty(&usage)?);
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
    let pool = crate::pool::AccountPool::load(db.clone()).await?;
    let client = crate::codex::client::CodexClient::new(cfg.codex.clone());

    let models = match pool.any_active_credentials().await {
        Some((access, account_id)) => match client.list_models(&access, &account_id).await {
            Ok(models) => models,
            Err(e) => {
                eprintln!("could not fetch models from the codex backend: {e}");
                Vec::new()
            }
        },
        None => {
            eprintln!("no usable account; run `slop-proxy login`");
            Vec::new()
        }
    };

    let arr: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let mut v = serde_json::to_value(m).unwrap_or_else(|_| serde_json::json!({}));
            if let Some(o) = v.as_object_mut() {
                o.insert("listed".into(), serde_json::json!(m.listed()));
            }
            v
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&arr)?);
    Ok(())
}

async fn debug_refresh(db: &Db, account: &str) -> Result<()> {
    let acc = db
        .find_account(account)
        .await?
        .context("no account matched")?;
    let tokens = crate::oauth::refresh::refresh(&acc.refresh_token).await?;
    db.update_account_tokens(acc.id, &tokens).await?;
    println!(
        "refreshed account {} ({})",
        acc.id,
        acc.email.as_deref().unwrap_or("-")
    );
    Ok(())
}
