use std::path::PathBuf;

use eyre::{Result, bail, eyre};
use pound::Parse;

use crate::config::Config;
use crate::db::Db;
use crate::provider::Provider;

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
        #[pound(long, default = "codex")]
        provider: Provider,
    },
    /// Manage stored Codex accounts
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
    /// Remove an account by id or email
    Remove { account: String },
}

#[derive(Parse)]
pub enum TokenCommand {
    /// Issue a new API token for a user
    Create {
        #[pound(long)]
        user: String,
    },
    /// List issued tokens
    List,
    /// Revoke a token by id or prefix
    Revoke { token: String },
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
            Provider::Codex => crate::oauth::login(&db, label).await,
            Provider::Anthropic => crate::oauth::anthropic::login(&db, label).await,
        },
        Command::Accounts { command } => match command {
            AccountsCommand::List => accounts_list(&db).await,
            AccountsCommand::Remove { account } => accounts_remove(&db, &account).await,
        },
        Command::Token { command } => match command {
            TokenCommand::Create { user } => token_create(&db, &user).await,
            TokenCommand::List => token_list(&db).await,
            TokenCommand::Revoke { token } => token_revoke(&db, &token).await,
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

async fn accounts_list(db: &Db) -> Result<()> {
    // A struct keeps field order; serde_json alphabetizes json! maps.
    #[derive(serde::Serialize)]
    struct AccountRow<'a> {
        id: i64,
        provider: &'a str,
        email: Option<&'a str>,
        plan_type: Option<&'a str>,
        status: &'a str,
        label: Option<&'a str>,
        cooldown_seconds_left: Option<i64>,
        disabled_reason: Option<&'a str>,
    }

    let accounts = db.list_accounts().await?;
    let now = crate::clock::unix_now();
    let rows = accounts
        .iter()
        .map(|a| AccountRow {
            id: a.id,
            provider: a.provider.as_str(),
            email: a.email.as_deref(),
            plan_type: a.plan_type.as_deref(),
            status: &a.status,
            label: a.label.as_deref(),
            cooldown_seconds_left: (a.status == "cooldown")
                .then(|| a.cooldown_until.map(|c| (c - now).max(0)))
                .flatten(),
            disabled_reason: a.disabled_reason.as_deref(),
        })
        .collect::<Vec<AccountRow>>();
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

async fn token_create(db: &Db, user: &str) -> Result<()> {
    let (raw, prefix) = crate::db::tokens::generate();
    db.create_token(user, &raw, &prefix).await?;
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
