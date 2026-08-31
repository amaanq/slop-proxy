mod cli;
mod codex;
mod config;
mod db;
mod oauth;
mod pool;
mod server;
mod stats;
mod translate;

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                if args.verbose {
                    "slop_proxy=debug".into()
                } else {
                    "slop_proxy=info".into()
                }
            }),
        )
        .init();

    let cfg = config::Config::load(&args)?;
    cli::run(args, cfg).await
}
