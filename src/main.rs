#![cfg_attr(
    test,
    expect(
        clippy::panic,
        clippy::wildcard_enum_match_arm,
        reason = "a test matches the one variant it expects and dies loudly on any other"
    )
)]

mod anthropic;
mod cli;
mod clock;
mod codex;
mod config;
mod db;
mod gemini;
mod glm;
mod oauth;
mod pool;
mod pricing;
mod provider;
mod server;
mod stats;
mod translate;
mod upstream;
mod zen;

use eyre::Result;
use pound::Parse as _;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
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
