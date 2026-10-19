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
mod copilot;
mod db;
mod deepseek;
mod egress;
mod experiential;
mod gemini;
mod glm;
mod oauth;
mod openrouter;
mod pool;
mod pricing;
mod provider;
mod server;
mod stats;
mod translate;
mod upstream;
mod zen;

use std::env;

use misstep::Result;
use pound::Parse as _;
use tracing::Level;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

#[tokio::main]
async fn main() -> Result<()> {
   misstep::install_panic_hook();
   let args = cli::Cli::parse();

   let level = if args.verbose {
      Level::DEBUG
   } else {
      Level::INFO
   };
   let filter = env::var("RUST_LOG")
      .ok()
      .and_then(|directives| directives.parse::<Targets>().ok())
      .unwrap_or_else(|| Targets::new().with_target("slop_proxy", level));
   tracing_subscriber::registry()
      .with(fmt::layer())
      .with(filter)
      .init();

   let cfg = config::Config::load(&args)?;
   cli::run(args, cfg).await
}
