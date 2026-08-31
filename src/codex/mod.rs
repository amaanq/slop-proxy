pub mod client;
pub mod models;
pub mod rate_limits;
pub mod sse;
pub mod types;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;

use crate::config::Config;
use crate::db::Db;
use crate::pool::AccountPool;

pub async fn debug_models(db: &Db, cfg: &Config) -> Result<()> {
    let pool = AccountPool::load(db.clone()).await?;
    let client = client::CodexClient::new(cfg.codex.clone());
    let (access, account_id) = pool
        .any_active_credentials()
        .await
        .context("no usable account; run `slop-proxy login`")?;

    println!("GET {}", client.models_url());
    let (status, body) = client
        .models_raw(&access, &account_id)
        .await
        .map_err(|e| anyhow!(e))?;
    println!("status: {status}\n");
    match serde_json::from_str::<serde_json::Value>(&body) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap_or(body)),
        Err(_) => println!("{body}"),
    }
    Ok(())
}

pub async fn debug_ping(
    db: &Db,
    cfg: &Config,
    model: Option<String>,
    prompt: String,
) -> Result<()> {
    let accounts = db.list_accounts().await?;
    let account = accounts
        .iter()
        .find(|a| a.status != "disabled")
        .context("no usable account; run `slop-proxy login`")?
        .clone();

    let now = chrono::Utc::now().timestamp();
    let account = if account.access_expires_at.unwrap_or(0) < now + 60 {
        println!("refreshing access token for account {}...", account.id);
        let tokens = crate::oauth::refresh::refresh(&account.refresh_token).await?;
        db.update_account_tokens(account.id, &tokens).await?;
        db.find_account(&account.id.to_string())
            .await?
            .context("account vanished")?
    } else {
        account
    };

    let model = model
        .or_else(|| Some(cfg.models.default.clone()).filter(|m| !m.is_empty()))
        .context("pass --model (no default configured); see `slop-proxy models`")?;
    let mut req = types::ResponsesRequest::new(model.clone(), cfg.codex.instructions());
    req.input.push(types::InputItem::Message {
        role: "user".into(),
        content: vec![types::ContentPart::InputText { text: prompt }],
    });
    req.reasoning = Some(types::ReasoningConfig {
        effort: cfg
            .models
            .default_effort
            .clone()
            .unwrap_or_else(|| "low".into()),
        summary: "auto".into(),
    });

    println!(
        "POST {}/responses model={model} account={}",
        cfg.codex.base_url, account.id
    );
    let client = client::CodexClient::new(cfg.codex.clone());
    let req = serde_json::to_value(&req)?;
    let mut stream = match client
        .send(&account.access_token, &account.chatgpt_account_id, &req)
        .await
    {
        Ok(resp) => sse::event_stream(resp),
        Err(e) => bail!("upstream error: {e}"),
    };

    while let Some(ev) = stream.next().await {
        println!("{ev:?}");
    }
    Ok(())
}
