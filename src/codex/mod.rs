pub mod client;
pub mod models;
pub mod sse;
pub mod types;

use axum::body::Bytes;
use eyre::{Result, bail, eyre};
use futures_util::StreamExt as _;

use crate::clock;
use crate::config::Config;
use crate::db::Db;
use crate::db::accounts::AccountStatus;
use crate::oauth::refresh;
use crate::pool::codex::CodexPool;
use crate::provider::Provider;

pub async fn debug_models(db: &Db, cfg: &Config) -> Result<()> {
   let client = client::CodexClient::new(cfg.codex.clone());
   let pool = CodexPool::load(db.clone(), client).await?;
   let (access, account_id) = pool
      .any_active_credentials()
      .await
      .ok_or_else(|| eyre!("no usable account; run `slop-proxy login`"))?;

   println!("GET {}", pool.client().models_url());
   let (status, body) = pool
      .client()
      .models_raw(&access, &account_id)
      .await
      .map_err(|err| eyre!(err))?;
   println!("status: {status}\n");
   match serde_json::from_str::<serde_json::Value>(&body) {
      Ok(value) => println!("{}", serde_json::to_string_pretty(&value).unwrap_or(body)),
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
      .find(|account| {
         account.provider == Provider::OpenAi && account.status != AccountStatus::Disabled
      })
      .ok_or_else(|| eyre!("no usable account; run `slop-proxy login`"))?
      .clone();

   let now = clock::unix_now();
   let account = if account.access_expires_at.unwrap_or(0) < now + 60 {
      println!("refreshing access token for account {}...", account.id);
      let tokens = refresh::refresh(&account.refresh_token).await?;
      db.update_account_tokens(account.id, &tokens).await?;
      db.find_account(&account.id.to_string())
         .await?
         .ok_or_else(|| eyre!("account vanished"))?
   } else {
      account
   };

   let model = model
      .or_else(|| Some(cfg.models.default.clone()).filter(|configured| !configured.is_empty()))
      .ok_or_else(|| eyre!("pass --model (no default configured); see `slop-proxy models`"))?;
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
   let req = Bytes::from(serde_json::to_vec(&req)?);
   let mut stream = match client
      .post(
         &account.access_token,
         &account.provider_account_id,
         &req,
         &uuid::Uuid::new_v4().to_string(),
      )
      .await
   {
      Ok(resp) => sse::event_stream(resp),
      Err(err) => bail!("upstream error: {err}"),
   };

   while let Some(event) = stream.next().await {
      println!(
         "{}",
         serde_json::to_string_pretty(&event)
            .unwrap_or_else(|_| String::from("<unserializable event>"))
      );
   }
   Ok(())
}
