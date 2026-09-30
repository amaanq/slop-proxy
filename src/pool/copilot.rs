use axum::body::Bytes;

use super::{AccountUsage, AuthPolicy, Backend, Pool, Route, Slot, UsageWindow};
use crate::copilot::{CopilotClient, QuotaReport};
use crate::provider::Provider;
use crate::upstream::SendError;

pub type CopilotPool = Pool<CopilotClient>;

#[derive(Clone)]
pub struct Call {
   pub body: Bytes,
   /// Copilot bills a replayed assistant or tool turn under a different
   /// initiator than a fresh user turn.
   pub agent: bool,
   pub vision: bool,
}

impl Backend for CopilotClient {
   const PROVIDER: Provider = Provider::Copilot;
   const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
   type Request = Call;
   type Response = reqwest::Response;

   fn soft_limit(&self) -> f64 {
      self.soft_utilization_limit()
   }

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      _route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      self.post(token, &req.body, req.agent, req.vision).await
   }
}

impl Pool<CopilotClient> {
   pub async fn models(&self) -> Vec<String> {
      self
         .first_answer(async |backend, key, _| backend.models(key).await)
         .await
         .unwrap_or_default()
   }

   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.quota(&token).await {
            Ok(report) => {
               let usage = AccountUsage {
                  windows: quota_windows(&report),
                  model_windows: Vec::new(),
                  locked: quota_locked(&report),
                  observed_at: 0,
               };
               self.slots.note_usage(&slot, usage).await;
            },
            Err(err) => tracing::debug!("usage for {}: {err}", slot.display),
         }
      }
   }
}

/// Only `premium_interactions` is billed. `chat` and `completions` are
/// unlimited on most plans and would flatten the band if mixed in.
fn quota_windows(report: &QuotaReport) -> Vec<UsageWindow> {
   let Some(premium) = report.quota_snapshots.premium_interactions.as_ref() else {
      return Vec::new();
   };
   vec![UsageWindow {
      name: "30d".into(),
      utilization: premium.utilization(),
      resets_at: report.resets_at(),
   }]
}

fn quota_locked(report: &QuotaReport) -> bool {
   report
      .quota_snapshots
      .premium_interactions
      .as_ref()
      .is_some_and(|premium| !premium.unlimited && premium.remaining <= 0)
}
