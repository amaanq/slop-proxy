use crate::anthropic::Model;
use crate::glm::{GlmClient, QuotaLimit, QuotaUnit};
use crate::pool::{AccountUsage, Backend, Pool, Relay, Route, Slot, UsageWindow};
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Z.ai keys.
pub type GlmPool = Pool<GlmClient>;

impl GlmPool {
   pub async fn models(&self) -> Option<Vec<Model>> {
      self
         .first_answer(async |backend, key, _| backend.models(key).await)
         .await
   }

   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.quota(&token).await {
            Ok(report) => {
               let usage = AccountUsage {
                  windows: report.limits.iter().map(window).collect(),
                  locked: report.limits.iter().any(|limit| limit.remaining <= 0.0_f64),
                  ..AccountUsage::default()
               };
               self.slots.note_usage(&slot, usage).await;
            },
            Err(err) => tracing::debug!("usage for {}: {err}", slot.display),
         }
      }
   }
}

fn window(limit: &QuotaLimit) -> UsageWindow {
   let name = match limit.unit {
      QuotaUnit::Hour => format!("{}h", limit.number),
      QuotaUnit::Week => format!("{}d", limit.number * 7),
   };
   UsageWindow {
      name,
      utilization: if limit.allowance > 0.0_f64 {
         limit.used / limit.allowance
      } else {
         0.0_f64
      },
      resets_at: limit.next_reset_ms.map(|millis| millis / 1000),
   }
}

impl Backend for GlmClient {
   const PROVIDER: Provider = Provider::Glm;
   type Request = Relay;
   type Response = reqwest::Response;

   async fn send(
      &self,
      token: &str,
      _slot: &Slot,
      route: Route<'_>,
      req: &Self::Request,
   ) -> Result<Self::Response, SendError> {
      Self::post(self, token, req.path, &req.body, route.session_key).await
   }
}
