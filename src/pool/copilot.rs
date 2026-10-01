use axum::body::Bytes;

use super::{AccountUsage, AuthPolicy, Backend, Cooldown, Pool, Route, Slot, UsageWindow};
use crate::copilot::client::{CopilotClient, QuotaReport};
use crate::provider::Provider;
use crate::translate::chat::ChatError;
use crate::upstream::SendError;

/// Pool over Copilot accounts. The pool hands out minted Copilot tokens and
/// each send carries the chat-completions body the caller already wrote.
pub type CopilotPool = Pool<CopilotClient>;

#[derive(Clone)]
pub struct Call {
   pub body: Bytes,
}

impl Backend for CopilotClient {
   const PROVIDER: Provider = Provider::Copilot;
   const RATE_LIMIT: Cooldown = Cooldown {
      max: 3600,
      base: 60,
   };
   const ON_AUTH: AuthPolicy = AuthPolicy::RefreshOnce;
   type Request = Call;
   type Response = reqwest::Response;

   fn reason(body: String) -> String {
      ChatError::reason(body)
   }

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
      self.post(token, &req.body).await
   }
}

impl Pool<CopilotClient> {
   /// The first account that answers. Every seat sees the same catalog, so
   /// there is nothing to merge across accounts.
   pub async fn models(&self) -> Vec<String> {
      for slot in self.slots.list().await {
         let Ok(key) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.models(&key).await {
            Ok(listed) => return listed,
            Err(err) => tracing::debug!("models for {}: {err}", slot.display),
         }
      }
      Vec::new()
   }

   /// Reads each account's quota from the provider. This needs no inference
   /// request, so idle accounts report real numbers and the router ranks on
   /// them.
   pub async fn poll_usage(&self) {
      for slot in self.slots.list().await {
         let Ok(token) = self.slots.fresh_token(&slot, false).await else {
            continue;
         };
         match self.backend.quota(&token).await {
            Ok(report) => {
               self
                  .slots
                  .note_usage(
                     &slot,
                     AccountUsage {
                        windows: quota_windows(&report),
                        model_windows: Vec::new(),
                        locked: quota_locked(&report),
                        observed_at: 0,
                     },
                  )
                  .await;
            },
            Err(err) => tracing::debug!("usage for {}: {err}", slot.display),
         }
      }
   }
}

/// The billed budget as the one routing window. Only `premium_interactions`
/// is metered against spend here; chat and completions ride along as
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

#[cfg(test)]
mod tests {
   use super::*;
   use crate::copilot::client::{QuotaDetail, QuotaSnapshots};

   #[test]
   fn empty_quota_reports_no_windows() {
      assert!(quota_windows(&QuotaReport::default()).is_empty());
      assert!(!quota_locked(&QuotaReport::default()));
   }

   #[test]
   fn spent_premium_locks_the_account() {
      let report = QuotaReport {
         quota_snapshots: QuotaSnapshots {
            premium_interactions: Some(QuotaDetail {
               entitlement: 100,
               remaining: 0,
               ..QuotaDetail::default()
            }),
            ..QuotaSnapshots::default()
         },
         ..QuotaReport::default()
      };
      assert!(quota_locked(&report));
      let window = &quota_windows(&report)[0];
      assert_eq!(window.name.as_str(), "30d");
      assert!((window.utilization - 1.0_f64).abs() < f64::EPSILON);
   }
}
