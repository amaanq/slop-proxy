use axum::body::Bytes;
use reqwest::StatusCode;
use reqwest::header::{CONTENT_TYPE, HeaderMap};

use crate::config::AnthropicConfig;
use crate::egress::Egresses;
use crate::provider::AuthMode;
use crate::upstream::{Classify, SendError, classify};

const OAUTH_BETA: &str = "oauth-2025-04-20";

const RULES: Classify = Classify {
   pass: |status| !matches!(status, 401 | 429 | 500..=599),
   auth: &[401],
   reset_headers: &[
      "anthropic-ratelimit-unified-reset",
      "anthropic-ratelimit-requests-reset",
   ],
   account_faults: &["Your credit balance is too low"],
};

/// Every claim a request counts against gets its own `anthropic-ratelimit-unified-<claim>-status` header.
fn sub_limit_rejected(headers: &HeaderMap) -> bool {
   let claim_rejected = |claim: &str| {
      headers
         .get(format!("anthropic-ratelimit-unified-{claim}-status"))
         .is_some_and(|value| value == "rejected")
   };
   let sub_limit = headers.iter().any(|(name, value)| {
      value == "rejected"
         && name
            .as_str()
            .strip_prefix("anthropic-ratelimit-unified-")
            .and_then(|rest| rest.strip_suffix("-status"))
            .is_some_and(|claim| !matches!(claim, "5h" | "7d" | "overage"))
   });
   sub_limit && !claim_rejected("5h") && !claim_rejected("7d")
}

/// Rolling-window usage as the subscription reports it, without spending an
/// inference request. `locked_reason` is set when the window is exhausted
/// rather than merely busy.
#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct Window {
   #[serde(default)]
   pub utilization: f64,
   #[serde(default)]
   pub locked_reason: Option<String>,
   #[serde(default)]
   pub resets_at: Option<String>,
}

impl Window {
   pub fn resets_at_unix(&self) -> Option<i64> {
      self
         .resets_at
         .as_deref()?
         .parse::<jiff::Timestamp>()
         .ok()
         .map(jiff::Timestamp::as_second)
   }
}

/// A model with a quota of its own, carved out of the account's weekly
/// allowance. The API names the model inside `scope` rather than in the key,
/// unlike the codenamed top-level fields, so this survives a model rename.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Limit {
   #[serde(default)]
   pub group: String,
   #[serde(default)]
   pub percent: f64,
   #[serde(default)]
   pub scope: Option<Scope>,
   #[serde(default)]
   pub is_active: Option<bool>,
   #[serde(default)]
   pub resets_at: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Scope {
   #[serde(default)]
   pub model: Option<ScopedModel>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScopedModel {
   #[serde(default)]
   pub display_name: Option<String>,
}

impl Limit {
   pub fn resets_at_unix(&self) -> Option<i64> {
      self
         .resets_at
         .as_deref()?
         .parse::<jiff::Timestamp>()
         .ok()
         .map(jiff::Timestamp::as_second)
   }

   fn window_name(&self) -> &'static str {
      match self.group.as_str() {
         "session" => "5h",
         _ => "7d",
      }
   }
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct Usage {
   #[serde(default)]
   pub five_hour: Option<Window>,
   #[serde(default)]
   pub seven_day: Option<Window>,
   #[serde(default)]
   pub limits: Vec<Limit>,
}

impl Usage {
   pub fn locked(&self) -> bool {
      [&self.five_hour, &self.seven_day]
         .into_iter()
         .flatten()
         .any(|window| window.locked_reason.is_some())
   }

   pub fn windows(&self) -> impl Iterator<Item = (&'static str, &Window)> {
      let dormant: Vec<&'static str> = self
         .limits
         .iter()
         .filter(|limit| {
            limit.scope.is_none() && limit.is_active == Some(false) && limit.percent == 0.0_f64
         })
         .map(Limit::window_name)
         .collect();
      [("5h", &self.five_hour), ("7d", &self.seven_day)]
         .into_iter()
         .filter_map(|(name, slot)| slot.as_ref().map(|window| (name, window)))
         .filter(move |&(ref name, _)| !dormant.contains(name))
   }

   pub fn cooldown_is_obsolete(&self, until: i64) -> bool {
      if self
         .windows()
         .any(|(_, window)| window.utilization >= 100.0_f64 || window.locked_reason.is_some())
         || self
            .limits
            .iter()
            .any(|limit| limit.is_active != Some(false) && limit.percent >= 100.0_f64)
      {
         return false;
      }
      self.limits.iter().any(|limit| {
         limit.scope.is_none()
            && limit.is_active == Some(false)
            && limit
               .resets_at_unix()
               .is_some_and(|reset| reset.abs_diff(until) <= 1)
      })
   }

   /// Per-model sub-limits, measured against their own allowance rather than
   /// the account's, so they are reported apart from `windows`.
   pub fn model_windows(&self) -> impl Iterator<Item = (String, &'static str, &Limit)> {
      self.limits.iter().filter_map(|limit| {
         let name = limit
            .scope
            .as_ref()?
            .model
            .as_ref()?
            .display_name
            .as_deref()?;
         Some((name.to_lowercase(), limit.window_name(), limit))
      })
   }
}

/// Client headers worth carrying through to the upstream request.
#[derive(Debug, Default, Clone)]
pub struct RelayHeaders {
   pub version: Option<String>,
   pub beta: Option<String>,
   pub user_agent: Option<String>,
}

pub struct AnthropicClient {
   direct: Egresses,
   /// `None` when no egress proxies are configured, so the direct pool is the
   /// only one that exists.
   proxied: Option<Egresses>,
   cfg: AnthropicConfig,
}

impl AnthropicClient {
   pub fn new(cfg: AnthropicConfig) -> eyre::Result<Self> {
      let urls = cfg.egress.urls()?;
      Ok(Self {
         direct: Egresses::new(&[], "anthropic", None)?,
         proxied: (!urls.is_empty())
            .then(|| Egresses::new(&urls, "anthropic", None))
            .transpose()?,
         cfg,
      })
   }

   /// An account marked with `accounts egress` leaves through the proxies, and
   /// everything else, the pooled seats included, dials Anthropic directly.
   const fn egresses(&self, via_proxy: bool) -> &Egresses {
      if via_proxy && let Some(proxied) = self.proxied.as_ref() {
         return proxied;
      }
      &self.direct
   }

   pub const fn soft_utilization_limit(&self) -> f64 {
      self.cfg.soft_utilization_limit
   }

   pub async fn usage(&self, access_token: &str) -> Result<Usage, SendError> {
      let resp = self
         .direct
         .http(0)
         .get(format!(
            "{}/api/oauth/usage",
            self.cfg.base_url.trim_end_matches('/')
         ))
         .bearer_auth(access_token)
         .header("anthropic-beta", OAUTH_BETA)
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      let resp = classify(resp, Classify::STRICT).await?;
      let status = resp.status().as_u16();
      resp.json().await.map_err(|err| SendError::Upstream {
         status,
         body: format!("parsing usage response: {err}"),
      })
   }

   /// The catalog exactly as the backend sends it. Relayed rather than
   /// rebuilt so a client sees the same model ids and display names it would
   /// talking to Anthropic directly.
   pub async fn models_raw(
      &self,
      access_token: &str,
   ) -> Result<(reqwest::StatusCode, String), SendError> {
      let resp = self
         .direct
         .http(0)
         .get(format!(
            "{}/v1/models?limit=100",
            self.cfg.base_url.trim_end_matches('/')
         ))
         .bearer_auth(access_token)
         .header("anthropic-beta", OAUTH_BETA)
         .header("anthropic-version", "2023-06-01")
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      let status = resp.status();
      let status_u16 = status.as_u16();
      let body = resp.text().await.map_err(|err| SendError::Upstream {
         status: status_u16,
         body: format!("reading models response: {err}"),
      })?;
      Ok((status, body))
   }

   /// Statuses other than 401/429/5xx come back as `Ok` so the caller can
   /// forward them verbatim, and only failures worth retrying on another
   /// account become errors.
   pub async fn post(
      &self,
      credential: &str,
      mode: AuthMode,
      via_egress: bool,
      path: &str,
      body: &Bytes,
      hdrs: &RelayHeaders,
   ) -> Result<reqwest::Response, SendError> {
      let beta = match mode {
         AuthMode::OAuth => Some(match hdrs.beta.as_ref() {
            Some(beta) if beta.split(',').any(|part| part.trim() == OAUTH_BETA) => beta.clone(),
            Some(beta) => format!("{OAUTH_BETA},{beta}"),
            None => OAUTH_BETA.into(),
         }),
         // A key cannot claim the subscription beta, but the caller's own flags
         // still gate the fields it sent, `context_management` among them.
         AuthMode::ApiKey => hdrs.beta.clone(),
      };
      let url = format!("{}{path}", self.cfg.base_url.trim_end_matches('/'));
      let resp = self
         .egresses(via_egress)
         .send(|http| {
            let url = url.clone();
            let beta = beta.clone();
            async move {
               let mut req = http
                  .post(url)
                  .header(
                     "anthropic-version",
                     hdrs.version.as_deref().unwrap_or("2023-06-01"),
                  )
                  .header(CONTENT_TYPE, "application/json")
                  .body(body.clone());
               req = match mode {
                  AuthMode::OAuth => req.bearer_auth(credential),
                  AuthMode::ApiKey => req.header("x-api-key", credential),
               };
               if let Some(beta) = beta {
                  req = req.header("anthropic-beta", beta);
               }
               if let Some(agent) = hdrs.user_agent.as_ref() {
                  req = req.header("user-agent", agent);
               }
               req.send()
                  .await
                  .map_err(|err| SendError::Network(err.to_string()))
            }
         })
         .await?;
      let model_limited =
         resp.status() == StatusCode::TOO_MANY_REQUESTS && sub_limit_rejected(resp.headers());
      match classify(resp, RULES).await {
         Err(SendError::RateLimited {
            retry_after,
            body: text,
         }) if model_limited => Err(SendError::ModelLimited {
            retry_after,
            body: text,
         }),
         other => other,
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   /// Trimmed from a live `/api/oauth/usage` response.
   const USAGE: &str = r#"{
      "five_hour": {"utilization": 3.0, "resets_at": "2026-09-01T08:30:00.007788+00:00"},
      "seven_day": {"utilization": 89.0, "resets_at": "2026-09-02T19:00:00.007811+00:00"},
      "seven_day_opus": null,
      "nimbus_quill": {"utilization": 0.0},
      "limits": [
        {"kind": "session", "group": "session", "percent": 3, "scope": null, "is_active": true},
        {"kind": "weekly_all", "group": "weekly", "percent": 89, "scope": null, "is_active": true},
        {"kind": "weekly_scoped", "group": "weekly", "percent": 63, "is_active": true,
         "scope": {"model": {"id": null, "display_name": "Fable"}, "surface": null}}
      ]
    }"#;

   #[test]
   fn a_scoped_model_is_named_from_its_scope() {
      let usage: Usage = serde_json::from_str(USAGE).unwrap();
      let got: Vec<_> = usage
         .model_windows()
         .map(|(model, window, limit)| (model, window, limit.percent / 100.0_f64))
         .collect();
      assert_eq!(got, vec![("fable".to_owned(), "7d", 0.63_f64)]);
   }

   #[test]
   fn a_dormant_window_is_not_reported() {
      let usage: Usage = serde_json::from_str(
         r#"{
              "five_hour": {"utilization": 1.0},
              "seven_day": {"utilization": 0.0},
              "limits": [
                {"group": "session", "percent": 1, "is_active": true},
                {"group": "weekly", "percent": 0, "is_active": false}
              ]
            }"#,
      )
      .unwrap();
      let got: Vec<_> = usage.windows().map(|(name, _)| name).collect();
      assert_eq!(got, vec!["5h"]);
   }

   /// The codenamed top-level fields come and go, so a payload without the
   /// array must not start reporting sub-limits that are not there.
   #[test]
   fn a_payload_without_limits_reports_none() {
      let usage: Usage = serde_json::from_str(r#"{"seven_day": {"utilization": 10.0}}"#).unwrap();
      assert_eq!(usage.model_windows().count(), 0);
   }

   #[test]
   fn only_an_explicitly_inactive_matching_window_clears_a_cooldown() {
      let reset = "2026-09-06T17:00:00.213156+00:00";
      let until = reset.parse::<jiff::Timestamp>().unwrap().as_second();
      let limit = serde_json::json!({
         "group": "weekly", "is_active": false, "percent": 0_i32, "resets_at": reset
      });
      let parse = |value| serde_json::from_value::<Usage>(value).unwrap();
      let usage = parse(serde_json::json!({
         "five_hour": {"utilization": 26.0_f64}, "limits": [limit]
      }));
      assert!(usage.cooldown_is_obsolete(until));
      assert!(usage.cooldown_is_obsolete(until - 1));
      assert!(!usage.cooldown_is_obsolete(until - 3600));

      for (field, value) in [
         ("is_active", serde_json::json!(true)),
         ("is_active", serde_json::Value::Null),
         ("resets_at", serde_json::Value::Null),
         ("resets_at", serde_json::json!("invalid")),
         (
            "scope",
            serde_json::json!({"model": {"display_name": "Fable"}}),
         ),
      ] {
         let mut changed = limit.clone();
         changed[field] = value;
         assert!(!parse(serde_json::json!({"limits": [changed]})).cooldown_is_obsolete(until));
      }
      for window in [
         serde_json::json!({"utilization": 100.0_f64}),
         serde_json::json!({"utilization": 0.0_f64, "locked_reason": "quota"}),
      ] {
         assert!(
            !parse(serde_json::json!({
               "five_hour": window, "limits": [limit]
            }))
            .cooldown_is_obsolete(until)
         );
      }
      let scoped = serde_json::json!({
         "group": "weekly", "is_active": true, "percent": 100_i32,
         "scope": {"model": {"display_name": "Fable"}}
      });
      assert!(!parse(serde_json::json!({"limits": [limit, scoped]})).cooldown_is_obsolete(until));
      assert!(
         !parse(serde_json::json!({"five_hour": {"utilization": 0.0_f64}}))
            .cooldown_is_obsolete(until)
      );
   }
}
