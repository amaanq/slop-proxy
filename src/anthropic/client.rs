use std::time::Duration;

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;

use crate::config::AnthropicConfig;
use crate::upstream::{Classify, SendError, classify};

const OAUTH_BETA: &str = "oauth-2025-04-20";

const RULES: Classify = Classify {
   pass: |status| !matches!(status, 401 | 429 | 500..=599),
   auth: &[401],
   reset_headers: &[
      "anthropic-ratelimit-unified-reset",
      "anthropic-ratelimit-requests-reset",
   ],
};

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
   pub is_active: bool,
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

   /// Max plans leave `weekly_all` inactive, and a dormant window reports a
   /// flat zero however much the account spends.
   pub fn windows(&self) -> impl Iterator<Item = (&'static str, &Window)> {
      let dormant: Vec<&'static str> = self
         .limits
         .iter()
         .filter(|limit| limit.scope.is_none() && !limit.is_active)
         .map(Limit::window_name)
         .collect();
      [("5h", &self.five_hour), ("7d", &self.seven_day)]
         .into_iter()
         .filter_map(|(name, slot)| slot.as_ref().map(|window| (name, window)))
         .filter(move |&(ref name, _)| !dormant.contains(name))
   }

   /// Per-model sub-limits, measured against their own allowance rather than
   /// the account's, so they are reported apart from `windows`.
   pub fn model_windows(&self) -> impl Iterator<Item = (String, &'static str, f64)> {
      self.limits.iter().filter_map(|limit| {
         let name = limit
            .scope
            .as_ref()?
            .model
            .as_ref()?
            .display_name
            .as_deref()?;
         Some((
            name.to_lowercase(),
            limit.window_name(),
            limit.percent / 100.0_f64,
         ))
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
   http: reqwest::Client,
   cfg: AnthropicConfig,
}

impl AnthropicClient {
   pub fn new(cfg: AnthropicConfig) -> Self {
      let http = reqwest::Client::builder()
         .connect_timeout(Duration::from_secs(30))
         .tcp_keepalive(Duration::from_secs(30))
         .build()
         .expect("building http client");
      Self { http, cfg }
   }

   pub const fn soft_utilization_limit(&self) -> f64 {
      self.cfg.soft_utilization_limit
   }

   pub async fn usage(&self, access_token: &str) -> Result<Usage, SendError> {
      let resp = self
         .http
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
         .http
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
      access_token: &str,
      path: &str,
      body: &Bytes,
      hdrs: &RelayHeaders,
   ) -> Result<reqwest::Response, SendError> {
      let beta = match hdrs.beta.as_ref() {
         Some(beta) if beta.split(',').any(|part| part.trim() == OAUTH_BETA) => beta.clone(),
         Some(beta) => format!("{OAUTH_BETA},{beta}"),
         None => OAUTH_BETA.into(),
      };
      let mut req = self
         .http
         .post(format!("{}{path}", self.cfg.base_url.trim_end_matches('/')))
         .bearer_auth(access_token)
         .header(
            "anthropic-version",
            hdrs.version.as_deref().unwrap_or("2023-06-01"),
         )
         .header("anthropic-beta", beta)
         .header(CONTENT_TYPE, "application/json")
         .body(body.clone());
      if let Some(agent) = hdrs.user_agent.as_ref() {
         req = req.header("user-agent", agent);
      }
      let resp = req
         .send()
         .await
         .map_err(|err| SendError::Network(err.to_string()))?;
      classify(resp, RULES).await
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
      let got: Vec<_> = usage.model_windows().collect();
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
}
