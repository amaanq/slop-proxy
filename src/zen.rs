//! `OpenCode` Zen speaks both the `Responses` and the messages API, one
//! dialect per model, so nothing here translates. The request goes up as the
//! caller wrote it and comes back as frames that caller already understands.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, RwLock};

use axum::body::Bytes;
use rand::distributions::Alphanumeric;
use rand::{Rng as _, thread_rng};
use reqwest::header::CONTENT_TYPE;

use crate::clock::{unix_now, unix_now_ms};
use crate::config::ZenConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, IdList, SendError, classify, json};

const CONTEXT_REFRESH_SECS: i64 = 12 * 60 * 60;

pub struct ZenClient {
   base_url: String,
   models_dev_url: String,
   egresses: Egresses,
   context_windows: RwLock<Arc<HashMap<String, i64>>>,
   context_fetched_at: AtomicI64,
}

pub struct ZenModel {
   pub id: String,
   pub context_window: Option<i64>,
}

impl ZenClient {
   pub fn new(cfg: ZenConfig) -> eyre::Result<Self> {
      let egresses = Egresses::new(&cfg.egress, "zen", Some(&cfg.user_agent))?;
      Ok(Self {
         base_url: cfg.base_url,
         models_dev_url: cfg.models_dev_url,
         egresses,
         context_windows: RwLock::default(),
         context_fetched_at: AtomicI64::new(0),
      })
   }

   pub const fn egresses(&self) -> &Egresses {
      &self.egresses
   }

   fn base_url(&self) -> &str {
      self.base_url.trim_end_matches('/')
   }

   /// The free contributor models answer without any credential at all, so
   /// the key is optional and only attached when an account supplies one.
   pub async fn post(
      &self,
      key: Option<&str>,
      session: &str,
      path: &str,
      req: &Bytes,
   ) -> Result<reqwest::Response, SendError> {
      let key = key.filter(|key| !key.is_empty());
      let request_id = request_id();
      let request_id = request_id.as_str();
      let attempt = move |http: reqwest::Client| async move {
         let mut builder = http
            .post(format!("{}{path}", self.base_url()))
            .header("Accept", "*/*")
            .header("x-opencode-session", session)
            .header("x-opencode-request", request_id)
            .header("x-opencode-client", "cli")
            .header("x-opencode-project", "global");
         builder = match key {
            Some(key) => builder.bearer_auth(key),
            None => builder.header("x-api-key", "public"),
         };
         if path.ends_with("/messages") {
            builder = builder.header("anthropic-version", "2023-06-01");
         }
         let response = builder
            .header(CONTENT_TYPE, "application/json")
            .body(req.clone())
            .send()
            .await?;
         classify(response, Classify::STRICT).await
      };
      if key.is_some() {
         self.egresses.send(attempt).await
      } else {
         self.egresses.send_anonymous(attempt).await
      }
   }

   pub async fn models(&self) -> Result<Vec<ZenModel>, SendError> {
      let response = self
         .egresses
         .send_anonymous(|http| http.get(format!("{}/models", self.base_url())).send())
         .await?;
      let listing: IdList = json(response, Classify::STRICT).await?;
      let windows = self.context_windows().await;
      Ok(listing
         .data
         .into_iter()
         .map(|entry| ZenModel {
            context_window: windows.get(&entry.id).copied(),
            id: entry.id,
         })
         .collect())
   }

   async fn context_windows(&self) -> Arc<HashMap<String, i64>> {
      let now = unix_now();
      let fetched = self.context_fetched_at.load(Ordering::Relaxed);
      if now - fetched >= CONTEXT_REFRESH_SECS
         && self
            .context_fetched_at
            .compare_exchange(fetched, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
      {
         match fetch_context_windows(&self.models_dev_url).await {
            Ok(windows) => *self.context_windows.write().unwrap() = Arc::new(windows),
            Err(error) => tracing::warn!("fetching zen context windows: {error}"),
         }
      }
      self.context_windows.read().unwrap().clone()
   }
}

async fn fetch_context_windows(url: &str) -> reqwest::Result<HashMap<String, i64>> {
   #[derive(serde::Deserialize)]
   struct Catalog {
      opencode: Provider,
   }
   #[derive(serde::Deserialize)]
   struct Provider {
      models: HashMap<String, Model>,
   }
   #[derive(serde::Deserialize)]
   struct Model {
      limit: Option<Limit>,
   }
   #[derive(serde::Deserialize)]
   struct Limit {
      context: i64,
   }

   let catalog: Catalog = reqwest::get(url).await?.error_for_status()?.json().await?;
   Ok(catalog
      .opencode
      .models
      .into_iter()
      .filter_map(|(id, model)| Some((id, model.limit?.context)))
      .collect())
}

fn request_id() -> String {
   let counter = thread_rng().gen_range(1..=0xfff_i64);
   let timestamp = (unix_now_ms().saturating_mul(0x1000) + counter) & 0xffff_ffff_ffff;
   let random = thread_rng()
      .sample_iter(Alphanumeric)
      .take(14)
      .map(char::from)
      .collect::<String>();
   format!("msg_{timestamp:012x}{random}")
}

#[cfg(test)]
mod tests {
   use std::sync::Arc;

   use axum::Router;
   use axum::body::Body;
   use axum::extract::Request;
   use axum::http::{HeaderValue, Response, StatusCode};
   use axum::routing::any;
   use tokio::net::TcpListener;
   use tokio::sync::Mutex;

   use super::*;
   use crate::config::EgressConfig;
   use crate::egress::ATTEMPTS;

   type Requests = Arc<Mutex<Vec<(String, Option<String>)>>>;

   async fn spawn_proxy(status: StatusCode) -> (String, Requests) {
      let requests = Requests::default();
      let seen = Arc::clone(&requests);
      let app = Router::new().fallback(any(move |request: Request| {
         let seen = Arc::clone(&seen);
         async move {
            let uri = request.uri().to_string();
            let auth = request
               .headers()
               .get("proxy-authorization")
               .and_then(|value| value.to_str().ok())
               .map(str::to_owned);
            seen.lock().await.push((uri.clone(), auth));
            let body = if status == StatusCode::TOO_MANY_REQUESTS {
               r#"{"type":"error"}"#
            } else if uri.ends_with("/models") {
               r#"{"data":[{"id":"muse-test"}]}"#
            } else {
               "{}"
            };
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = status;
            if status == StatusCode::TOO_MANY_REQUESTS {
               response
                  .headers_mut()
                  .insert("retry-after", HeaderValue::from_static("3600"));
            }
            response
         }
      }));
      let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
      let address = listener.local_addr().unwrap();
      tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
      (format!("http://{address}"), requests)
   }

   fn authenticated(url: &str) -> String {
      url.replacen("http://", "http://user:password@", 1)
   }

   #[tokio::test]
   async fn configured_proxies_rotate_across_models_and_responses() {
      let (first_url, first_requests) = spawn_proxy(StatusCode::OK).await;
      let (second_url, second_requests) = spawn_proxy(StatusCode::OK).await;
      let client = ZenClient::new(ZenConfig {
         base_url: "http://zen.invalid/v1".into(),
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: vec![authenticated(&first_url), authenticated(&second_url)],
            proxy_urls_file: None,
         },
      })
      .unwrap();

      let listed = client.models().await.unwrap();
      assert_eq!(
         listed
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
         ["muse-test"]
      );
      client
         .post(
            None,
            "sess-test",
            "/responses",
            &Bytes::from_static(br#"{"model":"muse-test"}"#),
         )
         .await
         .unwrap();

      let first = first_requests.lock().await;
      let second = second_requests.lock().await;
      assert_eq!(first.len(), 1);
      assert_eq!(second.len(), 1);
      assert_eq!(first[0].0, "http://zen.invalid/v1/models");
      assert_eq!(second[0].0, "http://zen.invalid/v1/responses");
   }

   #[tokio::test]
   async fn anonymous_rate_limits_cool_one_proxy_and_fail_over() {
      let (limited_url, limited_requests) = spawn_proxy(StatusCode::TOO_MANY_REQUESTS).await;
      let (working_url, working_requests) = spawn_proxy(StatusCode::OK).await;
      let client = ZenClient::new(ZenConfig {
         base_url: "http://zen.invalid/v1".into(),
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: vec![authenticated(&limited_url), authenticated(&working_url)],
            proxy_urls_file: None,
         },
      })
      .unwrap();

      client
         .post(
            None,
            "sess-test",
            "/responses",
            &Bytes::from_static(br#"{"model":"muse-test"}"#),
         )
         .await
         .unwrap();
      client
         .post(
            None,
            "sess-test",
            "/responses",
            &Bytes::from_static(br#"{"model":"muse-test"}"#),
         )
         .await
         .unwrap();

      assert_eq!(limited_requests.lock().await.len(), 1);
      assert_eq!(working_requests.lock().await.len(), 2);
   }

   #[test]
   fn invalid_proxy_errors_do_not_expose_credentials() {
      let error = ZenClient::new(ZenConfig {
         egress: EgressConfig {
            proxy_urls: vec!["http://user:secret@[".into()],
            proxy_urls_file: None,
         },
         ..ZenConfig::default()
      })
      .err()
      .unwrap()
      .to_string();
      assert_eq!(error, "invalid zen proxy URL at position 1");
      assert!(!error.contains("secret"));
   }

   #[tokio::test]
   async fn an_exhausted_walk_stops_at_the_cap_and_asks_for_a_quick_retry() {
      async fn seen(proxies: &[(String, Requests)]) -> usize {
         let mut total = 0;
         for &(_, ref requests) in proxies {
            total += requests.lock().await.len();
         }
         total
      }
      let mut proxies = Vec::new();
      for _ in 0..ATTEMPTS + 4 {
         proxies.push(spawn_proxy(StatusCode::TOO_MANY_REQUESTS).await);
      }
      let client = ZenClient::new(ZenConfig {
         base_url: "http://zen.invalid/v1".into(),
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: proxies
               .iter()
               .map(|&(ref url, _)| authenticated(url))
               .collect(),
            proxy_urls_file: None,
         },
      })
      .unwrap();

      let err = client
         .post(None, "sess-test", "/responses", &Bytes::from_static(b"{}"))
         .await
         .unwrap_err();
      assert!(
         matches!(
            err,
            SendError::RateLimited {
               retry_after: Some(1),
               ..
            }
         ),
         "{err}"
      );
      assert_eq!(seen(&proxies).await, ATTEMPTS);

      let exhausted = client
         .post(None, "sess-test", "/responses", &Bytes::from_static(b"{}"))
         .await
         .unwrap_err();
      assert!(
         matches!(exhausted, SendError::RateLimited { retry_after: Some(secs), .. } if secs > 1),
         "{exhausted}"
      );
      assert_eq!(seen(&proxies).await, ATTEMPTS + 4);
   }
}
