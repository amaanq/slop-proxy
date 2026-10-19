//! `OpenCode` Zen speaks both the `Responses` and the messages API, one
//! dialect per model, so nothing here translates. The request goes up as the
//! caller wrote it and comes back as frames that caller already understands.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::body::Bytes;
use reqwest::header::CONTENT_TYPE;
use tokio::time::sleep;

use crate::clock::{unix_now, unix_now_ms};
use crate::config::ZenConfig;
use crate::egress::Egresses;
use crate::upstream::{Classify, IdList, SendError, classify, json};

const CONTEXT_REFRESH_SECS: i64 = 12 * 60 * 60;

pub struct ZenClient {
   hosts: Vec<Host>,
   models_dev_url: String,
   context_windows: RwLock<Arc<HashMap<String, i64>>>,
   context_fetched_at: AtomicI64,
}

/// Each host keeps its own free-tier allowance per source address, so each
/// gets its own egresses and their own record of which are throttled.
struct Host {
   base_url: String,
   egresses: Egresses,
   /// Throttled on every address a request tried, so later anonymous
   /// requests skip it rather than spend their attempts rediscovering that.
   drained_until: AtomicI64,
}

/// A walk that runs out reports a one second retry while untried addresses
/// remain, too short to be worth skipping the host for.
const DRAINED_SECS: i64 = 60;

/// Zen relays this `DeepSeek` thinking-mode 400 for byte-identical bodies it
/// served minutes earlier, so a resend on the same exit usually clears it.
const REPLAY_REJECTION: &str = "thinking mode must be passed back";
const REPLAY_TRIES: u32 = 3;
const REPLAY_PAUSE: Duration = Duration::from_millis(250);
const UNAVAILABLE_MAX_SECS: i64 = 60;

pub struct ZenModel {
   pub id: String,
   pub context_window: Option<i64>,
}

impl ZenClient {
   pub fn new(cfg: ZenConfig) -> misstep::Result<Self> {
      misstep::ensure!(!cfg.base_urls.is_empty(), "zen needs at least one base url");
      let hosts = cfg
         .base_urls
         .into_iter()
         .map(|base_url| {
            Ok(Host {
               base_url: base_url.trim_end_matches('/').to_owned(),
               egresses: Egresses::new(&cfg.egress, "zen", Some(&cfg.user_agent))?,
               drained_until: AtomicI64::new(0),
            })
         })
         .collect::<misstep::Result<_>>()?;
      Ok(Self {
         hosts,
         models_dev_url: cfg.models_dev_url,
         context_windows: RwLock::default(),
         context_fetched_at: AtomicI64::new(0),
      })
   }

   pub fn egresses(&self) -> impl Iterator<Item = &Egresses> {
      self.hosts.iter().map(|host| &host.egresses)
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
      let now = unix_now();
      let last = self.hosts.len() - 1;
      let mut outcome = Err(SendError::Network("no zen hosts".into()));
      for (index, host) in self.hosts.iter().enumerate() {
         if key.is_none() && index < last && host.drained_until.load(Ordering::Relaxed) > now {
            continue;
         }
         let url = format!("{}{path}", host.base_url);
         let url = url.as_str();
         let attempt = move |http: reqwest::Client| async move {
            let mut tries = 1;
            loop {
               let mut builder = http
                  .post(url)
                  .header("Accept", "*/*")
                  .header("x-opencode-session", session)
                  .header("x-opencode-session-id", session)
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
               match classify(response, Classify::STRICT).await {
                  Err(SendError::BadRequest(body))
                     if tries < REPLAY_TRIES && body.contains(REPLAY_REJECTION) =>
                  {
                     tracing::warn!(tries, "zen refused a thinking replay, resending: {body}");
                     tries += 1;
                     sleep(REPLAY_PAUSE).await;
                  },
                  Err(SendError::BadRequest(body)) if body.contains("Model is unavailable") => {
                     return Err(SendError::ModelLimited {
                        retry_after: retry_after_ms(&body),
                        body,
                     });
                  },
                  other => return other,
               }
            }
         };
         outcome = if key.is_some() {
            host.egresses.send(attempt).await
         } else {
            host.egresses.send_anonymous(attempt).await
         };
         let Err(SendError::RateLimited { retry_after, .. }) = outcome else {
            break;
         };
         if key.is_none() {
            let until = now + retry_after.unwrap_or(0).max(DRAINED_SECS);
            host.drained_until.store(until, Ordering::Relaxed);
         }
      }
      outcome
   }

   pub async fn models(&self) -> Result<Vec<ZenModel>, SendError> {
      let host = &self.hosts[0];
      let listing_url = format!("{}/models", host.base_url);
      let url = listing_url.as_str();
      let response = host
         .egresses
         .send_anonymous(|http| async move {
            classify(http.get(url).send().await?, Classify::STRICT).await
         })
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

   let catalog = reqwest::get(url).await?.error_for_status()?.json::<Catalog>().await?;
   Ok(catalog
      .opencode
      .models
      .into_iter()
      .filter_map(|(id, model)| Some((id, model.limit?.context)))
      .collect())
}

/// Zen puts its backoff in the message text, as `retry-after-ms=<n>`. Its
/// "Model is unavailable" hints run near 19 minutes while re-asking costs one
/// 400, so a client honouring `Retry-After` waits a minute at most.
fn retry_after_ms(body: &str) -> Option<i64> {
   let (_, rest) = body.split_once("retry-after-ms=")?;
   let digits = rest.split(|digit: char| !digit.is_ascii_digit()).next()?;
   let millis = digits.parse::<i64>().ok()?;
   (millis > 0).then(|| ((millis + 999) / 1000).min(UNAVAILABLE_MAX_SECS))
}

fn request_id() -> String {
   const ALPHANUMERIC: &[u8; 62] =
      b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
   let mut bytes = [0_u8; 14];
   getrandom::fill(&mut bytes).expect("the OS random source is readable");
   // Zen's free tier checks that the per-millisecond counter in the low 12
   // bits is OpenCode's first value, 1, for both message and session ids.
   let timestamp = (unix_now_ms().saturating_mul(0x1000) + 1) & 0xffff_ffff_ffff;
   let random = bytes
      .iter()
      .map(|&byte| char::from(ALPHANUMERIC[usize::from(byte) % ALPHANUMERIC.len()]))
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
         base_urls: vec!["http://zen.invalid/v1".into()],
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: vec![authenticated(&first_url), authenticated(&second_url)],
            proxy_urls_file: None,
            source_prefixes: Vec::new(),
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
         base_urls: vec!["http://zen.invalid/v1".into()],
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: vec![authenticated(&limited_url), authenticated(&working_url)],
            proxy_urls_file: None,
            source_prefixes: Vec::new(),
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
            source_prefixes: Vec::new(),
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
         base_urls: vec!["http://zen.invalid/v1".into()],
         models_dev_url: "http://models.invalid".into(),
         user_agent: "opencode/1.18.31".into(),
         egress: EgressConfig {
            proxy_urls: proxies
               .iter()
               .map(|&(ref url, _)| authenticated(url))
               .collect(),
            proxy_urls_file: None,
            source_prefixes: Vec::new(),
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
