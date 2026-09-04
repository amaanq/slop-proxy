//! OpenCode Zen speaks the Responses API, so nothing here translates. The
//! request goes up as the caller wrote it and comes back as frames the codex
//! parser already understands.

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use axum::body::Bytes;

use crate::config::ZenConfig;
use crate::upstream::{SendError, retry_after_secs};

pub struct ZenClient {
    base_url: String,
    egresses: Vec<Egress>,
    next: AtomicUsize,
}

struct Egress {
    http: reqwest::Client,
    unavailable_until: AtomicI64,
    anonymous_cooldown_until: AtomicI64,
}

impl Egress {
    fn new(proxy_url: Option<&str>, index: usize) -> eyre::Result<Self> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(30));
        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|_| eyre::eyre!("invalid zen proxy URL at position {}", index + 1))?;
            builder = builder.proxy(proxy);
        }
        let http = builder
            .build()
            .map_err(|_| eyre::eyre!("building zen HTTP client"))?;
        Ok(Self {
            http,
            unavailable_until: AtomicI64::new(0),
            anonymous_cooldown_until: AtomicI64::new(0),
        })
    }
}

impl ZenClient {
    pub fn new(cfg: ZenConfig) -> eyre::Result<Self> {
        let proxy_urls = cfg.proxy_urls()?;
        let egresses = if proxy_urls.is_empty() {
            vec![Egress::new(None, 0)?]
        } else {
            proxy_urls
                .iter()
                .enumerate()
                .map(|(index, url)| Egress::new(Some(url), index))
                .collect::<eyre::Result<Vec<_>>>()?
        };
        Ok(Self {
            base_url: cfg.base_url,
            egresses,
            next: AtomicUsize::new(0),
        })
    }

    /// The free contributor models answer without any credential at all, so
    /// the key is optional and only attached when an account supplies one.
    pub async fn send(
        &self,
        key: Option<&str>,
        req: &Bytes,
    ) -> Result<reqwest::Response, SendError> {
        let anonymous = key.is_none_or(str::is_empty);
        let mut rate_limit_body = None;
        let mut network_error = None;
        for index in self.available_egresses(anonymous)? {
            match self.send_via(index, key, req).await {
                Ok(response) => return Ok(response),
                Err(SendError::RateLimited { retry_after, body }) if anonymous => {
                    self.cool_anonymous(index, retry_after.unwrap_or(60));
                    rate_limit_body = Some(body);
                }
                Err(SendError::Network(error)) => {
                    self.cool_unavailable(index, 30);
                    network_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(self.exhausted_error(rate_limit_body, network_error))
    }

    async fn send_via(
        &self,
        index: usize,
        key: Option<&str>,
        req: &Bytes,
    ) -> Result<reqwest::Response, SendError> {
        let mut builder = self.egresses[index]
            .http
            .post(format!("{}/responses", self.base_url.trim_end_matches('/')))
            .header("Accept", "text/event-stream");
        if let Some(key) = key.filter(|key| !key.is_empty()) {
            builder = builder.bearer_auth(key);
        }
        let response = builder
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(req.clone())
            .send()
            .await
            .map_err(|error| SendError::Network(error.to_string()))?;
        classify(response).await
    }

    pub async fn models(&self) -> Result<Vec<String>, String> {
        #[derive(serde::Deserialize)]
        struct Entry {
            id: String,
        }
        #[derive(serde::Deserialize)]
        struct Listing {
            data: Vec<Entry>,
        }
        let mut rate_limit_body = None;
        let mut network_error = None;
        for index in self
            .available_egresses(true)
            .map_err(|error| error.to_string())?
        {
            let response = match self.egresses[index]
                .http
                .get(format!("{}/models", self.base_url.trim_end_matches('/')))
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    self.cool_unavailable(index, 30);
                    network_error = Some(error.to_string());
                    continue;
                }
            };
            match classify(response).await {
                Ok(response) => {
                    let listing: Listing =
                        response.json().await.map_err(|error| error.to_string())?;
                    return Ok(listing.data.into_iter().map(|entry| entry.id).collect());
                }
                Err(SendError::RateLimited { retry_after, body }) => {
                    self.cool_anonymous(index, retry_after.unwrap_or(60));
                    rate_limit_body = Some(body);
                }
                Err(SendError::Network(error)) => {
                    self.cool_unavailable(index, 30);
                    network_error = Some(error);
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Err(self
            .exhausted_error(rate_limit_body, network_error)
            .to_string())
    }

    fn available_egresses(&self, anonymous: bool) -> Result<Vec<usize>, SendError> {
        let now = crate::clock::unix_now();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let indices = (0..self.egresses.len())
            .map(|offset| start.wrapping_add(offset) % self.egresses.len())
            .filter(|&index| {
                let egress = &self.egresses[index];
                egress.unavailable_until.load(Ordering::Relaxed) <= now
                    && (!anonymous
                        || egress.anonymous_cooldown_until.load(Ordering::Relaxed) <= now)
            })
            .collect::<Vec<_>>();
        if !indices.is_empty() {
            return Ok(indices);
        }
        let has_rate_limit = anonymous
            && self
                .egresses
                .iter()
                .any(|egress| egress.anonymous_cooldown_until.load(Ordering::Relaxed) > now);
        if has_rate_limit {
            return Err(SendError::RateLimited {
                retry_after: Some(self.retry_after(now, true)),
                body: "all zen egresses are cooling down".into(),
            });
        }
        Err(SendError::Network(
            "all zen egresses are temporarily unavailable".into(),
        ))
    }

    fn cool_unavailable(&self, index: usize, seconds: i64) {
        self.egresses[index].unavailable_until.store(
            crate::clock::unix_now().saturating_add(seconds.max(1)),
            Ordering::Relaxed,
        );
    }

    fn cool_anonymous(&self, index: usize, seconds: i64) {
        self.egresses[index].anonymous_cooldown_until.store(
            crate::clock::unix_now().saturating_add(seconds.max(1)),
            Ordering::Relaxed,
        );
    }

    fn retry_after(&self, now: i64, anonymous: bool) -> i64 {
        self.egresses
            .iter()
            .map(|egress| {
                let unavailable = egress.unavailable_until.load(Ordering::Relaxed);
                let rate_limited = if anonymous {
                    egress.anonymous_cooldown_until.load(Ordering::Relaxed)
                } else {
                    0
                };
                unavailable.max(rate_limited) - now
            })
            .filter(|seconds| *seconds > 0)
            .min()
            .unwrap_or(30)
    }

    fn exhausted_error(
        &self,
        rate_limit_body: Option<String>,
        network_error: Option<String>,
    ) -> SendError {
        if let Some(body) = rate_limit_body {
            return SendError::RateLimited {
                retry_after: Some(self.retry_after(crate::clock::unix_now(), true)),
                body,
            };
        }
        SendError::Network(network_error.unwrap_or_else(|| "all zen egresses failed".into()))
    }
}

async fn classify(response: reqwest::Response) -> Result<reqwest::Response, SendError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = retry_after_secs(response.headers(), &[]);
    let body = response.text().await.unwrap_or_default();
    let body = body.chars().take(2000).collect::<String>();
    Err(match status.as_u16() {
        401 | 403 => SendError::Auth(body),
        407 => SendError::Network("proxy authentication failed".into()),
        429 => SendError::RateLimited { retry_after, body },
        400 => SendError::BadRequest(body),
        status => SendError::Upstream { status, body },
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::Router;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::{HeaderValue, Response, StatusCode};
    use axum::routing::any;
    use tokio::sync::Mutex;

    use super::*;

    type Requests = Arc<Mutex<Vec<(String, Option<String>)>>>;

    async fn spawn_proxy(status: StatusCode) -> (String, Requests) {
        let requests = Requests::default();
        let seen = requests.clone();
        let app = Router::new().fallback(any(move |request: Request| {
            let seen = seen.clone();
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
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
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
            proxy_urls: vec![authenticated(&first_url), authenticated(&second_url)],
            proxy_urls_file: None,
        })
        .unwrap();

        assert_eq!(client.models().await.unwrap(), ["muse-test"]);
        client
            .send(None, &Bytes::from_static(br#"{"model":"muse-test"}"#))
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
            proxy_urls: vec![authenticated(&limited_url), authenticated(&working_url)],
            proxy_urls_file: None,
        })
        .unwrap();

        client
            .send(None, &Bytes::from_static(br#"{"model":"muse-test"}"#))
            .await
            .unwrap();
        client
            .send(None, &Bytes::from_static(br#"{"model":"muse-test"}"#))
            .await
            .unwrap();

        assert_eq!(limited_requests.lock().await.len(), 1);
        assert_eq!(working_requests.lock().await.len(), 2);
    }

    #[test]
    fn invalid_proxy_errors_do_not_expose_credentials() {
        let error = ZenClient::new(ZenConfig {
            proxy_urls: vec!["http://user:secret@[".into()],
            ..ZenConfig::default()
        })
        .err()
        .unwrap()
        .to_string();
        assert_eq!(error, "invalid zen proxy URL at position 1");
        assert!(!error.contains("secret"));
    }
}
