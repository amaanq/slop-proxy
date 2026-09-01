use serde_json::Value;

use crate::config::GeminiConfig;
use crate::gemini::native;
use crate::upstream::{SendError, retry_after_secs};

const RESET_HEADERS: &[&str] = &["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"];

pub struct GeminiClient {
    http: reqwest::Client,
    cfg: GeminiConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiProtocol {
    OpenAi,
    Native,
}

pub struct GeminiResponse {
    pub response: reqwest::Response,
    pub protocol: GeminiProtocol,
}

impl GeminiClient {
    pub fn new(cfg: GeminiConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("building http client");
        Self { http, cfg }
    }

    pub fn soft_utilization_limit(&self) -> f64 {
        self.cfg.soft_utilization_limit
    }

    /// Google's OpenAI-compatible surface drops `Referer` before API-key
    /// validation, so origin-restricted keys have to use the native surface.
    pub async fn send(
        &self,
        api_key: &str,
        account_referer: Option<&str>,
        body: &Value,
    ) -> Result<GeminiResponse, SendError> {
        let configured_referer = self
            .cfg
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("referer"))
            .map(|(_, value)| value.as_str());
        let referer = account_referer.or(configured_referer);
        let (mut req, protocol) = if let Some(referer) = referer {
            let translated = native::request(body).map_err(SendError::BadRequest)?;
            let base = self.cfg.base_url.trim_end_matches('/');
            let base = base.strip_suffix("/openai").unwrap_or(base);
            let action = if translated.streaming {
                "streamGenerateContent?alt=sse"
            } else {
                "generateContent"
            };
            let url = format!("{base}/models/{}:{action}", translated.model);
            (
                self.http
                    .post(url)
                    .header("x-goog-api-key", api_key)
                    .header("referer", referer)
                    .json(&translated.body),
                GeminiProtocol::Native,
            )
        } else {
            (
                self.http
                    .post(format!(
                        "{}/chat/completions",
                        self.cfg.base_url.trim_end_matches('/')
                    ))
                    .bearer_auth(api_key)
                    .json(body),
                GeminiProtocol::OpenAi,
            )
        };
        for (name, value) in &self.cfg.headers {
            if name.eq_ignore_ascii_case("referer") && referer.is_some() {
                continue;
            }
            req = req.header(name, value);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SendError::Network(e.to_string()))?;

        let status = resp.status().as_u16();
        if !matches!(status, 401 | 403 | 429 | 500..=599) {
            return Ok(GeminiResponse {
                response: resp,
                protocol,
            });
        }
        let retry_after = retry_after_secs(resp.headers(), RESET_HEADERS);
        let body = resp.text().await.unwrap_or_default();
        let body = body.chars().take(2000).collect::<String>();
        Err(match status {
            // A key restricted to an origin or with the API disabled answers
            // 403, and no retry on another account makes that key work.
            401 | 403 => SendError::Auth(body),
            429 => SendError::RateLimited { retry_after, body },
            s => SendError::Upstream { status: s, body },
        })
    }

    /// A caller that already speaks the native dialect is relayed as-is, so
    /// nothing round-trips through the OpenAI shape and back.
    pub async fn send_native(
        &self,
        api_key: &str,
        account_referer: Option<&str>,
        model: &str,
        action: &str,
        query: Option<&str>,
        body: &Value,
    ) -> Result<reqwest::Response, SendError> {
        let base = self.cfg.base_url.trim_end_matches('/');
        let base = base.strip_suffix("/openai").unwrap_or(base);
        let query = forwarded_query(query);
        let mut req = self
            .http
            .post(format!("{base}/models/{model}:{action}{query}"))
            .header("x-goog-api-key", api_key)
            .json(body);
        let referer = account_referer.or_else(|| {
            self.cfg
                .headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("referer"))
                .map(|(_, v)| v.as_str())
        });
        if let Some(referer) = referer {
            req = req.header("referer", referer);
        }
        for (name, value) in &self.cfg.headers {
            if name.eq_ignore_ascii_case("referer") && referer.is_some() {
                continue;
            }
            req = req.header(name, value);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SendError::Network(e.to_string()))?;
        let status = resp.status().as_u16();
        if !matches!(status, 401 | 403 | 429 | 500..=599) {
            return Ok(resp);
        }
        let retry_after = retry_after_secs(resp.headers(), RESET_HEADERS);
        let body = resp.text().await.unwrap_or_default();
        let body = body.chars().take(2000).collect::<String>();
        Err(match status {
            401 | 403 => SendError::Auth(body),
            429 => SendError::RateLimited { retry_after, body },
            s => SendError::Upstream { status: s, body },
        })
    }

    /// The catalog carries no per-key state, so a restricted key can read it
    /// from the native surface with the same referer the send path uses.
    pub async fn models(
        &self,
        api_key: &str,
        account_referer: Option<&str>,
    ) -> Result<Vec<String>, String> {
        let base = self.cfg.base_url.trim_end_matches('/');
        let mut req = if let Some(referer) = account_referer {
            let base = base.strip_suffix("/openai").unwrap_or(base);
            self.http
                .get(format!("{base}/models"))
                .header("x-goog-api-key", api_key)
                .header("referer", referer)
        } else {
            self.http.get(format!("{base}/models")).bearer_auth(api_key)
        };
        for (name, value) in &self.cfg.headers {
            if name.eq_ignore_ascii_case("referer") && account_referer.is_some() {
                continue;
            }
            req = req.header(name, value);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(resp.status().to_string());
        }
        let body: Value = resp.json().await.map_err(|e| e.to_string())?;
        // The two surfaces name the array differently, and the native one
        // prefixes every id with `models/`.
        let ids = body
            .get("data")
            .or_else(|| body.get("models"))
            .and_then(Value::as_array)
            .ok_or_else(|| "no model array in the catalog".to_string())?
            .iter()
            .filter_map(|m| m.get("id").or_else(|| m.get("name"))?.as_str())
            .map(|id| id.trim_start_matches("models/").to_string())
            .collect();
        Ok(ids)
    }
}

/// `alt=sse` decides whether the reply streams, so it has to survive, but
/// `key` holds the caller's own token and upstream would reject it.
fn forwarded_query(query: Option<&str>) -> String {
    query
        .map(|q| {
            q.split('&')
                .filter(|p| !p.starts_with("key="))
                .collect::<Vec<_>>()
                .join("&")
        })
        .filter(|q| !q.is_empty())
        .map(|q| format!("?{q}"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::Json;
    use axum::extract::Request;
    use axum::routing::post;
    use serde_json::json;

    use super::*;

    #[test]
    fn the_callers_own_token_is_not_forwarded_upstream() {
        assert_eq!(forwarded_query(Some("key=sp-secret")), "");
        assert_eq!(forwarded_query(Some("alt=sse&key=sp-secret")), "?alt=sse");
        assert_eq!(forwarded_query(Some("key=sp-secret&alt=sse")), "?alt=sse");
        assert_eq!(forwarded_query(Some("alt=sse")), "?alt=sse");
        assert_eq!(forwarded_query(None), "");
    }

    #[tokio::test]
    async fn restricted_keys_use_the_native_auth_surface() {
        let seen = Arc::new(Mutex::new(None));
        let captured = seen.clone();
        let app = axum::Router::new().fallback(post(move |request: Request| {
            let captured = captured.clone();
            async move {
                let headers = request.headers().clone();
                let uri = request.uri().clone();
                *captured.lock().unwrap() = Some((headers, uri));
                Json(json!({"candidates": []}))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = GeminiClient::new(GeminiConfig {
            base_url: format!("http://{address}/v1beta/openai"),
            ..GeminiConfig::default()
        });
        let response = client
            .send(
                "test-key",
                Some("https://example.test/"),
                &json!({
                    "model": "gemini-flash-latest",
                    "messages": [{"role": "user", "content": "hi"}]
                }),
            )
            .await
            .unwrap();
        assert_eq!(response.protocol, GeminiProtocol::Native);

        let (headers, uri) = seen.lock().unwrap().take().unwrap();
        assert_eq!(headers["x-goog-api-key"], "test-key");
        assert_eq!(headers["referer"], "https://example.test/");
        assert!(!headers.contains_key("authorization"));
        assert_eq!(
            uri.path(),
            "/v1beta/models/gemini-flash-latest:generateContent"
        );
    }
}

