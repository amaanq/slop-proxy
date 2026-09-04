use reqwest::header::HeaderMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SendError {
    #[error("upstream auth failed: {0}")]
    Auth(String),
    #[error("upstream rate limited")]
    RateLimited {
        retry_after: Option<i64>,
        body: String,
    },
    #[error("upstream error {status}: {body}")]
    Upstream { status: u16, body: String },
    #[error("bad request upstream: {0}")]
    BadRequest(String),
    #[error("network error: {0}")]
    Network(String),
}

/// Seconds until the caller may retry, from `retry-after` or the first of the
/// backend-specific reset headers that parses.
pub fn retry_after_secs(headers: &HeaderMap, reset_headers: &[&str]) -> Option<i64> {
    let get = |name: &str| headers.get(name)?.to_str().ok();
    if let Some(v) = get("retry-after").and_then(|v| v.parse::<i64>().ok()) {
        return Some(v);
    }
    let reset = reset_headers.iter().find_map(|h| get(h))?;
    let now = crate::clock::unix_now();
    if let Ok(ts) = reset.parse::<jiff::Timestamp>() {
        return Some((ts.as_second() - now).max(1));
    }
    let secs = reset.parse::<f64>().ok()? as i64;
    // Reset headers have been observed both as an absolute epoch and as
    // seconds-from-now.
    Some(if secs > now { secs - now } else { secs }.max(1))
}

/// How one backend's statuses read. Every client used to spell this out
/// by hand and they only ever differed in these three fields.
#[derive(Clone, Copy)]
pub struct Classify {
    /// Non-2xx statuses handed back as a response, for a relay that wants
    /// the body verbatim.
    pub pass: fn(u16) -> bool,
    pub auth: &'static [u16],
    pub reset_headers: &'static [&'static str],
}

impl Classify {
    pub const STRICT: Self = Self {
        pass: |_| false,
        auth: &[401, 403],
        reset_headers: &[],
    };
}

pub async fn classify(
    resp: reqwest::Response,
    rules: Classify,
) -> Result<reqwest::Response, SendError> {
    let status = resp.status().as_u16();
    if resp.status().is_success() || (rules.pass)(status) {
        return Ok(resp);
    }
    let retry_after = retry_after_secs(resp.headers(), rules.reset_headers);
    let body = resp.text().await.unwrap_or_default();
    let body = body.chars().take(2000).collect::<String>();
    Err(if rules.auth.contains(&status) {
        SendError::Auth(body)
    } else {
        match status {
            407 => SendError::Network("proxy authentication failed".into()),
            429 => SendError::RateLimited { retry_after, body },
            400 => SendError::BadRequest(body),
            status => SendError::Upstream { status, body },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(status: u16, body: &'static str) -> reqwest::Response {
        axum::http::Response::builder()
            .status(status)
            .header("retry-after", "7")
            .body(body)
            .unwrap()
            .into()
    }

    #[tokio::test]
    async fn a_passed_status_keeps_its_body_for_the_relay() {
        let rules = Classify {
            pass: |s| !matches!(s, 401 | 429 | 500..=599),
            ..Classify::STRICT
        };
        assert_eq!(
            classify(response(404, "x"), rules).await.unwrap().status(),
            404
        );
        assert!(matches!(
            classify(response(429, "slow"), rules).await,
            Err(SendError::RateLimited {
                retry_after: Some(7),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn the_auth_list_decides_what_a_403_means() {
        assert!(matches!(
            classify(response(403, "no"), Classify::STRICT).await,
            Err(SendError::Auth(_))
        ));
        let cloudflare = Classify {
            auth: &[401],
            ..Classify::STRICT
        };
        assert!(matches!(
            classify(response(403, "no"), cloudflare).await,
            Err(SendError::Upstream { status: 403, .. })
        ));
    }
}
