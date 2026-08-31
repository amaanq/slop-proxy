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
