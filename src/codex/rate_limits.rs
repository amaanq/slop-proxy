use reqwest::header::HeaderMap;

pub fn retry_after_secs(headers: &HeaderMap) -> Option<i64> {
    if let Some(v) = header_i64(headers, "retry-after") {
        return Some(v);
    }
    if let Some(reset_at) = header_str(headers, "x-codex-primary-reset-at") {
        if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(reset_at) {
            return Some((ts.timestamp() - chrono::Utc::now().timestamp()).max(1));
        }
        if let Ok(secs) = reset_at.parse::<f64>() {
            let now = chrono::Utc::now().timestamp() as f64;
            // Header has been observed both as an absolute epoch and as seconds-from-now.
            return Some(
                if secs > now {
                    (secs - now) as i64
                } else {
                    secs as i64
                }
                .max(1),
            );
        }
    }
    None
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
}

fn header_i64(headers: &HeaderMap, name: &str) -> Option<i64> {
    header_str(headers, name)?.parse().ok()
}
