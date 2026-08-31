pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn rfc3339(unix_secs: i64) -> String {
    jiff::Timestamp::from_second(unix_secs)
        .unwrap_or(jiff::Timestamp::UNIX_EPOCH)
        .to_string()
}

pub fn unix_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
