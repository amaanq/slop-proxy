use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_now() -> i64 {
   SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map_or(0, |dur| dur.as_secs() as i64)
}

pub fn rfc3339(unix_secs: i64) -> String {
   jiff::Timestamp::from_second(unix_secs)
      .unwrap_or(jiff::Timestamp::UNIX_EPOCH)
      .to_string()
}

pub fn unix_now_ms() -> i64 {
   SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map_or(0, |dur| dur.as_millis() as i64)
}
