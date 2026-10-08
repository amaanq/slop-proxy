use std::time::{SystemTime, UNIX_EPOCH};

use jiff::civil::Date;
use jiff::tz::TimeZone;

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

/// `YYYY-MM-DD` of the UTC day `day` days after the epoch.
pub fn date(day: i64) -> String {
   jiff::Timestamp::from_second(day * 86400)
      .unwrap_or(jiff::Timestamp::UNIX_EPOCH)
      .strftime("%Y-%m-%d")
      .to_string()
}

/// Days from the epoch to a `YYYY-MM-DD` date, the inverse of [`date`].
pub fn day(date: &str) -> Option<i64> {
   let midnight = date.parse::<Date>().ok()?.to_zoned(TimeZone::UTC).ok()?;
   Some(midnight.timestamp().as_second().div_euclid(86400))
}

pub fn unix_seconds(text: Option<&str>) -> Option<i64> {
   text?
      .parse::<jiff::Timestamp>()
      .ok()
      .map(jiff::Timestamp::as_second)
}

pub fn unix_now_ms() -> i64 {
   SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map_or(0, |dur| dur.as_millis() as i64)
}
