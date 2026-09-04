use eyre::{Result, bail};
use serde::Serialize;

use crate::db::Db;
use crate::db::usage::{UsageAgg, UsageDim};

#[derive(Serialize)]
struct Tokens {
   requests: i64,
   errors: i64,
   input_tokens: i64,
   output_tokens: i64,
   cache_read_tokens: i64,
   reasoning_tokens: i64,
}

impl From<&UsageAgg> for Tokens {
   fn from(a: &UsageAgg) -> Self {
      Self {
         requests: a.requests,
         errors: a.errors,
         input_tokens: a.input_tokens,
         output_tokens: a.output_tokens,
         cache_read_tokens: a.cache_read_tokens,
         reasoning_tokens: a.reasoning_tokens,
      }
   }
}

#[derive(Serialize)]
struct KeyedTokens {
   name: String,
   #[serde(flatten)]
   tokens: Tokens,
}

#[derive(Serialize)]
struct Report {
   since: Option<String>,
   until: Option<String>,
   totals: Tokens,
   by_user: Vec<KeyedTokens>,
   by_account: Vec<KeyedTokens>,
   by_model: Vec<KeyedTokens>,
}

pub async fn run(db: &Db, since: Option<String>, until: Option<String>) -> Result<()> {
   let now = crate::clock::unix_now();
   let since_ts = match &since {
      Some(s) => parse_time(s, now)?,
      None => 0,
   };
   let until_ts = match &until {
      Some(s) => parse_time(s, now)?,
      None => now + 1,
   };

   let totals = db.usage_totals(since_ts, until_ts).await?;
   let by_user = db.usage_by(UsageDim::User, since_ts, until_ts).await?;
   let by_account = db.usage_by(UsageDim::Account, since_ts, until_ts).await?;
   let by_model = db.usage_by(UsageDim::Model, since_ts, until_ts).await?;

   let keyed = |rows: &[UsageAgg]| {
      rows
         .iter()
         .map(|a| KeyedTokens {
            name: a.key.clone(),
            tokens: a.into(),
         })
         .collect()
   };
   let stamp = |ts| jiff::Timestamp::from_second(ts).ok().map(|t| t.to_string());
   let report = Report {
      since: stamp(since_ts),
      until: stamp(until_ts),
      totals: (&totals).into(),
      by_user: keyed(&by_user),
      by_account: keyed(&by_account),
      by_model: keyed(&by_model),
   };
   println!("{}", serde_json::to_string_pretty(&report)?);
   Ok(())
}

fn parse_time(s: &str, now: i64) -> Result<i64> {
   if let Ok(ts) = s.parse::<jiff::Timestamp>() {
      return Ok(ts.as_second());
   }
   let units = [('m', 60), ('h', 3600), ('d', 86400), ('w', 7 * 86400)];
   let secs = units.iter().find_map(|&(unit, mult)| {
      let n = s.strip_suffix(unit)?.parse::<i64>().ok()?;
      Some(n * mult)
   });
   if let Some(secs) = secs {
      return Ok(now - secs);
   }
   bail!("cannot parse time {s:?}; use RFC3339 or 30m/24h/7d/2w");
}
