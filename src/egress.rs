//! One HTTP client per configured proxy, rotated per request and cooled off
//! when it stops answering. A provider with no proxies configured gets a
//! single direct client and the same code path.

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::clock;
use crate::upstream::SendError;

/// How many egresses one request may burn before it gives up. A long proxy
/// list otherwise turns a dead upstream into a very slow failure.
pub const ATTEMPTS: usize = 8;

const UNREACHABLE_COOLDOWN: i64 = 30;

pub struct Egresses {
   entries: Vec<Egress>,
   next: AtomicUsize,
   label: &'static str,
}

struct Egress {
   http: reqwest::Client,
   unavailable_until: AtomicI64,
   /// Zen's free tier rate-limits per source address, so an anonymous 429
   /// benches the egress for anonymous traffic only.
   anonymous_until: AtomicI64,
}

/// Rides the response so a stream that dies halfway can name the proxy it
/// died on, which the headers cannot.
#[derive(Clone, Copy)]
pub struct EgressIndex(pub usize);

pub fn egress_of(response: &reqwest::Response) -> Option<usize> {
   response
      .extensions()
      .get::<EgressIndex>()
      .map(|index| index.0)
}

impl Egresses {
   pub fn new(
      proxy_urls: &[String],
      label: &'static str,
      user_agent: Option<&str>,
   ) -> eyre::Result<Self> {
      let build = |proxy_url: Option<&str>, index: usize| -> eyre::Result<Egress> {
         let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(30));
         if let Some(agent) = user_agent {
            builder = builder.user_agent(agent);
         }
         if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(proxy_url)
               .map_err(|_| eyre::eyre!("invalid {label} proxy URL at position {}", index + 1))?;
            builder = builder.proxy(proxy);
         }
         Ok(Egress {
            http: builder
               .build()
               .map_err(|_| eyre::eyre!("building {label} HTTP client"))?,
            unavailable_until: AtomicI64::new(0),
            anonymous_until: AtomicI64::new(0),
         })
      };

      let entries = if proxy_urls.is_empty() {
         vec![build(None, 0)?]
      } else {
         proxy_urls
            .iter()
            .enumerate()
            .map(|(index, url)| build(Some(url), index))
            .collect::<eyre::Result<Vec<_>>>()?
      };

      Ok(Self {
         entries,
         next: AtomicUsize::new(0),
         label,
      })
   }

   pub fn http(&self, index: usize) -> &reqwest::Client {
      &self.entries[index].http
   }

   /// Every egress still in service, starting one past the last request so
   /// concurrent callers spread across the list rather than stacking on the
   /// first healthy one.
   fn order(&self, anonymous: bool) -> Vec<usize> {
      let now = clock::unix_now();
      let start = self.next.fetch_add(1, Ordering::Relaxed);
      (0..self.entries.len())
         .map(|offset| start.wrapping_add(offset) % self.entries.len())
         .filter(|&index| {
            let egress = &self.entries[index];
            egress.unavailable_until.load(Ordering::Relaxed) <= now
               && (!anonymous || egress.anonymous_until.load(Ordering::Relaxed) <= now)
         })
         .collect()
   }

   fn cool(until: &AtomicI64, seconds: i64) {
      until.store(
         clock::unix_now().saturating_add(seconds.max(1)),
         Ordering::Relaxed,
      );
   }

   fn retry_after(&self, now: i64) -> i64 {
      self
         .entries
         .iter()
         .map(|egress| {
            let unavailable = egress.unavailable_until.load(Ordering::Relaxed);
            unavailable.max(egress.anonymous_until.load(Ordering::Relaxed)) - now
         })
         .filter(|&seconds| seconds > 0)
         .min()
         .unwrap_or(UNREACHABLE_COOLDOWN)
   }

   /// Retries the next egress when one cannot be reached at all. Anything the
   /// upstream itself answered is the upstream's verdict and stops the walk,
   /// since a second egress would only ask the same question again.
   pub async fn send<Attempt, Fut, Error>(
      &self,
      attempt: Attempt,
   ) -> Result<reqwest::Response, SendError>
   where
      Attempt: Fn(reqwest::Client) -> Fut,
      Fut: Future<Output = Result<reqwest::Response, Error>>,
      Error: Into<SendError>,
   {
      self.walk(false, attempt).await
   }

   /// As `send`, but a rate limit is the egress's address being throttled,
   /// so it benches that egress and the walk moves on.
   pub async fn send_anonymous<Attempt, Fut, Error>(
      &self,
      attempt: Attempt,
   ) -> Result<reqwest::Response, SendError>
   where
      Attempt: Fn(reqwest::Client) -> Fut,
      Fut: Future<Output = Result<reqwest::Response, Error>>,
      Error: Into<SendError>,
   {
      self.walk(true, attempt).await
   }

   async fn walk<Attempt, Fut, Error>(
      &self,
      anonymous: bool,
      attempt: Attempt,
   ) -> Result<reqwest::Response, SendError>
   where
      Attempt: Fn(reqwest::Client) -> Fut,
      Fut: Future<Output = Result<reqwest::Response, Error>>,
      Error: Into<SendError>,
   {
      let order = self.order(anonymous);
      if order.is_empty() {
         let now = clock::unix_now();
         let throttled = anonymous
            && self
               .entries
               .iter()
               .any(|egress| egress.anonymous_until.load(Ordering::Relaxed) > now);
         tracing::warn!(
            total = self.entries.len(),
            "no {} egress available",
            self.label
         );
         let body = format!(
            "all {} {} egresses are cooling down",
            self.entries.len(),
            self.label
         );
         return Err(if throttled {
            SendError::RateLimited {
               retry_after: Some(self.retry_after(now)),
               body,
            }
         } else {
            SendError::Network(body)
         });
      }

      let mut throttled = None;
      let mut unreachable = None;
      let mut tried = 0_usize;
      for index in order.iter().copied().take(ATTEMPTS) {
         tried += 1;
         match attempt(self.entries[index].http.clone())
            .await
            .map_err(Into::into)
         {
            Ok(mut response) => {
               response.extensions_mut().insert(EgressIndex(index));
               if tried > 1 {
                  tracing::info!(
                     egress = index,
                     failed = tried - 1,
                     "{} egress served after failover",
                     self.label
                  );
               }
               return Ok(response);
            },
            Err(SendError::RateLimited { retry_after, body }) if anonymous => {
               tracing::warn!(
                  egress = index,
                  "{} egress rate limited: {}",
                  self.label,
                  body.chars().take(200).collect::<String>()
               );
               Self::cool(
                  &self.entries[index].anonymous_until,
                  retry_after.unwrap_or(60),
               );
               throttled = Some(body);
            },
            Err(SendError::Network(error)) => {
               tracing::warn!(egress = index, "{} egress unreachable: {error}", self.label);
               if self.entries.len() > 1 {
                  Self::cool(&self.entries[index].unavailable_until, UNREACHABLE_COOLDOWN);
               }
               unreachable = Some(error);
            },
            other => return other,
         }
      }

      let untried = order.len().saturating_sub(tried);
      tracing::warn!(
         tried,
         untried,
         total = self.entries.len(),
         "{} egresses exhausted for this request",
         self.label
      );
      if let Some(body) = throttled {
         // Untried egresses can still serve, so the caller should come straight back.
         let retry_after = if untried > 0 {
            1
         } else {
            self.retry_after(clock::unix_now())
         };
         return Err(SendError::RateLimited {
            retry_after: Some(retry_after),
            body,
         });
      }
      let error = unreachable.unwrap_or_else(|| format!("all {} egresses failed", self.label));
      Err(SendError::Network(if untried > 0 {
         format!("{error}, {untried} egresses untried")
      } else {
         error
      }))
   }
}
