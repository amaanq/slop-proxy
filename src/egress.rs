//! One HTTP client per configured proxy, rotated per request and cooled off
//! when it stops answering. A provider with no proxies configured gets a
//! single direct client and the same code path.

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::clock;
use crate::upstream::SendError;

/// How many egresses one request may burn before it gives up. A long proxy
/// list otherwise turns a dead upstream into a very slow failure.
const ATTEMPTS: usize = 8;

/// Seconds an egress sits out after refusing to connect.
const UNREACHABLE_COOLDOWN: i64 = 30;

pub struct Egresses {
   entries: Vec<Egress>,
   next: AtomicUsize,
   label: &'static str,
}

struct Egress {
   http: reqwest::Client,
   unavailable_until: AtomicI64,
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
   pub fn order(&self) -> Vec<usize> {
      let now = clock::unix_now();
      let start = self.next.fetch_add(1, Ordering::Relaxed);
      (0..self.entries.len())
         .map(|offset| start.wrapping_add(offset) % self.entries.len())
         .filter(|index| {
            self.entries[*index]
               .unavailable_until
               .load(Ordering::Relaxed)
               <= now
         })
         .collect()
   }

   pub fn cool(&self, index: usize, seconds: i64) {
      self.entries[index]
         .unavailable_until
         .store(clock::unix_now() + seconds, Ordering::Relaxed);
   }

   /// Retries the next egress when one cannot be reached at all. Anything the
   /// upstream itself answered is the upstream's verdict and stops the walk,
   /// since a second egress would only ask the same question again.
   pub async fn send<Attempt, Fut>(&self, attempt: Attempt) -> Result<reqwest::Response, SendError>
   where
      Attempt: Fn(reqwest::Client) -> Fut,
      Fut: Future<Output = Result<reqwest::Response, SendError>>,
   {
      let order = self.order();
      let mut unreachable = None;
      let mut tried = 0_usize;
      for index in order.iter().copied().take(ATTEMPTS) {
         tried += 1;
         match attempt(self.http(index).clone()).await {
            Ok(response) => {
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
            Err(SendError::Network(error)) => {
               tracing::warn!(egress = index, "{} egress unreachable: {error}", self.label);
               self.cool(index, UNREACHABLE_COOLDOWN);
               unreachable = Some(error);
            },
            other => return other,
         }
      }
      tracing::warn!(
         tried,
         total = self.entries.len(),
         "{} egresses exhausted for this request",
         self.label
      );
      Err(SendError::Network(unreachable.unwrap_or_else(|| {
         format!(
            "all {} {} egresses are cooling down",
            self.entries.len(),
            self.label
         )
      })))
   }
}
