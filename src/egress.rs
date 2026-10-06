//! One HTTP client per configured proxy, rotated per request and cooled off
//! when it stops answering. A provider with no proxies configured gets a
//! single direct client and the same code path. A proxy file that empties
//! leaves no egress, never the host's own address.

use std::net::{IpAddr, Ipv6Addr};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use ipnet::Ipv6Net;
use rand::RngCore as _;
use rand::seq::index::sample;

use crate::clock;
use crate::config::EgressConfig;
use crate::upstream::SendError;

/// How many egresses one request may burn before it gives up. A long proxy
/// list otherwise turns a dead upstream into a very slow failure.
pub const ATTEMPTS: usize = 8;

const UNREACHABLE_COOLDOWN: i64 = 30;

const SOURCE_EGRESSES: usize = 1 << 16;

pub struct Egresses {
   entries: RwLock<Arc<[Arc<Egress>]>>,
   next: AtomicUsize,
   label: &'static str,
   user_agent: Option<String>,
}

struct Egress {
   route: Route,
   /// Built on first send, so a /48 of sources does not build 65536 TLS
   /// configs before the server binds.
   http: OnceLock<reqwest::Client>,
   unavailable_until: AtomicI64,
   /// Zen's free tier rate-limits per source address, so an anonymous 429
   /// benches the egress for anonymous traffic only.
   anonymous_until: AtomicI64,
}

enum Route {
   Direct,
   Proxy(String),
   Source(Ipv6Addr),
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
      cfg: &EgressConfig,
      label: &'static str,
      user_agent: Option<&str>,
   ) -> eyre::Result<Self> {
      let egresses = Self {
         entries: RwLock::new(Arc::new([])),
         next: AtomicUsize::new(0),
         label,
         user_agent: user_agent.map(str::to_owned),
      };
      let proxy_urls = cfg.urls()?;
      if let Some(prefix) = cfg.source_prefix {
         eyre::ensure!(
            proxy_urls.is_empty() && cfg.proxy_urls_file.is_none(),
            "{label} sets both proxies and a source prefix"
         );
         let sources = source_addresses(prefix)?
            .into_iter()
            .map(|address| Arc::new(Egress::lazy(Route::Source(address))))
            .collect::<Arc<[_]>>();
         egresses.build(&sources[0].route, 0)?;
         egresses.store(sources);
      } else if proxy_urls.is_empty() && cfg.proxy_urls_file.is_none() {
         let direct = egresses.built(Route::Direct, 0)?;
         egresses.store(Arc::new([Arc::new(direct)]));
      } else {
         egresses.replace(&proxy_urls)?;
      }
      Ok(egresses)
   }

   fn built(&self, route: Route, index: usize) -> eyre::Result<Egress> {
      let http = self.build(&route, index)?;
      Ok(Egress {
         http: OnceLock::from(http),
         ..Egress::lazy(route)
      })
   }

   fn client(&self, egress: &Egress) -> reqwest::Client {
      egress
         .http
         .get_or_init(|| {
            self
               .build(&egress.route, 0)
               .expect("egress client config is proven at startup")
         })
         .clone()
   }

   fn build(&self, route: &Route, index: usize) -> eyre::Result<reqwest::Client> {
      let label = self.label;
      let mut builder = reqwest::Client::builder()
         .connect_timeout(Duration::from_secs(30))
         .tcp_keepalive(Duration::from_secs(30));
      if let Some(agent) = self.user_agent.as_deref() {
         builder = builder.user_agent(agent);
      }
      match *route {
         Route::Direct => {},
         Route::Proxy(ref proxy_url) => {
            let proxy = reqwest::Proxy::all(proxy_url)
               .map_err(|_| eyre::eyre!("invalid {label} proxy URL at position {}", index + 1))?;
            builder = builder.proxy(proxy);
         },
         Route::Source(address) => {
            builder = builder
               .local_address(IpAddr::V6(address))
               .pool_max_idle_per_host(0);
         },
      }
      builder
         .build()
         .map_err(|_| eyre::eyre!("building {label} HTTP client"))
   }

   pub fn replace(&self, proxy_urls: &[String]) -> eyre::Result<()> {
      let current = self.snapshot();
      let mut added = 0_usize;
      let entries = proxy_urls
         .iter()
         .enumerate()
         .map(|(index, url)| {
            if let Some(kept) = current
               .iter()
               .find(|egress| matches!(&egress.route, Route::Proxy(kept) if kept == url))
            {
               return Ok(Arc::clone(kept));
            }
            added += 1;
            self.built(Route::Proxy(url.clone()), index).map(Arc::new)
         })
         .collect::<eyre::Result<Arc<[_]>>>()?;
      let dropped = current.len() + added - entries.len();
      if added > 0 || dropped > 0 {
         tracing::info!(
            added,
            dropped,
            total = entries.len(),
            "{} proxy list changed",
            self.label
         );
      }
      self.store(entries);
      Ok(())
   }

   fn store(&self, entries: Arc<[Arc<Egress>]>) {
      *self.entries.write().expect("egress lock poisoned") = entries;
   }

   fn snapshot(&self) -> Arc<[Arc<Egress>]> {
      Arc::clone(&self.entries.read().expect("egress lock poisoned"))
   }

   pub fn http(&self, index: usize) -> reqwest::Client {
      self.client(&self.snapshot()[index])
   }

   /// Every egress still in service, starting one past the last request so
   /// concurrent callers spread across the list rather than stacking on the
   /// first healthy one.
   fn order(&self, entries: &[Arc<Egress>], anonymous: bool) -> Vec<usize> {
      let now = clock::unix_now();
      let start = self.next.fetch_add(1, Ordering::Relaxed);
      (0..entries.len())
         .map(|offset| start.wrapping_add(offset) % entries.len())
         .filter(|&index| {
            let egress = &entries[index];
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

   fn retry_after(entries: &[Arc<Egress>], now: i64) -> i64 {
      entries
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
      let entries = self.snapshot();
      let order = self.order(&entries, anonymous);
      if order.is_empty() {
         let now = clock::unix_now();
         let throttled = anonymous
            && entries
               .iter()
               .any(|egress| egress.anonymous_until.load(Ordering::Relaxed) > now);
         tracing::warn!(total = entries.len(), "no {} egress available", self.label);
         let body = format!(
            "all {} {} egresses are cooling down",
            entries.len(),
            self.label
         );
         return Err(if throttled {
            SendError::RateLimited {
               retry_after: Some(Self::retry_after(&entries, now)),
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
         match attempt(self.client(&entries[index]))
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
               Self::cool(&entries[index].anonymous_until, retry_after.unwrap_or(60));
               throttled = Some(body);
            },
            Err(SendError::Network(error)) => {
               tracing::warn!(egress = index, "{} egress unreachable: {error}", self.label);
               if entries.len() > 1 {
                  Self::cool(&entries[index].unavailable_until, UNREACHABLE_COOLDOWN);
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
         total = entries.len(),
         "{} egresses exhausted for this request",
         self.label
      );
      if let Some(body) = throttled {
         // Untried egresses can still serve, so the caller should come straight back.
         let retry_after = if untried > 0 {
            1
         } else {
            Self::retry_after(&entries, clock::unix_now())
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

fn source_addresses(prefix: Ipv6Net) -> eyre::Result<Vec<Ipv6Addr>> {
   eyre::ensure!(
      (32..=64).contains(&prefix.prefix_len()),
      "source prefix {prefix} must be between /32 and /64"
   );
   let subnets = 1_usize << (64 - prefix.prefix_len());
   let network = u128::from(prefix.network());
   let mut rng = rand::thread_rng();
   Ok(sample(&mut rng, subnets, subnets.min(SOURCE_EGRESSES))
      .into_iter()
      .map(|subnet| {
         let host = u128::from(rng.next_u64().max(1));
         Ipv6Addr::from(network | ((subnet as u128) << 64_u32) | host)
      })
      .collect())
}

impl Egress {
   const fn lazy(route: Route) -> Self {
      Self {
         route,
         http: OnceLock::new(),
         unavailable_until: AtomicI64::new(0),
         anonymous_until: AtomicI64::new(0),
      }
   }
}
