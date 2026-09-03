use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use eyre::{Result, WrapErr};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    pub input: f64,
    pub output: f64,
    pub cache_write: f64,
    pub cache_read: f64,
}

/// Tokens as the proxy stores them, with `input` already net of anything the
/// provider served from cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tokens {
    pub input: i64,
    pub output: i64,
    pub cache_read: i64,
    pub cache_write: i64,
}

impl Tokens {
    fn context(&self) -> i64 {
        self.input
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub base: Rates,
    /// Threshold in tokens and the rates that replace `base` once a request's
    /// whole context crosses it. Both vendors switch the entire request rather
    /// than billing the excess separately, so this is not a marginal tier.
    pub long_context: Option<(i64, Rates)>,
}

impl ModelPrice {
    pub fn cost(&self, t: Tokens) -> f64 {
        let r = match self.long_context {
            Some((threshold, above)) if t.context() > threshold => above,
            _ => self.base,
        };
        t.input as f64 * r.input
            + t.output as f64 * r.output
            + t.cache_read as f64 * r.cache_read
            + t.cache_write as f64 * r.cache_write
    }
}

#[derive(Debug, Default)]
pub struct PriceTable(HashMap<String, ModelPrice>);

impl PriceTable {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// LiteLLM keys a model under both its bare name and vendor-prefixed
    /// spellings, so a lookup that misses is retried against the prefixed
    /// forms before giving up.
    pub fn find(&self, model: &str) -> Option<ModelPrice> {
        if let Some(p) = self.0.get(model) {
            return Some(*p);
        }
        let suffix = format!("/{model}");
        let dotted = format!(".{model}");
        self.0
            .iter()
            .find(|(k, _)| k.ends_with(&suffix) || k.ends_with(&dotted))
            .map(|(_, p)| *p)
    }

    pub fn cost(&self, model: &str, t: Tokens) -> f64 {
        self.find(model)
            .or_else(|| unpublished(model, crate::clock::unix_now()))
            .map_or(0.0, |p| p.cost(t))
    }

    fn parse(body: &str) -> Result<Self> {
        let raw: HashMap<String, Entry> =
            serde_json::from_str(body).wrap_err("parsing the price table")?;
        Ok(Self(
            raw.into_iter()
                .filter_map(|(name, e)| e.into_price().map(|p| (name, p)))
                .collect(),
        ))
    }
}

/// Rates litellm has not published, consulted only when the fetched table
/// misses so an upstream entry takes over the moment one appears. Google is
/// running Gemini 3.8 Flash at an introductory rate that doubles on
/// 2027-01-01, per https://ai.google.dev/gemini-api/docs/pricing.
pub fn unpublished(model: &str, now: i64) -> Option<ModelPrice> {
    const INTRO_ENDS: i64 = 1_798_761_600;
    let scale = if now < INTRO_ENDS { 1.0 } else { 2.0 };
    let base = match model {
        "gemini-3.8-flash" => Rates {
            input: 0.75e-6 * scale,
            output: 3.75e-6 * scale,
            cache_write: 0.0,
            cache_read: 0.075e-6 * scale,
        },
        _ => return None,
    };
    Some(ModelPrice {
        base,
        long_context: None,
    })
}

#[derive(Debug, Deserialize)]
struct Entry {
    #[serde(default)]
    input_cost_per_token: f64,
    #[serde(default)]
    output_cost_per_token: f64,
    #[serde(default)]
    cache_creation_input_token_cost: f64,
    #[serde(default)]
    cache_read_input_token_cost: f64,
    /// The tier threshold is encoded in the key rather than a value, as in
    /// `input_cost_per_token_above_272k_tokens`, so the above-tier rates can
    /// only be found by scanning field names.
    #[serde(flatten)]
    rest: HashMap<String, serde_json::Value>,
}

impl Entry {
    fn into_price(self) -> Option<ModelPrice> {
        let base = Rates {
            input: self.input_cost_per_token,
            output: self.output_cost_per_token,
            cache_write: self.cache_creation_input_token_cost,
            cache_read: self.cache_read_input_token_cost,
        };
        if base == Rates::default() {
            return None;
        }
        Some(ModelPrice {
            base,
            long_context: self.long_context(base),
        })
    }

    /// Ignores the `_flex`, `_priority` and `_batches` variants, which price a
    /// different service tier than the one the proxy sends.
    fn long_context(&self, base: Rates) -> Option<(i64, Rates)> {
        let mut threshold = None;
        let mut above = base;
        for (key, value) in &self.rest {
            let Some((prefix, tail)) = key.split_once("_above_") else {
                continue;
            };
            let Some(k) = tail
                .strip_suffix("k_tokens")
                .and_then(|n| n.parse::<i64>().ok())
            else {
                continue;
            };
            let Some(rate) = value.as_f64() else { continue };
            let slot = match prefix {
                "input_cost_per_token" => &mut above.input,
                "output_cost_per_token" => &mut above.output,
                "cache_creation_input_token_cost" => &mut above.cache_write,
                "cache_read_input_token_cost" => &mut above.cache_read,
                _ => continue,
            };
            *slot = rate;
            threshold = Some(threshold.map_or(k, |t: i64| t.min(k)));
        }
        threshold.map(|k| (k * 1000, above))
    }
}

/// Holds the current table and keeps it fresh. The last good fetch is written
/// beside the database so a restart without network still bills correctly.
pub struct Prices {
    table: RwLock<Arc<PriceTable>>,
    cache_path: PathBuf,
    url: String,
}

impl Prices {
    pub fn new(db_path: &Path, url: String) -> Self {
        let cache_path = db_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("litellm-prices.json");
        Self {
            table: RwLock::new(Arc::new(PriceTable::default())),
            cache_path,
            url,
        }
    }

    pub fn table(&self) -> Arc<PriceTable> {
        self.table.read().unwrap().clone()
    }

    pub fn cost(&self, model: &str, t: Tokens) -> f64 {
        self.table().cost(model, t)
    }

    /// Loads the cached copy first so pricing is available before the network
    /// is, then refreshes from upstream.
    pub async fn load(&self) {
        if self.table().is_empty()
            && let Ok(body) = tokio::fs::read_to_string(&self.cache_path).await
            && let Ok(table) = PriceTable::parse(&body)
        {
            tracing::info!("loaded {} cached model prices", table.len());
            *self.table.write().unwrap() = Arc::new(table);
        }
        if let Err(e) = self.refresh().await {
            tracing::warn!("refreshing model prices: {e}");
        }
    }

    pub async fn refresh(&self) -> Result<()> {
        let body = reqwest::get(&self.url)
            .await
            .wrap_err("fetching the price table")?
            .error_for_status()?
            .text()
            .await?;
        let table = PriceTable::parse(&body)?;
        if table.is_empty() {
            eyre::bail!("price table has no priced models");
        }
        tracing::info!("loaded {} model prices", table.len());
        let _ = tokio::fs::write(&self.cache_path, &body).await;
        *self.table.write().unwrap() = Arc::new(table);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "claude-opus-5": {
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000025,
        "cache_creation_input_token_cost": 0.00000625,
        "cache_read_input_token_cost": 0.0000005,
        "litellm_provider": "anthropic"
      },
      "gpt-5.6-sol": {
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.00002,
        "cache_creation_input_token_cost": 0.000005,
        "cache_read_input_token_cost": 0.0000004,
        "input_cost_per_token_above_272k_tokens": 0.000008,
        "output_cost_per_token_above_272k_tokens": 0.00003,
        "cache_read_input_token_cost_above_272k_tokens": 0.0000008,
        "input_cost_per_token_flex": 0.000002,
        "litellm_provider": "openai"
      },
      "chatgpt/gpt-5.3-codex-spark": { "litellm_provider": "chatgpt" },
      "vertex_ai/claude-opus-5": {
        "input_cost_per_token": 0.000006,
        "output_cost_per_token": 0.00003,
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 0.0
      }
    }"#;

    fn table() -> PriceTable {
        PriceTable::parse(SAMPLE).unwrap()
    }

    #[test]
    fn an_unpublished_model_still_bills() {
        let t = table();
        let tokens = Tokens {
            input: 1_000_000,
            output: 1_000_000,
            cache_read: 1_000_000,
            cache_write: 0,
        };
        assert!((t.cost("gemini-3.8-flash", tokens) - 4.575).abs() < 1e-9);
        assert_eq!(t.cost("gemini-3.8-pro", tokens), 0.0);
    }

    #[test]
    fn the_introductory_rate_expires() {
        let intro = unpublished("gemini-3.8-flash", 1_798_761_599).unwrap();
        let after = unpublished("gemini-3.8-flash", 1_798_761_600).unwrap();
        assert!((after.base.input - intro.base.input * 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_published_price_beats_the_builtin() {
        let t = PriceTable::parse(
            r#"{"gemini-3.8-flash": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.0}}"#,
        )
        .unwrap();
        assert_eq!(t.find("gemini-3.8-flash").unwrap().base.input, 0.001);
    }

    #[test]
    fn unpriced_models_are_dropped() {
        let t = table();
        assert!(t.find("chatgpt/gpt-5.3-codex-spark").is_none());
        assert_eq!(t.cost("gpt-5.3-codex-spark", Tokens::default()), 0.0);
    }

    #[test]
    fn a_bare_name_beats_a_vendor_prefixed_one() {
        let t = table();
        let p = t.find("claude-opus-5").unwrap();
        assert_eq!(p.base.input, 0.000005);
    }

    #[test]
    fn a_prefixed_key_is_found_by_bare_name() {
        let t = PriceTable::parse(
            r#"{"chatgpt/gpt-x": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}}"#,
        )
        .unwrap();
        assert_eq!(t.find("gpt-x").unwrap().base.input, 0.001);
    }

    #[test]
    fn cost_bills_every_bucket() {
        let t = table();
        let cost = t.cost(
            "claude-opus-5",
            Tokens {
                input: 1_000,
                output: 2_000,
                cache_read: 50_000,
                cache_write: 4_000,
            },
        );
        let want =
            1_000.0 * 0.000005 + 2_000.0 * 0.000025 + 50_000.0 * 0.0000005 + 4_000.0 * 0.00000625;
        assert!((cost - want).abs() < 1e-12, "{cost} != {want}");
    }

    #[test]
    fn crossing_the_threshold_reprices_the_whole_request() {
        let t = table();
        let under = t.cost(
            "gpt-5.6-sol",
            Tokens {
                input: 10_000,
                output: 1_000,
                cache_read: 100_000,
                cache_write: 0,
            },
        );
        assert!(
            (under - (10_000.0 * 0.000004 + 1_000.0 * 0.00002 + 100_000.0 * 0.0000004)).abs()
                < 1e-12
        );

        let over = t.cost(
            "gpt-5.6-sol",
            Tokens {
                input: 10_000,
                output: 1_000,
                cache_read: 300_000,
                cache_write: 0,
            },
        );
        let want = 10_000.0 * 0.000008 + 1_000.0 * 0.00003 + 300_000.0 * 0.0000008;
        assert!((over - want).abs() < 1e-12, "{over} != {want}");
    }

    #[test]
    fn a_missing_above_rate_keeps_the_base_one() {
        let t = table();
        let p = t.find("gpt-5.6-sol").unwrap();
        let (threshold, above) = p.long_context.unwrap();
        assert_eq!(threshold, 272_000);
        assert_eq!(above.cache_write, p.base.cache_write);
    }
}
