use axum::Json;
use axum::http::HeaderMap;
use axum::routing::get;
use serde_json::{Value, json};

use super::*;
use crate::server::desktop::{ACCEPTED_WRITES, EMPTY_READS};

fn claims(jwt: &str) -> Value {
   let payload = jwt.split('.').nth(1).unwrap();
   let bytes = data_encoding::BASE64URL_NOPAD
      .decode(payload.as_bytes())
      .unwrap();
   serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn codex_auth_carries_the_tokens_identity_and_refreshes_through_its_key() {
   let (base, db) = spawn_proxy().await;
   let client = reqwest::Client::new();
   let token_id = db.auth_token("sp-test").await.unwrap().unwrap().id;
   let file: Value = client
      .get(format!("{base}/config/codex/auth.json"))
      .bearer_auth("sp-test")
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
   let id_token = claims(file["tokens"]["id_token"].as_str().unwrap());
   assert_eq!(id_token["name"], "alice");
   assert_eq!(
      id_token["https://api.openai.com/auth"]["chatgpt_user_id"],
      format!("slop-proxy-token-{token_id}")
   );
   let access = file["tokens"]["access_token"].as_str().unwrap();
   assert_eq!(claims(access)["api_key"], "sp-test");
   assert_eq!(claims(access)["name"], "alice");

   // The access token authenticates as the key it wraps.
   let check = client
      .get(format!("{base}/backend-api/wham/accounts/check"))
      .bearer_auth(access)
      .send()
      .await
      .unwrap();
   assert_eq!(check.status(), StatusCode::OK);

   let refreshed: Value = client
      .post(format!("{base}/oauth/token"))
      .json(&json!({
         "client_id": "app",
         "grant_type": "refresh_token",
         "refresh_token": file["tokens"]["refresh_token"],
      }))
      .send()
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
   assert_eq!(
      claims(refreshed["access_token"].as_str().unwrap())["api_key"],
      "sp-test"
   );
   assert_eq!(refreshed["access_token"], refreshed["refresh_token"]);
   assert_eq!(
      claims(refreshed["id_token"].as_str().unwrap())["sub"],
      format!("slop-proxy-token-{token_id}")
   );

   db.revoke_token("sp-test").await.unwrap();
   let revoked = client
      .post(format!("{base}/oauth/token"))
      .json(&json!({"refresh_token": access}))
      .send()
      .await
      .unwrap();
   assert_eq!(revoked.status(), StatusCode::UNAUTHORIZED);
}

/// One request per window, spent before the desktop starts reading.
async fn spend_quota(db: &Db) {
   let limits = TokenLimits {
      requests: Some(1),
      window_seconds: 3600,
      ..TokenLimits::default()
   };
   db.set_token_limits("sp-test", &limits).await.unwrap();
   let token = db.auth_token("sp-test").await.unwrap().unwrap();
   db.admit_token(token.id, &limits).await.unwrap().unwrap();
}

#[tokio::test]
async fn account_reads_name_the_owner_and_answer_once_quota_is_spent() {
   let (base, db) = spawn_proxy().await;
   spend_quota(&db).await;
   let client = reqwest::Client::new();
   let get = |path: &str, key: Option<&str>| {
      let request = client.get(format!("{base}{path}"));
      match key {
         Some(key) => request.bearer_auth(key),
         None => request,
      }
      .send()
   };

   for path in [
      "/backend-api/me",
      "/backend-api/wham/accounts/check",
      "/backend-api/accounts/check/v4-2023-04-27",
      "/backend-api/accounts/optimized/check",
   ] {
      assert_eq!(
         get(path, None).await.unwrap().status(),
         StatusCode::UNAUTHORIZED
      );
      assert_eq!(
         get(path, Some("sp-test")).await.unwrap().status(),
         StatusCode::OK,
         "{path}"
      );
   }

   let check: Value = get("/backend-api/wham/accounts/check", Some("sp-test"))
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
   assert_eq!(check["accounts"][0]["id"], "slop-proxy");
   assert_eq!(check["accounts"][0]["name"], "alice");
   assert_eq!(
      check["accounts"][0]["workspace_backend_origin"],
      "NO_CONSTRAINT"
   );
   let accounts: Value = get("/backend-api/accounts/check/v4-2023-04-27", Some("sp-test"))
      .await
      .unwrap()
      .json()
      .await
      .unwrap();
   assert_eq!(accounts["account_ordering"][0], "slop-proxy");
   let member = &accounts["accounts"]["slop-proxy"];
   assert_eq!(member["account"]["name"], "alice");
   assert!(member["entitlement"]["subscription_plan"].is_null());

   assert_eq!(
      db.token_meter("sp-test").await.unwrap().unwrap().requests,
      1
   );
}

/// A codex backend answering the desktop's feature gates for `acct-1`, and
/// counting how often it was asked.
async fn spawn_backend(reads: Arc<Mutex<Vec<Value>>>) -> String {
   let app = Router::new()
      .route(
         "/backend-api/wham/statsig/bootstrap",
         post(move |headers: HeaderMap, Json(context): Json<Value>| {
            let reads = Arc::clone(&reads);
            async move {
               assert_eq!(headers["authorization"], "Bearer at");
               assert_eq!(headers["chatgpt-account-id"], "acct-1");
               reads.lock().unwrap().push(context);
               let payload = json!({
                  "hash_used": "djb2",
                  "user": {"email": "upstream@example.com"},
                  "feature_gates": {
                     "410262010": {"name": "410262010", "value": true, "rule_id": "browser"},
                     "1506311413": {"name": "1506311413", "value": false, "rule_id": "computer"},
                     "1315865107": {"name": "1315865107", "value": true, "rule_id": "cloud"},
                     "99": {"name": "99", "value": true, "rule_id": "unrelated"},
                  },
               });
               Json(json!({"statsigPayload": payload.to_string()}))
            }
         }),
      )
      .route(
         "/backend-api/aura/site_status",
         get(|| async { Json(json!({"feature_status": {}})) }),
      );
   let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
   let addr = listener.local_addr().unwrap();
   tokio::spawn(async move {
      axum::serve(listener, app).await.unwrap();
   });
   format!("http://{addr}/backend-api/codex")
}

fn statsig_payload(bootstrap: &Value) -> Value {
   serde_json::from_str(bootstrap["statsigPayload"].as_str().unwrap()).unwrap()
}

#[tokio::test]
async fn statsig_bootstrap_relays_only_the_local_feature_gates() {
   let reads = Arc::new(Mutex::new(Vec::new()));
   let base_url = spawn_backend(Arc::clone(&reads)).await;
   let (base, db) = spawn_proxy_at(ModelsConfig::default(), None, base_url).await;
   spend_quota(&db).await;
   let client = reqwest::Client::new();
   let bootstrap = || {
      client
         .post(format!("{base}/backend-api/wham/statsig/bootstrap"))
         .bearer_auth("sp-test")
         .json(&json!({
            "window_type": "electron",
            "stable_id": "device",
            "app_version": "26.1",
            "desktop_app_beta_enabled": true,
         }))
         .send()
   };

   let payload = statsig_payload(&bootstrap().await.unwrap().json().await.unwrap());
   let gates = &payload["feature_gates"];
   assert_eq!(gates["410262010"]["value"], true);
   assert_eq!(gates["1506311413"]["value"], false);
   assert_eq!(gates["1506311413"]["rule_id"], "computer");
   assert_eq!(gates["1315865107"]["value"], false);
   assert!(gates.get("99").is_none());
   assert_eq!(payload["user"]["email"], "alice");
   assert_eq!(payload["user"]["customIDs"]["stableID"], "device");
   assert_eq!(payload["user"]["appVersion"], "26.1");
   assert_eq!(payload["user"]["custom"]["desktop_app_beta_enabled"], true);
   assert!(!payload.to_string().contains("upstream@"));
   assert_eq!(reads.lock().unwrap()[0]["stable_id"], "device");

   let site = client
      .get(format!(
         "{base}/backend-api/aura/site_status?site_url=https://example.com"
      ))
      .bearer_auth("sp-test")
      .send()
      .await
      .unwrap();
   assert_eq!(site.status(), StatusCode::OK);
   assert_eq!(
      db.token_meter("sp-test").await.unwrap().unwrap().requests,
      1
   );

   // A token kept off the codex backend gets no upstream evaluation at all.
   db.set_token_limits(
      "sp-test",
      &TokenLimits {
         providers: vec![Provider::Anthropic],
         ..TokenLimits::default()
      },
   )
   .await
   .unwrap();
   let scoped = statsig_payload(&bootstrap().await.unwrap().json().await.unwrap());
   assert!(scoped["feature_gates"].get("410262010").is_none());
   assert_eq!(scoped["feature_gates"]["1315865107"]["value"], false);
   assert_eq!(reads.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn reads_of_chatgpt_only_features_answer_empty_once_quota_is_spent() {
   let (base, db) = spawn_proxy().await;
   spend_quota(&db).await;
   let client = reqwest::Client::new();
   let reads = EMPTY_READS.iter().map(|&(path, _)| (path, false));
   let writes = ACCEPTED_WRITES.iter().map(|&(path, _)| (path, true));
   for (path, write) in reads.chain(writes) {
      let url = format!(
         "{base}{}",
         path
            .replace("{account}", "slop-proxy")
            .replace("{category}", "featured")
      );
      let request = |key: Option<&str>| {
         let request = if write {
            client.post(&url).json(&json!({"events": []}))
         } else {
            client.get(&url)
         };
         match key {
            Some(key) => request.bearer_auth(key),
            None => request,
         }
         .send()
      };
      assert_eq!(
         request(None).await.unwrap().status(),
         StatusCode::UNAUTHORIZED,
         "{path}"
      );
      let response = request(Some("sp-test")).await.unwrap();
      assert_eq!(response.status(), StatusCode::OK, "{path}");
      response.json::<Value>().await.unwrap();
   }
}

#[tokio::test]
async fn usage_and_profile_count_the_owners_tokens_once_quota_is_spent() {
   let (base, db) = spawn_proxy().await;
   let client = reqwest::Client::new();
   let complete = || {
      client
         .post(format!("{base}/v1/chat/completions"))
         .bearer_auth("sp-test")
         .json(&json!({"model": "gpt-5-codex", "messages": [{"role": "user", "content": "hi"}]}))
         .send()
   };
   assert_eq!(complete().await.unwrap().status(), StatusCode::OK);
   db.flush().await.unwrap();
   db.set_token_limits(
      "sp-test",
      &TokenLimits {
         requests: Some(1),
         window_seconds: 3600,
         ..TokenLimits::default()
      },
   )
   .await
   .unwrap();
   assert_eq!(
      complete().await.unwrap().status(),
      StatusCode::TOO_MANY_REQUESTS
   );
   let read = |path: &str| {
      let url = format!("{base}{path}");
      let client = client.clone();
      async move {
         let response = client.get(url).bearer_auth("sp-test").send().await.unwrap();
         (response.status(), response.json::<Value>().await.unwrap())
      }
   };
   let today = clock::date(clock::unix_now().div_euclid(86400));

   // 80 fresh input, 20 cached and 25 output, reasoning already inside it.
   let (status, profile) = read("/backend-api/wham/profiles/me").await;
   assert_eq!(status, StatusCode::OK);
   assert_eq!(profile["stats"]["lifetime_tokens"], 125_i32);
   assert_eq!(profile["stats"]["current_streak_days"], 1_i32);
   assert_eq!(
      profile["stats"]["daily_usage_buckets"][0]["start_date"],
      today
   );
   assert_eq!(profile["profile"]["username"], "alice");

   let (_, page) = read("/backend-api/profiles/me/page").await;
   assert_eq!(page["page"]["stats"]["agentic"]["lifetime_tokens"], 125_i32);
   assert_eq!(
      page["page"]["activity_graph"]["weekly_usage_buckets"][0]["tokens"],
      125_i32
   );
   assert_eq!(
      read("/backend-api/profiles/alice/page").await.0,
      StatusCode::OK
   );
   assert_eq!(
      read("/backend-api/profiles/bob/page").await.0,
      StatusCode::NOT_FOUND
   );

   let (_, history) = read("/backend-api/wham/usage/daily-token-usage-breakdown").await;
   assert_eq!(history["units"], "tokens");
   let day = &history["data"][0];
   assert_eq!(day["date"], today);
   assert_eq!(day["models"][0]["uncached_text_input_tokens"], 80_i32);
   assert_eq!(day["models"][0]["cached_text_input_tokens"], 20_i32);
   assert_eq!(day["models"][0]["text_output_tokens"], 25_i32);
   assert_eq!(day["product_surface_usage_values"]["codex"], 125_i32);
   let (_, past) =
      read("/backend-api/wham/usage/daily-token-usage-breakdown?start_date=2020-01-01&end_date=2020-01-31").await;
   assert_eq!(past["data"], json!([]));
   for range in [
      "start_date=2026-1-1",
      "start_date=2020-02-01&end_date=2020-01-01",
      "start_date=2020-01-01&end_date=2021-12-31",
   ] {
      let (rejected, _) = read(&format!(
         "/backend-api/wham/usage/daily-token-usage-breakdown?{range}"
      ))
      .await;
      assert_eq!(rejected, StatusCode::BAD_REQUEST, "{range}");
   }

   let (limited, limits) = read("/backend-api/wham/usage").await;
   assert_eq!(limited, StatusCode::OK);
   assert_eq!(limits["plan_type"], "pro");
   assert_eq!(limits["rate_limit"]["allowed"], true);
}
