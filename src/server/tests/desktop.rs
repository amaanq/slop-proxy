use super::*;
use serde_json::{Value, json};

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
