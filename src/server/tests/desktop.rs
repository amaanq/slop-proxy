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
