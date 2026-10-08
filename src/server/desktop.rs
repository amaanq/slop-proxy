//! The account reads codex makes against `chatgpt_base_url`, answered for
//! the one workspace every token belongs to.
use std::collections::{BTreeMap, HashMap};

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::clock;
use crate::provider::Provider;
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::clientcfg::{ACCOUNT_ID, PLAN, user_id};

/// Browser, the external browser and computer use, then ultra effort and
/// the model features settings. The desktop checks each before offering
/// it, and the upstream account's plan decides them.
const RELAYED_GATES: [&str; 5] = [
   "410262010",
   "410065390",
   "1506311413",
   "1186680773",
   "3693343337",
];

/// Cloud tasks run on `ChatGPT`'s executor, which no proxy token reaches.
const CLOUD_GATE: &str = "1315865107";

pub fn routes() -> Router<AppState> {
   Router::new()
      .route("/backend-api/me", get(user))
      .route("/backend-api/wham/accounts/check", get(accounts_check))
      .route("/backend-api/accounts/check/{version}", get(accounts))
      .route("/backend-api/accounts/optimized/check", get(account))
      .route(
         "/backend-api/wham/statsig/bootstrap",
         post(statsig_bootstrap),
      )
}

/// The token's membership of the proxy's workspace, which is personal and
/// named for the token's owner.
#[derive(Serialize)]
#[expect(
   clippy::struct_excessive_bools,
   reason = "the flags the desktop reads off a workspace"
)]
struct Workspace<'a> {
   account_id: &'static str,
   account_user_id: String,
   account_user_role: &'static str,
   structure: &'static str,
   plan_type: &'static str,
   name: &'a str,
   is_deactivated: bool,
   is_zdr: bool,
   is_openai_internal: bool,
   is_fedramp_compliant_workspace: bool,
   /// `NO_CONSTRAINT` keeps requests on the `chatgpt_base_url` origin,
   /// which is the proxy.
   workspace_backend_origin: &'static str,
   account_routing_override: &'static str,
}

impl<'a> Workspace<'a> {
   fn new(auth: &'a AuthInfo) -> Self {
      Self {
         account_id: ACCOUNT_ID,
         account_user_id: user_id(auth.token_id),
         account_user_role: "account-owner",
         structure: "personal",
         plan_type: PLAN,
         name: &auth.user,
         is_deactivated: false,
         is_zdr: false,
         is_openai_internal: false,
         is_fedramp_compliant_workspace: false,
         workspace_backend_origin: "NO_CONSTRAINT",
         account_routing_override: "NO_CONSTRAINT",
      }
   }
}

#[derive(Serialize)]
struct Membership<'a> {
   account: Workspace<'a>,
   can_access_with_session: bool,
   features: [(); 0],
   entitlement: Entitlement,
   last_active_subscription: LastSubscription,
}

impl<'a> Membership<'a> {
   fn new(auth: &'a AuthInfo) -> Self {
      Self {
         account: Workspace::new(auth),
         can_access_with_session: true,
         features: [],
         entitlement: Entitlement::default(),
         last_active_subscription: LastSubscription::default(),
      }
   }
}

/// No token has a subscription of its own to show or bill.
#[derive(Default, Serialize)]
struct Entitlement {
   has_active_subscription: bool,
   billing_currency: Option<()>,
   subscription_plan: Option<()>,
   scheduled_plan_change: Option<()>,
   cancels_at: Option<()>,
}

#[derive(Default, Serialize)]
struct LastSubscription {
   subscription_id: Option<()>,
   purchase_origin_platform: Option<()>,
}

#[derive(Serialize)]
struct AccountsCheck<'a> {
   accounts: [AccountEntry<'a>; 1],
   default_account_id: &'static str,
   account_ordering: [&'static str; 1],
}

#[derive(Serialize)]
struct AccountEntry<'a> {
   id: &'static str,
   #[serde(flatten)]
   workspace: Workspace<'a>,
   can_access_with_session: bool,
}

#[derive(Serialize)]
struct Accounts<'a> {
   account_ordering: [&'static str; 1],
   accounts: BTreeMap<&'static str, Membership<'a>>,
}

#[derive(Serialize)]
struct User<'a> {
   id: String,
   object: &'static str,
   email: &'a str,
   name: &'a str,
}

async fn user(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(User {
      id: user_id(auth.token_id),
      object: "user",
      email: &auth.user,
      name: &auth.user,
   })
   .into_response()
}

/// Codex 0.156 refuses to start a ChatGPT-auth session until this workspace
/// discovery succeeds for the `auth.json` account id.
async fn accounts_check(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(AccountsCheck {
      accounts: [AccountEntry {
         id: ACCOUNT_ID,
         workspace: Workspace::new(&auth),
         can_access_with_session: true,
      }],
      default_account_id: ACCOUNT_ID,
      account_ordering: [ACCOUNT_ID],
   })
   .into_response()
}

/// The desktop's account switcher.
async fn accounts(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(Accounts {
      account_ordering: [ACCOUNT_ID],
      accounts: BTreeMap::from([(ACCOUNT_ID, Membership::new(&auth))]),
   })
   .into_response()
}

/// The desktop's read of the signed-in account alone.
async fn account(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(Membership::new(&auth)).into_response()
}

/// What the desktop says about itself. The gates are evaluated against it,
/// and the user it gets back has to agree with it, or the desktop sends the
/// difference straight to Statsig and drops these evaluations.
#[derive(Default, Deserialize)]
struct BootstrapContext {
   #[serde(default)]
   desktop_app_beta_enabled: bool,
   stable_id: Option<String>,
   app_version: Option<String>,
   locale: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct Bootstrap {
   #[serde(rename = "statsigPayload")]
   payload: String,
}

#[derive(Deserialize)]
struct Evaluations {
   #[serde(default)]
   feature_gates: HashMap<String, Evaluation>,
}

#[derive(Deserialize)]
struct Evaluation {
   value: bool,
   rule_id: String,
}

#[derive(Serialize)]
struct Payload<'a> {
   user: StatsigUser<'a>,
   feature_gates: BTreeMap<&'a str, Gate<'a>>,
   dynamic_configs: Map<String, Value>,
   layer_configs: Map<String, Value>,
   has_updates: bool,
   /// The gate names are already the hashes the desktop looks up.
   hash_used: &'static str,
   time: i64,
}

#[derive(Serialize)]
struct Gate<'a> {
   name: &'a str,
   value: bool,
   secondary_exposures: [(); 0],
   rule_id: &'a str,
}

#[derive(Serialize)]
struct StatsigUser<'a> {
   #[serde(rename = "userID")]
   user_id: String,
   email: &'a str,
   #[serde(rename = "customIDs")]
   custom_ids: CustomIds<'a>,
   custom: Custom,
   #[serde(rename = "appVersion", skip_serializing_if = "Option::is_none")]
   app_version: Option<&'a str>,
   #[serde(skip_serializing_if = "Option::is_none")]
   locale: Option<&'a str>,
}

#[derive(Serialize)]
struct CustomIds<'a> {
   account_id: &'static str,
   #[serde(rename = "stableID", skip_serializing_if = "Option::is_none")]
   stable_id: Option<&'a str>,
   #[serde(skip_serializing_if = "Option::is_none")]
   source_surface_stable_id: Option<&'a str>,
}

#[derive(Serialize)]
struct Custom {
   plan_type: &'static str,
   desktop_app_beta_enabled: bool,
}

/// The desktop's feature gates. Only the few that decide local features come
/// from an upstream account, so nothing turns on that needs a `ChatGPT` service
/// the proxy does not run, and the upstream identity stays behind.
async fn statsig_bootstrap(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   body: Bytes,
) -> Response {
   let context: BootstrapContext = serde_json::from_slice(&body).unwrap_or_default();
   let upstream = upstream_gates(&state, &auth, body).await;
   let mut feature_gates = BTreeMap::from([(
      CLOUD_GATE,
      Gate {
         name: CLOUD_GATE,
         value: false,
         rule_id: "slop-proxy",
         secondary_exposures: [],
      },
   )]);
   for gate in RELAYED_GATES {
      if let Some(evaluation) = upstream.get(gate) {
         feature_gates.insert(
            gate,
            Gate {
               name: gate,
               value: evaluation.value,
               rule_id: &evaluation.rule_id,
               secondary_exposures: [],
            },
         );
      }
   }

   let stable_id = context.stable_id.as_deref();
   let payload = Payload {
      user: StatsigUser {
         user_id: user_id(auth.token_id),
         email: &auth.user,
         custom_ids: CustomIds {
            account_id: ACCOUNT_ID,
            stable_id,
            source_surface_stable_id: stable_id,
         },
         custom: Custom {
            plan_type: PLAN,
            desktop_app_beta_enabled: context.desktop_app_beta_enabled,
         },
         app_version: context.app_version.as_deref(),
         locale: context.locale.as_deref(),
      },
      feature_gates,
      dynamic_configs: Map::new(),
      layer_configs: Map::new(),
      has_updates: true,
      hash_used: "none",
      time: clock::unix_now_ms(),
   };
   Json(Bootstrap {
      payload: serde_json::to_string(&payload).expect("bootstrap payload serializes"),
   })
   .into_response()
}

/// None when the token may not reach the codex backend or the account's
/// answer is unusable, which leaves every relayed feature off.
async fn upstream_gates(
   state: &AppState,
   auth: &AuthInfo,
   body: Bytes,
) -> HashMap<String, Evaluation> {
   if !auth.limits.may_use(Provider::OpenAi) {
      return HashMap::new();
   }
   let route = auth.route(&auth.user, "");
   let evaluations = async {
      let served = state
         .pools
         .codex
         .backend(route, "/wham/statsig/bootstrap".into(), Some(body))
         .await
         .map_err(|err| err.to_string())?;
      let bootstrap: Bootstrap = served
         .response
         .json()
         .await
         .map_err(|err| err.to_string())?;
      serde_json::from_str::<Evaluations>(&bootstrap.payload).map_err(|err| err.to_string())
   };
   match evaluations.await {
      Ok(evaluations) => evaluations.feature_gates,
      Err(err) => {
         tracing::warn!("desktop feature gates from the codex backend: {err}");
         HashMap::new()
      },
   }
}
