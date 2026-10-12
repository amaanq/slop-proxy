//! The account reads codex makes against `chatgpt_base_url`, answered for
//! the one workspace every token belongs to.
use std::collections::BTreeMap;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::{Deserialize, Serialize};

use crate::clock;
use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::clientcfg::{ACCOUNT_ID, PLAN};

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

impl<'a> From<&'a AuthInfo> for Workspace<'a> {
   fn from(auth: &'a AuthInfo) -> Self {
      Self {
         account_id: ACCOUNT_ID,
         account_user_id: auth.codex_user_id(),
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
   account_user: Seat,
   can_access_with_session: bool,
   features: [(); 0],
   entitlement: Entitlement,
   last_active_subscription: LastSubscription,
}

impl<'a> From<&'a AuthInfo> for Membership<'a> {
   fn from(auth: &'a AuthInfo) -> Self {
      Self {
         account: Workspace::from(auth),
         account_user: Seat::default(),
         can_access_with_session: true,
         features: [],
         entitlement: Entitlement::default(),
         last_active_subscription: LastSubscription::default(),
      }
   }
}

/// The token's seat, a full one with no trial. The desktop reads these
/// fields without checking the object is there, and crashes without it.
#[derive(Serialize)]
struct Seat {
   is_trial: bool,
   seat_type: &'static str,
   trial_state: Option<()>,
   trial_expires_at: Option<()>,
   pending_seat_upgrade_request: bool,
}

impl Default for Seat {
   fn default() -> Self {
      Self {
         is_trial: false,
         seat_type: "default",
         trial_state: None,
         trial_expires_at: None,
         pending_seat_upgrade_request: false,
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
      id: auth.codex_user_id(),
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
         workspace: Workspace::from(&auth),
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
      accounts: BTreeMap::from([(ACCOUNT_ID, Membership::from(&auth))]),
   })
   .into_response()
}

/// The desktop's read of the signed-in account alone.
async fn account(Extension(auth): Extension<AuthInfo>) -> Response {
   Json(Membership::from(&auth)).into_response()
}

/// What the desktop says about itself. The user it gets back has to agree
/// with it, or the desktop sends the difference straight to Statsig and
/// drops these evaluations.
#[derive(Default, Deserialize)]
struct BootstrapContext {
   #[serde(default)]
   desktop_app_beta_enabled: bool,
   stable_id: Option<String>,
   app_version: Option<String>,
   locale: Option<String>,
}

#[derive(Serialize)]
struct Bootstrap {
   #[serde(rename = "statsigPayload")]
   payload: String,
}

#[derive(Serialize)]
struct Payload<'a> {
   user: StatsigUser<'a>,
   feature_gates: BTreeMap<&'a str, Gate<'a>>,
   dynamic_configs: BTreeMap<&'static str, bool>,
   layer_configs: BTreeMap<&'static str, bool>,
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
   rule_id: &'static str,
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

/// The desktop's feature gates, cloud tasks off and the rest as the config
/// sets them. Relaying an upstream account's evaluation would hand `OpenAI`
/// this token's identity under that account.
async fn statsig_bootstrap(
   State(state): State<AppState>,
   Extension(auth): Extension<AuthInfo>,
   body: Bytes,
) -> Response {
   let context = serde_json::from_slice::<BootstrapContext>(&body).unwrap_or_default();
   let gate = |name, value, rule_id| Gate {
      name,
      value,
      secondary_exposures: [],
      rule_id,
   };
   let mut feature_gates = BTreeMap::from([(CLOUD_GATE, gate(CLOUD_GATE, false, "slop-proxy"))]);
   for (name, &value) in &state.cfg.codex.desktop_gates {
      feature_gates.insert(name, gate(name, value, "slop-proxy-config"));
   }

   let stable_id = context.stable_id.as_deref();
   let payload = Payload {
      user: StatsigUser {
         user_id: auth.codex_user_id(),
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
      dynamic_configs: BTreeMap::new(),
      layer_configs: BTreeMap::new(),
      has_updates: true,
      hash_used: "none",
      time: clock::unix_now_ms(),
   };
   Json(Bootstrap {
      payload: serde_json::to_string(&payload).expect("bootstrap payload serializes"),
   })
   .into_response()
}
