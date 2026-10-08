//! The account reads codex makes against `chatgpt_base_url`, answered for
//! the one workspace every token belongs to.
use std::collections::BTreeMap;

use axum::response::{IntoResponse as _, Response};
use axum::routing::get;
use axum::{Extension, Json, Router};
use serde::Serialize;

use crate::server::AppState;
use crate::server::auth::AuthInfo;
use crate::server::clientcfg::{ACCOUNT_ID, PLAN, user_id};

pub fn routes() -> Router<AppState> {
   Router::new()
      .route("/backend-api/me", get(user))
      .route("/backend-api/wham/accounts/check", get(accounts_check))
      .route("/backend-api/accounts/check/{version}", get(accounts))
      .route("/backend-api/accounts/optimized/check", get(account))
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
