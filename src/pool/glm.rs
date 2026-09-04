use axum::body::Bytes;

use super::{AuthPolicy, Backend, Cooldown, Pool, Route, Slot};
use crate::glm::client::GlmClient;
use crate::provider::Provider;
use crate::upstream::SendError;

/// Session-sticky pool over Z.ai keys.
pub type GlmPool = Pool<GlmClient>;

#[derive(Clone)]
pub struct Relay {
    pub path: &'static str,
    pub body: Bytes,
}

impl Backend for GlmClient {
    const PROVIDER: Provider = Provider::Glm;
    const RATE_LIMIT: Cooldown = Cooldown {
        max: 3600,
        base: 60,
    };
    const ON_AUTH: AuthPolicy = AuthPolicy::CoolKey(15 * 60);
    type Request = Relay;
    type Response = reqwest::Response;

    fn reason(body: String) -> String {
        crate::translate::chat::ChatError::reason(body)
    }

    async fn send(
        &self,
        token: &str,
        _slot: &Slot,
        _route: Route<'_>,
        req: &Self::Request,
    ) -> Result<Self::Response, SendError> {
        Self::post(self, token, req.path, &req.body).await
    }
}
