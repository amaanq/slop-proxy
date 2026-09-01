use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::header::{CONTENT_ENCODING, CONTENT_LENGTH};
use axum::middleware::Next;
use axum::response::Response;

/// Codex zstd-encodes request bodies whenever it talks to the built-in
/// `openai` provider, and a turn's context is mostly repeated text, so the
/// wire form runs about a third of the JSON. Agent turns carrying a 200k
/// token context land well past axum's 2MB default either way.
pub const MAX_BODY: usize = 64 * 1024 * 1024;

pub async fn zstd_requests(req: Request, next: Next) -> Response {
    let encoded = req
        .headers()
        .get(CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("zstd"));
    if !encoded {
        return next.run(req).await;
    }

    let (mut parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MAX_BODY).await else {
        return super::error::error_response(
            super::error::Dialect::OpenAi,
            413,
            "invalid_request_error",
            "request body too large",
        );
    };
    let Some(plain) = decode(&bytes) else {
        return super::error::error_response(
            super::error::Dialect::OpenAi,
            400,
            "invalid_request_error",
            "malformed zstd request body",
        );
    };

    parts.headers.remove(CONTENT_ENCODING);
    parts
        .headers
        .insert(CONTENT_LENGTH, plain.len().to_string().parse().unwrap());
    next.run(Request::from_parts(parts, Body::from(plain)))
        .await
}

fn decode(bytes: &Bytes) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::with_capacity(bytes.len() * 4);
    ruzstd::decoding::StreamingDecoder::new(&mut &bytes[..])
        .ok()?
        .take(MAX_BODY as u64)
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::routing::post;

    #[tokio::test]
    async fn a_zstd_body_reaches_the_handler_as_json() {
        // `{"model":"m"}` compressed with `zstd -19`.
        let frame: &[u8] = &[
            0x28, 0xb5, 0x2f, 0xfd, 0x04, 0x58, 0x69, 0x00, 0x00, 0x7b, 0x22, 0x6d, 0x6f, 0x64,
            0x65, 0x6c, 0x22, 0x3a, 0x22, 0x6d, 0x22, 0x7d, 0x9c, 0x14, 0xb0, 0xa2,
        ];
        let app = Router::new()
            .route(
                "/",
                post(|body: String| async move { format!("got:{body}") }),
            )
            .layer(axum::middleware::from_fn(super::zstd_requests));

        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/")
            .header("content-encoding", "zstd")
            .body(axum::body::Body::from(frame.to_vec()))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), 200);
        let out = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        assert_eq!(&out[..], b"got:{\"model\":\"m\"}");
    }

    #[tokio::test]
    async fn an_unencoded_body_is_untouched() {
        let app = Router::new()
            .route(
                "/",
                post(|body: String| async move { format!("got:{body}") }),
            )
            .layer(axum::middleware::from_fn(super::zstd_requests));
        let req = axum::extract::Request::builder()
            .method("POST")
            .uri("/")
            .body(axum::body::Body::from("{\"model\":\"m\"}"))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        let out = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        assert_eq!(&out[..], b"got:{\"model\":\"m\"}");
    }
}
