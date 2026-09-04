use axum::body::{Body, Bytes, to_bytes};
use axum::extract::Request;
use axum::http::header::{CONTENT_ENCODING, CONTENT_LENGTH};
use axum::middleware::Next;
use axum::response::Response;
use ruzstd::decoding::StreamingDecoder;

/// Codex zstd-encodes request bodies whenever it talks to the built-in
/// `openai` provider, and a turn's context is mostly repeated text, so the
/// wire form runs about a third of the JSON. Agent turns carrying a 200k
/// token context land well past axum's 2MB default either way.
pub const MAX_BODY: usize = 192 * 1024 * 1024;

pub async fn zstd_requests(req: Request, next: Next) -> Response {
   let encoded = req
      .headers()
      .get(CONTENT_ENCODING)
      .and_then(|value| value.to_str().ok())
      .is_some_and(|value| value.eq_ignore_ascii_case("zstd"));
   if !encoded {
      return next.run(req).await;
   }

   let (mut parts, body) = req.into_parts();
   let Ok(bytes) = to_bytes(body, MAX_BODY).await else {
      return super::error::error_response(
         super::error::Dialect::OpenAi,
         413,
         "invalid_request_error",
         "request body too large",
      );
   };
   let plain = match decode(&bytes) {
      Ok(plain) => plain,
      Err(DecodeError::TooLarge) => {
         tracing::warn!(
            "rejecting a body that unpacks past {MAX_BODY} bytes from {} compressed",
            bytes.len()
         );
         return super::error::error_response(
            super::error::Dialect::OpenAi,
            413,
            "invalid_request_error",
            "request body too large",
         );
      },
      Err(DecodeError::Malformed) => {
         return super::error::error_response(
            super::error::Dialect::OpenAi,
            400,
            "invalid_request_error",
            "malformed zstd request body",
         );
      },
   };

   parts.headers.remove(CONTENT_ENCODING);
   parts
      .headers
      .insert(CONTENT_LENGTH, plain.len().to_string().parse().unwrap());
   next
      .run(Request::from_parts(parts, Body::from(plain)))
      .await
}

enum DecodeError {
   TooLarge,
   Malformed,
}

/// Reads one byte past the ceiling so a body that would exceed it is refused
/// rather than truncated.
fn decode(bytes: &Bytes) -> Result<Vec<u8>, DecodeError> {
   use std::io::Read as _;
   let mut out = Vec::with_capacity(bytes.len() * 4);
   StreamingDecoder::new(&mut &**bytes)
      .map_err(|_| DecodeError::Malformed)?
      .take(MAX_BODY as u64 + 1)
      .read_to_end(&mut out)
      .map_err(|_| DecodeError::Malformed)?;
   if out.len() > MAX_BODY {
      return Err(DecodeError::TooLarge);
   }
   Ok(out)
}

#[cfg(test)]
mod tests {
   use std::ptr;

   use axum::Router;
   use axum::body::{Body, Bytes, to_bytes};
   use axum::extract::Request;
   use axum::middleware::from_fn;
   use axum::routing::post;

   #[tokio::test]
   async fn a_zstd_body_reaches_the_handler_as_json() {
      // `{"model":"m"}` compressed with `zstd -19`.
      let frame: &[u8] = &[
         0x28, 0xb5, 0x2f, 0xfd, 0x04, 0x58, 0x69, 0x00, 0x00, 0x7b, 0x22, 0x6d, 0x6f, 0x64, 0x65,
         0x6c, 0x22, 0x3a, 0x22, 0x6d, 0x22, 0x7d, 0x9c, 0x14, 0xb0, 0xa2,
      ];
      let app = Router::new()
         .route(
            "/",
            post(|body: String| async move { format!("got:{body}") }),
         )
         .layer(from_fn(super::zstd_requests));

      let req = Request::builder()
         .method("POST")
         .uri("/")
         .header("content-encoding", "zstd")
         .body(Body::from(frame.to_vec()))
         .unwrap();
      let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
      let out = to_bytes(resp.into_body(), 4096).await.unwrap();
      assert_eq!(out.as_ref(), b"got:{\"model\":\"m\"}");
   }

   /// A body over the ceiling has to be refused outright. Truncating it
   /// yields half a JSON document, which surfaces as a parse error naming
   /// the wrong cause.
   #[test]
   fn an_oversized_body_is_refused_rather_than_truncated() {
      let big = vec![b'a'; super::MAX_BODY + 4096];
      let framed = zstd_frame(&big);
      assert!(matches!(
         super::decode(&Bytes::from(framed)),
         Err(super::DecodeError::TooLarge)
      ));
   }

   /// zstd's raw-block form, so the test needs no compressor.
   fn zstd_frame(payload: &[u8]) -> Vec<u8> {
      let mut out = vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00];
      for chunk in payload.chunks(1 << 17) {
         let last = ptr::eq(chunk.as_ptr_range().end, payload.as_ptr_range().end);
         let header = ((chunk.len() as u32) << 3_u32) | u32::from(last);
         out.extend_from_slice(&header.to_le_bytes()[..3]);
         out.extend_from_slice(chunk);
      }
      out
   }
}
