use super::types::ApiError;

/// When Google gives up mid-answer it keeps the 200 and appends a bare JSON
/// error after the frames, `{"error":..}` natively and `[{"error":..}]` on the
/// `OpenAI` surface, where no `SSE` parser looks.
#[derive(Default)]
pub struct Frames {
    buf: Vec<u8>,
    trailer: Vec<u8>,
}

impl Frames {
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(bytes);
        let mut payloads = Vec::new();
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            match line.strip_prefix(b"data:") {
                Some(data) => payloads.push(data.strip_prefix(b" ").unwrap_or(data).to_vec()),
                None if line.is_empty() => {}
                None => {
                    self.trailer.extend_from_slice(line);
                    self.trailer.push(b'\n');
                }
            }
        }
        payloads
    }

    pub fn cutoff(&self) -> Option<ApiError> {
        let text = [self.trailer.as_slice(), self.buf.as_slice()].concat();
        if text.is_empty() {
            return None;
        }
        // A struct also deserializes from a sequence, so the list goes first.
        if let Ok(list) = serde_json::from_slice::<Vec<Envelope>>(&text) {
            return list.into_iter().next().map(|e| e.error);
        }
        serde_json::from_slice::<Envelope>(&text)
            .ok()
            .map(|e| e.error)
    }
}

#[derive(serde::Deserialize)]
struct Envelope {
    error: ApiError,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(bytes: &[u8]) -> (usize, Option<ApiError>) {
        let mut frames = Frames::default();
        let mut n = 0;
        for chunk in bytes.chunks(97) {
            n += frames.feed(chunk).len();
        }
        (n, frames.cutoff())
    }

    #[test]
    fn a_native_cutoff_is_seen_after_its_frames() {
        let (frames, cutoff) = run(include_bytes!("fixtures/native_cutoff.sse"));
        assert_eq!(frames, 5);
        assert_eq!(cutoff.unwrap().status.as_deref(), Some("UNAVAILABLE"));
    }

    #[test]
    fn the_openai_surface_wraps_its_cutoff_in_a_list() {
        let (frames, cutoff) = run(include_bytes!("fixtures/openai_cutoff.sse"));
        assert_eq!(frames, 3);
        assert_eq!(cutoff.unwrap().code, Some(503));
    }

    #[test]
    fn a_clean_stream_has_no_cutoff() {
        let (frames, cutoff) = run(b"data: {\"a\":1}\r\n\r\ndata: [DONE]\n\n");
        assert_eq!(frames, 2);
        assert!(cutoff.is_none());
    }
}
