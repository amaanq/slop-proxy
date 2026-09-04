use std::pin::Pin;

use eventsource_stream::Eventsource as _;
use futures_util::{Stream, StreamExt as _};

use super::types::ResponsesEvent;

pub type EventStream = Pin<Box<dyn Stream<Item = ResponsesEvent> + Send>>;

pub fn event_stream(resp: reqwest::Response) -> EventStream {
    let stream = resp
        .bytes_stream()
        .eventsource()
        .filter_map(|ev| async move {
            match ev {
                Ok(ev) => {
                    if ev.data == "[DONE]" {
                        return None;
                    }
                    match serde_json::from_str::<ResponsesEvent>(&ev.data) {
                        Ok(parsed) => Some(parsed),
                        Err(e) => {
                            tracing::debug!("unparsed SSE event {:?}: {e}", ev.event);
                            Some(ResponsesEvent::Other)
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("SSE stream error: {e}");
                    None
                }
            }
        });
    Box::pin(stream)
}
