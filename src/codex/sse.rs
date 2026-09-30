use std::pin::Pin;

use eventsource_stream::Eventsource as _;
use futures_util::{Stream, StreamExt as _};

use crate::codex::types::ResponsesEvent;

pub type EventStream = Pin<Box<dyn Stream<Item = ResponsesEvent> + Send>>;

pub fn event_stream(resp: reqwest::Response) -> EventStream {
   let stream = resp
      .bytes_stream()
      .eventsource()
      .filter_map(|event| async move {
         let event = event
            .inspect_err(|err| tracing::warn!("SSE stream error: {err}"))
            .ok()?;
         if event.data == "[DONE]" {
            return None;
         }
         Some(
            serde_json::from_str::<ResponsesEvent>(&event.data).unwrap_or_else(|err| {
               tracing::debug!("unparsed SSE event {:?}: {err}", event.event);
               ResponsesEvent::Other
            }),
         )
      });
   Box::pin(stream)
}
