use std::pin::Pin;

use eventsource_stream::Eventsource as _;
use futures_util::{Stream, StreamExt as _};

use super::types::ResponsesEvent;

pub type EventStream = Pin<Box<dyn Stream<Item = ResponsesEvent> + Send>>;

pub fn event_stream(resp: reqwest::Response) -> EventStream {
   let stream = resp
      .bytes_stream()
      .eventsource()
      .filter_map(|event| async move {
         match event {
            Ok(event) => {
               if event.data == "[DONE]" {
                  return None;
               }
               match serde_json::from_str::<ResponsesEvent>(&event.data) {
                  Ok(parsed) => Some(parsed),
                  Err(err) => {
                     tracing::debug!("unparsed SSE event {:?}: {err}", event.event);
                     Some(ResponsesEvent::Other)
                  },
               }
            },
            Err(err) => {
               tracing::warn!("SSE stream error: {err}");
               None
            },
         }
      });
   Box::pin(stream)
}
