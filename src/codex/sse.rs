use std::mem;
use std::pin::Pin;

use axum::body::Bytes;
use futures_util::{Stream, StreamExt as _, stream};

use crate::codex::types::ResponsesEvent;

pub type EventStream = Pin<Box<dyn Stream<Item = ResponsesEvent> + Send>>;

pub struct Event {
   pub event: String,
   pub data: String,
}

/// Splits a byte stream into server-sent events. An event cut off by the end
/// of the stream is dropped, as the spec has it.
pub fn events<S>(source: S) -> impl Stream<Item = Result<Event, reqwest::Error>>
where
   S: Stream<Item = reqwest::Result<Bytes>> + Unpin,
{
   let state = (source, Vec::<u8>::new(), String::new(), None::<String>);
   stream::unfold(
      state,
      |(mut bytes, mut buf, mut event, mut data)| async move {
         loop {
            while let Some(newline) = buf.iter().position(|&byte| byte == b'\n') {
               let line = buf.drain(..=newline).collect::<Vec<u8>>();
               let line = String::from_utf8_lossy(&line);
               let line = line.trim_end_matches(['\n', '\r']);
               if line.is_empty() {
                  if let Some(data) = data.take() {
                     let item = Event {
                        event: mem::take(&mut event),
                        data,
                     };
                     return Some((Ok(item), (bytes, buf, event, None)));
                  }
                  event.clear();
                  continue;
               }
               let (field, value) = line.split_once(':').unwrap_or((line, ""));
               let value = value.strip_prefix(' ').unwrap_or(value);
               match field {
                  "event" => value.clone_into(&mut event),
                  "data" => match data {
                     Some(ref mut joined) => {
                        joined.push('\n');
                        joined.push_str(value);
                     },
                     None => data = Some(value.to_owned()),
                  },
                  _ => {},
               }
            }
            match bytes.next().await? {
               Ok(chunk) => buf.extend_from_slice(&chunk),
               Err(err) => return Some((Err(err), (bytes, buf, event, data))),
            }
         }
      },
   )
}

pub fn event_stream(resp: reqwest::Response) -> EventStream {
   let stream = events(resp.bytes_stream()).filter_map(|event| async move {
      let event = event
         .inspect_err(|err| tracing::warn!("SSE stream error: {err}"))
         .ok()?;
      if event.data == "[DONE]" {
         return None;
      }
      Some(
         serde_json::from_slice::<ResponsesEvent>(event.data.as_bytes()).unwrap_or_else(|err| {
            tracing::debug!("unparsed SSE event {:?}: {err}", event.event);
            ResponsesEvent::Other
         }),
      )
   });
   Box::pin(stream)
}
