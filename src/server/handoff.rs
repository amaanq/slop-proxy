//! Listening sockets that outlive the process serving them, passed by the
//! `LISTEN_FDS` protocol from systemd or from a predecessor on reload.

use std::collections::HashMap;
use std::convert::Infallible;
use std::env;
use std::fmt;
use std::fs;
use std::io;
use std::iter;
use std::net::TcpListener as StdListener;
use std::os::fd::{AsFd as _, AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};
use std::os::linux::net::SocketAddrExt as _;
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::path::{Path, PathBuf};
use std::process;
use std::str::FromStr;
use std::time::Duration;

use misstep::{Result, ResultExt as _, bail};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::process::Command;
use tokio::sync::watch;
use tokio::time;

const LISTEN_FDS_START: RawFd = 3;
const PREDECESSOR: &str = "SLOP_PROXY_PREDECESSOR";
const READY_LIMIT: Duration = Duration::from_mins(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Socket {
   Api,
   Metrics,
}

impl fmt::Display for Socket {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str(match *self {
         Self::Api => "api",
         Self::Metrics => "metrics",
      })
   }
}

impl FromStr for Socket {
   type Err = ();

   fn from_str(name: &str) -> Result<Self, ()> {
      match name {
         "api" => Ok(Self::Api),
         "metrics" => Ok(Self::Metrics),
         _ => Err(()),
      }
   }
}

/// Sockets passed in by systemd or a predecessor.
pub struct Inherited(HashMap<Socket, StdListener>);

impl Inherited {
   pub fn take() -> Result<Self> {
      let ours = env::var("LISTEN_PID").map_or(true, |pid| pid.parse::<u32>() == Ok(process::id()));
      let count = env::var("LISTEN_FDS")
         .ok()
         .filter(|_| ours)
         .and_then(|count| count.parse::<RawFd>().ok())
         .unwrap_or_default();
      let names = env::var("LISTEN_FDNAMES").unwrap_or_default();
      let mut sockets = HashMap::new();
      for (descriptor, name) in (LISTEN_FDS_START..LISTEN_FDS_START + count).zip(names.split(':')) {
         // SAFETY: the LISTEN_FDS protocol hands this process sole ownership
         // of every descriptor from 3 up to the count it names.
         let listener = unsafe { StdListener::from_raw_fd(descriptor) };
         let Ok(socket) = name.parse::<Socket>() else {
            tracing::warn!("ignoring inherited socket named {name:?}");
            continue;
         };
         listener.set_nonblocking(true)?;
         sockets.insert(socket, listener);
      }
      Ok(Self(sockets))
   }

   pub async fn listen(&mut self, socket: Socket, addr: &str) -> Result<TcpListener> {
      if let Some(listener) = self.0.remove(&socket) {
         tracing::info!("serving {socket} on an inherited socket");
         return Ok(TcpListener::from_std(listener)?);
      }
      TcpListener::bind(addr)
         .await
         .with_context(|| format!("binding {socket} listener {addr}"))
   }
}

/// Tells systemd this process is the unit's main pid and ready, which also
/// completes a handoff.
pub fn notify_ready() {
   let Some(path) = env::var_os("NOTIFY_SOCKET") else {
      return;
   };
   let path = path.to_string_lossy();
   let addr = path.strip_prefix('@').map_or_else(
      || SocketAddr::from_pathname(path.as_ref()),
      SocketAddr::from_abstract_name,
   );
   let state = format!("MAINPID={}\nREADY=1", process::id());
   let sent = addr.and_then(|addr| UnixDatagram::unbound()?.send_to_addr(state.as_bytes(), &addr));
   if let Err(err) = sent {
      tracing::warn!("notifying systemd: {err}");
   }
}

/// The process that started this one on its own sockets.
pub struct Predecessor(libc::pid_t);

impl Predecessor {
   pub fn from_env() -> Option<Self> {
      env::var(PREDECESSOR).ok()?.parse().ok().map(Self)
   }

   /// Lets it stop accepting and drain.
   pub fn relieve(self) {
      // SAFETY: kill has no memory effects.
      if unsafe { libc::kill(self.0, libc::SIGTERM) } != 0_i32 {
         tracing::warn!(
            "signalling predecessor {}: {}",
            self.0,
            io::Error::last_os_error()
         );
      }
   }
}

/// The command a running server starts as its replacement. Travels as its
/// arguments joined by NUL.
pub struct Successor(Vec<String>);

impl Successor {
   pub fn new<Args>(program: &Path, args: Args) -> Self
   where
      Args: IntoIterator<Item = String>,
   {
      Self(
         iter::once(program.to_string_lossy().into_owned())
            .chain(args)
            .collect(),
      )
   }

   /// Run as `ExecReload`, after the unit has been reloaded, so this is the
   /// new binary with the new arguments.
   pub async fn request(&self) -> Result<()> {
      let Some(path) = control_path() else {
         bail!("RUNTIME_DIRECTORY is unset, nothing to hand off from");
      };
      let mut stream = UnixStream::connect(&path)
         .await
         .with_context(|| format!("connecting to {}", path.display()))?;
      stream.write_all(self.to_string().as_bytes()).await?;
      stream.shutdown().await?;
      let mut reply = String::new();
      stream.read_to_string(&mut reply).await?;
      if reply != "ok" {
         bail!("{reply}");
      }
      Ok(())
   }

   fn command(&self, sockets: &[(Socket, OwnedFd)]) -> Result<Command> {
      let Some((program, args)) = self.0.split_first() else {
         bail!("empty handoff request");
      };
      let names = sockets
         .iter()
         .map(|&(socket, _)| socket.to_string())
         .collect::<Vec<_>>()
         .join(":");
      let descriptors = sockets
         .iter()
         .map(|&(_, ref socket)| socket.as_raw_fd())
         .collect::<Vec<_>>();
      let mut command = Command::new(program);
      command
         .args(args)
         .env("LISTEN_FDS", descriptors.len().to_string())
         .env("LISTEN_FDNAMES", names)
         .env(PREDECESSOR, process::id().to_string())
         .env_remove("LISTEN_PID");
      // SAFETY: the hook runs between fork and exec and calls only fcntl,
      // dup2 and close, which are async-signal-safe.
      unsafe {
         command.pre_exec(move || place_sockets(&descriptors));
      }
      Ok(command)
   }
}

impl fmt::Display for Successor {
   fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str(&self.0.join("\0"))
   }
}

impl FromStr for Successor {
   type Err = Infallible;

   fn from_str(request: &str) -> Result<Self, Infallible> {
      Ok(Self(request.split('\0').map(str::to_owned).collect()))
   }
}

fn control_path() -> Option<PathBuf> {
   env::var_os("RUNTIME_DIRECTORY").map(|dir| PathBuf::from(dir).join("control"))
}

/// Accepts handoff requests until one succeeds.
pub fn accept(listeners: &[(Socket, &TcpListener)], drain: watch::Receiver<bool>) -> Result<()> {
   let Some(path) = control_path() else {
      return Ok(());
   };
   let sockets = listeners
      .iter()
      .map(|&(socket, listener)| Ok((socket, listener.as_fd().try_clone_to_owned()?)))
      .collect::<io::Result<Vec<_>>>()?;
   let _ = fs::remove_file(&path);
   let control = UnixListener::bind(&path)
      .with_context(|| format!("binding handoff socket {}", path.display()))?;
   tokio::spawn(async move {
      loop {
         let Ok((mut stream, _)) = control.accept().await else {
            continue;
         };
         match hand_off(&mut stream, &sockets, drain.clone()).await {
            Ok(()) => {
               let _ = stream.write_all(b"ok").await;
               return;
            },
            Err(err) => {
               tracing::error!("handoff failed: {err:#}");
               let _ = stream.write_all(format!("{err:#}").as_bytes()).await;
            },
         }
      }
   });
   Ok(())
}

async fn hand_off(
   stream: &mut UnixStream,
   sockets: &[(Socket, OwnedFd)],
   mut drain: watch::Receiver<bool>,
) -> Result<()> {
   let mut request = String::new();
   stream.read_to_string(&mut request).await?;
   let Ok(successor) = request.parse::<Successor>();
   tracing::info!("handing off to {}", successor.0[0]);
   let mut child = successor
      .command(sockets)?
      .spawn()
      .context("starting successor")?;

   tokio::select! {
      status = child.wait() => bail!("successor exited before it was ready: {}", status?),
      () = async {
         let _ = drain.wait_for(|&draining| draining).await;
      } => Ok(()),
      () = time::sleep(READY_LIMIT) => {
         let _ = child.kill().await;
         bail!("successor was not ready within {}s", READY_LIMIT.as_secs())
      },
   }
}

/// Moves the sockets to 3, 4, ... in the child. Copying them above that
/// range first keeps one from landing on another before it has moved.
fn place_sockets(descriptors: &[RawFd]) -> io::Result<()> {
   let floor =
      RawFd::try_from(descriptors.len()).map_or(RawFd::MAX, |count| LISTEN_FDS_START + count);
   let mut copies = Vec::with_capacity(descriptors.len());
   for &descriptor in descriptors {
      // SAFETY: fcntl on a descriptor this process owns.
      let copy = unsafe { libc::fcntl(descriptor, libc::F_DUPFD, floor) };
      if copy < 0_i32 {
         return Err(io::Error::last_os_error());
      }
      copies.push(copy);
   }
   for (target, copy) in (LISTEN_FDS_START..).zip(copies) {
      // SAFETY: dup2 on descriptors this process owns. It leaves the target
      // without close-on-exec, which is what lets it survive.
      if unsafe { libc::dup2(copy, target) } < 0_i32 {
         return Err(io::Error::last_os_error());
      }
      // SAFETY: the copy is ours and nothing else refers to it.
      unsafe {
         libc::close(copy);
      }
   }
   Ok(())
}
