//! Shell execution backend abstraction.
//!
//! [`ShellBackend`] is the trait boundary between the `bash` tool and the underlying
//! process management. Two v0 implementations:
//! - `defect_tools::shell::LocalShellBackend`: spawns child processes directly
//! - `defect_acp::shell::AcpShellBackend`: delegates to the client via ACP `terminal/*`
//!   reverse requests
//!
//! Assembly is handled in the `defect-acp` `session/new` handler — it selects the backend
//! based on the client's [`ClientCapabilities::terminal`] negotiation result and injects
//! it into [`crate::session::AgentCore::create_session`].

//! [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities

use std::path::PathBuf;

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// A terminal handle. Internally, in the backend, it maps to a PID + monotonic counter
/// (local) or an ACP schema's `TerminalId` (acp).
///
/// A newtype rather than a bare `String`: callers see a "terminal handle" at trait
/// boundaries, not a plain string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TerminalId(String);

impl TerminalId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<TerminalId> for String {
    fn from(value: TerminalId) -> Self {
        value.0
    }
}

/// A snapshot result of [`ShellBackend::output`].
#[derive(Debug, Clone)]
pub struct ShellOutput {
    /// Accumulated combined stdout/stderr text up to this call. The backend guarantees
    /// valid UTF-8.
    pub text: String,
    /// Whether the output was truncated by the backend due to a byte limit.
    pub truncated: bool,
    /// Set to the actual exit status when the process has exited, or `None` if it is
    /// still running.
    pub exit_status: Option<TerminalExitStatus>,
}

/// Exit status of a terminal process.
#[derive(Debug, Clone)]
pub struct TerminalExitStatus {
    /// Exit code of the process. `None` if killed by a signal; see `signal`.
    ///
    /// Internally uses `i32` to match `BashOutput.exit_code`. When `AcpShellBackend`
    /// receives
    /// `Option<u32>` from the schema, it uses `i32::try_from`; values exceeding
    /// `i32::MAX` degrade to
    /// `-1` (the actual exit code range is 0..=255, so this never overflows).
    pub exit_code: Option<i32>,
    /// Signal name (e.g. `SIGKILL`). The local backend obtains it from
    /// `signal_name(sig)`; the ACP backend passes through the schema's `signal:
    /// Option<String>`.
    pub signal: Option<String>,
}

/// Shell backend trait.
///
/// v0 semantics: each command gets an independent terminal — `create` → run →
/// `wait_for_exit` for the exit status → `output` for the full output → `release` to free
/// resources. Persistent terminals reused across turns are not exposed; interactive
/// terminal tooling is left for future evolution.
///
/// Parameters use owned `String` / `PathBuf` to confine the future's lifetime to `&'_
/// self`, avoiding explicit lifetime parameters — the same trade-off as
/// [`crate::fs::FsBackend`].
pub trait ShellBackend: Send + Sync {
    /// Creates a terminal and starts the command.
    ///
    /// `command` is a full shell command line (v0 runs it via `sh -c` on the backend).
    /// `cwd` must be an absolute path already validated to be inside the workspace — the
    /// agent tool layer enforces this boundary; the backend does not perform business
    /// validation.
    fn create(
        &self,
        command: String,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>>;

    /// Take a snapshot of the terminal's current accumulated output.
    ///
    /// **Idempotent and safe to call repeatedly** — the backend does not drain the buffer
    /// here. `exit_status = Some(_)` indicates the process has exited, but `output`
    /// itself does not block waiting for exit (use [`ShellBackend::wait_for_exit`] for
    /// blocking).
    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>>;

    /// Blocks until the terminal process exits.
    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>>;

    /// Release terminal resources (close file descriptors / remove internal bookkeeping).
    ///
    /// Idempotent: releasing the same `id` multiple times does not return an error
    /// (silently succeeds if already released).
    fn release(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>>;

    /// Forcefully kill the terminal process. Does **not** release resources — subsequent
    /// calls to [`ShellBackend::output`] / [`ShellBackend::wait_for_exit`] are still
    /// valid; releasing is handled by [`ShellBackend::release`].
    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>>;
}

/// Errors from the shell backend.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ShellError {
    /// The terminal ID refers to a non-existent or already-released terminal.
    #[error("terminal not found: {0:?}")]
    NotFound(TerminalId),

    /// Backend failed to spawn a child process or communicate with the client.
    #[error("shell backend failure: {0}")]
    Backend(#[source] BoxError),

    /// Operation not permitted: cwd out of bounds, client denied, insufficient
    /// permissions, etc.
    #[error("operation not permitted: {0}")]
    NotPermitted(String),
}

/// A no-op shell backend for testing only. All methods return
/// [`ShellError::NotPermitted`],
/// allowing test scenarios that require an `Arc<dyn ShellBackend>` (without actually
/// running
/// a shell tool) to skip setup.
///
/// For real use, use `defect_tools::shell::LocalShellBackend` or
/// `defect_acp::shell::AcpShellBackend`.
pub struct NoopShellBackend;

impl ShellBackend for NoopShellBackend {
    fn create(
        &self,
        _command: String,
        _cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>> {
        Box::pin(async {
            Err(ShellError::NotPermitted(
                "NoopShellBackend cannot spawn".to_string(),
            ))
        })
    }

    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>> {
        let id = id.clone();
        Box::pin(async move { Err(ShellError::NotFound(id)) })
    }

    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>> {
        let id = id.clone();
        Box::pin(async move { Err(ShellError::NotFound(id)) })
    }

    fn release(&self, _id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        // Release is idempotent — the no-op backend never holds resources, so it always
        // succeeds.
        Box::pin(async { Ok(()) })
    }

    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let id = id.clone();
        Box::pin(async move { Err(ShellError::NotFound(id)) })
    }
}
