//! Opens a session running directly on the local machine — a neutral helper shared by the
//! interactive REPL (`repl`) and the single-shot (`oneshot`) paths.
//!
//! Both frontend paths execute files and commands directly on the local machine: fs and
//! shell both use the local backend, and the frontend is marked [`Frontend::Cli`]. They
//! depend on this module at the same level, not on each other — this avoids making
//! oneshot depend on the repl module (which would tie oneshot to the `repl` feature).

use std::path::Path;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use defect_agent::session::{AgentCore, Frontend, Session, new_session_id};
use defect_tools::{LocalFsBackend, LocalShellBackend};

/// Knobs for opening a local CLI session — everything tied to the local backends or the
/// open/resume choice. Grouped into one struct so the frontend run functions
/// (`repl::run` / `oneshot::run`) forward a single value instead of growing a new
/// positional argument every time a local-backend setting becomes configurable.
#[derive(Debug, Clone)]
pub struct LocalSessionOpts {
    /// `Some(id)` resumes that session; `None` creates a fresh one.
    pub resume: Option<SessionId>,
    /// Captured-output cap (bytes) for the local shell backend, from
    /// `[tools.bash].output_max_bytes`.
    pub shell_output_max_bytes: usize,
}

/// Opens a session running directly on the local machine (both fs and shell use local
/// backends, frontend is [`Frontend::Cli`]).
///
/// # Errors
///
/// Returns an error if `load_session` / `create_session` fails (session does not exist,
/// duplicate id, cwd unavailable, etc.).
pub async fn open_local_session(
    agent: &Arc<dyn AgentCore>,
    cwd: &Path,
    opts: LocalSessionOpts,
) -> anyhow::Result<Arc<dyn Session>> {
    let fs = Arc::new(LocalFsBackend::new(cwd.to_path_buf()));
    let shell = Arc::new(LocalShellBackend::with_max_output_bytes(
        opts.shell_output_max_bytes,
    ));
    match opts.resume {
        Some(id) => agent
            .load_session(id, fs, shell, Frontend::Cli)
            .await
            .map_err(|e| anyhow::anyhow!("load_session failed: {e}")),
        None => {
            let session_id = SessionId::new(new_session_id());
            agent
                .create_session(
                    session_id,
                    cwd.to_path_buf(),
                    // Per-session MCP increment left empty: the configured `[mcp]` servers
                    // already live in the McpToolFactory's `default_servers`. This second
                    // channel exists for an ACP client to attach extra servers per
                    // `session/new`; the CLI is single-session with process-level config
                    // and has no per-session dimension to fill here.
                    Vec::new(),
                    fs,
                    shell,
                    Frontend::Cli,
                )
                .await
                .map_err(|e| anyhow::anyhow!("create_session failed: {e}"))
        }
    }
}
