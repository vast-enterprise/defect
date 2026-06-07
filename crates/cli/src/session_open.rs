//! Logic for opening a local session — a neutral helper shared by the interactive REPL
//! (`repl`) and the single-shot (`oneshot`) paths.
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

/// Opens a session running directly on the local machine (both fs and shell use local
/// backends, frontend is `Cli`).
/// `resume = Some(id)` resumes that session; otherwise creates a new one.
///
/// # Errors
///
/// Returns an error if `load_session` / `create_session` fails (session does not exist,
/// duplicate id, cwd unavailable, etc.).
pub async fn open_session(
    agent: &Arc<dyn AgentCore>,
    cwd: &Path,
    resume: Option<SessionId>,
) -> anyhow::Result<Arc<dyn Session>> {
    let fs = Arc::new(LocalFsBackend::new(cwd.to_path_buf()));
    let shell = Arc::new(LocalShellBackend::new());
    match resume {
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
