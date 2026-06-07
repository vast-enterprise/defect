//! [`AcpShellBackend`] delegates shell execution of the `bash` tool to the ACP client.
//!
//! ACP reverse requests (`terminal/create`, `terminal/output`, `terminal/wait_for_exit`,
//! `terminal/release`, `terminal/kill`) are initiated by the agent and handled by the
//! client. In this delegation chain, clients with integrated terminal UIs (such as Zed or
//! VS Code) run commands in the client's PTY.
//!
//! ACP shell backend — delegates shell execution to the client via ACP.

use std::path::PathBuf;

use agent_client_protocol::Client;
use agent_client_protocol::ConnectionTo;
use agent_client_protocol::schema::{
    CreateTerminalRequest, KillTerminalRequest, ReleaseTerminalRequest, SessionId,
    TerminalId as AcpTerminalId, TerminalOutputRequest, WaitForTerminalExitRequest,
};
use defect_agent::error::BoxError;
use defect_agent::shell::{ShellBackend, ShellError, ShellOutput, TerminalExitStatus, TerminalId};
use futures::future::BoxFuture;

/// Delegation-pattern shell backend.
///
/// Holds an ACP reverse channel [`ConnectionTo<Client>`] + session id + workspace root:
/// - `cx`: handle for sending requests to the client; it is an `Arc<...>` newtype, cheap
///   to clone
/// - `session_id`: required on every reverse request; the client uses it to route in
///   multi-session scenarios
/// - `workspace_root`: the agent guards the workspace boundary independently, avoiding
///   reliance on the client as a fallback.
pub struct AcpShellBackend {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
    workspace_root: PathBuf,
}

impl AcpShellBackend {
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId, workspace_root: PathBuf) -> Self {
        Self {
            cx,
            session_id,
            workspace_root,
        }
    }

    fn acp_terminal_id(id: &TerminalId) -> AcpTerminalId {
        // The schema's `TerminalId` is an `Arc<str>` newtype; copy via the standard
        // library `From<&str> for Arc<str>`.
        AcpTerminalId::new(id.as_str())
    }
}

impl ShellBackend for AcpShellBackend {
    fn create(
        &self,
        command: String,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>> {
        Box::pin(async move {
            // Consistent with `fs`: the agent enforces a second boundary — the bash tool
            // layer already validates `workdir`, and this additional check makes the
            // backend's safety guarantees symmetric (see docs §5 "double fence").
            // Note: `fs`'s `resolve_workspace_path` is designed for a "target file +
            // parent directory" model (splitting out `file_name`), so it cannot be
            // directly applied to a directory `cwd`.
            let abs_cwd = resolve_workspace_dir(&self.workspace_root, &cwd)?;

            // v0: all shell commands go through `sh -c <command>`, matching
            // `LocalShellBackend` (see docs §2.1 "command + args separation").
            let req = CreateTerminalRequest::new(self.session_id.clone(), "/bin/sh")
                .args(vec!["-c".into(), command])
                .cwd(abs_cwd);
            let resp = self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(TerminalId::new(resp.terminal_id.0.to_string()))
        })
    }

    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>> {
        let acp_id = Self::acp_terminal_id(id);
        Box::pin(async move {
            let req = TerminalOutputRequest::new(self.session_id.clone(), acp_id);
            let resp = self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(ShellOutput {
                text: resp.output,
                truncated: resp.truncated,
                exit_status: resp.exit_status.map(map_acp_exit_status),
            })
        })
    }

    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>> {
        let acp_id = Self::acp_terminal_id(id);
        Box::pin(async move {
            let req = WaitForTerminalExitRequest::new(self.session_id.clone(), acp_id);
            let resp = self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(map_acp_exit_status(resp.exit_status))
        })
    }

    fn release(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let acp_id = Self::acp_terminal_id(id);
        Box::pin(async move {
            let req = ReleaseTerminalRequest::new(self.session_id.clone(), acp_id);
            self.cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(())
        })
    }

    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let acp_id = Self::acp_terminal_id(id);
        Box::pin(async move {
            let req = KillTerminalRequest::new(self.session_id.clone(), acp_id);
            self.cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(())
        })
    }
}

/// Maps ACP schema's `TerminalExitStatus` (`exit_code: Option<u32>`) to the agent's
/// internal [`TerminalExitStatus`] (`exit_code: Option<i32>`).
/// Standard exit codes are in 0..=255, which fits in `i32`; values exceeding `i32::MAX`
/// are degraded to -1.
fn map_acp_exit_status(s: agent_client_protocol::schema::TerminalExitStatus) -> TerminalExitStatus {
    TerminalExitStatus {
        exit_code: s.exit_code.map(|n| i32::try_from(n).unwrap_or(-1)),
        signal: s.signal,
    }
}

/// Maps a wire `Error` returned by the client into [`ShellError::Backend`]. Forwards the
/// wire `code` / `message` as the source of the [`BoxError`] so the LLM can access the
/// original text in `tool_result` for debugging. Same trade-off as
/// `acp::fs::map_wire_error`.
fn map_wire_error(err: agent_client_protocol::Error) -> ShellError {
    ShellError::Backend(BoxError::new(err))
}

/// Resolves the requested cwd to an absolute path within the workspace and validates that
/// it does not escape. Same approach as
/// `defect_agent::fs::resolve_workspace_path`, but since the target is already a
/// directory (cwd), there is no need to split off a file_name — so the entire target is
/// canonicalized and then checked with `starts_with` against the root.
fn resolve_workspace_dir(
    workspace_root: &std::path::Path,
    requested: &std::path::Path,
) -> Result<std::path::PathBuf, ShellError> {
    let target = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        workspace_root.join(requested)
    };

    let canon_target =
        std::fs::canonicalize(&target).map_err(|e| ShellError::Backend(BoxError::new(e)))?;
    let canon_root =
        std::fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());

    if !canon_target.starts_with(&canon_root) {
        return Err(ShellError::NotPermitted(format!(
            "workdir {} escapes workspace root {}",
            canon_target.display(),
            canon_root.display()
        )));
    }
    Ok(canon_target)
}
