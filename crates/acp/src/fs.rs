//! [`AcpFsBackend`]: delegates fs tool reads and writes to the ACP client.
//!
//! ACP reverse requests `fs/read_text_file` / `fs/write_text_file` are initiated by the
//! agent and handled by the client (clients with a workspace UI, such as zed / vscode,
//! use this delegation chain to keep unsaved buffers in sync with agent changes).
//!
//! ACP filesystem backend — delegates fs operations to the client via ACP.

use std::path::PathBuf;

use agent_client_protocol::Client;
use agent_client_protocol::ConnectionTo;
use agent_client_protocol::schema::{ReadTextFileRequest, SessionId, WriteTextFileRequest};
use defect_agent::error::BoxError;
use defect_agent::fs::{FsBackend, FsError, resolve_workspace_path};
use futures::future::BoxFuture;

/// ACP delegation-based filesystem backend.
///
/// Holds an ACP reverse channel [`ConnectionTo<Client>`], a session ID, and a workspace
/// root:
/// - `cx`: handle for sending requests to the client; it is an `Arc<...>` newtype, cheap
///   to clone
/// - `session_id`: included with every reverse request so the client can route it in
///   multi-session scenarios
/// - `workspace_root`: the agent independently guards the workspace boundary.
pub struct AcpFsBackend {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
    workspace_root: PathBuf,
}

impl AcpFsBackend {
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId, workspace_root: PathBuf) -> Self {
        Self {
            cx,
            session_id,
            workspace_root,
        }
    }
}

impl FsBackend for AcpFsBackend {
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>> {
        Box::pin(async move {
            // The agent enforces its own boundary; even if the client may re-enforce it,
            // do not rely on the client as a fallback.
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;

            let mut req = ReadTextFileRequest::new(self.session_id.clone(), abs);
            if let Some(l) = line {
                req = req.line(l);
            }
            if let Some(k) = limit {
                req = req.limit(k);
            }
            let resp = self
                .cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(resp.content)
        })
    }

    fn write_text(&self, path: PathBuf, content: String) -> BoxFuture<'_, Result<(), FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;
            let req = WriteTextFileRequest::new(self.session_id.clone(), abs, content);
            self.cx
                .send_request(req)
                .block_task()
                .await
                .map_err(map_wire_error)?;
            Ok(())
        })
    }
}

/// Maps a wire `Error` from the client into [`FsError::Backend`].
///
/// v0 does not subdivide by `code` (the ACP does not mandate specific error-code
/// semantics for `fs/*`). The wire `code` and `message` are forwarded as the source of
/// [`BoxError`] so that the LLM can inspect the original text in `tool_result` for
/// troubleshooting. This mapping will be expanded once the client stabilizes error codes
/// such as deny, quota, and read-only.
fn map_wire_error(err: agent_client_protocol::Error) -> FsError {
    FsError::Backend(BoxError::new(err))
}
