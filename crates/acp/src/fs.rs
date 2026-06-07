//! [`AcpFsBackend`]：把 fs 工具的读写委托给 ACP 客户端。
//!
//! ACP 反向请求 `fs/read_text_file` / `fs/write_text_file` 由 agent 发起、
//! client 处理（zed / vscode 这类有 workspace UI 的客户端在这条委托链上
//! 让 unsaved buffer 与 agent 改动对齐）。
//!
//! ACP filesystem backend — delegates fs operations to the client via ACP.

use std::path::PathBuf;

use agent_client_protocol::Client;
use agent_client_protocol::ConnectionTo;
use agent_client_protocol::schema::{ReadTextFileRequest, SessionId, WriteTextFileRequest};
use defect_agent::error::BoxError;
use defect_agent::fs::{FsBackend, FsError, resolve_workspace_path};
use futures::future::BoxFuture;

/// 委托模式 fs 后端。
///
/// 持有 ACP 反向通道 [`ConnectionTo<Client>`] + session id + workspace root：
/// - `cx`：把请求送给客户端的句柄；本身是 `Arc<...>` newtype，clone 廉价
/// - `session_id`：每条反向请求都要带，客户端用它在多 session 场景里路由
/// - `workspace_root`：agent 自己守工作区边界（agent guards the workspace boundary independently）。
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
            // agent 自己守边界：即便客户端可能再 enforce 一遍，也不依赖客户端兜底。
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

/// 客户端返回的 wire `Error` → [`FsError::Backend`]。
///
/// v0 不按 `code` 细分（ACP 没硬性规定 `fs/*` 的错误码语义）；wire `code` /
/// `message` 透传到 [`BoxError`] 的 source，让 LLM 在 tool_result 里
/// 能拿到原文排障。等客户端实现收敛 deny / quota / read-only 等错误码后再扩。
fn map_wire_error(err: agent_client_protocol::Error) -> FsError {
    FsError::Backend(BoxError::new(err))
}
