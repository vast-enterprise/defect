//! [`AcpShellBackend`]：把 `bash` 工具的 shell 执行委托给 ACP 客户端。
//!
//! ACP 反向请求 `terminal/create` / `terminal/output` / `terminal/wait_for_exit`
//! / `terminal/release` / `terminal/kill` 由 agent 发起、client 处理（zed /
//! vscode 这类有集成终端 UI 的客户端在这条委托链上让命令在客户端 PTY 里跑）。
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

/// 委托模式 shell 后端。
///
/// 持有 ACP 反向通道 [`ConnectionTo<Client>`] + session id + workspace root：
/// - `cx`：把请求送给客户端的句柄；本身是 `Arc<...>` newtype，clone 廉价
/// - `session_id`：每条反向请求都要带，客户端用它在多 session 场景里路由
/// - `workspace_root`：agent 自己守工作区边界，避免依赖客户端兜底
///   （agent guards the workspace boundary independently）。
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
        // schema 的 TerminalId 是 Arc<str> newtype；从 &str 走标准库
        // `From<&str> for Arc<str>` 拷贝一份。
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
            // 与 fs 一致：agent 自己再守一道边界——bash 工具层已校验过 workdir，
            // 这里再校验一次让 backend 的安全保证对称（详见 docs §5 "双层栅栏"）。
            // 注意：fs 的 `resolve_workspace_path` 是按"目标文件 + 父目录"模型
            // 设计的（split 出 file_name），不能直接套到 directory cwd 上。
            let abs_cwd = resolve_workspace_dir(&self.workspace_root, &cwd)?;

            // v0：所有 shell 命令都走 `sh -c <command>`，与 LocalShellBackend
            // 对齐（详见 docs §2.1 "command + args 分离"）。
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

/// ACP schema 的 `TerminalExitStatus`（`exit_code: Option<u32>`）→ agent
/// 内部的 [`TerminalExitStatus`]（`exit_code: Option<i32>`）。
/// exit code 标准值域 0..=255，i32 域
/// 足够装下；超过 i32::MAX 的退化为 -1。
fn map_acp_exit_status(s: agent_client_protocol::schema::TerminalExitStatus) -> TerminalExitStatus {
    TerminalExitStatus {
        exit_code: s.exit_code.map(|n| i32::try_from(n).unwrap_or(-1)),
        signal: s.signal,
    }
}

/// 客户端返回的 wire `Error` → [`ShellError::Backend`]。透传 wire `code` /
/// `message` 到 [`BoxError`] 的 source，让 LLM 在 tool_result 里能拿到原文
/// 排障。与 `acp::fs::map_wire_error` 同款取舍。
fn map_wire_error(err: agent_client_protocol::Error) -> ShellError {
    ShellError::Backend(BoxError::new(err))
}

/// 把请求 cwd 解析到工作区内的绝对目录路径，并校验未越界。与
/// `defect_agent::fs::resolve_workspace_path` 同款思路，但目标本身就是
/// directory（cwd），不需要再 split 出 file_name——因此整个目标 canonicalize，
/// 再 starts_with 检查根。
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
