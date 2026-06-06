//! 本机直跑 session 的开启逻辑——交互 REPL（`repl`）与单轮 oneshot
//! （`oneshot`）共用的中立 helper。
//!
//! 两条前端路径都在本机直接执行文件与命令：fs / shell 都用 local 后端，
//! frontend 标 [`Frontend::Cli`]。它们**平级依赖**本模块，互不依赖——避免
//! 让 oneshot 反过来依赖 repl 模块（那会把 oneshot 绑死在 `repl` feature 上）。

use std::path::Path;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use defect_agent::session::{AgentCore, Frontend, Session, new_session_id};
use defect_tools::{LocalFsBackend, LocalShellBackend};

/// 开一个本机直跑的 session（fs/shell 都用 local 后端，frontend 标 Cli）。
/// `resume = Some(id)` 恢复该 session，否则新建。
///
/// # Errors
///
/// `load_session` / `create_session` 失败（session 不存在、id 重复、cwd 不可用等）。
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
