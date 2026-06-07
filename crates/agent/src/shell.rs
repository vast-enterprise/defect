//! Shell 执行后端抽象。
//!
//! [`ShellBackend`] 是 `bash` 工具与底层进程管理之间的 trait 边界。两个 v0
//! 实现：
//! - `defect_tools::shell::LocalShellBackend`：直接 spawn 子进程
//! - `defect_acp::shell::AcpShellBackend`：走 ACP `terminal/*` 反向请求
//!   委托给客户端
//!
//! 装配权在 `defect-acp` 的 `session/new` handler——按客户端的
//! [`ClientCapabilities::terminal`] 协商结果选择后端，注入给
//! [`crate::session::AgentCore::create_session`]。
//!

//!
//! [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities

use std::path::PathBuf;

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// terminal 句柄。在 backend 内部映射到 PID + 单调计数器（local）或 ACP
/// schema 的 `TerminalId`（acp）。
///
/// 用 newtype 而非裸 `String`：调用方在 trait 边界上看到的就是"terminal 句柄"，
/// 不会与普通字符串混淆。
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

/// 一次 [`ShellBackend::output`] 的快照结果。
#[derive(Debug, Clone)]
pub struct ShellOutput {
    /// 截至本次调用累积的合并 stdout/stderr 文本。后端保证 UTF-8 合法性。
    pub text: String,
    /// 输出是否被后端按字节上限截断。
    pub truncated: bool,
    /// 进程已退出时填实际退出态；仍在跑则为 `None`。
    pub exit_status: Option<TerminalExitStatus>,
}

/// terminal 进程的退出态。
#[derive(Debug, Clone)]
pub struct TerminalExitStatus {
    /// 进程 exit code。被信号杀掉时为 `None`，看 `signal`。
    ///
    /// 内部用 `i32` 与 `BashOutput.exit_code` 一致。`AcpShellBackend` 收到
    /// schema 的 `Option<u32>` 时用 `i32::try_from`，超过 `i32::MAX` 退化为
    /// `-1`（实际 exit code 域是 0..=255，不会越界）。
    pub exit_code: Option<i32>,
    /// 信号名（如 `SIGKILL`）。本地后端来自 `signal_name(sig)`；ACP 后端透传
    /// schema 的 `signal: Option<String>`。
    pub signal: Option<String>,
}

/// shell 后端 trait。
///
/// v0 语义：每条命令一个独立 terminal——`create` → 跑 → `wait_for_exit`
/// 拿退出态 → `output` 拿全量输出 → `release` 释放资源。不暴露"持久 terminal
/// 跨 turn 复用"——交互式 terminal 工具是后续演进。
///
/// 入参用 owned `String` / `PathBuf`：把 future 的生命周期收敛到 `&'_ self`，
/// 避免显式生命周期参数；与 [`crate::fs::FsBackend`] 同款取舍。
pub trait ShellBackend: Send + Sync {
    /// 创建 terminal 并启动命令。
    ///
    /// `command` 是一整行 shell 命令（v0 由后端用 `sh -c` 跑）。`cwd` 必须是
    /// 已校验在工作区内的绝对路径——agent 工具层负责守边界，backend 不再做
    /// 业务校验。
    fn create(
        &self,
        command: String,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>>;

    /// 取 terminal 当前累积输出的快照。
    ///
    /// **幂等可重复调用**——后端不在此处 drain 缓冲。`exit_status = Some(_)`
    /// 表示进程已退出，但 `output` 本身不阻塞等待退出（要阻塞等用
    /// [`ShellBackend::wait_for_exit`]）。
    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>>;

    /// 阻塞等待 terminal 进程退出。
    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>>;

    /// 释放 terminal 资源（关闭 fd / 移除内部记录）。
    ///
    /// 幂等：重复 release 同一个 `id` 不返回错误（已被释放时静默成功）。
    fn release(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>>;

    /// 强制终止 terminal 进程。**不**释放资源——后续仍可调
    /// [`ShellBackend::output`] / [`ShellBackend::wait_for_exit`]，
    /// 释放由 [`ShellBackend::release`] 负责。
    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>>;
}

/// shell 后端错误。
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ShellError {
    /// 引用了不存在 / 已释放的 terminal_id。
    #[error("terminal not found: {0:?}")]
    NotFound(TerminalId),

    /// 后端 spawn 子进程 / 与客户端通信等失败。
    #[error("shell backend failure: {0}")]
    Backend(#[source] BoxError),

    /// 操作被拒：cwd 越界 / 客户端 deny / 权限不足等。
    #[error("operation not permitted: {0}")]
    NotPermitted(String),
}

/// 仅用于测试的 no-op shell 后端。所有方法返回 [`ShellError::NotPermitted`]，
/// 让需要 `Arc<dyn ShellBackend>` 的测试场景（不实际跑 shell 工具）能跳过装配。
///
/// 真实运行时用 `defect_tools::shell::LocalShellBackend` 或
/// `defect_acp::shell::AcpShellBackend`。
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
        // 释放语义是幂等的——no-op 后端从不持有资源，直接成功。
        Box::pin(async { Ok(()) })
    }

    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let id = id.clone();
        Box::pin(async move { Err(ShellError::NotFound(id)) })
    }
}
