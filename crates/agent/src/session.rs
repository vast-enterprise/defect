//! Session：单个对话会话的状态容器与生命周期接口。
//!
//! 设计详见 `docs/internal/session.md`。
//!
//! ## 抽象层次
//!
//! - [`AgentCore`]：进程级"agent 实例"，持有内置工具集与全局配置；
//!   是 `defect-cli` 装配出来后注入给 `defect-acp::serve` 的根对象
//! - [`Session`]：单次对话的生命周期单元；持有历史、per-session 工具
//!   表（含 MCP）、cancel token、事件流
//! - [`History`]：消息历史的封装，预留压缩 / token 计数 / resume 钩子
//!
//! 三者**全部以 trait 暴露**，具体实现在 crate 内的 `session/` 子模块
//! 与 `defect-cli` 的装配点完成；`defect-acp` 只通过 trait 与之打交道。

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use futures::future::BoxFuture;

use crate::error::BoxError;
use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::llm::{Message, ProviderError};
use crate::tool::{Tool, ToolSchema};

mod default;
mod events;
mod history;
mod permissions;
mod tool_registry;
mod turn;

pub use default::{DefaultAgentCore, DefaultAgentCoreBuilder, DefaultSession, uuid_like};
pub use events::EventEmitter;
pub use history::VecHistory;
pub use permissions::PermissionGate;
pub use tool_registry::{CompositeRegistry, StaticToolRegistry, StaticToolRegistryBuilder};
pub use turn::{TurnConfig, TurnRequestLimit, TurnRunner};

/// 进程级 agent 根对象。
///
/// `defect-cli` 在启动时构造一个具体实现（持有 LLM provider 注册表、
/// 内置工具集、配置），把 `Arc<dyn AgentCore>` 注入给 `defect-acp::serve`。
///
/// 抽 trait 的考量：
/// - 测试时可注入 mock，不必拉起真实 LLM
/// - 未来出现"嵌入式 agent"（lib 模式被宿主应用调用）等形态时，
///   可有第二个具体实现，不动 acp 桥接代码
pub trait AgentCore: Send + Sync {
    /// 创建一个新 session。
    ///
    /// `id` 由调用方（`defect-acp` 的 `session/new` handler）生成并传入——
    /// fs 后端在 [`AgentCore::create_session`] 之外构造时已经需要 SessionId
    /// 了（见 `docs/inbound/acp-fs.md` §3.2）。具体实现把它当作外部权威 id
    /// 用，重复时返回 [`AgentError::DuplicateSessionId`]。
    ///
    /// `mcp_servers` 是 `session/new` 请求里携带的 per-session MCP server
    /// 列表；具体实现在初始化阶段拉起子进程 / 建立 SSE 连接，把每个 MCP
    /// 工具包装成 [`Tool`] 加入会话工具表。
    ///
    /// `fs` 是 session 级文件系统后端——`defect-acp` 装配时按客户端的
    /// [`FileSystemCapabilities`] 选择 `LocalFsBackend` 或 `AcpFsBackend`。
    /// session 持有它的 `Arc`，所有 fs 工具调用都走它。
    ///
    /// # Errors
    ///
    /// MCP 启动失败、cwd 不存在、id 重复等。
    ///
    /// [`FileSystemCapabilities`]: agent_client_protocol::schema::FileSystemCapabilities
    fn create_session(
        &self,
        id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    /// 按 id 查找已存在的 session。
    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>>;
}

/// 单次会话。
///
/// 所有方法都是 trait 对象友好（`&self` + `BoxFuture`）。`Arc<dyn Session>`
/// 在 `defect-acp` 与主循环之间共享。
pub trait Session: Send + Sync {
    fn id(&self) -> &SessionId;

    /// 订阅事件流。三个独立消费者（acp / storage / tracing）各自调一次，
    /// 互不影响——内部用 mpsc 配 fan-out 保证慢消费者只 backpressure
    /// 自己、不丢事件。具体技术细节见 `docs/internal/session.md` §5。
    fn subscribe(&self) -> EventStream;

    /// 启动一次 turn。
    ///
    /// 返回的 future 在 turn 结束时 resolve：
    /// - `Ok(StopReason)`：正常结束（含 Cancelled），驱动 ACP 的
    ///   `PromptResponse`
    /// - `Err(TurnError)`：fatal 错误（鉴权过期 / 模型不可用等），
    ///   驱动 ACP 的 JSON-RPC `Error` 返回
    ///
    /// 期间产生的 [`AgentEvent`] 通过 [`Session::subscribe`] 推送，
    /// **不**通过此 future。`TurnEnded` 事件仍然在事件流上发（给
    /// storage / tracing），但 ACP 桥接以本 future 的 outcome 为准。
    ///
    /// 同一 session 同时只能有一个进行中的 turn；并发调用返回
    /// [`TurnError::TurnInProgress`]。
    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>>;

    /// 取消当前 turn。幂等：没有 turn 在跑时是 no-op。
    fn cancel_turn(&self);

    /// 把 ACP 反向 request `session/request_permission` 的客户端响应
    /// 回写给主循环。
    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution);
}

/// 事件流。类型擦除以支持 trait 对象返回。
pub type EventStream = futures::stream::BoxStream<'static, AgentEvent>;

/// 消息历史的抽象。
///
/// v0 实现仅是 `Vec<Message>` + `Mutex` 的最小封装，但 trait 留出
/// 后续上压缩 / token 计数 / resume 的口子。turn 主循环不直接接触
/// `Vec<Message>`，只通过这个 trait 拼下一轮的输入。
pub trait History: Send + Sync {
    /// 追加一条消息。
    fn append(&self, msg: Message);

    /// 当前历史的快照，用于喂给下一轮 LLM 调用。
    fn snapshot(&self) -> Vec<Message>;

    /// 估算当前历史的 token 数。`None` 表示估算不可用（v0 默认行为）。
    fn token_estimate(&self) -> Option<u64>;

    /// 主循环触发的"压缩"钩子。v0 实现可以是 no-op；
    /// 真正实现压缩时通过此入口（详见 `docs/internal/turn-loop.md`）。
    fn compact(&self) -> BoxFuture<'_, Result<CompactionReport, BoxError>>;
}

/// 压缩报告。压缩前后的 token 数据被主循环包成 [`AgentEvent::ContextCompressed`]。
#[derive(Debug, Clone, Copy)]
pub struct CompactionReport {
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// 进程级 agent 错误。
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("invalid working directory: {0}")]
    InvalidCwd(PathBuf),

    /// MCP server 启动失败（stdio 进程拉不起来 / sse 连不上）。
    #[error("mcp startup failed for {server}: {source}")]
    McpStartup {
        server: String,
        #[source]
        source: BoxError,
    },

    /// 调用方传入的 [`SessionId`] 已经存在于 session 表中。
    /// 单调递增 + 时间戳的 id 生成器理论上不会冲突；这是安全网。
    #[error("session id already in use: {0}")]
    DuplicateSessionId(SessionId),

    #[error(transparent)]
    Other(#[from] BoxError),
}

/// 一次 turn 的失败原因。
///
/// 划线规则：**只把"导致 turn 无法继续"的错误归到这里**。turn 内部
/// 工具失败、单次 LLM 重试失败等不归这里，归 [`AgentEvent`] 与历史
/// 的状态机。
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TurnError {
    /// 该 session 上已经有 turn 在跑。
    #[error("turn already in progress for this session")]
    TurnInProgress,

    /// 重试用尽后仍失败的 provider 错误。
    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// 主循环内部 invariant 被破坏（理应是 bug）。
    #[error("internal turn error: {0}")]
    Internal(#[source] BoxError),
}

/// 工具注册表的抽象。
///
/// 进程级（[`AgentCore`] 持有，内置工具）与会话级（[`Session`] 持有，
/// MCP 工具）共用同一形状；turn 主循环通过 `Session` 暴露的
/// composite registry 查工具。
pub trait ToolRegistry: Send + Sync {
    /// 列出注册表内所有工具的 schema，用于装配 LLM 请求的 `tools` 字段。
    fn schemas(&self) -> Vec<ToolSchema>;

    /// 按名查找工具。
    fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
}
