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

use agent_client_protocol_schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use futures::future::BoxFuture;

use crate::error::BoxError;
use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::llm::{
    Message, ModelCandidate, ModelInfo, ProviderError, ProviderInfo, ReasoningEffort,
};
use crate::shell::ShellBackend;
use crate::tool::{Tool, ToolSchema};

mod background;
mod capabilities;
mod context;
mod default;
mod events;
mod goal;
mod history;
mod permissions;
mod prompt;
mod tool_registry;
mod turn;

pub use background::{
    BackgroundOutcome, BackgroundProgressConfig, BackgroundResult, BackgroundTasks, BlockKind,
    ProgressBlock, TaskHandle, TaskSnapshot, TaskStatus, format_background_outcome,
};
pub use capabilities::{
    ResolvedSessionCapabilities, SessionCapabilitiesConfig, WebSearchCapabilityConfig,
    WebSearchCapabilityMode,
};
pub use context::{Frontend, RunningContext};
pub use default::{DefaultAgentCore, DefaultAgentCoreBuilder, DefaultSession, new_session_id};
pub use events::EventEmitter;
pub use goal::GoalState;
pub use history::VecHistory;
pub use permissions::PermissionGate;
pub use prompt::resolve_system_prompt;
pub use tool_registry::{CompositeRegistry, StaticToolRegistry, StaticToolRegistryBuilder};
/// crate 内部复用：`spawn_agent` 子 agent 工具构造嵌套 [`TurnRunner`] 时需要
/// 一个 `RequestAuditTracker` 实例。它对外不公开（诊断用内部状态），但同 crate
/// 的 `crate::tool::spawn_agent` 要能 `new()`。
pub(crate) use turn::RequestAuditTracker;
pub use turn::{
    BasePromptConfig, CompactionSlot, PromptConfig, TurnConfig, TurnRequestLimit, TurnRunner,
};

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
    /// `shell` 是 session 级 shell 后端——`defect-acp` 装配时按客户端的
    /// [`ClientCapabilities::terminal`] 选择 `LocalShellBackend` 或
    /// `AcpShellBackend`。session 持有它的 `Arc`，`bash` 工具调用都走它。
    ///
    /// `frontend` 标记 agent 被如何接入（[`Frontend::Acp`] 携带 ACP 握手协商
    /// 出的 fs / shell 委托状态），用于注入 system prompt 的 `# Environment` 段。
    ///
    /// # Errors
    ///
    /// MCP 启动失败、cwd 不存在、id 重复等。
    ///
    /// [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities
    /// [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities
    fn create_session(
        &self,
        id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        frontend: Frontend,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    /// 从持久化状态恢复一个已存在的 session。
    ///
    /// `frontend` 同 [`AgentCore::create_session`]——恢复出的 session 也要据此
    /// 注入运行环境信息。
    ///
    /// # Errors
    ///
    /// session 不存在、持久化数据损坏、恢复出的 cwd 不可用等。
    fn load_session(
        &self,
        id: SessionId,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        frontend: Frontend,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    /// 按 id 查找已存在的 session。
    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>>;
}

/// 从持久化存储恢复 session 的抽象。
///
/// 具体实现通常来自 `defect-storage`。
pub trait SessionLoader: Send + Sync {
    /// 按 session id 读回恢复所需状态。
    ///
    /// # Errors
    ///
    /// session 不存在、存储损坏、或回放失败。
    fn load_session(&self, id: SessionId) -> BoxFuture<'_, Result<LoadedSession, BoxError>>;
}

/// 为单个 session 构建附加工具表的抽象。
///
/// 典型实现来自 `defect-mcp`：按 `session/new` 或 `session/load` 提供的
/// MCP server 列表建立连接，并把远端工具包装成 [`ToolRegistry`]。
pub trait SessionToolFactory: Send + Sync {
    /// 为当前 session 构建一份会话级工具表。
    ///
    /// # Errors
    ///
    /// 外部工具源初始化失败、远端 inventory 拉取失败、或配置不受支持。
    fn build_registry(
        &self,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> BoxFuture<'_, Result<Arc<dyn ToolRegistry>, BoxError>>;
}

/// `AgentCore::create_session` 成功后的观察器。
///
/// 典型用途：
/// - 启动 `defect-storage` 的事件订阅落盘
/// - 挂 tracing / metrics 的 per-session 旁路消费者
pub trait SessionObserver: Send + Sync {
    /// 在 session 创建成功后调用。
    ///
    /// # Errors
    ///
    /// 初始化旁路消费者失败时返回错误，阻止该 session 对外可见。
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError>;
}

/// 一个可选权限模式的对外描述。`defect-acp` 用它构造 ACP `SessionMode`。
///
/// 是 [`crate::policy::PolicyMode`] 的"无 policy"投影——只暴露 id / 展示
/// 字段，不泄露内部决策器。
#[derive(Debug, Clone)]
pub struct ModeDescriptor {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

/// 模型选择键：`(provider vendor, model id)` 对。
///
/// 同一 model id 可由多个 provider 声明（多网关同模型），故选择必须同时带上
/// provider vendor 与 model id。`provider` 即 [`ProviderInfo::vendor`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    pub provider: String,
    pub model: String,
}

/// 单次会话。
///
/// 所有方法都是 trait 对象友好（`&self` + `BoxFuture`）。`Arc<dyn Session>`
/// 在 `defect-acp` 与主循环之间共享。
pub trait Session: Send + Sync {
    fn id(&self) -> &SessionId;

    /// 当前 session 使用的 provider 元信息。
    fn provider_info(&self) -> ProviderInfo;

    /// 当前 session 使用的模型 id。
    fn current_model(&self) -> String;

    /// 列出当前 provider 对此 session 可用的模型候选。
    ///
    /// # Errors
    ///
    /// 当 provider 拉取模型列表失败时返回 [`ProviderError`]。
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>>;

    /// 列出 session 可见的 (provider, model) 候选对——多 provider 装配下
    /// 同一 session 可能跨 provider 切换 model，ACP 渲染时需要在每条候选
    /// 旁边注明所属 provider。
    ///
    /// # Errors
    ///
    /// 同 [`Self::list_models`]：provider 列表拉取失败时返回 [`ProviderError`]。
    fn list_candidates(&self) -> BoxFuture<'_, Result<Vec<ModelCandidate>, ProviderError>>;

    /// 切换当前 session 的模型。
    ///
    /// 选择键是 `(provider vendor, model)` 对——同一 model id 可由多个 provider
    /// 声明（多网关同模型），故必须显式带上 provider。当前进行中的 turn 保持原
    /// 选择；后续 turn 使用新选择。
    ///
    /// # Errors
    ///
    /// 当 provider 拉取模型列表失败，或请求的 `(provider, model)` 对不存在时返回
    /// [`ProviderError`]。
    fn set_model(&self, selection: ModelSelection) -> BoxFuture<'_, Result<(), ProviderError>>;

    /// 当前生效的权限模式 id。未装配模式目录时返回 `None`。
    ///
    /// 映射到 ACP `SessionModeState::current_mode_id`。
    fn current_mode(&self) -> Option<String>;

    /// 本 session 可选的权限模式列表（顺序即装配顺序）。未装配模式目录时
    /// 返回空。映射到 ACP `SessionModeState::available_modes`。
    fn available_modes(&self) -> Vec<ModeDescriptor>;

    /// 切换当前权限模式。后续 turn 生效，进行中的 turn 保持原 policy
    /// （与 [`Self::set_model`] 同语义——run_turn 启动时快照 policy）。
    ///
    /// # Errors
    ///
    /// `mode_id` 未命中任一可选模式，或本 session 未装配模式目录时返回
    /// [`AgentError::ModeNotFound`]。
    fn set_mode(&self, mode_id: String) -> Result<(), AgentError>;

    /// 当前的 `reasoning_effort` 等级（`None` = 未设置，沿用 provider 默认）。
    /// 映射到 ACP thought-level 配置项的当前值。
    fn current_reasoning_effort(&self) -> Option<ReasoningEffort>;

    /// 设置 `reasoning_effort` 等级。`None` 清除覆盖（回落 provider 默认）。
    /// 后续 turn 生效。不支持该概念的 provider 在装配请求时忽略。
    fn set_reasoning_effort(&self, effort: Option<ReasoningEffort>);

    /// 订阅事件流。三个独立消费者（acp / storage / tracing）各自调一次，
    /// 互不影响——内部用 mpsc 配 fan-out 保证慢消费者只 backpressure
    /// 自己、不丢事件。具体技术细节见 `docs/internal/session.md` §5。
    fn subscribe(&self) -> EventStream;

    /// 当前历史的只读快照，用于 session/load 后向客户端 replay transcript。
    fn history_snapshot(&self) -> Vec<Message>;

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

/// 创建成功后给 [`SessionObserver`] 的稳定信息。
#[derive(Debug, Clone)]
pub struct SessionCreateInfo {
    pub id: SessionId,
    pub cwd: PathBuf,
    pub mcp_servers: Vec<McpServer>,
}

/// 从持久化存储恢复出来的最小 session 数据。
#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub info: SessionCreateInfo,
    pub history: Vec<Message>,
}

/// 消息历史的抽象——**纯存储 + token 计量**。
///
/// 压缩这件事**不在这里**：摘要需要调 LLM，而存储抽象够不到 provider。
/// 压缩编排在 turn 主循环（`session/turn/compact.rs`）里完成——它读
/// [`History::snapshot`]、调 LLM 摘要、再用 [`History::replace`] 把算好的
/// 新消息列表整体写回。本 trait 只负责：追加、快照、整体替换、以及给
/// 主循环报一个「当前历史值多少 token」的估算。
///
/// token 估算策略（详见 [`VecHistory`]）：以上一次 LLM 调用回报的**真实
/// 输入 token**为基线，叠加其后新追加消息的**字符启发式**增量；真实基线
/// 不可用时整份走字符启发式兜底。turn 主循环用它与压缩阈值比较。
pub trait History: Send + Sync {
    /// 追加一条消息。
    fn append(&self, msg: Message);

    /// 当前历史的快照，用于喂给下一轮 LLM 调用。
    fn snapshot(&self) -> Vec<Message>;

    /// 压缩后整体替换消息列表。turn 主循环算好「摘要 + 保留尾部」的新列表后
    /// 调它回写。实现应同时重置 token 估算基线（旧的真实 token 已不适用新列表）。
    fn replace(&self, messages: Vec<Message>);

    /// 前缀替换：用 `summary` 这一条消息换掉**当前**列表最前面 `drop_count` 条，
    /// 保留其后全部。返回实际丢弃的消息数（`drop_count` 被 clamp 到当前长度）。
    ///
    /// 这是**后台压缩**回写的原语：后台任务在某时刻的 snapshot 上算出
    /// `drop_count`（= 待摘要前缀长度）与 `summary`，但摘要 LLM 调用耗时期间，
    /// 前台 turn 仍在往**尾部** `append`。回写时绝不能 `replace(整表)`——那会冲掉
    /// 这期间新增的尾部消息。`splice_prefix` 只动**当前**列表的前 `drop_count` 条，
    /// 保留 `drop_count..` 的全部（含期间新增的尾部），故回写正确。
    ///
    /// **并发不变式**（务必维持）：`drop_count` 在旧 snapshot 上算得、对**当前**列表
    /// 合法，前提是「飞行期间只发生尾插（`append`）与原地内容替换（微压缩
    /// `replace` 同长度重建），不增删中段消息」。唯一会删中段的操作是压缩本身，
    /// 而压缩**单飞**（同时至多一个在跑）——故该不变式成立。
    ///
    /// 同 [`Self::replace`]，回写后重置 token 基线（新前缀的真实 token 未知）。
    fn splice_prefix(&self, drop_count: usize, summary: Message) -> usize;

    /// 喂入上一次 LLM 调用的真实输入 token 数
    /// （`input + cache_read + cache_creation`）。作为 [`Self::token_estimate`]
    /// 的精确基线；其后 [`Self::append`] 的消息走字符启发式增量叠加。
    fn record_input_tokens(&self, tokens: u64);

    /// 估算当前历史的 token 数。`None` 表示历史为空 / 无可用估算。
    fn token_estimate(&self) -> Option<u64>;
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

    #[error("session observer failed: {0}")]
    Observer(#[source] BoxError),

    #[error("session not found in storage: {0}")]
    SessionNotFound(SessionId),

    /// `set_mode` 收到的 `mode_id` 不在该 session 的模式目录里（或目录未装配）。
    #[error("permission mode not found: {0}")]
    ModeNotFound(String),

    #[error("session restore failed: {0}")]
    Restore(#[source] BoxError),

    /// session 启动期能力裁决失败。详见 [`SessionInitError`]。
    #[error(transparent)]
    Init(#[from] SessionInitError),

    #[error(transparent)]
    Other(#[from] BoxError),
}

/// session 启动期一次性裁决失败。
///
/// 设计详见 `docs/internal/capabilities.md` §5。
/// 当 `capabilities.<name>.mode = "delegate"` 但当前 provider 的
/// [`crate::llm::LlmProvider::hosted_capabilities`] 不支持该 capability
/// 时，拒绝启动 session。
#[non_exhaustive]
#[derive(Debug)]
pub enum SessionInitError {
    /// 用户显式选择 `Delegate`，但 provider 不支持对应 hosted capability。
    CapabilityUnsatisfied {
        /// 出问题的 capability 名（例如 `"web_search"`）。
        capability: &'static str,
        /// 当前 session 绑定的 provider 名。
        provider: String,
    },
}

impl std::fmt::Display for SessionInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityUnsatisfied {
                capability,
                provider,
            } => {
                writeln!(
                    f,
                    "{capability} capability is unsatisfied: provider `{provider}` does not support hosted {capability}."
                )?;
                writeln!(f)?;
                writeln!(f, "To fix this, choose one of:")?;
                writeln!(f, "  1. Disable hosted {capability} for this provider:")?;
                writeln!(f, "       [providers.{provider}.capabilities.{capability}]")?;
                writeln!(f, "       mode = \"disabled\"")?;
                writeln!(
                    f,
                    "  2. Change global default to `disabled` and only delegate where supported:"
                )?;
                writeln!(f, "       [capabilities.{capability}]")?;
                writeln!(f, "       mode = \"disabled\"")?;
                writeln!(
                    f,
                    "       [providers.<hosted-supported>.capabilities.{capability}]"
                )?;
                write!(f, "       mode = \"delegate\"")
            }
        }
    }
}

impl std::error::Error for SessionInitError {}

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
