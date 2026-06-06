//! 工具抽象。
//!
//! 内置工具（`defect-tools`）与 MCP 适配器（`defect-mcp`）都通过实现
//! [`Tool`] trait 接入 agent 主循环。设计详见
//! `docs/internal/tool-trait.md`。
//!
//! ## ACP 对位
//!
//! [`Tool::describe`] 与 [`ToolEvent::Progress`] / [`ToolEvent::Completed`]
//! 直接复用 ACP 的 [`ToolCallUpdateFields`]，避免重复造一份字段。
//! 主循环把工具产出的字段拼上 [`ToolCallId`]、[`raw_input`] 等元信息后
//! 转手发出 `session/update` 与 `session/request_permission`。
//!
//! [`ToolCallId`]: agent_client_protocol_schema::ToolCallId
//! [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
//! [`raw_input`]: agent_client_protocol_schema::ToolCallUpdateFields::raw_input

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{ToolCallId, ToolCallUpdateFields};
use futures::Stream;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;
use crate::fs::FsBackend;
use crate::http::HttpClient;
use crate::session::EventEmitter;
use crate::shell::ShellBackend;

mod background_tasks;
mod goal_done;
mod skill;
mod spawn_agent;
pub use background_tasks::{CancelBackgroundTaskTool, InspectBackgroundTaskTool};
pub use goal_done::{GOAL_DONE_TOOL_NAME, GoalDoneTool};
pub use skill::{SkillEntry, SkillTool, SkillTriggers};
pub use spawn_agent::{SpawnAgentTool, SubagentProfile};

/// 工具的"对外名片"：只描述参数形状，不带任何执行能力。
///
/// [`crate::llm::CompletionRequest::tools`] 接受 `Vec<ToolSchema>`，
/// provider 不持有 `dyn Tool`，只把 schema 序列化进 wire JSON。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// 参数 JSON Schema。约定使用 Draft 2020-12 的子集（具体子集
    /// 与转义规则在 `tool-trait.md` 中沉淀）。
    pub input_schema: serde_json::Value,
}

/// 工具调用的"自描述"。直接对位 ACP 的 [`ToolCallUpdateFields`]。
///
/// 用途（同一份数据驱动三种 ACP 消息）：
/// - 首次推送 `ToolCall`（`status = Pending`）
/// - `RequestPermission` 请求中的 `tool_call` 字段
/// - 作为 [`ToolEvent::Progress`] 增量更新的基线
///
/// 字段约定：
/// - `tool_call_id` 不在此结构中；由主循环统一分配（用 LLM 给的
///   `tool_use_id` 或自生成 UUID），工具不关心。
/// - `raw_input` 由主循环在外层填充原始 args，工具实现不应自己塞，
///   避免与 wire 上的真实参数发散。
/// - `status` 由 [`ToolEvent`] 的 variant 推断：`Progress` → `InProgress`，
///   `Completed` → `Completed`，`Failed` → `Failed`。工具不应自行设置。
///
/// [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
#[derive(Debug, Clone)]
pub struct ToolCallDescription {
    pub fields: ToolCallUpdateFields,
}

/// 工具的安全等级。
///
/// 仅作为**提示**喂给外部 sandbox policy；最终的 Allow / Deny / Ask
/// 决策由 policy（结合用户配置、历史授权等）作出，trait 自身不做策略。
///
/// `serde` 形态使用 `snake_case`（`read_only` / `mutating` / `destructive` /
/// `network`），方便 `defect-config` 在 hook matcher 等场景里直接从 TOML
/// 反序列化。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyClass {
    /// 纯读：列目录、读文件、查询元数据。
    ReadOnly,
    /// 修改：写文件、改 state，副作用可逆性视情况。
    Mutating,
    /// 破坏性：删文件、移动、执行命令。
    Destructive,
    /// 出网：HTTP / DNS / 任意远程 IO。
    Network,
}

/// [`Tool::execute`] 产出的事件流元素。
///
/// 终态语义：流中**至多有一个** [`ToolEvent::Completed`] 或
/// [`ToolEvent::Failed`]，且必须是流的最后一个事件。主循环看到终态
/// 后即视为本次工具调用结束，不再消费后续元素。
#[non_exhaustive]
#[derive(Debug)]
pub enum ToolEvent {
    /// 进度增量：主循环转手发 ACP `session/update` 的 `tool_call_update`。
    /// 仅含本次变化的字段，符合 [`ToolCallUpdateFields`] 的"patch"语义。
    ///
    /// [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
    Progress(ToolCallUpdateFields),

    /// 成功结束。`fields` 是最终态的剩余字段（如最终 content / locations /
    /// raw_output）；主循环负责把 `status` 设为 `Completed`。
    Completed(ToolCallUpdateFields),

    /// 失败结束。携带 Rust 侧错误便于上层做 retry / log；映射到 ACP 时
    /// 由主循环把 `status` 设为 `Failed` 并把 [`ToolError`] 文本塞进
    /// `content`。
    Failed(ToolError),
}

/// [`Tool::execute`] 的事件流。类型擦除以便 `dyn Tool` 直接可用。
pub type ToolStream = Pin<Box<dyn Stream<Item = ToolEvent> + Send>>;

/// 注入给 [`Tool::execute`] 的运行环境。
///
/// 显式 struct 而非环境变量 / thread-local，方便测试时构造、避免隐式
/// 全局状态。字段标注 `non_exhaustive` 是为了允许后续追加（sandbox
/// 句柄、ACP 反向通道等）而不破坏现有实现。
#[non_exhaustive]
pub struct ToolContext<'a> {
    /// 工具默认的工作目录（通常是 ACP session 的 cwd）。
    pub cwd: &'a Path,
    /// 取消令牌：上层 `session/cancel`、用户 Ctrl+C、超时等都会触发。
    /// 工具实现应在长循环 / await 点检查 `cancel.is_cancelled()` 并尽快
    /// 退出。
    pub cancel: CancellationToken,
    /// 文件系统后端。fs 工具家族（`read_file` / `write_file` /
    /// `edit_file`）通过它读写文件；装配时由 `defect-acp` 按客户端
    /// 协商的 [`FileSystemCapabilities`] 选择 `LocalFsBackend` 或
    /// `AcpFsBackend`，工具实现完全不感知。
    ///
    /// 用 [`Arc`] 而非借用：`Tool::execute` 返回 `'static` future / stream,
    /// 工具内部通常 `clone` 一份 fs 进入异步任务。借用形式无法跨过 await。
    ///
    /// [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities
    pub fs: Arc<dyn FsBackend>,
    /// Shell 执行后端。`bash` 工具通过它创建 terminal 跑命令；装配时由
    /// `defect-acp` 按客户端协商的 [`ClientCapabilities::terminal`] 选择
    /// `LocalShellBackend` 或 `AcpShellBackend`，工具实现不感知。
    ///
    /// 与 `fs` 同款 `Arc` 取舍——`Tool::execute` 是 `'static` future。
    ///
    /// [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities
    pub shell: Arc<dyn ShellBackend>,
    /// HTTP fetch 后端。`fetch` 工具通过它发起网络读取；装配在 CLI 入口完成
    /// （按 `HttpClientConfig` 构造一个进程级 [`HttpClient`] 实例并复用）。
    /// 工具实现拿到的是 [`Arc`] 副本；`Tool::execute` 是 `'static` future，
    /// 借用形式无法跨过 await。
    pub http: Arc<dyn HttpClient>,
    /// 当前 turn 选中的 model id。绝大多数工具用不到；`spawn_agent`
    /// 子 agent 工具用它做"model 回落到父会话当前选择"——`ToolContext`
    /// 不携带 provider registry，但携带这个字符串就够 `spawn_agent` 在
    /// 自己捕获的 registry 上 `entry_for_model` 解析出父此刻用的 provider。
    /// 由 [`TurnRunner`](crate::session::TurnRunner) 构造 ctx 时填入 `config.model`。
    pub current_model: &'a str,
    /// session 级后台任务句柄。`Some` 时工具可 fire-and-forget 地 spawn 一个
    /// 活过当前 turn 的任务（首要场景：`spawn_agent { run_in_background: true }`）；
    /// `None` 表示本上下文不支持后台（子 agent 嵌套 turn / 测试），工具应回退到
    /// 同步执行。
    ///
    /// 用 owned [`Arc`]-backed 句柄而非借用：`Tool::execute` 返回 `'static`
    /// future，借用无法跨过 await。由顶层 [`TurnRunner`](crate::session::TurnRunner)
    /// 在构造 ctx 时注入；嵌套子 agent turn 不注入（结构性禁止后台任务自我繁殖）。
    pub background: Option<crate::session::BackgroundTasks>,
    /// subagent 事件桥：`Some` 时工具可把自己内部派生的子 turn 事件包成
    /// [`crate::event::AgentEvent::Subagent`] 转发回父 session 的事件流，供
    /// observability 嵌套展示。当前唯一使用者是 `spawn_agent`。由
    /// `session::turn` 的 turn runner 在驱动每个工具时按该工具的
    /// [`ToolCallId`] 注入——**顶层与子 agent 嵌套 turn 都注入**（递归桥接），
    /// 挂载坐标由 [`SubagentBridge::parent_tool_call_id`] 表达。
    pub subagent_bridge: Option<SubagentBridge>,
    /// 本 turn 快照的 active sandbox policy。`spawn_agent` 用它做"子 agent 包
    /// 父此刻的真实策略"——`session/set_mode` 切换后新起的 turn 把新策略经此
    /// 传下去，子 agent 不会拿到陈旧的进程级默认。`None` 时 `spawn_agent`
    /// 回退到构造期捕获的 policy（测试 / 未注入场景）。绝大多数工具忽略本字段。
    pub policy: Option<Arc<dyn crate::policy::SandboxPolicy>>,
    /// `--goal` 目标驱动循环的共享状态。`Some` 时本 session 跑在目标模式下，
    /// `goal_done` 工具调用 [`crate::session::GoalState::mark_reached`] 置位；
    /// `goal-gate` hook 在 turn 自愿停止时据此决定放行还是续命。`None` = 非目标
    /// 模式（默认），`goal_done` 工具不会被注册，本字段无人读。
    pub goal: Option<Arc<crate::session::GoalState>>,
    /// 从当前层起还能再派发多少层 subagent。顶层 turn = 配置的初始上限；
    /// `spawn_agent` 为子 agent 嵌套 turn 注入时减一。`0` ⇒ 子 agent 拿不到
    /// `spawn_agent` 工具（深度耗尽，结构性禁止继续递归）——取代旧的"白名单永不
    /// 含 spawn_agent"硬编码。功能性闸门，与 observability 无关，故独立于可空的
    /// [`Self::subagent_bridge`]，在测试 / 无桥场景下同样生效。默认 `0`（最保守：
    /// 不显式注入即不可派发；顶层 turn 必须显式 [`Self::with_subagent_depth`]）。
    pub subagent_depth: u32,
}

/// 把工具内部派生的子 turn 事件桥接回父 session 事件流所需的句柄。
///
/// 持有父 session 的 [`EventEmitter`] 与发起本工具调用的 [`ToolCallId`]。`Clone`
/// 廉价（内部 `Arc` + 小字符串）。
///
/// ## 递归桥接：每层只 prepend 自己的 id
///
/// 完整祖先链不存在这里——它在事件**向上冒泡**时由各层桥接逐段累积。每一层的桥接
/// 订阅者（`spawn_agent` 的 `bridge_task`）：
/// - 收到子 turn 的**叶子**事件 ⇒ 包成
///   `Subagent{ ancestor_path: [parent_tool_call_id], agent_type: <本层 profile>, inner: 叶子 }`；
/// - 收到的**已是** `Subagent`（来自更深层、已带部分链）⇒ 把 `parent_tool_call_id`
///   **prepend** 到其 `ancestor_path` 链首、保留 `inner` 叶子与深层 `agent_type` 不变。
///
/// 于是事件穿过 N 层桥接后，`ancestor_path` 恰好是从顶层到叶子那层的完整 id 链。
/// 每层无需预知全链，只认自己这一跳——这也让前台 / 后台 / 任意深度共用同一逻辑。
///
/// 递归的**深度闸门**不在这里——它是功能性的、必须始终生效（含无 observability /
/// 测试场景），故走 [`ToolContext::subagent_depth`] 独立字段，而非这个可空的桥。
#[derive(Clone)]
pub struct SubagentBridge {
    /// 父 session 的事件总线。包好的 [`crate::event::AgentEvent::Subagent`] 投到这里。
    pub parent_events: Arc<EventEmitter>,
    /// 发起子 agent 的那次工具调用 id（父 trace 里对应的 tool span）。本层桥接据它
    /// prepend，是该子 agent 在父 trace 里挂载点的坐标。
    pub parent_tool_call_id: ToolCallId,
}

impl<'a> ToolContext<'a> {
    /// 构造一个最小 `ToolContext`。`#[non_exhaustive]` 让外部 crate 不能直接
    /// 用结构体字面量构造——这个构造函数是 cross-crate 唯一入口。新增字段时
    /// 给签名加默认值或新构造函数，不破坏现有调用点。
    pub fn new(
        cwd: &'a Path,
        cancel: CancellationToken,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        http: Arc<dyn HttpClient>,
        current_model: &'a str,
    ) -> Self {
        Self {
            cwd,
            cancel,
            fs,
            shell,
            http,
            current_model,
            background: None,
            subagent_bridge: None,
            policy: None,
            goal: None,
            subagent_depth: 0,
        }
    }

    /// 注入本层起的剩余 subagent 派发深度。顶层 turn 的工具驱动用配置的初始上限
    /// 调用；`spawn_agent` 为子 agent 嵌套 turn 注入减一后的值。不调用 ⇒ `0`
    /// （最保守：不可派发 subagent）。
    #[must_use]
    pub fn with_subagent_depth(mut self, depth: u32) -> Self {
        self.subagent_depth = depth;
        self
    }

    /// 注入本 turn 快照的 active sandbox policy。顶层 turn 的工具驱动用它把
    /// 父此刻的策略传给 `spawn_agent`；不调用则 `policy` 为 `None`（子 agent
    /// 嵌套 / 测试），`spawn_agent` 回退到构造期捕获的 policy。
    #[must_use]
    pub fn with_policy(mut self, policy: Arc<dyn crate::policy::SandboxPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// 注入 session 级后台任务句柄。顶层 turn 的工具驱动用它开启 `run_in_background`
    /// 能力；不调用则 `background` 为 `None`（子 agent / 测试的默认），工具回退同步执行。
    #[must_use]
    pub fn with_background(mut self, background: crate::session::BackgroundTasks) -> Self {
        self.background = Some(background);
        self
    }

    /// 注入 `--goal` 目标驱动循环的共享状态。`goal_done` 工具据它置位 reached；
    /// 不调用则 `goal` 为 `None`（非目标模式，默认）。
    #[must_use]
    pub fn with_goal(mut self, goal: Arc<crate::session::GoalState>) -> Self {
        self.goal = Some(goal);
        self
    }

    /// 注入 subagent 事件桥。工具驱动 `session::turn` 为每个工具调用按其
    /// [`ToolCallId`] 注入，让 `spawn_agent` 能把子 turn 事件嵌套回父 trace。
    #[must_use]
    pub fn with_subagent_bridge(mut self, bridge: SubagentBridge) -> Self {
        self.subagent_bridge = Some(bridge);
        self
    }
}

/// agent 可调用的工具。
///
/// 实现者通常是无状态的（每次调用通过 `args` + [`ToolContext`] 拿到
/// 全部依赖）；如果需要持有连接 / 缓存等状态，把状态放在 `Self` 上、
/// 用 `Arc<Self>` 注册给主循环即可。
pub trait Tool: Send + Sync {
    /// 工具名片。返回引用避免每次调用都构造一份。
    fn schema(&self) -> &ToolSchema;

    /// 在不实际执行的前提下，给 sandbox policy 一个安全等级提示。
    ///
    /// `args` 是已经反序列化好的 JSON Value——同一个工具的安全等级
    /// 可能依参数而异（例如 `bash` 工具在 `command` 含 `rm` 时升为
    /// [`SafetyClass::Destructive`]）。实现应当**纯函数**，不做 IO。
    fn safety_hint(&self, args: &serde_json::Value) -> SafetyClass;

    /// 在执行前生成一份"自描述"，用于推送给 ACP 客户端展示。
    ///
    /// 异步签名 + [`ToolContext`] 注入：实现可以在 describe 阶段做轻量
    /// IO（典型用例：`write_file` 在请求授权前先读旧内容，给客户端画
    /// 精确 old↔new diff——比"全新内容"更利于审查）。
    ///
    /// 性能约束：describe 在每次 ACP `ToolCall` 推送前都会跑一次，
    /// 实现仍应保持快速且对失败 graceful（IO 失败时降级返回基础字段，
    /// 不要让 describe 自己抛错——签名也没给错误通道）。
    ///
    /// 具体由谁填什么字段见 [`ToolCallDescription`] 的字段约定。
    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription>;

    /// 启动一次工具调用，返回事件流。
    ///
    /// 流的元素见 [`ToolEvent`]；终态事件之后流应立即结束。drop 流
    /// 视为取消（与 `ctx.cancel.cancel()` 等价）。
    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream;
}

/// 工具执行错误。
///
/// 粒度故意保持粗——细化到具体错误类型由内置工具自己在 `Execution`
/// 的 source 里携带。这里只区分主循环需要差异化处理的几大类。
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ToolError {
    /// 上层取消（[`ToolContext::cancel`] 触发）。
    #[error("tool canceled")]
    Canceled,

    /// 参数 JSON 解析失败 / schema 校验不过。主循环可把它送回 LLM
    /// 让模型修正参数后重试。
    #[error("invalid tool arguments: {0}")]
    InvalidArgs(#[source] BoxError),

    /// 执行期错误（IO 失败、子进程非零退出、网络错误等）。
    #[error("tool execution failed: {0}")]
    Execution(#[source] BoxError),
}
