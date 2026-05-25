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
//! [`ToolCallId`]: agent_client_protocol::schema::ToolCallId
//! [`ToolCallUpdateFields`]: agent_client_protocol::schema::ToolCallUpdateFields
//! [`raw_input`]: agent_client_protocol::schema::ToolCallUpdateFields::raw_input

use std::path::Path;
use std::pin::Pin;

use agent_client_protocol::schema::ToolCallUpdateFields;
use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;

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
/// [`ToolCallUpdateFields`]: agent_client_protocol::schema::ToolCallUpdateFields
#[derive(Debug, Clone)]
pub struct ToolCallDescription {
    pub fields: ToolCallUpdateFields,
}

/// 工具的安全等级。
///
/// 仅作为**提示**喂给外部 sandbox policy；最终的 Allow / Deny / Ask
/// 决策由 policy（结合用户配置、历史授权等）作出，trait 自身不做策略。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// [`ToolCallUpdateFields`]: agent_client_protocol::schema::ToolCallUpdateFields
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
}

impl<'a> ToolContext<'a> {
    /// 构造一个最小 `ToolContext`。`#[non_exhaustive]` 让外部 crate 不能直接
    /// 用结构体字面量构造——这个构造函数是 cross-crate 唯一入口。新增字段时
    /// 给签名加默认值或新构造函数，不破坏现有调用点。
    pub fn new(cwd: &'a Path, cancel: CancellationToken) -> Self {
        Self { cwd, cancel }
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
    /// 实现应当快速返回（不做 IO），仅基于 args 决定 title / kind /
    /// locations 等字段。具体由谁填什么字段见 [`ToolCallDescription`]
    /// 的字段约定。
    fn describe(&self, args: &serde_json::Value) -> ToolCallDescription;

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
