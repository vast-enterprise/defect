//! The event stream published by the agent main loop to external consumers.
//!
//! ## Decoupling-by-shape
//!
//! 主循环只产生 [`AgentEvent`]——内部 enum，三个独立消费者各取所需：
//!
//! ```text
//!                ┌──► defect-acp     (翻译成 SessionUpdate / PromptResponse)
//! AgentEvent ────┼──► defect-storage (jsonl 持久化)
//!                └──► tracing        (结构化日志、observability)
//! ```
//!
//! enum **变体由我们定义**（持久化格式与 wire 解耦、能表达 wire 上没有
//! 的语义如 turn 边界与 LLM 调用），但**字段类型尽量直接复用 ACP 的被动
//! 数据结构**（`ToolCallUpdateFields`、`ContentBlock`、`StopReason` 等），
//! 避免重新发明字段。

use std::sync::Arc;

use agent_client_protocol_schema::{
    ContentBlock, PermissionOptionId, StopReason as AcpStopReason, ToolCallId, ToolCallUpdateFields,
};
use serde::{Deserialize, Serialize};

use crate::llm::{Message, Usage};
use crate::policy::PolicyDecision;

/// agent 主循环对外发布的事件。
///
/// 终态语义：一次 turn 的事件流以 [`AgentEvent::TurnStarted`] 开始、
/// 以 [`AgentEvent::TurnEnded`] 结束。`TurnEnded` 之后流不再产出本轮
/// 事件——`defect-acp` 看到它即停止推 `session/update` 并 respond
/// `PromptResponse`。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    // ---------- turn 边界 ----------
    /// 一次 prompt turn 开始。
    TurnStarted,

    /// 用户 prompt 已被主循环提交到 history。
    UserPromptCommitted { content: Vec<ContentBlock> },

    /// 一次 prompt turn 结束。`reason` 直接借 ACP 的语义类别。
    TurnEnded {
        reason: AcpStopReason,
        /// 本 turn 的累计 token 用量（来自 [`crate::llm::ProviderChunk::Usage`] 的逐字段累加）。
        usage: Usage,
    },

    // ---------- 助手输出（推给 wire） ----------
    /// 助手文本增量。映射到 ACP `SessionUpdate::AgentMessageChunk`。
    AssistantText { content: ContentBlock },

    /// 助手思考链增量。映射到 ACP `SessionUpdate::AgentThoughtChunk`。
    AssistantThought { content: ContentBlock },

    // ---------- 工具调用（推给 wire） ----------
    /// 一次工具调用开始声明。
    /// 映射到 ACP `SessionUpdate::ToolCall`（status = Pending）。
    ToolCallStarted {
        id: ToolCallId,
        name: String,
        fields: ToolCallUpdateFields,
    },

    /// 工具调用进度增量。
    /// 映射到 ACP `SessionUpdate::ToolCallUpdate`。
    ToolCallProgress {
        id: ToolCallId,
        fields: ToolCallUpdateFields,
    },

    /// 工具调用结束（成功 / 失败由 `fields.status` 表达）。
    /// 映射到 ACP `SessionUpdate::ToolCallUpdate`（含终态 status）。
    ToolCallFinished {
        id: ToolCallId,
        fields: ToolCallUpdateFields,
    },

    // ---------- 权限决策（部分推给 wire） ----------
    /// sandbox policy 对工具调用作出决策。`Ask` 触发 ACP
    /// `session/request_permission`；`Allow` / `Deny` 仅作审计、不入 wire。
    PolicyDecision {
        id: ToolCallId,
        decision: PolicyDecision,
    },

    /// 用户对 [`PolicyDecision::Ask`] 的应答。仅审计，不入 wire。
    PermissionResolved {
        id: ToolCallId,
        outcome: PermissionResolution,
    },

    // ---------- 主循环编排（不入 wire，仅 storage / tracing） ----------
    /// 一次 LLM provider 调用开始。
    LlmCallStarted {
        model: String,
        /// 第几次尝试（首次为 1）。重试由主循环驱动。
        attempt: u32,
        /// 本次调用发给 provider 的请求快照（system + 完整 messages 历史）。
        ///
        /// 供 observability 把 generation 的 `input` 还原成标准 chat messages
        /// 数组（含 system 一条）。不入 wire；storage 当前忽略此字段。
        ///
        /// 用 `Arc` 包裹：事件经 [`crate::session::EventEmitter`] fan-out 给每个
        /// 订阅者时会 `clone` 一次，长上下文下整份 messages 历史被多次深拷贝代价
        /// 很高。快照进事件后只读，`Arc` 让 clone 退化成引用计数。
        /// `#[serde(skip)]`：`AgentEvent` 的 serde derive 目前无人实际使用，且
        /// 不想为它启用 serde 的 `rc` feature——反序列化时此字段取默认空快照。
        #[serde(skip)]
        request: Arc<LlmRequestSnapshot>,
    },

    /// 一次 LLM provider 调用结束。`error` 为 `Some` 表示失败（按 retry
    /// hint 决定是否进入下一次 attempt）。
    LlmCallFinished {
        model: String,
        attempt: u32,
        usage: Usage,
        /// 失败时的错误描述（不放完整错误对象——它进 tracing）。
        error: Option<String>,
    },

    /// 主循环对历史做了压缩 / 截断。
    ContextCompressed {
        tokens_before: u64,
        tokens_after: u64,
    },

    /// 主循环做了一次**微压缩**：清理较旧轮次里的超大 `tool_result` 正文（不调
    /// LLM、不删消息）。`cleared` = 实际清理的 `tool_result` 条数。与
    /// [`Self::ContextCompressed`] 区分以便 observability / UI 分别展示。
    ContextMicrocompacted {
        tokens_before: u64,
        tokens_after: u64,
        cleared: usize,
    },

    // ---------- subagent 嵌套（仅 observability） ----------
    /// 一个 `spawn_agent` 子 agent turn 内部产生的**叶子**事件，从子 turn 的隔离
    /// 事件流桥接到父 session 的事件流。
    ///
    /// 设计意图：子 agent 在 fresh、隔离上下文里跑（自己的 [`crate::session::EventEmitter`]），
    /// 父 agent **看不到**它的中间过程——这是 `spawn_agent` 的隔离契约。但
    /// observability（langfuse）希望把子 turn 的 LLM 调用 / 工具调用嵌套展示在
    /// 父那次 `spawn_agent` 工具调用的 span 之下。于是 `spawn_agent` 在子 emitter
    /// 上挂一个桥接订阅者，把每个子事件包成本变体转发给父 emitter。
    ///
    /// ## 扁平化（支持递归 subagent）
    ///
    /// `inner` **永远是叶子事件**（绝不再是另一个 `Subagent`）。嵌套深度由
    /// [`Self::Subagent::ancestor_path`] 这条**祖先链**表达，而非层层 `Box` 包裹：
    /// 链里从顶层 `spawn_agent` 工具调用 id 到当前层的 id 依次排列。每经过一层桥接，
    /// 该层把自己的 `parent_tool_call_id` **prepend** 到链首、保留叶子 `inner` 不变
    /// （见 `spawn_agent.rs` 桥接闭包）。projector 用整条链定位父挂载点——链全局唯一，
    /// 天然避开跨子 session 的 `ToolCallId` 撞号，且 projector 无需递归剥壳。
    ///
    /// **消费约定**：只有 langfuse projector 处理它（投成挂在父 tool span 下的
    /// 嵌套 generation / span）。其余消费者（`defect-storage` 落盘、`defect-acp`
    /// wire 投射、REPL 渲染）一律**忽略**——隔离契约对它们不变。
    Subagent {
        /// 从顶层 `spawn_agent` 工具调用到当前 subagent 层的 `ToolCallId` 祖先链。
        /// 链首是顶层（直接挂在父 turn trace 里的那次 spawn_agent），链尾是发起
        /// 本叶子事件那一层的 spawn_agent。深度 = `ancestor_path.len()`。
        ancestor_path: Vec<ToolCallId>,
        /// 发起本叶子事件那一层 subagent 的 profile 名（如 `weebs-in`），进嵌套
        /// span 的命名 / 元数据。
        agent_type: String,
        /// 被桥接的子 turn **叶子**事件（绝不是另一个 `Subagent`）。`Box` 避免 enum
        /// 因自引用而无界膨胀。
        inner: Box<AgentEvent>,
    },
}

/// 一次 LLM 调用的请求快照——只带 observability 还原 generation `input`
/// 所需的部分（system + 完整 messages 历史）。不含 tools / sampling 等。
///
/// 单独定义而非直接塞 `CompletionRequest`：避免 `AgentEvent` 依赖整个请求类型，
/// 也让快照保持最小、序列化稳定。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmRequestSnapshot {
    /// 系统提示词（若有）。observability 把它还原成 `{role:"system"}` 一条。
    pub system: Option<Arc<str>>,
    /// 本次发给 provider 的完整 messages 历史。
    pub messages: Vec<Message>,
}

/// 用户对 `Ask` 的应答。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionResolution {
    /// 用户选择了某个选项；`option_id` 由 ACP `PermissionOption` 携带。
    Selected { option_id: PermissionOptionId },
    /// 用户在选项作出前取消了 turn。
    Cancelled,
}
