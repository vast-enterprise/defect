//! agent 主循环对外发布的事件流。
//!
//! 设计详见 `docs/internal/event-model.md`。
//!
//! ## 形式上的解耦
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

use agent_client_protocol::schema::{
    ContentBlock, PermissionOptionId, StopReason as AcpStopReason, ToolCallId, ToolCallUpdateFields,
};
use serde::{Deserialize, Serialize};

use crate::llm::Usage;
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
