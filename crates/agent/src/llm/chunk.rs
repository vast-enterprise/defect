//! Unified shape for streaming output chunks.

use serde::{Deserialize, Serialize};

/// provider 流式生成产生的单个事件。
///
/// 语义约束（见文档第 1.2 节）：
/// - 流的首事件总是 [`ProviderChunk::MessageStart`]
/// - [`ProviderChunk::Stop`] 是最后一个语义事件，之后只可能再出现
///   [`ProviderChunk::Usage`]
/// - 同一 `tool_use_id` 的 [`ProviderChunk::ToolUseArgsDelta`] 按到达
///   顺序拼接得到完整参数 JSON；并发 tool_use 按 id 分桶累积，不依赖
///   wire 顺序
/// - [`ProviderChunk::Usage`] 可多次到达，调用方应逐字段累加
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderChunk {
    /// 流的第一个事件，仅含会话级元信息。
    MessageStart { id: String, model: String },

    /// 助手文本增量。
    TextDelta { text: String },

    /// 思考链文本增量（Anthropic extended thinking / DeepSeek
    /// `reasoning_content` / o1-style 等的统一抽象）。
    ThinkingDelta { text: String },

    /// 思考链签名（Anthropic 多轮保留 thinking 时的校验数据）。
    /// 不是文本，不应与 [`ProviderChunk::ThinkingDelta`] 合并。
    ThinkingSignature { signature: String },

    /// 工具调用开始：声明一个新 tool_use，后续
    /// [`ProviderChunk::ToolUseArgsDelta`] 与 [`ProviderChunk::ToolUseEnd`]
    /// 都通过 `id` 关联到本次调用。
    ToolUseStart { id: String, name: String },

    /// 工具调用参数片段。
    ///
    /// `fragment` 为裸字节片段，**不保证是合法 JSON 子串**；调用方必须
    /// 在收到对应 [`ProviderChunk::ToolUseEnd`] 之后才能整体解析为 JSON。
    ToolUseArgsDelta { id: String, fragment: String },

    /// 工具调用结束：此 `id` 对应的所有 ArgsDelta 已发完。
    ToolUseEnd { id: String },

    /// 本次生成结束。
    Stop { reason: StopReason },

    /// token 使用统计。可能在一次流中多次到达，调用方应累加而非覆盖。
    Usage(Usage),
}

/// 生成结束的语义类别。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// 模型自然结束本轮。
    EndTurn,
    /// 命中 max_tokens。
    MaxTokens,
    /// 命中 stop_sequence。
    StopSequence,
    /// 模型请求工具调用，调用方应执行 tool_use 后续轮。
    ToolUse,
    /// 安全策略拒绝输出。
    Refusal,
}

/// token 使用统计。各字段为 `Option`，表达 provider 不报告该字段的情况。
///
/// provider 多次发送时调用方逐字段相加（`Option::None` 视为 0）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}
