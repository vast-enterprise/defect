//! provider 入参。
//!
//! 设计详见 `docs/internal/llm-trait.md` 第 3 节。

use serde::{Deserialize, Serialize};

use crate::tool::ToolSchema;

/// 一次完整生成请求的输入。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingParams,
}

/// 对话历史中的一条消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<MessageContent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// 消息体内的内容片段。
///
/// 把"上一轮模型说要调工具"和"这一轮告诉模型工具结果"都放在 messages
/// 数组中，与 Anthropic Messages API 形态一致；OpenAI 是分离的"assistant
/// message with tool_calls" + "tool message"，由 codec 在编码时翻译。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
    },
    /// 上一轮模型产出的思考链。仅出现在 [`Role::Assistant`] 消息里。
    ///
    /// `signature` 是 Anthropic extended thinking 的防伪签名：必须与
    /// 文本同进同出。DeepSeek-v4-pro 等纯文本 echo 的 provider 这里
    /// 为 [`None`]。
    Thinking {
        text: String,
        signature: Option<String>,
    },
    /// 历史轮次的工具调用：发出请求时把上一轮 tool_use 与 tool_result
    /// 都放在 messages 里，让 provider 重建上下文。
    ToolUse {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        output: ToolResultBody,
        is_error: bool,
    },
    /// 多模态输入。*(P2)*
    Image {
        mime: String,
        data: ImageData,
    },
}

/// 工具结果载荷。codec 在序列化时按 wire 需要转换：部分 wire 只支持
/// 字符串，会把 [`ToolResultBody::Json`] stringify。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultBody {
    Text { text: String },
    Json { value: serde_json::Value },
}

/// 多模态图片载荷的占位形态。具体形态在 v0 之后再敲定。*(P2)*
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageData {
    /// base64 编码的图片字节。
    Base64 { encoded: String },
    /// 远程 URL。
    Url { url: String },
}

/// 工具选择策略。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    /// 模型自行决定。
    #[default]
    Auto,
    /// 强制至少调用一个工具。
    Required,
    /// 强制调用指定工具。
    Named { name: String },
    /// 禁止工具调用，只产文本。
    None,
}

/// 采样参数。
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub thinking: ThinkingConfig,
}

/// 思考链配置。不支持思考链概念的 provider 应忽略 `Enabled` 的预算字段，
/// 或在能力矩阵中报告 [`super::FeatureSupport::Unsupported`]。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ThinkingConfig {
    #[default]
    Disabled,
    /// 启用思考链；`budget_tokens` 仅 Anthropic 等支持预算的 provider 使用。
    Enabled { budget_tokens: Option<u32> },
}
