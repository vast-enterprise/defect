//! Provider request parameters.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::capability::HostedCapabilities;
use crate::tool::ToolSchema;

/// 一次完整生成请求的输入。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// 系统提示词。用 `Arc<str>` 而非 `String`：请求在 turn 主循环里会被
    /// `clone`（发给 provider、随 `LlmCallStarted` 事件 fan-out），长 system
    /// prompt 反复深拷贝代价高；`Arc` 让 clone 退化成引用计数。
    pub system: Option<Arc<str>>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingParams,
    /// 本轮允许 provider 自行使用的 hosted capability 集合。
    ///
    /// session 启动期一次性裁决（详见
    ///
    /// 每轮 turn 装配请求时直接复用 session 上的标记。
    /// provider adapter 据此决定是否在 wire 上声明 hosted tool。
    #[serde(default)]
    pub hosted_capabilities: HostedCapabilities,
}

/// 对话历史中的一条消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// 内容片段。用 `Arc<[_]>` 而非 `Vec`：历史 `snapshot()`、`complete()` 的
    /// 请求 clone、`LlmCallStarted` 事件 fan-out 都会整体 clone messages，长上下文
    /// 下深拷贝代价高；`Arc` 让 clone 退化成引用计数。进了历史即只读，正合适。
    pub content: Arc<[MessageContent]>,
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
    /// Provider-hosted capability activity (e.g. hosted web_search, hosted code execution).
    /// The agent does not interpret `payload`: it passes it through when retrying the same
    /// provider, or the codec decides how to degrade when switching providers.
    ///
    /// `payload` uses `#[serde(skip)]`: it is dropped when persisting across processes;
    /// on session resume, if the model re-triggers the same hosted call, a new hosted
    /// call is made without relying on the old payload.
    ProviderActivity {
        provider_id: String,
        kind: ProviderActivityKind,
        #[serde(skip)]
        payload: serde_json::Value,
    },
}

/// hosted activity 的种类。仅在 [`MessageContent::ProviderActivity`]
/// 内出现。
///
/// `#[non_exhaustive]` 是为后续追加 `CodeExecution` / `ImageGeneration`
/// 等不构成 breaking。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderActivityKind {
    /// hosted web search。
    Search,
}

/// 工具结果载荷。codec 在序列化时按 wire 需要转换：部分 wire 只支持
/// 字符串，会把 [`ToolResultBody::Json`] stringify。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultBody {
    Text {
        text: String,
    },
    Json {
        value: serde_json::Value,
    },
    /// 多模态工具结果：文本与图片块混排。`read_file` 读图片、未来的截图
    /// 工具等走这条。
    ///
    /// 各 provider 的物化由 codec 负责，形态不一：
    /// - Anthropic 的 `tool_result` 块原生支持 image，逐块塞进去即可
    /// - OpenAI 的 tool message 只接受文本——codec 把图片块剥出来挂到
    ///   紧随其后的 user message 里，tool message 仅留文本（含占位提示）
    Content {
        blocks: Vec<ToolResultContent>,
    },
}

/// [`ToolResultBody::Content`] 里的单块。文本沿用 [`ToolResultBody::Text`]
/// 的语义；图片复用 [`MessageContent::Image`] 的 `(mime, data)` 形态。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { mime: String, data: ImageData },
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
    /// OpenAI 兼容协议的 `reasoning_effort` 等级。`Some(_)` 时 codec 直接
    /// 写入 wire；`None` 时 codec 回退到从 [`Self::thinking`] 推导。
    ///
    /// 这是该值的**运行时权威表示**——能 per-session 切换（ACP
    /// `session/set_config_option`，category=ThoughtLevel）。配置文件层另有
    /// `defect_config::ReasoningEffort` 负责反序列化，装配时翻成本枚举填进
    /// 初始 `SamplingParams`。不支持该概念的 provider 应忽略此字段。
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// OpenAI 兼容协议 `reasoning_effort` 的运行时等级枚举。
///
/// 与 OpenAI 官方 wire 枚举 1:1 对齐：`xhigh` 仅 `gpt-5.1-codex-max` 之后
/// 支持，`none` 仅 `gpt-5.1` 之后支持；本层不区分模型，原样下发由上游
/// 校验。`defect-llm` 的 wire codec import 本枚举做物化映射。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
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
