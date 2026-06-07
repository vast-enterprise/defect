//! Provider and model metadata.

use serde::{Deserialize, Serialize};

use super::capability::ModelCapabilityOverrides;

/// 厂商元信息（厂商名、API 风格、tracing 标签等）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// 厂商标识（"anthropic" / "openai" / "bedrock" / ...）。
    pub vendor: String,
    /// 底层协议。
    pub protocol: ProtocolId,
    /// 用于 tracing 的人读名（"Anthropic Claude"）。
    pub display_name: String,
}

/// 底层协议标识。`defect-llm::protocol` 中的 codec 与此一一对应。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolId {
    AnthropicMessages,
    OpenAiChat,
}

/// 单个模型的元信息。
///
/// `context_window` 与 `max_output_tokens` 用 `Option` 是因为部分
/// provider 不公开这些字段；调用方按未知处理（不强制裁剪）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub deprecated: bool,
    /// 此模型相对 provider 全局 [`super::Capabilities`] 的差异。
    pub capabilities_overrides: ModelCapabilityOverrides,
}
