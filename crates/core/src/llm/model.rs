//! Metadata for providers and models.

use serde::{Deserialize, Serialize};

use super::capability::ModelCapabilityOverrides;

/// Provider metadata (vendor name, API style, tracing labels, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Vendor identifier (e.g. "anthropic", "openai", "bedrock", …).
    pub vendor: String,
    /// Underlying protocol.
    pub protocol: ProtocolId,
    /// Human-readable name for tracing (e.g. "Anthropic Claude").
    pub display_name: String,
}

/// Protocol identifier. Each variant corresponds one-to-one with a codec in
/// `defect-llm::protocol`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolId {
    AnthropicMessages,
    OpenAiChat,
}

/// Metadata for a single model.
///
/// `context_window` and `max_output_tokens` are `Option` because some providers do not
/// expose these fields; callers should treat them as unknown (no forced truncation).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub deprecated: bool,
    /// Differences from the provider-wide [`super::Capabilities`] for this model.
    pub capabilities_overrides: ModelCapabilityOverrides,
}
