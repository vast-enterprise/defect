//! Unified shape for streaming output chunks.

use serde::{Deserialize, Serialize};

/// A single event produced by a provider's streaming generation.
///
/// Semantic constraints:
/// - The first event of a stream is always [`ProviderChunk::MessageStart`]
/// - [`ProviderChunk::Stop`] is the last semantic event; only [`ProviderChunk::Usage`]
///   may appear after it
/// - [`ProviderChunk::ToolUseArgsDelta`] chunks with the same `tool_use_id` are
///   concatenated in arrival order to form the complete argument JSON; concurrent tool
///   uses are accumulated per id, independent of wire order
/// - [`ProviderChunk::Usage`] may arrive multiple times; callers should accumulate each
///   field
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderChunk {
    /// First event of the stream, containing only session-level metadata.
    MessageStart { id: String, model: String },

    /// Incremental assistant text.
    TextDelta { text: String },

    /// Thinking-chain text delta (unified abstraction for Anthropic extended thinking,
    /// DeepSeek `reasoning_content`, o1-style reasoning, etc.).
    ThinkingDelta { text: String },

    /// Thinking chain signature (Anthropic verification data for preserving thinking
    /// across multiple turns).
    /// Not text; should not be merged with [`ProviderChunk::ThinkingDelta`].
    ThinkingSignature { signature: String },

    /// Tool call start: declares a new `tool_use`; subsequent
    /// [`ProviderChunk::ToolUseArgsDelta`] and [`ProviderChunk::ToolUseEnd`]
    /// are linked to this call via `id`.
    ToolUseStart { id: String, name: String },

    /// A fragment of tool-use arguments.
    ///
    /// `fragment` is a raw byte slice; it is **not guaranteed to be a valid JSON
    /// substring**. Callers must wait until the corresponding
    /// [`ProviderChunk::ToolUseEnd`] is received before parsing the complete payload as
    /// JSON.
    ToolUseArgsDelta { id: String, fragment: String },

    /// Tool call end: all `ArgsDelta` for this `id` have been sent.
    ToolUseEnd { id: String },

    /// The generation has ended.
    Stop { reason: StopReason },

    /// Token usage statistics. May arrive multiple times in a single stream; callers
    /// should accumulate rather than overwrite.
    Usage(Usage),
}

/// Semantic category of generation termination.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model naturally ended the current turn.
    EndTurn,
    /// Hit the max_tokens limit.
    MaxTokens,
    /// Hit stop_sequence.
    StopSequence,
    /// Model requested a tool call; the caller should proceed with a subsequent
    /// `tool_use` turn.
    ToolUse,
    /// Refusal due to safety policy.
    Refusal,
}

/// Token usage statistics. Each field is `Option` to indicate that the provider does not
/// report that field.
///
/// When the provider sends multiple responses, the caller should sum each field
/// individually (treating `Option::None` as 0).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}
