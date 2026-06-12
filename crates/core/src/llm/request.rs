//! Provider request parameters.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::capability::HostedCapabilities;
use crate::tool::ToolSchema;

/// Input for a single generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// System prompt. Uses `Arc<str>` instead of `String`: the request is `clone`d in the
    /// turn main loop (sent to the provider, fanned out with the `LlmCallStarted` event),
    /// and deep-copying a long system prompt repeatedly is expensive; `Arc` reduces clone
    /// to a reference-count bump.
    pub system: Option<Arc<str>>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingParams,
    /// The set of hosted capabilities the provider may use in this turn.
    ///
    /// Determined once at session startup (see
    ///
    /// Reused from the session marker when assembling each turn's request.
    /// The provider adapter uses this to decide whether to advertise a hosted tool on the
    /// wire.
    #[serde(default)]
    pub hosted_capabilities: HostedCapabilities,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// Content fragments. Uses `Arc<[_]>` instead of `Vec`: cloning the entire messages
    /// list (e.g. for history `snapshot()`, `complete()`, or fan-out of `LlmCallStarted`
    /// events) is expensive with deep copies under long contexts; `Arc` reduces clone to
    /// reference counting. Messages are read-only once in history, so this is
    /// appropriate.
    pub content: Arc<[MessageContent]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// A piece of content inside a message body.
///
/// Both "the model requesting a tool call in the previous turn" and "the tool result
/// reported back in the current turn" are placed in the `messages` array, matching the
/// shape of the Anthropic Messages API. OpenAI uses separate `assistant message with
/// tool_calls` + `tool message`; the codec translates between the two during encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
    },
    /// The thinking chain produced by the model in the previous turn. Only present in
    /// [`Role::Assistant`] messages.
    ///
    /// `signature` is the anti-forgery signature for Anthropic extended thinking: it must
    /// be kept together with the text. For providers that echo plain text (e.g.
    /// DeepSeek-v4-pro), this is [`None`].
    Thinking {
        text: String,
        signature: Option<String>,
    },
    /// Tool call from a previous turn: when sending a request, include both the prior
    /// `tool_use` and `tool_result` in `messages` so the provider can reconstruct the
    /// context.
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
    /// Multimodal input. *(P2)*
    Image {
        mime: String,
        data: ImageData,
    },
    /// Provider-hosted capability activity (e.g. hosted web_search, hosted code
    /// execution).
    /// The agent does not interpret `payload`; it passes it through when retrying the
    /// same
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

/// The kind of hosted activity. Only appears inside [`MessageContent::ProviderActivity`].
///
/// Adding `CodeExecution` / `ImageGeneration` etc. later is a deliberate breaking change:
/// downstream provider crates that depend on `defect-core` should re-compile and handle
/// the new variant rather than silently fall through a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderActivityKind {
    /// Hosted web search.
    Search,
}

/// Tool result payload. The codec converts it for the wire during serialization: some
/// wires only support strings, so they stringify [`ToolResultBody::Json`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultBody {
    Text {
        text: String,
    },
    Json {
        value: serde_json::Value,
    },
    /// Multimodal tool result: a mix of text and image blocks. Used by `read_file` for
    /// images and future screenshot tools.
    ///
    /// Materialization per provider is handled by the codec, with different shapes:
    /// - Anthropic's `tool_result` block natively supports images; just insert each block
    ///   as-is.
    /// - OpenAI's tool message only accepts text — the codec strips image blocks and
    ///   attaches them to the following user message, leaving only text (including
    ///   placeholder hints) in the tool message.
    Content {
        blocks: Vec<ToolResultContent>,
    },
}

/// A single block inside [`ToolResultBody::Content`]. Text follows the same semantics as
/// [`ToolResultBody::Text`]; images reuse the `(mime, data)` shape from
/// [`MessageContent::Image`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { mime: String, data: ImageData },
}

/// Placeholder shape for multimodal image payloads. The exact shape is not yet
/// finalized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageData {
    /// Base64-encoded image bytes.
    Base64 { encoded: String },
    /// A remote URL.
    Url { url: String },
}

/// Tool selection strategy.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides on its own.
    #[default]
    Auto,
    /// Forces at least one tool to be called.
    Required,
    /// Force the model to call the specified tool.
    Named { name: String },
    /// Disables tool calls; only text output is allowed.
    None,
}

/// Sampling parameters.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub thinking: ThinkingConfig,
    /// The `reasoning_effort` level in the OpenAI-compatible protocol. When `Some(_)`,
    /// the codec writes it directly to the wire; when `None`, the codec falls back to
    /// deriving it from [`Self::thinking`].
    ///
    /// This is the **runtime authoritative representation** of the value — it can be
    /// switched per-session (ACP `session/set_config_option`, category=ThoughtLevel). The
    /// config layer has its own `defect_config::ReasoningEffort` for deserialization,
    /// which is converted into this enum during assembly and placed into the initial
    /// `SamplingParams`. Providers that do not support this concept should ignore this
    /// field.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Runtime-level enum for the OpenAI-compatible `reasoning_effort` protocol.
///
/// Maps 1:1 to the official OpenAI wire enum: `xhigh` is only supported after
/// `gpt-5.1-codex-max`, and `none` only after `gpt-5.1`. This layer does not distinguish
/// between models; the value is passed through as-is for upstream validation. The
/// `defect-llm` wire codec imports this enum for materialization mapping.
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

/// Thinking chain configuration. Providers that do not support the concept of a thinking
/// chain should ignore the budget field of `Enabled`, or report
/// [`super::FeatureSupport::Unsupported`] in the capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ThinkingConfig {
    #[default]
    Disabled,
    /// Enable thinking chain; `budget_tokens` is only used by providers that support
    /// budgets, such as Anthropic.
    Enabled { budget_tokens: Option<u32> },
}
