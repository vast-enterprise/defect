//! Provider and model capability matrix.

use serde::{Deserialize, Serialize};

/// Provider-level capability matrix.
///
/// Model-level differences are expressed via [`ModelCapabilityOverrides`]; the main loop
/// merges them as needed:
/// a model-level `Some(_)` overrides the provider level, while `None` falls back to the
/// provider level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Tool calls (content_block contains `tool_use` / `tool_calls` fields).
    pub tool_calls: FeatureSupport,
    /// Multiple concurrent tool_use calls within a single turn.
    pub parallel_tool_calls: FeatureSupport,
    /// Chain of thought.
    pub thinking: FeatureSupport,
    /// Multimodal input (images).
    pub vision: FeatureSupport,
    /// Prompt cache.
    pub prompt_cache: FeatureSupport,
    /// Thinking content replay strategy. See [`ThinkingEcho`].
    pub thinking_echo: ThinkingEcho,
}

/// Model-level overrides. `None` means fall back to the provider-level [`Capabilities`]
/// field.
///
/// The field set is limited to properties that actually vary per model in practice, and
/// does not mechanically mirror [`Capabilities`]. Additional fields may be added later as
/// new differences emerge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilityOverrides {
    pub thinking: Option<FeatureSupport>,
    pub vision: Option<FeatureSupport>,
    pub prompt_cache: Option<FeatureSupport>,
    pub parallel_tool_calls: Option<FeatureSupport>,
    pub thinking_echo: Option<ThinkingEcho>,
}

/// Policy for replaying thinking content.
///
/// `Required` — the previous assistant turn's thinking must be included in the next
/// request (Anthropic extended thinking, DeepSeek-v4-pro). `Forbidden` — replay is
/// rejected by the server (DeepSeek-R1, official OpenAI o1/o3). `Optional` — the server
/// accepts either behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEcho {
    #[default]
    Forbidden,
    Required,
    Optional,
}

/// Tri-state feature support declaration.
///
/// Using a tri-state instead of `bool` allows expressing
/// [`FeatureSupport::PassthroughAsTool`] — pseudo-support via adaptation. Even though v0
/// has no implementation that produces this value, defining a tri-state from the start is
/// simpler than upgrading from `bool` to an enum later.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSupport {
    Supported,
    Unsupported,
    /// Passthrough support via adapter.
    ///
    /// For example, a provider may not natively support `web_search`, but the agent wraps
    /// it as a tool exposed to the LLM, thereby "pretending" to support it.
    PassthroughAsTool,
}

/// The set of hosted capabilities that the provider advertises.
///
/// Distinguished from [`Capabilities`]:
/// - [`Capabilities`] describes model-level abilities (thinking, vision, tool_calls,
///   etc.)
/// - [`HostedCapabilities`] describes the provider adapter's own implementation state:
///   whether the current adapter can declare hosted `web_search`, `fetch`, or
///   `code_execution` on the wire.
///
/// At session startup, this struct is obtained via
/// [`super::LlmProvider::hosted_capabilities`] and, together with
/// `capabilities.web_search.mode`, determines the source of web search capability for the
/// session. Note that local grep/glob tools (the `search` tool) are not part of the
/// capability layer and are managed separately by `[tools.search]`.
///
/// Native metadata returned by the model after a completions call.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedCapabilities {
    /// Whether the provider adapter supports hosted web search.
    ///
    /// The hosted tool version is hardcoded internally by the adapter to always use the
    /// latest (Anthropic `web_search_20260209`, OpenAI Responses API `web_search`); the
    /// agent is unaware of the specific version field.
    pub web_search: bool,
}

impl HostedCapabilities {
    /// Constructs from a single field. Cross-crate tests or adapter implementations need
    /// this entry point because the struct is `#[non_exhaustive]` and cannot be built
    /// with a struct literal directly.
    #[must_use]
    pub const fn with_web_search(web_search: bool) -> Self {
        Self { web_search }
    }
}
