//! LLM provider abstraction.
//!
//! The protocol-level types (chunk / request / model / capability / error / provider) now
//! live in `defect-core::llm` so `defect-llm` can implement providers without depending on
//! the agent runtime. They are re-exported here so existing `defect_agent::llm::*` paths
//! keep working.
//!
//! The `registry` module stays in this crate: it depends on the session capabilities
//! config (a runtime concern) and is not needed by the provider implementations.

pub(crate) mod registry;

// Protocol types from defect-core, re-exported under the original `defect_agent::llm::*`
// paths so call sites are unaffected.
pub use defect_core::llm::{
    Capabilities, CompletionRequest, FeatureSupport, HostedCapabilities, ImageData, LlmProvider,
    Message, MessageContent, ModelCapabilityOverrides, ModelInfo, ProtocolId, ProviderActivityKind,
    ProviderChunk, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream, RateLimitScope,
    ReasoningEffort, RetryAction, RetryHint, Role, SamplingParams, StopReason, ThinkingConfig,
    ThinkingEcho, TimeoutPhase, ToolChoice, ToolResultBody, ToolResultContent, Usage,
};

// Registry stays local (depends on `crate::session::SessionCapabilitiesConfig`).
pub use registry::{ModelCandidate, ProviderEntry, ProviderRegistry, ProviderRegistryError};
