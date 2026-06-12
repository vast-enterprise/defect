//! LLM provider abstraction — protocol-level types shared across the workspace.
//!
//! Sub-modules are split by responsibility (chunk / request / model / capability /
//! error / provider). External access goes through this module's re-exports.
//!
//! The `registry` module is **not** here — it depends on the session capabilities config
//! (a runtime concern) and lives in `defect-agent::llm::registry`. This module holds only
//! the provider-protocol types that `defect-llm` needs to implement a provider without
//! pulling in the agent runtime.

pub mod capability;
pub mod chunk;
pub mod error;
pub mod model;
pub mod provider;
pub mod request;

pub use capability::{
    Capabilities, FeatureSupport, HostedCapabilities, ModelCapabilityOverrides, ThinkingEcho,
};
pub use chunk::{ProviderChunk, StopReason, Usage};
pub use error::{
    ProviderError, ProviderErrorKind, RateLimitScope, RetryAction, RetryHint, TimeoutPhase,
};
pub use model::{ModelInfo, ProtocolId, ProviderInfo};
pub use provider::{LlmProvider, ProviderStream};
pub use request::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderActivityKind, ReasoningEffort,
    Role, SamplingParams, ThinkingConfig, ToolChoice, ToolResultBody, ToolResultContent,
};
