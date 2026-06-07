//! LLM provider abstraction.
//!
//! Sub-modules are split by responsibility (chunk / request / model / capability /
//! error / provider). External access goes through this module's re-exports;
//! sub-modules are not visible outside the crate.

pub(crate) mod capability;
pub(crate) mod chunk;
pub(crate) mod error;
pub(crate) mod model;
pub(crate) mod provider;
pub(crate) mod registry;
pub(crate) mod request;

pub use capability::{
    Capabilities, FeatureSupport, HostedCapabilities, ModelCapabilityOverrides, ThinkingEcho,
};
pub use chunk::{ProviderChunk, StopReason, Usage};
pub use error::{
    ProviderError, ProviderErrorKind, RateLimitScope, RetryAction, RetryHint, TimeoutPhase,
};
pub use model::{ModelInfo, ProtocolId, ProviderInfo};
pub use provider::{LlmProvider, ProviderStream};
pub use registry::{ModelCandidate, ProviderEntry, ProviderRegistry, ProviderRegistryError};
pub use request::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderActivityKind, ReasoningEffort,
    Role, SamplingParams, ThinkingConfig, ToolChoice, ToolResultBody, ToolResultContent,
};
