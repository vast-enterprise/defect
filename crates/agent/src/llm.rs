//! LLM provider 抽象。
//!
//! 设计沉淀于 `docs/internal/llm-trait.md`。
//! 子模块按职责切分（chunk / request / model / capability / error / provider），
//! 外部仅通过本模块顶层访问公共类型，子模块本身对 crate 外不可见。

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
