//! Vendor layer: implements [`defect_core::llm::LlmProvider`].
//!
//! The vendor layer handles transport, auth, URL templates, capability declarations,
//! error hints,
//! model metadata tables, and vendor-specific tracing. Each vendor gets its own
//! submodule.

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-bedrock")]
pub mod bedrock;
#[cfg(feature = "provider-deepseek")]
pub mod deepseek;
#[cfg(feature = "provider-openai")]
pub mod openai;
