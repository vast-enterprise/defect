//! LLM provider abstraction layer.
//!
//! Implements [`defect_core::llm::LlmProvider`] for Anthropic and OpenAI.
//! Architecture: protocol layer + vendor layer.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod protocol;
pub mod provider;
pub mod wire;
