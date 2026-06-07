//! LLM provider 适配层。
//!
//! 实现 [`defect_agent::llm::LlmProvider`]，对接 Anthropic 与 OpenAI
//! LLM provider abstraction. Architecture: protocol layer + vendor layer.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod protocol;
pub mod provider;
pub mod wire;
