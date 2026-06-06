//! LLM provider 适配层。
//!
//! 实现 [`defect_agent::llm::LlmProvider`]，对接 Anthropic 与 OpenAI
//! 兼容接口。架构按"协议层 + 厂商层"切分，详见 `docs/outbound/llm.md`。

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod protocol;
pub mod provider;
pub mod wire;
