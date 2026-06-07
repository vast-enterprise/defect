//! Protocol layer: bidirectional conversion between wire JSON and [`defect_agent::llm`]
//! internal representations.
//!
//! This layer handles only encoding/decoding; it does not include transport, auth, or URL
//! templates. Each submodule corresponds to a [`defect_agent::llm::ProtocolId`].

// anthropic_messages is shared by anthropic and bedrock (bedrock uses the Anthropic
// Messages shape).
#[cfg(any(feature = "provider-anthropic", feature = "provider-bedrock"))]
pub mod anthropic_messages;
// deepseek_chat is used only by deepseek and depends on openai_chat.
#[cfg(feature = "provider-deepseek")]
pub mod deepseek_chat;
// openai_chat is shared by openai and deepseek.
#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
pub mod openai_chat;
