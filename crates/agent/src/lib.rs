//! Defect agent core.
//!
//! Defines the abstractions that the agent main loop depends on: [`llm::LlmProvider`],
//! [`tool::Tool`],
//! [`event::AgentEvent`], and the session state container. Concrete provider/tool
//! implementations live in
//! sibling crates (`defect-llm`, `defect-tools`, `defect-mcp`, etc.) and are plugged in
//! through

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]
//! These traits are consumed here.
//!
//! Modules are organized by responsibility and are **exposed only at the module level**
//! (no flat re-exports at the lib root). Callers write `defect_agent::llm::LlmProvider`
//! rather than `defect_agent::LlmProvider`.

pub mod error;
pub mod event;
pub mod fs;
pub mod hooks;
pub mod http;
pub mod llm;
pub mod policy;
pub mod session;
pub mod shell;
pub mod tool;
