//! `defect-core` — foundational shared types for the defect workspace.
//!
//! This crate holds the **pure types and traits** that both the agent runtime
//! (`defect-agent`) and the provider/transport crates (`defect-llm`, `defect-http`) need,
//! with **zero dependency on the agent runtime** (no session / turn loop / hooks). Pulling
//! these out of `defect-agent` lets a provider or HTTP-stack crate be depended on
//! independently — e.g. via a git dependency — without dragging in the whole session
//! machinery.
//!
//! `defect-agent` re-exports everything here under its original paths
//! (`defect_agent::error`, `defect_agent::llm::*`, `defect_agent::http`,
//! `defect_agent::tool::ToolSchema`), so existing call sites are unaffected.
//!
//! What lives here: [`error::BoxError`], the LLM protocol types in [`llm`], the
//! [`http::HttpClient`] trait, and [`tool::ToolSchema`]. What does **not**: the
//! `ProviderRegistry` (depends on session capabilities config), the full `Tool` trait /
//! `ToolContext`, events, policy, and the session runtime — those stay in `defect-agent`.

pub mod error;
pub mod http;
pub mod llm;
pub mod tool;
