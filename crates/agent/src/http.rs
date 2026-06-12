//! HTTP client abstraction.
//!
//! The trait and its types now live in `defect-core` so `defect-http` can implement them
//! without depending on the agent runtime. Re-exported here so existing
//! `defect_agent::http::*` paths keep working — the CLI still injects
//! `Arc<dyn HttpClient>` into [`crate::session::AgentCore`], propagated through
//! [`crate::tool::ToolContext`] to tools.

pub use defect_core::http::*;
