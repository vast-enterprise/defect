//! ACP (Agent Client Protocol) server implementation.
//!
//! Bridges the event stream exposed by [`defect_agent`] with the ACP wire protocol; does
//! not participate in business logic,
//! only protocol adaptation and transport (v0 = stdio).

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]
//! ACP (Agent Client Protocol) bridge — translates between internal events and ACP wire
//! messages.

mod echo_provider;
pub mod fs;
mod project;
mod serve;
pub mod shell;

pub use echo_provider::EchoProvider;
pub use fs::AcpFsBackend;
pub use serve::{AcpError, serve, serve_on, serve_on_with_resume, serve_with_resume};
pub use shell::AcpShellBackend;
