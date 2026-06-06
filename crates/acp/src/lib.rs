//! ACP（Agent Client Protocol）服务端实现。
//!
//! 桥接 [`defect_agent`] 暴露的事件流与 ACP 线上协议；不参与业务逻辑，
//! 仅做协议适配与传输（v0 = stdio）。

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]
//!
//! 设计详见 `docs/inbound/acp-bridge.md`。

mod echo_provider;
pub mod fs;
mod project;
mod serve;
pub mod shell;

pub use echo_provider::EchoProvider;
pub use fs::AcpFsBackend;
pub use serve::{AcpError, serve, serve_on, serve_on_with_resume, serve_with_resume};
pub use shell::AcpShellBackend;
