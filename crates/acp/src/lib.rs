//! ACP（Agent Client Protocol）服务端实现。
//!
//! 桥接 [`defect_agent`] 暴露的事件流与 ACP 线上协议；不参与业务逻辑，
//! 仅做协议适配与传输（v0 = stdio）。
//!
//! 设计详见 `docs/inbound/acp-bridge.md`。

mod echo_provider;
mod project;
mod serve;

pub use echo_provider::EchoProvider;
pub use serve::{serve, serve_on, AcpError};
