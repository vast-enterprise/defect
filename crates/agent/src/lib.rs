//! Defect agent 核心。
//!
//! 定义 agent 主循环所依赖的抽象：[`llm::LlmProvider`]、[`tool::Tool`]、
//! [`event::Event`]，以及会话状态容器。具体的 provider / tool 实现位于
//! sibling crate（`defect-llm`、`defect-tools`、`defect-mcp` 等），通过

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]
//! 这里的 trait 接入。
//!
//! 模块按职责切分，对外**仅以模块为单位暴露**（不在 lib 顶层平铺 re-export），
//! 调用方写 `defect_agent::llm::LlmProvider` 而非 `defect_agent::LlmProvider`。

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
