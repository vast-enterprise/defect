//! 内置工具集。
//!
//! 实现 [`defect_agent`] 中的 `Tool` trait，提供文件读写、bash 执行、
//! 文本检索等基础能力。每个工具放在独立子模块中，后续按需启用。

pub mod bash;

pub use bash::BashTool;
