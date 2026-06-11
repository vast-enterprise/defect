//! Built-in tool set.
//!
//! Implements the `Tool` trait from [`defect_agent`], providing basic capabilities such
//! as file I/O, bash execution, and text search. Each tool resides in its own submodule
//! and can be enabled as needed.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod bash;
pub mod fetch;
pub mod fs;
pub mod search;
pub mod shell;

pub use bash::BashTool;
pub use fetch::FetchTool;
pub use fs::{EditFileTool, LocalFsBackend, ReadFileTool, WriteFileTool};
pub use search::SearchTool;
pub use shell::{DEFAULT_MAX_OUTPUT_BYTES, LocalShellBackend};
