//! `fs` 内置工具家族：`read_file` / `write_file` / `edit_file`。
//!
//! 设计详见 `docs/internal/tools-fs.md`。

mod edit;
mod local_backend;
mod read;
mod write;

#[cfg(test)]
mod tests;

pub use edit::EditFileTool;
pub use local_backend::{LocalFsBackend, MAX_FS_BYTES};
pub use read::ReadFileTool;
pub use write::WriteFileTool;
