//! 沙箱与权限策略层。
//!
//! v0 仅提供策略级决策（read-only / auto / full + 路径白名单），
//! OS 级隔离（landlock/seatbelt/seccomp）作为后续可插拔后端引入。

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]
