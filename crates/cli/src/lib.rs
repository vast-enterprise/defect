//! `defect-cli` 装配库——给 `defect` 二进制与下游二次开发者复用。
//!
//! 本 crate 的目标是让"组装一个 ACP server"是几行代码的事：把
//! `defect-config` 的 typed 配置翻译成 `defect-agent` / `defect-llm` /
//! `defect-tools` / `defect-mcp` 等模块需要的运行期结构。
//!
//! ## 二次开发入口
//!
//! - [`args::CliArgs`] / [`args::CliArgs::to_overrides`]：标准 CLI 参数
//! - [`providers::build_registry`]：装配 [`ProviderRegistry`] + [`TurnConfig`]
//! - [`http_stack::build_http_stack_config`]：把 typed http 配置翻成
//!   `defect_http::HttpStackConfig`
//! - [`tools::build_process_tools`] / [`mcp_servers::build_default_mcp_servers`]
//! - [`hooks::build_engine_arc`]：装配 hook 引擎
//! - [`policy::build_policy`] / [`paths::default_sessions_root`]
//! - tracing 初始化已搬到 `defect-obs`（`defect_obs::init_tracing`）
//!
//! 主二进制 `src/bin/cli.rs` 仅做拼装，不持有任何 helper 实现——下游可以
//! 替换其中任何一步而不必 fork 整套 helper。
//!
//! [`ProviderRegistry`]: defect_agent::llm::ProviderRegistry
//! [`TurnConfig`]: defect_agent::session::TurnConfig

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]

pub mod args;
pub mod hooks;
pub mod http_stack;
pub mod mcp_servers;
pub mod observability;
pub mod paths;
pub mod policy;
pub mod providers;
pub mod tools;

#[cfg(test)]
mod test;
