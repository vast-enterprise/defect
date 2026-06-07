//! `defect-cli` assembly library — reusable by the `defect` binary and downstream
//! developers.
//!
//! This crate aims to make "assembling an ACP server" a few lines of code: it translates
//! the typed configuration from `defect-config` into the runtime structures needed by
//! `defect-agent`, `defect-llm`, `defect-tools`, `defect-mcp`, and other modules.
//!
//! ## Extension points for downstream development
//!
//! - [`args::CliArgs`] / [`args::CliArgs::to_overrides`]: standard CLI arguments
//! - [`providers::build_registry`]: assembles [`ProviderRegistry`] + [`TurnConfig`]
//! - [`http_stack::build_http_stack_config`]: translates typed HTTP configuration into
//!   `defect_http::HttpStackConfig`
//! - [`tools::build_process_tools`] / [`mcp_servers::build_default_mcp_servers`]
//! - [`hooks::build_engine_arc`]: assembles the hook engine
//! - [`policy::build_policy`] / [`paths::default_sessions_root`]
//! - tracing initialization has moved to `defect-obs` (`defect_obs::init_tracing`)
//!
//! The main binary `src/bin/cli.rs` only performs assembly and holds no helper
//! implementations — downstream consumers can replace any step without forking the entire
//! helper set.
//!
//! [`ProviderRegistry`]: defect_agent::llm::ProviderRegistry
//! [`TurnConfig`]: defect_agent::session::TurnConfig

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod args;
pub mod assembly;
pub mod hooks;
pub mod http_stack;
pub mod mcp_servers;
pub mod observability;
#[cfg(feature = "oneshot")]
pub mod oneshot;
pub mod paths;
pub mod policy;
pub mod providers;
#[cfg(feature = "repl")]
pub mod repl;
#[cfg(any(feature = "repl", feature = "oneshot"))]
pub mod session_open;
pub mod tools;

#[cfg(test)]
mod tests;
