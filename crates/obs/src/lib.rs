//! `defect-obs`：可观测性栈。
//!
//! 把 tracing 初始化与（规划中的）Langfuse 上报等遥测能力从 `defect-cli`
//! 抽出，集中到一个 crate。cli 只调一个入口，后续扩 Langfuse / OTLP 不再
//! 改 cli 装配。
//!
//! Observability crate — tracing, metrics, and Langfuse integration.
//!
//! ## 当前能力
//!
//! - [`tracing_init::init_tracing`]：进程级 `tracing-subscriber` 初始化。
//!
//! ## 规划中
//!
//! - Langfuse 上报（实现 `defect-agent` 的 `SessionObserver`，每 turn 一个
//!   trace，复用 `defect-http` 的 `HttpStack` 发 ingestion 请求）。
//! - OTLP 导出（复用 `defect-config` 的 `OtlpTracingConfig` 脚手架）。

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod langfuse;
pub mod tracing_init;

pub use langfuse::{LangfuseObserver, LangfuseSetup, build_observer};
pub use tracing_init::init_tracing;
