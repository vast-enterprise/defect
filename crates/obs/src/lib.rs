//! `defect-obs`: Observability stack.
//!
//! Extracts tracing initialization and (planned) Langfuse reporting from `defect-cli`
//! into a single crate. The CLI calls one entry point; future Langfuse / OTLP extensions
//! won't require changes to CLI assembly.
//!
//! Observability crate — tracing, metrics, and Langfuse integration.
//!
//! ## Current capabilities
//!
//! - [`tracing_init::init_tracing`]: process-level `tracing-subscriber` initialization.
//!
//! ## Planned
//!
//! - Langfuse reporting (implements `defect-agent`'s `SessionObserver`, one trace per
//!   turn, reuses `defect-http`'s `HttpStack` for ingestion requests).
//! - OTLP export (reuses `defect-config`'s `OtlpTracingConfig` scaffolding).

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

pub mod langfuse;
pub mod tracing_init;

pub use langfuse::{LangfuseObserver, LangfuseSetup, build_observer};
pub use tracing_init::init_tracing;
