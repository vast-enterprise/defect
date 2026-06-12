//! Cross-module error utilities.
//!
//! [`BoxError`] now lives in `defect-core` so provider/transport crates can use it without
//! depending on the agent runtime. Re-exported here so existing `defect_agent::error::*`
//! paths keep working.

pub use defect_core::error::BoxError;
