//! Langfuse integration — uploads an `AgentEvent` stream as Langfuse traces, generations,
//! and spans.
//!
//! Module layout:
//! - [`model`]: wire types for the ingestion API.
//! - [`ingest`]: buffered batching + background reporting (with optional drop
//!   degradation).
//! - [`projector`]: translates `AgentEvent` into ingestion events.
//! - [`observer`]: `SessionObserver` implementation that subscribes to the event stream
//!   per session.

use std::time::Duration;

use defect_http::HttpStack;

pub mod ingest;
pub mod model;
pub mod observer;
pub mod projector;

pub use ingest::{IngestConfig, LangfuseIngest};
pub use observer::LangfuseObserver;
pub use projector::TraceProjector;

/// Default Langfuse host.
pub const DEFAULT_HOST: &str = "https://cloud.langfuse.com";
/// Default periodic flush interval.
pub const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(2);
/// Default maximum number of events per batch.
pub const DEFAULT_MAX_BATCH: usize = 100;
/// Inbound channel capacity (backpressure boundary; drops when full, does not
/// backpressure the main loop).
pub const DEFAULT_QUEUE_CAPACITY: usize = 1024;

/// Parsed Langfuse upload parameters (credentials already validated as non-empty).
pub struct LangfuseSetup {
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
    pub flush_interval: Duration,
    pub max_batch: usize,
}

/// Starts the reporter with a [`LangfuseSetup`] and an already-built [`HttpStack`],
/// returning an observer.
///
/// The reporter's background flush task is started here; the returned
/// [`LangfuseObserver`] is passed to `AgentCore::observe_session`.
#[must_use]
pub fn build_observer(setup: LangfuseSetup, http: HttpStack) -> LangfuseObserver {
    let ingest = LangfuseIngest::spawn(IngestConfig {
        http,
        host: setup.host,
        public_key: setup.public_key,
        secret_key: setup.secret_key,
        max_batch: setup.max_batch,
        flush_interval: setup.flush_interval,
        queue_capacity: DEFAULT_QUEUE_CAPACITY,
    });
    LangfuseObserver::new(ingest)
}

// Tests heavily use chained indexing like `value["body"]["x"]` plus `.unwrap()`
// assertions — these are far more readable than `.get().expect()`, and panicking on test
// failure is exactly the desired behavior. Therefore, `indexing_slicing` and
// `unwrap_used` are suppressed for the test module.
#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests;
