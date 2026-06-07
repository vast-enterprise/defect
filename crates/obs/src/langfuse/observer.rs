//! `LangfuseObserver` reports each session's
//! [`AgentEvent`](defect_agent::event::AgentEvent) stream to Langfuse.
//!
//! The shape follows `defect-storage::StorageObserver` (`crates/storage/src/lib.rs`):
//! in [`SessionObserver::on_session_created`], `session.subscribe()` obtains an
//! independent
//! mpsc stream, `tokio::spawn` a consumer task that feeds each event to
//! [`TraceProjector`] for
//! translation and then to [`LangfuseIngest`] for reporting; after the stream ends
//! (session drop),
//! `flush` any remaining data.
//!
//! Key difference from storage: **degradable dropping**. Storage's slow consumption
//! backpressures the main loop
//! ("no-drop" semantics); Langfuse cannot do that — so the consumer loop only does
//! `enqueue` (non-blocking) +
//! lightweight translation. All real network I/O lives in [`LangfuseIngest`]'s background
//! task, which drops when full.
//! Any Langfuse failure must NOT affect the agent.

use std::sync::Arc;

use defect_agent::error::BoxError;
use defect_agent::session::{Session, SessionCreateInfo, SessionObserver};
use futures::StreamExt;

use super::ingest::LangfuseIngest;
use super::projector::TraceProjector;

/// Langfuse reporting observer. `Clone` is cheap (internally `Arc`).
#[derive(Clone)]
pub struct LangfuseObserver {
    ingest: LangfuseIngest,
}

impl LangfuseObserver {
    /// Constructs a new observer from an already-started ingester. The ingester's
    /// background task is launched by [`LangfuseIngest::spawn`]; this observer only wires
    /// in the per-session event stream.
    #[must_use]
    pub fn new(ingest: LangfuseIngest) -> Self {
        Self { ingest }
    }
}

impl SessionObserver for LangfuseObserver {
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError> {
        let mut events = session.subscribe();
        let ingest = self.ingest.clone();
        let session_id = info.id.0.to_string();

        tokio::spawn(async move {
            let mut projector = TraceProjector::new(session_id);
            // Each ingestion event gets a random UUID as its envelope ID / trace ID.
            let mut new_id = || uuid::Uuid::new_v4().to_string();

            while let Some(event) = events.next().await {
                // Use the receive time as an approximation of the event time (AgentEvent
                // has no timestamp; see design doc §3.4).
                let now = chrono::Utc::now().to_rfc3339();
                for ev in projector.project(event, &now, &mut new_id) {
                    ingest.enqueue(ev);
                }
            }

            // On stream end (session drop / process exit): best-effort flush of remaining
            // telemetry.
            ingest.flush().await;
        });

        Ok(())
    }
}
