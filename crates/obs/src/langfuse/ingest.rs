//! Langfuse batch uploader.
//!
//! Pipeline: `enqueue` (non-blocking) → bounded mpsc → background flush task (batch on N
//! items or every T seconds)
//! → reuses `defect-http`'s [`HttpStack`] POST `/api/public/ingestion`.
//!
//! ## Drop-safe degradation (hard constraint)
//!
//! Langfuse is out-of-band telemetry; **no failure may affect the agent's main loop**:
//! - `enqueue` uses `try_send`; when the channel is full, **drop and count a warning**,
//!   never block;
//! - POST failures only `warn!`, **no retry** (to avoid backpressure buildup);
//! - On 207 (partial success), read the body and log errors, but do not affect subsequent
//!   processing.
//!
//! Langfuse ingestion — batch upload of traces and observations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use defect_http::HttpStack;
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{Method, Request};
use http_body_util::{BodyExt, Full};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;

use super::model::{IngestionBatch, IngestionEvent, IngestionResponse};

/// Commands for the background task.
enum Cmd {
    /// An event to be reported.
    Event(Box<IngestionEvent>),
    /// Flush the buffer immediately and signal completion via the oneshot (used for
    /// flushing before shutdown).
    Flush(oneshot::Sender<()>),
}

/// Ingest handle. `Clone` is cheap (inner `Arc`) — each session's observer holds one.
#[derive(Clone)]
pub struct LangfuseIngest {
    tx: mpsc::Sender<Cmd>,
    /// Cumulative count of events dropped due to a full channel. Used only for throttling
    /// alerts.
    dropped: Arc<AtomicU64>,
}

/// Configuration for building the reporter.
pub struct IngestConfig {
    /// Pre-built HTTP stack (shared with the LLM provider, includes
    /// timeout/retry/proxy/UA/trace).
    pub http: HttpStack,
    /// Langfuse host, e.g. `https://cloud.langfuse.com` (without trailing slash).
    pub host: String,
    /// Public key.
    pub public_key: String,
    /// Secret key.
    pub secret_key: String,
    /// Flush when the batch reaches this many items.
    pub max_batch: usize,
    /// Periodic flush interval.
    pub flush_interval: Duration,
    /// Capacity of the enqueue channel (backpressure boundary; drops when full).
    pub queue_capacity: usize,
}

impl LangfuseIngest {
    /// Spawns the background flush task and returns a handle.
    pub fn spawn(config: IngestConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let dropped = Arc::new(AtomicU64::new(0));

        let auth = {
            let raw = format!("{}:{}", config.public_key, config.secret_key);
            format!("Basic {}", BASE64.encode(raw.as_bytes()))
        };
        let endpoint = format!("{}/api/public/ingestion", config.host.trim_end_matches('/'));

        let worker = Worker {
            rx,
            http: config.http,
            endpoint,
            auth,
            max_batch: config.max_batch.max(1),
            flush_interval: config.flush_interval,
        };
        tokio::spawn(worker.run());

        Self { tx, dropped }
    }

    /// Non‑blocking enqueue. Drops and counts when the channel is full — never blocks the
    /// caller (agent main loop).
    pub fn enqueue(&self, event: IngestionEvent) {
        if self.tx.try_send(Cmd::Event(Box::new(event))).is_err() {
            let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            // Throttle warnings: only warn once per batch of drops to avoid log storms.
            if n.is_multiple_of(256) {
                tracing::warn!(
                    dropped_total = n,
                    "langfuse ingest queue full; dropping telemetry events (agent unaffected)"
                );
            }
        }
    }

    /// Flushes the buffer and waits for completion. Used for best-effort delivery before
    /// a session stream ends or the process exits.
    ///
    /// Returns immediately if the background task has already exited (receiver closed) —
    /// best-effort, no delivery guarantee.
    pub async fn flush(&self) {
        let (ack_tx, ack_rx) = oneshot::channel();
        if self.tx.send(Cmd::Flush(ack_tx)).await.is_ok() {
            let _ = ack_rx.await;
        }
    }
}

/// State of the background flush task.
struct Worker {
    rx: mpsc::Receiver<Cmd>,
    http: HttpStack,
    endpoint: String,
    auth: String,
    max_batch: usize,
    flush_interval: Duration,
}

impl Worker {
    async fn run(mut self) {
        let mut buf: Vec<IngestionEvent> = Vec::new();
        let mut tick = tokio::time::interval(self.flush_interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                cmd = self.rx.recv() => match cmd {
                    Some(Cmd::Event(ev)) => {
                        buf.push(*ev);
                        if buf.len() >= self.max_batch {
                            self.send_batch(&mut buf).await;
                        }
                    }
                    Some(Cmd::Flush(ack)) => {
                        self.send_batch(&mut buf).await;
                        let _ = ack.send(());
                    }
                    // All senders dropped: flush remaining data and exit.
                    None => {
                        self.send_batch(&mut buf).await;
                        break;
                    }
                },
                _ = tick.tick() => {
                    self.send_batch(&mut buf).await;
                }
            }
        }
    }

    /// Sends the current buffer as a single request. An empty buffer is a no-op.
    async fn send_batch(&self, buf: &mut Vec<IngestionEvent>) {
        if buf.is_empty() {
            return;
        }
        let batch = std::mem::take(buf);
        let body = match serde_json::to_vec(&IngestionBatch { batch }) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(%err, "langfuse: failed to serialize ingestion batch; dropped");
                return;
            }
        };

        let request = match Request::builder()
            .method(Method::POST)
            .uri(&self.endpoint)
            .header(AUTHORIZATION, &self.auth)
            .header(CONTENT_TYPE, "application/json")
            .body(toac::body::Body::new(Full::new(Bytes::from(body))))
        {
            Ok(req) => req,
            Err(err) => {
                tracing::warn!(%err, "langfuse: failed to build ingestion request; dropped");
                return;
            }
        };

        // `HttpStack` is a cloneable tower service — clone an independent copy and call
        // `oneshot` on it.
        match self.http.clone().oneshot(request).await {
            Ok(resp) => self.inspect_response(resp).await,
            Err(err) => {
                tracing::warn!(%err, "langfuse: ingestion POST failed; batch dropped (no retry)");
            }
        }
    }

    /// Inspect the response.
    ///
    /// The Langfuse ingestion endpoint **always returns 207 Multi-Status** for batch
    /// requests, with per-item results in the body's `successes` / `errors` fields.
    /// Therefore:
    /// - **2xx (including 207)**: parse the body; warn only if `errors` is **non-empty**
    ///   (partial failure). If `errors` is empty (all succeeded), return silently — this
    ///   is the normal path, not an error.
    /// - **Non-2xx** (401/403/5xx etc., genuine errors): warn as-is.
    async fn inspect_response(&self, resp: http::Response<hyper::body::Incoming>) {
        let status = resp.status();
        let body = match resp.into_body().collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(err) => {
                tracing::warn!(%status, %err, "langfuse: ingestion response body unreadable");
                return;
            }
        };

        if status.is_success() {
            // Parse individual results; warn only when there are actual failures.
            match serde_json::from_slice::<IngestionResponse>(&body) {
                Ok(parsed) if parsed.errors.is_empty() => {
                    // Normal path: all succeeded, silent.
                    tracing::trace!(
                        succeeded = parsed.successes.len(),
                        "langfuse: ingestion batch accepted"
                    );
                }
                Ok(parsed) => {
                    tracing::warn!(
                        failed = parsed.errors.len(),
                        succeeded = parsed.successes.len(),
                        errors = ?parsed.errors,
                        "langfuse: some ingestion events rejected"
                    );
                }
                Err(err) => {
                    // 2xx but body is not the expected structure — log a debug line, do
                    // not treat as an error.
                    let snippet = String::from_utf8_lossy(&body);
                    let snippet = snippet.chars().take(512).collect::<String>();
                    tracing::debug!(%status, %err, body = %snippet, "langfuse: unrecognized ingestion response");
                }
            }
            return;
        }

        // Non-2xx: real error (auth failure / server error, etc.).
        let snippet = String::from_utf8_lossy(&body);
        let snippet = snippet.chars().take(1024).collect::<String>();
        tracing::warn!(%status, body = %snippet, "langfuse: ingestion request failed");
    }
}
