//! Transport jitter retry layer.
//!
//! **Strict scope**: only retries [`HttpStackError::Transport`] — DNS / connect / TLS /
//! hyper IO layer errors. Any HTTP status (200 / 4xx / 5xx) is passed through unchanged.
//!
//! Implementation notes:
//! - The request body is buffered into [`bytes::Bytes`] first (LLM requests are small
//!   JSON); on retry the same bytes are used to reconstruct [`toac::body::Body`],
//!   avoiding the stream-already-consumed problem.
//! - Backoff: `initial_backoff * 2^attempt ± 25% jitter`, capped at 30s.
//! - As soon as the inner future polls to a non-transport error (or success), the result
//!   is propagated upward immediately; timeouts / configuration errors are not swallowed.
//!
//! Currently does not retry mid-stream errors for **already-started streaming
//! responses** — the hyper-util `Client`'s `Future` returns `Err` for the entire future
//! when the error occurs before the status line, so this layer's semantics of "retry
//! only when the future returns `Err`" naturally applies only before the status line;
//! errors after the status line appear in the response body stream and never reach this
//! layer's future.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use rand::RngExt;
use tower::{Layer, Service};

use super::HttpStackError;

const MAX_BACKOFF: Duration = Duration::from_secs(30);
const JITTER_FRAC: f64 = 0.25;

#[derive(Debug, Clone)]
pub(crate) struct TransportRetryLayer {
    max_retries: u8,
    initial_backoff: Duration,
}

impl TransportRetryLayer {
    pub(crate) fn new(max_retries: u8, initial_backoff: Duration) -> Self {
        Self {
            max_retries,
            initial_backoff,
        }
    }
}

impl<S> Layer<S> for TransportRetryLayer {
    type Service = TransportRetry<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TransportRetry {
            inner,
            max_retries: self.max_retries,
            initial_backoff: self.initial_backoff,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TransportRetry<S> {
    inner: S,
    max_retries: u8,
    initial_backoff: Duration,
}

impl<S> Service<http::Request<toac::body::Body>> for TransportRetry<S>
where
    S: Service<http::Request<toac::body::Body>, Error = HttpStackError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
{
    type Response = S::Response;
    type Error = HttpStackError;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, HttpStackError>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<toac::body::Body>) -> Self::Future {
        // Per the Tower protocol, clone only after `poll_ready`. To avoid consuming the
        // locked version, clone a copy for the future and keep the original `self` for
        // subsequent calls.
        let clone = self.inner.clone();
        let mut svc = std::mem::replace(&mut self.inner, clone);
        let max_retries = self.max_retries;
        let initial_backoff = self.initial_backoff;

        Box::pin(async move {
            let (parts, body) = req.into_parts();
            // Collect the full body first so it can be reused on retries.
            let bytes = collect_body(body).await?;

            let mut attempt: u32 = 0;
            loop {
                let req = rebuild_request(&parts, bytes.clone());
                let result = svc.call(req).await;
                match result {
                    Ok(resp) => return Ok(resp),
                    Err(e) if !is_transport_retryable(&e) => return Err(e),
                    Err(e) => {
                        if attempt >= u32::from(max_retries) {
                            return Err(e);
                        }
                        let delay = backoff_delay(initial_backoff, attempt);
                        tracing::trace!(
                            attempt = attempt + 1,
                            max_retries,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "transport error; retrying",
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                    }
                }
            }
        })
    }
}

/// Collect the entire request body into [`Bytes`] — LLM requests are JSON, a few KB, so
/// the cost of collecting the whole body is negligible.
async fn collect_body(body: toac::body::Body) -> Result<Bytes, HttpStackError> {
    body.collect()
        .await
        .map(|c| c.to_bytes())
        .map_err(|e| HttpStackError::Transport(defect_agent::error::BoxError::from(e)))
}

/// Rebuild a new [`http::Request`] from the same `bytes` and original `parts` — clones
/// all headers, URI, and extensions, and wraps the new [`Bytes`] into a
/// [`toac::body::Body`] for the body.
fn rebuild_request(parts: &http::request::Parts, bytes: Bytes) -> http::Request<toac::body::Body> {
    let mut req = http::Request::new(toac::body::Body::new(http_body_util::Full::new(bytes)));
    *req.method_mut() = parts.method.clone();
    *req.uri_mut() = parts.uri.clone();
    *req.version_mut() = parts.version;
    *req.headers_mut() = parts.headers.clone();
    *req.extensions_mut() = parts.extensions.clone();
    req
}

/// Returns whether an `HttpStackError` is worth retrying.
///
/// Only [`HttpStackError::Transport`] is retried — other variants (Timeout /
/// Config / ProxyConnect) are structural errors that will recur on retry.
pub(crate) fn is_transport_retryable(err: &HttpStackError) -> bool {
    matches!(err, HttpStackError::Transport(_))
}

/// `initial * 2^attempt` with ±25% jitter, capped at 30s.
fn backoff_delay(initial: Duration, attempt: u32) -> Duration {
    let base_nanos = initial.as_nanos().saturating_mul(1u128 << attempt.min(20));
    let cap_nanos = MAX_BACKOFF.as_nanos();
    let clamped = base_nanos.min(cap_nanos);

    let mut rng = rand::rng();
    let factor: f64 = 1.0 + rng.random_range(-JITTER_FRAC..JITTER_FRAC);
    // f64 precision is sufficient for nanoseconds (30s = 3e10 ns, well below f64's exact
    // integer range).
    let nanos = (clamped as f64 * factor).round();
    let nanos = nanos.clamp(0.0, cap_nanos as f64) as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests;
