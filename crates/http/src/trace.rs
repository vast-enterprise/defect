//! Tracing layer：在 `tracing::trace` 级别打 method/uri/status/elapsed。
//!
//! v0 不用 [`tower-http`] 的 `TraceLayer`——它会把 `Response<B>` 包成
//! `Response<ResponseBody<B, _>>` 来挂 body chunk hook，跟我们
//! `BoxCloneSyncService<_, http::Response<hyper::body::Incoming>, _>`
//! 的签名打架。我们这层只看 status / elapsed，不需要 body chunk
//! 钩子，自起一份更简单。
//!
//! 用法：`RUST_LOG=defect_http=trace`。

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use tower::{Layer, Service};
use tracing::trace;

#[derive(Debug, Clone)]
pub(crate) struct TraceLayer;

impl<S> Layer<S> for TraceLayer {
    type Service = Trace<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Trace { inner }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Trace<S> {
    inner: S,
}

impl<S, B, RespBody> Service<http::Request<B>> for Trace<S>
where
    S: Service<http::Request<B>, Response = http::Response<RespBody>>,
    S::Future: Send + 'static,
    S::Error: std::fmt::Display + Send + 'static,
    RespBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = TraceFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        let method = req.method().clone();
        let uri = req.uri().clone();
        trace!(http.method = %method, http.uri = %uri, "request start");
        TraceFuture {
            inner: self.inner.call(req),
            started: Instant::now(),
            method,
            uri,
        }
    }
}

pin_project_lite::pin_project! {
    pub(crate) struct TraceFuture<F> {
        #[pin]
        inner: F,
        started: Instant,
        method: http::Method,
        uri: http::Uri,
    }
}

impl<F, RespBody, E> Future for TraceFuture<F>
where
    F: Future<Output = Result<http::Response<RespBody>, E>>,
    E: std::fmt::Display,
{
    type Output = Result<http::Response<RespBody>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.inner.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(resp)) => {
                trace!(
                    http.method = %this.method,
                    http.uri = %this.uri,
                    http.status = resp.status().as_u16(),
                    elapsed_ms = this.started.elapsed().as_millis() as u64,
                    "request done",
                );
                Poll::Ready(Ok(resp))
            }
            Poll::Ready(Err(e)) => {
                trace!(
                    http.method = %this.method,
                    http.uri = %this.uri,
                    error = %e,
                    elapsed_ms = this.started.elapsed().as_millis() as u64,
                    "request failed",
                );
                Poll::Ready(Err(e))
            }
        }
    }
}
