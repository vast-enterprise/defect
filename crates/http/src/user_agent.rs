//! A layer that injects a `User-Agent` header.
//!
//! Before each `inner.call(req)`, it writes `User-Agent` into `req.headers_mut()`,
//! skipping if the provider has already explicitly set it (i.e. `entry.or_insert(...)`
//! semantics). The fixed value is computed at construction time; subsequent clones
//! reuse the [`HeaderValue`]'s internal `Arc` sharing.
//!
//! The default value is given by [`default_user_agent`]:
//! `defect-http/{version} ({git_sha})`.

use std::task::{Context, Poll};

use http::HeaderValue;
use http::header::USER_AGENT;
use tower::{Layer, Service};

/// The default `User-Agent` value is `defect-http/{CARGO_PKG_VERSION}
/// ({DEFECT_HTTP_GIT_SHA})`.
///
/// `DEFECT_HTTP_GIT_SHA` is injected by `build.rs`: it first reads the build-time
/// environment variable `DEFECT_HTTP_BUILD_SHA` (for downstream packaging scenarios
/// without a `.git` directory), then falls back to running `git rev-parse`, and finally
/// degrades to `"unknown"` if neither is available.
pub fn default_user_agent() -> HeaderValue {
    let pkg = env!("CARGO_PKG_VERSION");
    let sha = env!("DEFECT_HTTP_GIT_SHA");
    let raw = format!("defect-http/{pkg} ({sha})");
    HeaderValue::from_str(&raw).unwrap_or_else(|_| HeaderValue::from_static("defect-http"))
}

#[derive(Debug, Clone)]
pub(crate) struct UserAgentLayer {
    value: HeaderValue,
}

impl UserAgentLayer {
    pub(crate) fn new(value: HeaderValue) -> Self {
        Self { value }
    }
}

impl<S> Layer<S> for UserAgentLayer {
    type Service = UserAgent<S>;

    fn layer(&self, inner: S) -> Self::Service {
        UserAgent {
            inner,
            value: self.value.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UserAgent<S> {
    inner: S,
    value: HeaderValue,
}

impl<S, B> Service<http::Request<B>> for UserAgent<S>
where
    S: Service<http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: http::Request<B>) -> Self::Future {
        req.headers_mut()
            .entry(USER_AGENT)
            .or_insert_with(|| self.value.clone());
        self.inner.call(req)
    }
}

#[cfg(test)]
mod tests;
