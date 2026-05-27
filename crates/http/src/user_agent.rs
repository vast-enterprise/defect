//! `User-Agent` 注入 layer。
//!
//! 在每次 `inner.call(req)` 之前，往 `req.headers_mut()` 写入
//! `User-Agent`——若 provider 已经显式写了就跳过（`entry.or_insert(...)`
//! 语义）。固定值在构造时算好，后续 clone 复用 [`HeaderValue`]
//! 自身的 `Arc` 共享。
//!
//! 默认值由 [`default_user_agent`] 给出：`defect-http/{version} ({git_sha})`。

use std::task::{Context, Poll};

use http::HeaderValue;
use http::header::USER_AGENT;
use tower::{Layer, Service};

/// `User-Agent` 默认值：`defect-http/{CARGO_PKG_VERSION} ({DEFECT_HTTP_GIT_SHA})`。
///
/// `DEFECT_HTTP_GIT_SHA` 由 `build.rs` 注入：优先读 build-time 环境变量
/// `DEFECT_HTTP_BUILD_SHA`（用于无 `.git` 的下游打包场景），其次跑
/// `git rev-parse`，都拿不到时退化为 `"unknown"`。
pub(crate) fn default_user_agent() -> HeaderValue {
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
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_valid_header_value() {
        // 既不应 panic 也不应退化到 fallback——build.rs 的 sha 必然是
        // ascii，version 也是 semver ascii。
        let v = default_user_agent();
        let s = v.to_str().expect("ascii header");
        assert!(s.starts_with("defect-http/"), "got {s}");
    }

    #[test]
    fn build_sha_is_present() {
        // build.rs 至少给出 `unknown` 兜底，所以 sha 段不应为空。
        // 这条主要防止后续把 fallback 误删。
        let sha = env!("DEFECT_HTTP_GIT_SHA");
        assert!(!sha.is_empty(), "DEFECT_HTTP_GIT_SHA must always be set");
    }
}
