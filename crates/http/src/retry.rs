//! Transport 抖动重试 layer。
//!
//! **范围严格**：只重试 [`HttpStackError::Transport`]——DNS / connect / TLS /
//! hyper IO 层错误。HTTP status 任意值（200 / 4xx / 5xx）一律放行，
//! 让上层 provider / turn-loop 解析。详见 [`docs/outbound/http.md`] §3.1。
//!
//! 实现要点：
//! - 请求体先 buffer 成 [`bytes::Bytes`]（LLM 请求都是小 JSON），重试
//!   时用同一份 bytes 重建 [`toac::body::Body`]，避免 stream 已消费问题。
//! - 退避：`initial_backoff * 2^attempt ± 25% jitter`，封顶 30s。
//! - 一旦 inner future poll 出非 transport 错（或成功）立刻向上抛，
//!   不会"吞"掉 timeout / 配置错误。
//!
//! v0 不重试**已开始流式响应**的中段错误——hyper-util `Client` 的
//! `Future` 在 status line 之前就出错时整个 future 返回 `Err`，所以这
//! 层的语义"future 出错才重试"已经天然落在 status line 之前；status line
//! 之后的错误体现在 response body stream 上，不会回到这层 future。

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
        // tower 协议：clone 之前要 poll_ready；为了避免持有锁版本被消费
        // 这里克隆一份给 future 使用，原 self 留给后续调用。
        let clone = self.inner.clone();
        let mut svc = std::mem::replace(&mut self.inner, clone);
        let max_retries = self.max_retries;
        let initial_backoff = self.initial_backoff;

        Box::pin(async move {
            let (parts, body) = req.into_parts();
            // 先把请求体收满，方便重试时复用。
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

/// 收满请求体到 [`Bytes`]——LLM 请求都是 JSON 几 KB，整体收上来代价
/// 可忽略。
async fn collect_body(body: toac::body::Body) -> Result<Bytes, HttpStackError> {
    body.collect()
        .await
        .map(|c| c.to_bytes())
        .map_err(|e| HttpStackError::Transport(defect_agent::error::BoxError::from(e)))
}

/// 用同一份 `bytes` + 原 `parts` 重建一个新 [`http::Request`]——header /
/// uri / extensions 全部克隆下来，body 走新 [`Bytes`] 包成
/// [`toac::body::Body`]。
fn rebuild_request(
    parts: &http::request::Parts,
    bytes: Bytes,
) -> http::Request<toac::body::Body> {
    let mut req = http::Request::new(toac::body::Body::new(http_body_util::Full::new(bytes)));
    *req.method_mut() = parts.method.clone();
    *req.uri_mut() = parts.uri.clone();
    *req.version_mut() = parts.version;
    *req.headers_mut() = parts.headers.clone();
    *req.extensions_mut() = parts.extensions.clone();
    req
}

/// 判定一个 `HttpStackError` 是否值得重试。
///
/// 仅 [`HttpStackError::Transport`] 走重试——其它错误形态（Timeout /
/// Config / ProxyConnect）都是结构性错误，重试只会复现。
pub(crate) fn is_transport_retryable(err: &HttpStackError) -> bool {
    matches!(err, HttpStackError::Transport(_))
}

/// `initial * 2^attempt`，加 ±25% jitter，封顶 30s。
fn backoff_delay(initial: Duration, attempt: u32) -> Duration {
    let base_nanos = initial
        .as_nanos()
        .saturating_mul(1u128 << attempt.min(20));
    let cap_nanos = MAX_BACKOFF.as_nanos();
    let clamped = base_nanos.min(cap_nanos);

    let mut rng = rand::rng();
    let factor: f64 = 1.0 + rng.random_range(-JITTER_FRAC..JITTER_FRAC);
    // f64 精度对 ns 足够（30s = 3e10 ns，远小于 f64 整数精度）。
    let nanos = (clamped as f64 * factor).round();
    let nanos = nanos.clamp(0.0, cap_nanos as f64) as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use defect_agent::error::BoxError;
    use std::io;

    #[test]
    fn transport_error_is_retryable() {
        let e = HttpStackError::Transport(BoxError::new(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            "x",
        )));
        assert!(is_transport_retryable(&e));
    }

    #[test]
    fn timeout_is_not_retryable() {
        let e = HttpStackError::Timeout {
            phase: super::super::TimeoutPhase::Total,
        };
        assert!(!is_transport_retryable(&e));
    }

    #[test]
    fn config_is_not_retryable() {
        let e = HttpStackError::Config {
            hint: "x".into(),
        };
        assert!(!is_transport_retryable(&e));
    }

    #[test]
    fn proxy_connect_is_not_retryable() {
        let e = HttpStackError::ProxyConnect { hint: "x".into() };
        assert!(!is_transport_retryable(&e));
    }

    #[tokio::test]
    async fn retries_transport_then_succeeds() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use tower::ServiceExt;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        // Inner service：前两次返回 transport 错，第 3 次返回 200。
        let inner = tower::service_fn(
            move |_req: http::Request<toac::body::Body>| {
                let attempts = attempts_clone.clone();
                async move {
                    let n = attempts.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err::<http::Response<()>, _>(HttpStackError::Transport(
                            BoxError::new(io::Error::new(
                                io::ErrorKind::ConnectionRefused,
                                format!("attempt {n}"),
                            )),
                        ))
                    } else {
                        Ok(http::Response::new(()))
                    }
                }
            },
        );

        let svc = TransportRetryLayer::new(3, Duration::from_millis(1)).layer(inner);
        let req = http::Request::builder()
            .method(http::Method::POST)
            .uri("/test")
            .body(toac::body::Body::empty())
            .expect("build req");
        let resp = svc.oneshot(req).await.expect("retry to succeed");
        assert_eq!(resp.status(), http::StatusCode::OK);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_non_transport_error() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use tower::ServiceExt;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();
        let inner = tower::service_fn(
            move |_req: http::Request<toac::body::Body>| {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<http::Response<()>, _>(HttpStackError::Timeout {
                        phase: super::super::TimeoutPhase::Total,
                    })
                }
            },
        );
        let svc = TransportRetryLayer::new(3, Duration::from_millis(1)).layer(inner);
        let req = http::Request::builder()
            .uri("/")
            .body(toac::body::Body::empty())
            .expect("build req");
        let err = svc.oneshot(req).await.expect_err("must error");
        assert!(matches!(err, HttpStackError::Timeout { .. }));
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "Timeout 不该被重试"
        );
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use tower::ServiceExt;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();
        let inner = tower::service_fn(
            move |_req: http::Request<toac::body::Body>| {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<http::Response<()>, _>(HttpStackError::Transport(BoxError::new(
                        io::Error::other("nope"),
                    )))
                }
            },
        );
        let svc = TransportRetryLayer::new(2, Duration::from_millis(1)).layer(inner);
        let req = http::Request::builder()
            .uri("/")
            .body(toac::body::Body::empty())
            .expect("build req");
        let err = svc.oneshot(req).await.expect_err("must error");
        assert!(matches!(err, HttpStackError::Transport(_)));
        // max_retries=2 → 共 3 次 attempt（首次 + 2 次重试）。
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn backoff_grows_and_caps() {
        let initial = Duration::from_millis(200);
        // attempt 0: ~200ms ± 25% → [150, 250]ms
        for _ in 0..50 {
            let d = backoff_delay(initial, 0);
            assert!(
                d >= Duration::from_millis(149) && d <= Duration::from_millis(251),
                "attempt 0 jitter range: {d:?}"
            );
        }
        // 大 attempt 必须封顶 30s（含 jitter ≤ 30s）。
        for _ in 0..50 {
            let d = backoff_delay(initial, 30);
            assert!(d <= MAX_BACKOFF, "cap broken: {d:?}");
        }
    }
}
