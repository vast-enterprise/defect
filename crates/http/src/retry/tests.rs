use super::*;
use defect_core::error::BoxError;
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
    let e = HttpStackError::Config { hint: "x".into() };
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

    // Inner service: first two calls return a transport error, third returns 200.
    let inner = tower::service_fn(move |_req: http::Request<toac::body::Body>| {
        let attempts = attempts_clone.clone();
        async move {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err::<http::Response<()>, _>(HttpStackError::Transport(BoxError::new(
                    io::Error::new(io::ErrorKind::ConnectionRefused, format!("attempt {n}")),
                )))
            } else {
                Ok(http::Response::new(()))
            }
        }
    });

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
    let inner = tower::service_fn(move |_req: http::Request<toac::body::Body>| {
        let attempts = attempts_clone.clone();
        async move {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err::<http::Response<()>, _>(HttpStackError::Timeout {
                phase: super::super::TimeoutPhase::Total,
            })
        }
    });
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
        "Timeout should not be retried"
    );
}

#[tokio::test]
async fn gives_up_after_max_retries() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tower::ServiceExt;

    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    let inner = tower::service_fn(move |_req: http::Request<toac::body::Body>| {
        let attempts = attempts_clone.clone();
        async move {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err::<http::Response<()>, _>(HttpStackError::Transport(BoxError::new(
                io::Error::other("nope"),
            )))
        }
    });
    let svc = TransportRetryLayer::new(2, Duration::from_millis(1)).layer(inner);
    let req = http::Request::builder()
        .uri("/")
        .body(toac::body::Body::empty())
        .expect("build req");
    let err = svc.oneshot(req).await.expect_err("must error");
    assert!(matches!(err, HttpStackError::Transport(_)));
    // max_retries=2 → 3 total attempts (initial + 2 retries).
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
    // Large attempts must cap at 30s (including jitter ≤ 30s).
    for _ in 0..50 {
        let d = backoff_delay(initial, 30);
        assert!(d <= MAX_BACKOFF, "cap broken: {d:?}");
    }
}
