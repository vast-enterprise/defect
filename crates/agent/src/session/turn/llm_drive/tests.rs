use super::*;

#[test]
fn non_backoff_hints_unchanged() {
    assert_eq!(retry_delay(RetryHint::No, 1), None);
    assert_eq!(retry_delay(RetryHint::Immediate, 5), Some(Duration::ZERO));
    let d = Duration::from_secs(7);
    assert_eq!(retry_delay(RetryHint::After(d), 3), Some(d));
}

#[test]
fn backoff_grows_exponentially_within_jitter() {
    // attempt 1 → ~500 ms ±25 % → [375, 625] ms
    for _ in 0..100 {
        let d = backoff_delay(1);
        assert!(
            d >= Duration::from_millis(374) && d <= Duration::from_millis(626),
            "attempt 1 out of jitter range: {d:?}"
        );
    }
    // attempt 3 → ~2000ms ±25% → [1500, 2500]ms
    for _ in 0..100 {
        let d = backoff_delay(3);
        assert!(
            d >= Duration::from_millis(1499) && d <= Duration::from_millis(2501),
            "attempt 3 out of jitter range: {d:?}"
        );
    }
}

#[test]
fn backoff_caps() {
    for _ in 0..100 {
        assert!(backoff_delay(40) <= BACKOFF_MAX, "cap broken");
    }
}
