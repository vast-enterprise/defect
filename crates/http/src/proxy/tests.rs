use super::*;

fn pats(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn empty_patterns_never_match() {
    assert!(!matches_no_proxy("api.openai.com", &[]));
}

#[test]
fn exact_host_match() {
    assert!(matches_no_proxy("example.com", &pats(&["example.com"])));
}

#[test]
fn suffix_match_with_dot() {
    assert!(matches_no_proxy("api.openai.com", &pats(&[".openai.com"])));
}

#[test]
fn suffix_match_without_dot() {
    // GNU style: a pattern without a leading dot still matches as a suffix.
    assert!(matches_no_proxy("api.openai.com", &pats(&["openai.com"])));
}

#[test]
fn substring_does_not_match() {
    // "openai" should not match "myopenai.com" — must end at a "." boundary.
    assert!(!matches_no_proxy("myopenai.com", &pats(&["openai.com"])));
}

#[test]
fn case_insensitive() {
    assert!(matches_no_proxy("API.OpenAI.COM", &pats(&["openai.com"])));
}

#[test]
fn multiple_patterns() {
    let p = pats(&["foo.com", ".bar.com", "baz.com"]);
    assert!(matches_no_proxy("foo.com", &p));
    assert!(matches_no_proxy("x.bar.com", &p));
    assert!(matches_no_proxy("baz.com", &p));
    assert!(!matches_no_proxy("qux.com", &p));
}

#[test]
fn empty_pattern_in_list_is_skipped() {
    // Corner-case input like `,foo.com,` is already filtered by `parse_no_proxy`, but
    // `matches_no_proxy` must also be idempotent when it receives such input.
    let p = pats(&["", "foo.com"]);
    assert!(matches_no_proxy("foo.com", &p));
    assert!(!matches_no_proxy("bar.com", &p));
}

#[test]
fn host_trailing_dot_normalised() {
    assert!(matches_no_proxy("api.openai.com.", &pats(&["openai.com"])));
}

#[test]
fn parse_no_proxy_splits_and_trims() {
    let v = parse_no_proxy("  foo.com , bar.com,, baz.com  ");
    assert_eq!(v, vec!["foo.com", "bar.com", "baz.com"]);
}

#[test]
fn parse_no_proxy_empty() {
    assert!(parse_no_proxy("").is_empty());
    assert!(parse_no_proxy("   ").is_empty());
}

#[test]
fn star_disables_all() {
    assert!(no_proxy_disables_all(&pats(&["*"])));
    assert!(no_proxy_disables_all(&pats(&["foo.com", "*"])));
    assert!(!no_proxy_disables_all(&pats(&["foo.com"])));
}

#[test]
fn resolve_explicit_with_star_returns_empty() {
    let settings = ProxySettings {
        http_proxy: Some("http://p:8080".parse().unwrap()),
        https_proxy: Some("http://p:8080".parse().unwrap()),
        no_proxy: pats(&["*"]),
    };
    let v = resolve_explicit(&settings).unwrap();
    assert!(v.is_empty(), "* must short-circuit to no-proxy");
}

#[test]
fn resolve_explicit_emits_two_entries() {
    let settings = ProxySettings {
        http_proxy: Some("http://p:8080".parse().unwrap()),
        https_proxy: Some("http://p:8443".parse().unwrap()),
        no_proxy: pats(&[".internal"]),
    };
    let v = resolve_explicit(&settings).unwrap();
    assert_eq!(v.len(), 2);
}

#[test]
fn resolve_explicit_only_http() {
    let settings = ProxySettings {
        http_proxy: Some("http://p:8080".parse().unwrap()),
        https_proxy: None,
        no_proxy: Vec::new(),
    };
    let v = resolve_explicit(&settings).unwrap();
    assert_eq!(v.len(), 1);
}

#[test]
fn resolve_explicit_no_proxies() {
    let settings = ProxySettings::default();
    let v = resolve_explicit(&settings).unwrap();
    assert!(v.is_empty());
}

#[tokio::test]
async fn build_proxy_connector_does_not_reject_https_when_no_proxy_match() {
    // Regression test: previously
    // `wrap_connector(ProxyConnector::new(HttpConnector::new()))` omitted
    // `enforce_http(false)`, causing HTTPS requests that did not match a proxy entry
    // to fail at the `HttpConnector::call` stage with `Err(InvalidUri/scheme is not
    // http)` before reaching TLS. Here we poll the connection once and assert that we
    // do **not** get that error — actual DNS / connection refusal errors are
    // acceptable (no network).
    use http::Uri;
    use tower::{Service, ServiceExt};

    let connector = build_proxy_connector(&ProxyConfig::Disabled).expect("build");
    let uri: Uri = "https://example.invalid/".parse().unwrap();
    let svc = connector.ready_oneshot().await;
    // A failure in `ready_oneshot` means the connector itself cannot become ready —
    // currently both `hyper-rustls` and `hyper-util` are always ready, so this should
    // not panic.
    let mut svc = svc.expect("connector ready");
    let res = svc.call(uri).await;
    if let Err(e) = res {
        let msg = e.to_string();
        assert!(
            !msg.contains("scheme is not http"),
            "https URI should not be rejected at HttpConnector layer; got: {msg}"
        );
    }
}

#[test]
fn intercept_closure_respects_scheme_and_no_proxy() {
    // Directly verify the closure semantics: scheme mismatch → false; scheme matches
    // but host is in NO_PROXY → false; scheme matches and host is not in NO_PROXY →
    // true.
    let no_proxy = Arc::<[String]>::from(pats(&[".openai.com"]));
    let intercept = scheme_intercept_with_no_proxy("https", no_proxy);
    // intercept_closure cannot be called directly — verified via
    // `Intercept::matches`.
    struct FakeUri<'a> {
        scheme: Option<&'a str>,
        host: Option<&'a str>,
        port: Option<u16>,
    }
    impl<'a> hyper_http_proxy::Dst for FakeUri<'a> {
        fn scheme(&self) -> Option<&str> {
            self.scheme
        }
        fn host(&self) -> Option<&str> {
            self.host
        }
        fn port(&self) -> Option<u16> {
            self.port
        }
    }
    let must_proxy = FakeUri {
        scheme: Some("https"),
        host: Some("anthropic.com"),
        port: Some(443),
    };
    let must_skip_no_proxy = FakeUri {
        scheme: Some("https"),
        host: Some("api.openai.com"),
        port: Some(443),
    };
    let must_skip_scheme = FakeUri {
        scheme: Some("http"),
        host: Some("anthropic.com"),
        port: Some(80),
    };
    assert!(intercept.matches(&must_proxy));
    assert!(!intercept.matches(&must_skip_no_proxy));
    assert!(!intercept.matches(&must_skip_scheme));
}
