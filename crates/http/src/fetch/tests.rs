use super::*;

#[test]
fn parse_http_uri_accepts_http_https() {
    parse_http_uri("http://example.com/").unwrap();
    parse_http_uri("https://example.com/path?q=1").unwrap();
}

#[test]
fn parse_http_uri_rejects_other_schemes() {
    assert!(matches!(
        parse_http_uri("file:///etc/passwd"),
        Err(HttpClientError::InvalidUrl(_))
    ));
    assert!(matches!(
        parse_http_uri("ftp://example.com/"),
        Err(HttpClientError::InvalidUrl(_))
    ));
    assert!(matches!(
        parse_http_uri("data:text/plain,hi"),
        Err(HttpClientError::InvalidUrl(_))
    ));
}

#[test]
fn parse_http_uri_rejects_missing_host() {
    assert!(matches!(
        parse_http_uri("http:///path"),
        Err(HttpClientError::InvalidUrl(_))
    ));
}

#[test]
fn resolve_redirect_absolute() {
    let base: Uri = "https://example.com/a".parse().unwrap();
    let r = resolve_redirect(&base, "https://other.test/x").unwrap();
    assert_eq!(r.to_string(), "https://other.test/x");
}

#[test]
fn resolve_redirect_path_absolute() {
    let base: Uri = "https://example.com/a/b".parse().unwrap();
    let r = resolve_redirect(&base, "/c/d").unwrap();
    assert_eq!(r.to_string(), "https://example.com/c/d");
}

#[test]
fn resolve_redirect_protocol_relative() {
    let base: Uri = "https://example.com/a".parse().unwrap();
    let r = resolve_redirect(&base, "//other.test/x").unwrap();
    assert_eq!(r.to_string(), "https://other.test/x");
}

#[test]
fn resolve_redirect_relative_rejected() {
    let base: Uri = "https://example.com/a/".parse().unwrap();
    assert!(resolve_redirect(&base, "other.html").is_err());
}

#[test]
fn resolve_redirect_rejects_non_http_scheme() {
    let base: Uri = "https://example.com/a".parse().unwrap();
    assert!(matches!(
        resolve_redirect(&base, "ftp://example.com/x"),
        Err(HttpClientError::InvalidUrl(_))
    ));
}
