//! HTTP/HTTPS proxy connector assembly.
//!
//! HTTP proxy implementation.
//!
//! Architecture: the connection layer wraps a
//! [`hyper_http_proxy::ProxyConnector<HttpConnector>`],
//! then wraps that with [`hyper_rustls::HttpsConnector`] for TLS. When the `proxies` list
//! is empty,
//! `ProxyConnector` transparently passes through (see the `match_proxy` branch in the
//! upstream
//! `Service<Uri>` impl), so the connector type stays the same regardless of whether the
//! user has
//! enabled a proxy — `HttpsConnector<ProxyConnector<HttpConnector>>` — avoiding two
//! different
//! connector types in [`build_http_stack`].
//!
//! NO_PROXY: each proxy entry's [`Intercept`] is written as an [`Intercept::Custom`]
//! closure that
//! matches scheme + host against the `NO_PROXY` suffix list. Matching follows
//! [GNU style](https://about.gitlab.com/blog/we-need-to-talk-no-proxy/):
//! comma-separated, domain suffixes (`api.openai.com` matches `*.openai.com`), `*` means
//! block all,
//! IP CIDR and port v0 are not supported.
//!
//! [`build_http_stack`]: super::build_http_stack

use std::env;
use std::sync::Arc;

use http::Uri;
use hyper_http_proxy::{Intercept, Proxy, ProxyConnector};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::connect::HttpConnector;

use super::{HttpStackError, ProxyConfig, ProxySettings};

/// The full connector type used by upper layers to construct a
/// [`hyper_util::client::legacy::Client`].
pub type ProxyAwareConnector = hyper_rustls::HttpsConnector<ProxyConnector<HttpConnector>>;

/// Build a full connector from [`ProxyConfig`].
///
/// - `Disabled` → still returns a `ProxyConnector`, but with no entries; `match_proxy`
///   always returns `None`, behaving equivalently to "no proxy".
/// - `FromEnv` → reads `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` (case-insensitive,
///   lowercase preferred, matching curl conventions).
/// - `Explicit` → uses the given values directly.
///
/// # Errors
///
/// Returns an error if loading native TLS roots fails, or if a proxy URL from the
/// environment cannot be parsed.
pub fn build_proxy_connector(config: &ProxyConfig) -> Result<ProxyAwareConnector, HttpStackError> {
    let entries = resolve_proxy(config)?;

    // ⚠ Must call `enforce_http(false)`: by default `HttpConnector` rejects `https`
    // schemes. When the outer `HttpsConnector` handles TLS for `https://` URLs, the inner
    // `HttpConnector` only manages TCP. `ProxyConnector` falls through to the inner
    // `HttpConnector` when no proxy entry matches (see the fallthrough branch in the
    // upstream `Service<Uri>` impl). With the default `enforce_http(true)`, all
    // `https://` requests immediately return `Err(InvalidUri)`. hyper-rustls's own
    // `HttpsConnectorBuilder::build()` applies this same change, but `wrap_connector(_)`
    // does not modify custom connectors — it must be done manually.
    let mut http_connector = HttpConnector::new();
    http_connector.enforce_http(false);

    // ⚠ Must use `unsecured`: when `__rustls` (any `rustls-tls-*-roots` feature) is
    // enabled, `ProxyConnector::new` embeds a `tokio_rustls::TlsConnector` that performs
    // its own TLS handshake over the CONNECT tunnel, returning `ProxyStream::Secured`.
    // Our outer `HttpsConnector::wrap_connector(_)` then wraps this already-encrypted
    // stream in another TLS layer — TLS-in-TLS — so the outer handshake never receives a
    // ServerHello and times out after ~14s. Using `unsecured` disables the proxy
    // connector's own TLS, making it handle only the CONNECT tunnel + raw TCP (returning
    // `ProxyStream::Regular`), while the outer `HttpsConnector` handles all TLS
    // (including HTTP/2 ALPN). That's why the workspace disables all `rustls-*-roots`
    // features on `hyper-http-proxy`.
    let mut proxy_connector = ProxyConnector::unsecured(http_connector);
    for entry in entries {
        proxy_connector.add_proxy(Proxy::new(entry.intercept, entry.uri));
    }

    let https = HttpsConnectorBuilder::new()
        .with_native_roots()
        .map_err(|e| HttpStackError::Config {
            hint: format!("load native TLS roots failed: {e}"),
        })?
        .https_or_http()
        .enable_all_versions()
        .wrap_connector(proxy_connector);

    Ok(https)
}

/// A single resolved proxy entry.
struct ResolvedProxy {
    intercept: Intercept,
    uri: Uri,
}

/// Converts a [`ProxyConfig`] into a list of `(Intercept, Uri)` pairs.
///
/// Returns an empty list when no proxy is configured (a valid state); returns
/// `HttpStackError::Config` if a URI fails to parse.
fn resolve_proxy(config: &ProxyConfig) -> Result<Vec<ResolvedProxy>, HttpStackError> {
    match config {
        ProxyConfig::Disabled => Ok(Vec::new()),
        ProxyConfig::FromEnv => resolve_from_env(),
        ProxyConfig::Explicit(settings) => resolve_explicit(settings),
    }
}

fn resolve_from_env() -> Result<Vec<ResolvedProxy>, HttpStackError> {
    let http_proxy = env_proxy("http_proxy", "HTTP_PROXY")?;
    let https_proxy = env_proxy("https_proxy", "HTTPS_PROXY")?;
    let no_proxy = parse_no_proxy(env_first("no_proxy", "NO_PROXY").as_deref().unwrap_or(""));

    let settings = ProxySettings {
        http_proxy,
        https_proxy,
        no_proxy,
    };
    resolve_explicit(&settings)
}

fn resolve_explicit(settings: &ProxySettings) -> Result<Vec<ResolvedProxy>, HttpStackError> {
    if no_proxy_disables_all(&settings.no_proxy) {
        return Ok(Vec::new());
    }

    let no_proxy = Arc::<[String]>::from(settings.no_proxy.clone());
    let mut entries = Vec::with_capacity(2);

    if let Some(uri) = settings.http_proxy.clone() {
        entries.push(ResolvedProxy {
            intercept: scheme_intercept_with_no_proxy("http", no_proxy.clone()),
            uri,
        });
    }
    if let Some(uri) = settings.https_proxy.clone() {
        entries.push(ResolvedProxy {
            intercept: scheme_intercept_with_no_proxy("https", no_proxy.clone()),
            uri,
        });
    }

    Ok(entries)
}

/// Reads an env variable and parses it into a [`Uri`]. Prefers the lowercase name,
/// falling back to uppercase — this is the de‑facto convention used by curl, Go,
/// requests, and other mainstream clients.
fn env_proxy(lower: &str, upper: &str) -> Result<Option<Uri>, HttpStackError> {
    let raw = match env_first(lower, upper) {
        Some(v) => v,
        None => return Ok(None),
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let uri = trimmed.parse::<Uri>().map_err(|e| HttpStackError::Config {
        hint: format!("invalid proxy URL `{trimmed}` from env: {e}"),
    })?;
    Ok(Some(uri))
}

fn env_first(lower: &str, upper: &str) -> Option<String> {
    if let Ok(v) = env::var(lower)
        && !v.trim().is_empty()
    {
        return Some(v);
    }
    if let Ok(v) = env::var(upper)
        && !v.trim().is_empty()
    {
        return Some(v);
    }
    None
}

/// `Intercept::Custom`: proxy only when the scheme matches and the host is not in the
/// NO_PROXY list.
fn scheme_intercept_with_no_proxy(scheme: &'static str, no_proxy: Arc<[String]>) -> Intercept {
    Intercept::Custom(
        (move |s: Option<&str>, h: Option<&str>, _p: Option<u16>| -> bool {
            if s != Some(scheme) {
                return false;
            }
            let host = match h {
                Some(h) => h,
                None => return true,
            };
            !matches_no_proxy(host, &no_proxy)
        })
        .into(),
    )
}

/// Parse a comma-separated `NO_PROXY` string, trimming whitespace and skipping empty
/// entries.
fn parse_no_proxy(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// A `*` in the list disables all proxies.
fn no_proxy_disables_all(patterns: &[String]) -> bool {
    patterns.iter().any(|p| p == "*")
}

/// Returns whether `host` is exempted by the NO_PROXY list.
///
/// GNU style: each pattern is a domain name (leading/trailing `.` are stripped);
/// `host` is exempt if it matches one of:
/// - `host == pattern` (after stripping leading dot)
/// - `host` ends with `.<pattern>`
///
/// `*` is already handled by [`no_proxy_disables_all`], so it is not checked here.
/// Ports (e.g. `example.com:8080`) and IP CIDR v0 are not supported — patterns
/// containing `:` or numeric subnets are compared literally; if they don't match,
/// the host is not exempted (safe behavior: prefer the proxy over a false match).
pub(crate) fn matches_no_proxy(host: &str, patterns: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    for raw in patterns {
        let pat = raw
            .trim_start_matches('.')
            .trim_end_matches('.')
            .to_ascii_lowercase();
        if pat.is_empty() {
            continue;
        }
        if host == pat {
            return true;
        }
        if host.ends_with(&format!(".{pat}")) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
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
}
