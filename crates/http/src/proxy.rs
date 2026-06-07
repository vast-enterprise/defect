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
//! IP CIDR and ports are not currently supported.
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
/// Ports (e.g. `example.com:8080`) and IP CIDR are not currently supported — patterns
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
mod tests;
