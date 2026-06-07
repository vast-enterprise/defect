//! HTTP client implementation for the `fetch` tool.
//!
//! Exists in parallel with `build_http_stack` (used by LLM providers, input is
//! `toac::Request`):
//! the provider path needs toac's wire body; the fetch path only sends bare GET requests
//! with
//! `Empty<Bytes>` as the body, and must handle URL validation, redirects, timeouts, and
//! body truncation at the client layer — these should not be pushed into the toac service
//! chain.
//!
//! Shared logic remains in [`super::proxy::build_proxy_connector`] — TLS, proxy, and
//! connection pool semantics are identical for both paths.
//!
//! HTTP fetch implementation.

use std::sync::Arc;

use bytes::Bytes;
use futures::future::BoxFuture;
use http::header::{CONTENT_TYPE, LOCATION, USER_AGENT};
use http::{HeaderValue, Method, Request, Uri};
use http_body_util::{BodyExt, Empty};
use hyper::body::Incoming;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;

use defect_agent::error::BoxError;
use defect_agent::http::{HttpClient, HttpClientError, HttpRequest, HttpResponse};

use super::proxy::{ProxyAwareConnector, build_proxy_connector};
use super::user_agent::default_user_agent;
use super::{HttpStackConfig, HttpStackError, ProxyConfig};

/// Internal hyper-util Client type alias — uses the same connector and body type family
/// as `build_http_stack` (here `Empty<Bytes>`, because fetch only issues GET requests).
type FetchHyperClient = HyperClient<ProxyAwareConnector, Empty<Bytes>>;

/// HTTP client backing the `fetch` utility.
///
/// See [`HttpClient`] for the behavioral contract: timeouts, redirects, and body
/// truncation are all implemented at this layer.
pub struct FetchHttpClient {
    inner: FetchHyperClient,
    user_agent: HeaderValue,
}

impl std::fmt::Debug for FetchHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FetchHttpClient")
            .field("user_agent", &self.user_agent)
            .finish()
    }
}

/// Builds a fetch client from an [`HttpStackConfig`].
///
/// Reuses the same [`ProxyConfig`] and default user agent. `transport_retries` /
/// `total_timeout` / `initial_backoff` do not apply directly in the fetch path —
/// fetch timeouts are determined per-call by [`HttpRequest::timeout`], and retries
/// are left to the caller ([`HttpClient`] does not retry, to avoid silently doubling
/// traffic for `GET` requests with side effects).
///
/// # Errors
///
/// Returns [`HttpStackError`] if the connector fails to build (e.g. TLS root loading,
/// proxy URL parsing).
pub fn build_fetch_client(config: &HttpStackConfig) -> Result<FetchHttpClient, HttpStackError> {
    let connector = build_proxy_connector(&config.proxy)?;
    let inner = HyperClient::builder(TokioExecutor::default()).build::<_, Empty<Bytes>>(connector);

    let user_agent = match &config.user_agent {
        Some(s) => HeaderValue::from_str(s).map_err(|e| HttpStackError::Config {
            hint: format!("invalid user_agent: {e}"),
        })?,
        None => default_user_agent(),
    };

    Ok(FetchHttpClient { inner, user_agent })
}

/// Convenience wrapper around `build_fetch_client` + `Arc::new` — the common path used at
/// CLI assembly points.
///
/// # Errors
///
/// Same as [`build_fetch_client`].
pub fn build_fetch_client_arc(
    config: &HttpStackConfig,
) -> Result<Arc<dyn HttpClient>, HttpStackError> {
    Ok(Arc::new(build_fetch_client(config)?))
}

impl HttpClient for FetchHttpClient {
    fn fetch(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, HttpClientError>> {
        Box::pin(async move {
            let timeout = req.timeout;
            let fut = self.execute(req);
            match timeout {
                Some(d) => match tokio::time::timeout(d, fut).await {
                    Ok(res) => res,
                    Err(_) => Err(HttpClientError::Timeout),
                },
                None => fut.await,
            }
        })
    }
}

impl FetchHttpClient {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, HttpClientError> {
        let mut current = parse_http_uri(&req.url)?;
        let mut redirects: u32 = 0;

        loop {
            let response = self.send_one(&current).await?;

            let status = response.status().as_u16();
            let is_redirect = (300..400).contains(&status) && status != 304;

            if is_redirect && req.follow_redirects {
                if redirects >= req.max_redirects {
                    return Err(HttpClientError::TooManyRedirects(redirects));
                }
                let Some(location) = response.headers().get(LOCATION) else {
                    // 3xx without Location: treat as terminal response (rare, matches
                    // most client behavior).
                    return collect_response(response, &current, redirects, req.max_response_bytes)
                        .await;
                };
                let raw = location
                    .to_str()
                    .map_err(|e| HttpClientError::Transport(BoxError::new(e)))?
                    .to_string();
                let next = resolve_redirect(&current, &raw)?;
                drop(response);
                current = next;
                redirects += 1;
                continue;
            }

            return collect_response(response, &current, redirects, req.max_response_bytes).await;
        }
    }

    async fn send_one(&self, uri: &Uri) -> Result<http::Response<Incoming>, HttpClientError> {
        let request = Request::builder()
            .method(Method::GET)
            .uri(uri.clone())
            .header(USER_AGENT, self.user_agent.clone())
            .body(Empty::<Bytes>::new())
            .map_err(|e| HttpClientError::Transport(BoxError::new(e)))?;

        self.inner
            .request(request)
            .await
            .map_err(|e| HttpClientError::Transport(BoxError::new(e)))
    }
}

fn parse_http_uri(raw: &str) -> Result<Uri, HttpClientError> {
    let uri: Uri = raw
        .parse()
        .map_err(|e: http::uri::InvalidUri| HttpClientError::InvalidUrl(e.to_string()))?;
    let scheme = uri
        .scheme_str()
        .ok_or_else(|| HttpClientError::InvalidUrl(format!("missing scheme in `{raw}`")))?;
    if !matches!(scheme, "http" | "https") {
        return Err(HttpClientError::InvalidUrl(format!(
            "unsupported scheme `{scheme}`: only http/https allowed"
        )));
    }
    if uri.host().is_none() {
        return Err(HttpClientError::InvalidUrl(format!(
            "missing host in `{raw}`"
        )));
    }
    Ok(uri)
}

/// Resolve the redirect target.
///
/// - Absolute URI (with scheme + authority) → parse and validate the scheme.
/// - Protocol-relative (`//host/path`) → reuse the original scheme.
/// - Absolute path (`/path`) → reuse the original scheme + authority.
/// - Relative path (`other.html`) → not currently supported, returns an error (rare
///   in practice and error-prone).
fn resolve_redirect(base: &Uri, location: &str) -> Result<Uri, HttpClientError> {
    let trimmed = location.trim();
    if trimmed.is_empty() {
        return Err(HttpClientError::Transport(BoxError::new(
            std::io::Error::other("empty Location header"),
        )));
    }

    if trimmed.contains("://") {
        // Absolute URI
        return parse_http_uri(trimmed);
    }

    let base_scheme = base.scheme_str().ok_or_else(|| {
        HttpClientError::Transport(BoxError::new(std::io::Error::other(
            "base URI missing scheme",
        )))
    })?;
    let base_authority = base.authority().ok_or_else(|| {
        HttpClientError::Transport(BoxError::new(std::io::Error::other(
            "base URI missing authority",
        )))
    })?;

    let composed = if let Some(rest) = trimmed.strip_prefix("//") {
        // Protocol-relative URL
        format!("{base_scheme}://{rest}")
    } else if trimmed.starts_with('/') {
        // Path-absolute
        format!("{base_scheme}://{base_authority}{trimmed}")
    } else {
        // path-relative: not currently supported
        return Err(HttpClientError::Transport(BoxError::new(
            std::io::Error::other(format!("relative redirect not supported: `{trimmed}`")),
        )));
    };

    parse_http_uri(&composed)
}

async fn collect_response(
    response: http::Response<Incoming>,
    final_uri: &Uri,
    redirects: u32,
    max_response_bytes: u64,
) -> Result<HttpResponse, HttpClientError> {
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let final_url = final_uri.to_string();

    let mut body_buf: Vec<u8> = Vec::new();
    let mut bytes_received: u64 = 0;
    let mut truncated = false;

    let mut frames = response.into_body();
    while let Some(frame) = frames
        .frame()
        .await
        .transpose()
        .map_err(|e| HttpClientError::Transport(BoxError::new(e)))?
    {
        if let Ok(data) = frame.into_data() {
            let len = data.len() as u64;
            bytes_received = bytes_received.saturating_add(len);
            if !truncated {
                let remaining = max_response_bytes.saturating_sub(body_buf.len() as u64);
                if (data.len() as u64) <= remaining {
                    body_buf.extend_from_slice(&data);
                } else {
                    let take = remaining as usize;
                    body_buf.extend_from_slice(&data[..take]);
                    truncated = true;
                }
            }
        }
    }

    Ok(HttpResponse {
        status,
        content_type,
        body: body_buf,
        bytes_received,
        truncated,
        redirects,
        final_url,
    })
}

/// Minimal version – allows the CLI to set up a fetch client even when a full
/// [`HttpStackConfig`] is not available.
///
/// Equivalent to `build_fetch_client(&HttpStackConfig::default())`.
///
/// # Errors
///
/// Same as [`build_fetch_client`].
pub fn build_default_fetch_client_arc() -> Result<Arc<dyn HttpClient>, HttpStackError> {
    build_fetch_client_arc(&HttpStackConfig {
        proxy: ProxyConfig::FromEnv,
        ..HttpStackConfig::default()
    })
}

#[cfg(test)]
mod tests;
