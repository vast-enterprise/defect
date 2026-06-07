//! HTTP fetch backend abstraction.
//!
//! [`HttpClient`] is the trait boundary between the `fetch` tool and the underlying HTTP
//! stack. The concrete implementation comes from [`defect-http`]; during session assembly
//! the CLI injects `Arc<dyn HttpClient>` into [`crate::session::AgentCore`], propagated
//! through [`crate::tool::ToolContext`] to tools.
//!
//! Unlike [`crate::fs::FsBackend`] / [`crate::shell::ShellBackend`], HTTP has no
//! per-client capability negotiation, so [`HttpClient`] is shared at the process level
//! rather than assembled per session; it simply reuses the same `Arc<dyn …>` injection
//! pattern to avoid introducing a new injection path.
//!
//! [`defect-http`]: ../../../crates/http

use std::time::Duration;

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// An HTTP fetch request.
///
/// v0 is `GET`-only — the `fetch` tool's schema also exposes only read semantics, not
/// method / header / body / auth.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// Absolute `http://` / `https://` URL. Other schemes are rejected early by the fetch
    /// tool layer.
    pub url: String,
    /// Per-request total timeout; `None` lets the backend use the stack-level default.
    pub timeout: Option<Duration>,
    /// Whether to follow 3xx `Location` redirects. When `false`, treat 3xx responses as
    /// terminal.
    pub follow_redirects: bool,
    /// Maximum number of redirect hops to follow; ignored when `follow_redirects` is
    /// `false`.
    pub max_redirects: u32,
    /// Maximum accumulated body size; if exceeded the body is truncated and
    /// `HttpResponse::truncated` is set to `true`.
    pub max_response_bytes: u64,
}

/// A response that was fetched successfully.
///
/// `status` is the status code of the final response (after following redirects);
/// `final_url` is analogous.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    /// The raw `content-type` header value (the tool layer should strip parameters like
    /// boundary/charset to get the main type); `None` if the server did not set it.
    pub content_type: Option<String>,
    /// Body truncated to `max_response_bytes`.
    pub body: Vec<u8>,
    /// Number of bytes the server actually sent (excluding bytes discarded by truncation
    /// — the backend stops reading when truncating, so this is approximate and for
    /// reference only).
    pub bytes_received: u64,
    /// `true` if the body was truncated because it exceeded `max_response_bytes`.
    pub truncated: bool,
    /// Number of redirects followed. 0 means the first response was final.
    pub redirects: u32,
    /// The final URL after following redirects; if no redirects were followed, this is
    /// the same as `request.url`.
    pub final_url: String,
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum HttpClientError {
    /// The URL could not be parsed (e.g., invalid scheme, missing host).
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// The request timed out as a whole.
    #[error("http request timed out")]
    Timeout,

    /// Exceeded `max_redirects` redirects. Contains the actual number of redirects
    /// attempted.
    #[error("too many redirects ({0})")]
    TooManyRedirects(u32),

    /// Transport-layer error (DNS / connect / TLS / IO); source is the underlying error.
    #[error("http transport error: {0}")]
    Transport(#[source] BoxError),
}

/// HTTP fetch backend trait.
///
/// Implementors must satisfy the following contract:
/// - `fetch` must internally enforce the total timeout from `req.timeout` (including
///   connect and read body);
///   on timeout, return [`HttpClientError::Timeout`].
/// - When `req.follow_redirects = true`, follow 3xx responses per RFC 7231, up to
///   `req.max_redirects` hops; exceeding that returns
///   [`HttpClientError::TooManyRedirects`].
/// - When reading the body, stop after accumulating `req.max_response_bytes` and set
///   `truncated = true` on the response.
/// - Any HTTP status (including 4xx/5xx) is considered a success
///   ([`HttpResponse::status`]
///   is returned as-is); only transport or decode failures should return `Err`.
pub trait HttpClient: Send + Sync {
    fn fetch(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, HttpClientError>>;
}

/// A placeholder implementation for testing or an `echo` provider. Every `fetch` call
/// returns
/// [`HttpClientError::Transport`], allowing assembly paths that require `Arc<dyn
/// HttpClient>`
/// to skip constructing a real HTTP stack.
pub struct NoopHttpClient;

impl HttpClient for NoopHttpClient {
    fn fetch(&self, _req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, HttpClientError>> {
        Box::pin(async move {
            Err(HttpClientError::Transport(BoxError::new(
                std::io::Error::other("NoopHttpClient: HTTP fetch not configured"),
            )))
        })
    }
}
