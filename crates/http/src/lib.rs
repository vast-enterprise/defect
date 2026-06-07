//! HTTP infrastructure shared across modules.
//!
//! A thin wrapper on top of `client_util::build_https_client` that adds: timeouts,
//! transport retry with jitter, HTTP/HTTPS proxy support, and a unified `User-Agent`.
//! HTTP client abstraction for the agent.
//!
//! Current consumers: `defect-llm` (various LLM providers); planned: `defect-tools`'
//! fetch tool. This layer is extracted into its own crate to prevent the latter from
//! depending on `defect-llm` (which would create an inverted dependency).
//!
//! Public entry points are only [`build_http_stack`], [`HttpStackConfig`], [`HttpStack`],
//! and [`HttpStackError`]. Concrete layer implementations live in submodules as
//! `pub(crate)` and are not exposed outside the crate — callers see only a type-erased
//! Service.

use std::time::Duration;

use http::HeaderValue;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;
use thiserror::Error;
use tower::ServiceBuilder;
use tower::util::BoxCloneSyncService;

use defect_agent::error::BoxError;

mod fetch;
mod proxy;
mod retry;
mod trace;
mod user_agent;

pub use fetch::{
    FetchHttpClient, build_default_fetch_client_arc, build_fetch_client, build_fetch_client_arc,
};
pub use proxy::{ProxyAwareConnector, build_proxy_connector};
pub use user_agent::default_user_agent;

/// Type-erased service returned by `build_http_stack`.
///
/// Takes a `toac::Request` and returns `http::Response<hyper::body::Incoming>`,
/// with errors unified as [`HttpStackError`]. Each provider passes this to
/// `toac::ApiClient::new`.
///
/// Uses [`BoxCloneSyncService`] instead of `BoxService`: toac's `tower::Service`
/// impl requires `S: Clone` so that after `poll_ready`, a lock-free clone can be
/// taken for the future — see the `mem::replace` pattern in toac's `lib.rs`.
pub type HttpStack =
    BoxCloneSyncService<toac::Request, http::Response<hyper::body::Incoming>, HttpStackError>;

/// HTTP stack configuration.
///
/// `Default::default()` provides v0 recommended values: `total_timeout = 600s`,
/// `transport_retries = 2`, `initial_backoff = 200ms`, `user_agent = None`
/// (compile-time default), `proxy = ProxyConfig::FromEnv`.
#[derive(Debug, Clone)]
pub struct HttpStackConfig {
    /// Total timeout for a single request. `None` means no limit. For SSE streaming
    /// responses, the timer starts after the first byte arrives and continues until the
    /// stream ends — the v0 default of 600s covers the maximum reasonable duration for
    /// Anthropic extended thinking.
    pub total_timeout: Option<Duration>,

    /// Maximum number of transport error retries (excluding the initial attempt). `0`
    /// disables the retry layer. Only retries transport-level jitter (DNS / TCP / TLS /
    /// hyper IO); any HTTP status code is treated as "success" and passed through —
    /// business-level retries are handled in turn-loop §7.
    pub transport_retries: u8,

    /// Initial backoff for retries. Each retry multiplies by 2, adds ±25% jitter, and
    /// caps at 30s.
    pub initial_backoff: Duration,

    /// `User-Agent` header value. When `None`, uses the compile-time default
    /// (`defect-http/{version} ({git_sha[..8]})`).
    pub user_agent: Option<String>,

    /// Proxy configuration.
    pub proxy: ProxyConfig,
}

impl Default for HttpStackConfig {
    fn default() -> Self {
        Self {
            total_timeout: Some(Duration::from_secs(600)),
            transport_retries: 2,
            initial_backoff: Duration::from_millis(200),
            user_agent: None,
            proxy: ProxyConfig::FromEnv,
        }
    }
}

/// Proxy configuration.
#[derive(Debug, Clone, Default)]
pub enum ProxyConfig {
    /// Reads `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` from the environment.
    #[default]
    FromEnv,
    /// Explicitly provided.
    Explicit(ProxySettings),
    /// Forcefully disable proxying, even if environment variables are set.
    Disabled,
}

/// Explicit proxy settings. `http_proxy` / `https_proxy` may each be `None`;
/// `no_proxy` is a list of domain suffixes (following the GNU `NO_PROXY` convention).
#[derive(Debug, Clone, Default)]
pub struct ProxySettings {
    pub http_proxy: Option<http::Uri>,
    pub https_proxy: Option<http::Uri>,
    pub no_proxy: Vec<String>,
}

/// HTTP stack-layer error.
///
/// Corresponds to the `E` in `toac::CallError<E>` — the provider translates this error
/// into `ProviderErrorKind` in `call_error_to_provider` (see HTTP retry/error semantics).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HttpStackError {
    /// Transport error (DNS, TCP, TLS, hyper I/O, etc.).
    #[error("HTTP transport error: {0}")]
    Transport(#[source] BoxError),

    /// Request timed out. `phase` indicates which stage timed out — v0 only supports
    /// `Total`.
    /// Staged timeouts for HTTP requests.
    #[error("HTTP request timed out (phase = {phase:?})")]
    Timeout { phase: TimeoutPhase },

    /// HTTP layer configuration error (e.g., proxy URL parsing failure).
    #[error("HTTP layer config invalid: {hint}")]
    Config { hint: String },

    /// Proxy CONNECT phase failed.
    #[error("proxy CONNECT failed: {hint}")]
    ProxyConnect { hint: String },
}

/// Timeout phase. Mirrors [`defect_agent::llm::TimeoutPhase`], but this crate does not
/// reference the agent's type internally to avoid coupling the layer implementation to
/// the LLM error model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimeoutPhase {
    Connect,
    ReadHeaders,
    ReadBody,
    Idle,
    Total,
}

/// Builds the full HTTP stack; the result can be fed directly to `toac::ApiClient::new`.
///
/// Current layer order (outer → inner, request direction):
/// `UserAgent → Trace → Timeout? → hyper-util Client`
///
/// `Timeout` is inserted only when `config.total_timeout = Some(_)` — when `None`,
/// the entire timeout layer is skipped. This avoids a type mismatch with `Identity`
/// when `tower::timeout` wraps the error as [`tower::BoxError`] (`option_layer`
/// does not change the error type on the `None` path).
pub fn build_http_stack(config: HttpStackConfig) -> Result<HttpStack, HttpStackError> {
    // The connector layer merges TLS + proxy in one pass: `ProxyConnector` transparently
    // passes through when no entries are configured, so `Disabled` also uses the same
    // connector type, avoiding two forked `HyperClient` types behind an `if`.
    let connector = proxy::build_proxy_connector(&config.proxy)?;
    let inner =
        HyperClient::builder(TokioExecutor::default()).build::<_, toac::body::Body>(connector);

    // Maps `hyper-util Client` errors to `HttpStackError::Transport`
    let transport = ServiceBuilder::new()
        .map_err(|e: hyper_util::client::legacy::Error| HttpStackError::Transport(BoxError::new(e)))
        .service(inner);

    let ua_value = match &config.user_agent {
        Some(s) => HeaderValue::from_str(s).map_err(|e| HttpStackError::Config {
            hint: format!("invalid user_agent: {e}"),
        })?,
        None => user_agent::default_user_agent(),
    };

    let retry_layer = (config.transport_retries > 0)
        .then(|| retry::TransportRetryLayer::new(config.transport_retries, config.initial_backoff));

    let retried = ServiceBuilder::new()
        .option_layer(retry_layer)
        .service(transport);

    let stack = if let Some(timeout) = config.total_timeout {
        let s = ServiceBuilder::new()
            .layer(user_agent::UserAgentLayer::new(ua_value))
            .layer(trace::TraceLayer)
            .map_err(map_timeout_error)
            .layer(tower::timeout::TimeoutLayer::new(timeout))
            .service(retried);
        BoxCloneSyncService::new(s)
    } else {
        let s = ServiceBuilder::new()
            .layer(user_agent::UserAgentLayer::new(ua_value))
            .layer(trace::TraceLayer)
            .service(retried);
        BoxCloneSyncService::new(s)
    };

    Ok(stack)
}

/// Converts a [`tower::BoxError`] from [`tower::timeout`] back into an
/// [`HttpStackError`]:
/// - [`tower::timeout::error::Elapsed`] → `Timeout { phase: Total }`
/// - Otherwise it should be an inner [`HttpStackError`]—[`tower::timeout`] boxes it, so
///   `downcast` recovers it
/// - Last resort (should not happen) → `Transport`, preserving the original source
fn map_timeout_error(err: tower::BoxError) -> HttpStackError {
    if err.is::<tower::timeout::error::Elapsed>() {
        return HttpStackError::Timeout {
            phase: TimeoutPhase::Total,
        };
    }
    match err.downcast::<HttpStackError>() {
        Ok(boxed) => *boxed,
        Err(other) => HttpStackError::Transport(BoxError::from(other)),
    }
}
