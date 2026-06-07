//! Provider error types.

use std::time::Duration;

use thiserror::Error;

use crate::error::BoxError;

/// A unified error for provider operation failures.
///
/// The top-level struct separates cross-cutting diagnostic information (e.g.
/// `request_id`)
/// from classification information (`kind`) to avoid duplicating `request_id` in every
/// variant.
#[derive(Debug, Error)]
#[error("{kind}")]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    /// The request ID returned by the server (e.g. Anthropic `request-id` header / OpenAI
    /// `x-request-id`). This is the primary signal for debugging; populate it whenever
    /// possible.
    pub request_id: Option<String>,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind) -> Self {
        Self {
            kind,
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Returns a retry hint for this error.
    pub fn retry_hint(&self) -> RetryHint {
        use ProviderErrorKind::*;
        match &self.kind {
            AuthMissing { .. }
            | AuthMalformed { .. }
            | AuthRejected { .. }
            | ModelNotFound { .. }
            | BadRequest { .. }
            | InvalidToolSchema { .. }
            | InputBlocked { .. }
            | OutputBlocked { .. }
            | ProtocolViolation { .. }
            | MaxTokensInvalid { .. }
            | QuotaExceeded { .. }
            | Canceled
            | Other(_) => RetryHint::No,

            AuthExpired => RetryHint::AfterAction(RetryAction::RefreshAuth),
            ContextOverflow { .. } => RetryHint::AfterAction(RetryAction::ReduceContext),

            RateLimit {
                retry_after: Some(d),
                ..
            } => RetryHint::After(*d),
            RateLimit {
                retry_after: None, ..
            } => RetryHint::Backoff,

            ServerError { .. }
            | ServerStreamAborted { .. }
            | Malformed(_)
            | Transport(_)
            | Timeout { .. } => RetryHint::Backoff,
        }
    }

    /// Convenience check: whether the agent should automatically retry.
    pub fn is_retryable(&self) -> bool {
        !matches!(self.retry_hint(), RetryHint::No)
    }
}

/// Semantic classification of provider errors.
///
/// Fallback principle: if a category of errors repeatedly falls into
/// [`ProviderErrorKind::Other`],
/// prefer to **extract it as a new variant** rather than letting `Other` become the
/// default.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ProviderErrorKind {
    // ---------- Authentication ----------
    /// Missing credential.
    #[error("missing credential{}", var_hint.as_deref().map(|h| format!(" (hint: {h})")).unwrap_or_default())]
    AuthMissing { var_hint: Option<String> },

    /// Malformed credential.
    #[error("malformed credential{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    AuthMalformed { hint: Option<String> },

    /// Credential rejected by the server (401).
    #[error("credential rejected by server{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    AuthRejected { hint: Option<String> },

    /// OAuth/STS token expired.
    #[error("auth token expired")]
    AuthExpired,

    // Quota
    /// Request-level rate limiting.
    #[error("rate limit hit ({scope:?}){}", retry_after.map(|d| format!(", retry after {}s", d.as_secs())).unwrap_or_default())]
    RateLimit {
        retry_after: Option<Duration>,
        scope: RateLimitScope,
    },

    /// Quota exhausted / monthly allowance depleted.
    #[error("quota exceeded{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    QuotaExceeded { hint: Option<String> },

    // ---------- Input ----------
    /// Context window overflow.
    #[error("context overflow{}", match (used, limit) {
        (Some(u), Some(l)) => format!(" ({u} > {l})"),
        _ => String::new(),
    })]
    ContextOverflow {
        used: Option<u64>,
        limit: Option<u64>,
    },

    /// The requested `max_tokens` exceeds the model's limit or was rejected by the
    /// server.
    #[error("max_tokens invalid{}", match (requested, limit) {
        (Some(r), Some(l)) => format!(" ({r} > {l})"),
        _ => String::new(),
    })]
    MaxTokensInvalid {
        requested: Option<u64>,
        limit: Option<u64>,
    },

    /// Model ID does not exist or is unavailable.
    #[error("model not found: {model}")]
    ModelNotFound { model: String },

    /// Request body rejected by the wire service validation (schema error, conflicting
    /// mutually exclusive fields).
    #[error("bad request{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    BadRequest { hint: Option<String> },

    /// The tool schema referenced in the request was rejected by the server.
    #[error("invalid tool schema for {tool}{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    InvalidToolSchema { tool: String, hint: Option<String> },

    // ---------- Security / Compliance ----------
    /// Input triggered a safety filter.
    #[error("input blocked{}", policy.as_deref().map(|p| format!(" by {p}")).unwrap_or_default())]
    InputBlocked { policy: Option<String> },

    /// Model output blocked by safety filter.
    #[error("output blocked{}", policy.as_deref().map(|p| format!(" by {p}")).unwrap_or_default())]
    OutputBlocked { policy: Option<String> },

    // ---------- Protocol / Server Faults ----------
    /// A 5xx or server-reported internal error.
    #[error("server error{}{}",
        status.map(|s| format!(" ({s})")).unwrap_or_default(),
        hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    ServerError {
        status: Option<u16>,
        hint: Option<String>,
    },

    /// The server aborted the stream during generation.
    #[error("server aborted stream{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    ServerStreamAborted { hint: Option<String> },

    /// Failed to parse wire JSON / SSE.
    #[error("malformed wire response: {0}")]
    Malformed(#[source] BoxError),

    /// The server responded with a wire type or field not defined in the protocol
    /// specification.
    #[error("protocol violation: {hint}")]
    ProtocolViolation { hint: String },

    // ---------- transport ----------
    /// Transport-layer error (DNS, TCP, TLS, HTTP).
    #[error("transport error: {0}")]
    Transport(#[source] BoxError),

    /// Request timed out.
    #[error("request timeout at {phase:?}")]
    Timeout { phase: TimeoutPhase },

    // ---------- control flow ----------
    /// Canceled by the user or upper layer.
    #[error("canceled")]
    Canceled,

    // ---------- Catch-all ----------
    /// Catch-all variant; prefer to extract cases from here when adding new categories.
    #[error("other provider error: {0}")]
    Other(#[source] BoxError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitScope {
    /// Requests per minute.
    Rpm,
    /// Requests per minute.
    Tpm,
    /// Reported by the server but not further subdivided.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutPhase {
    Connect,
    ReadHeaders,
    ReadBody,
    Idle,
    Total,
}

/// Retry hints for errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryHint {
    /// Not retryable.
    No,
    /// Retry immediately once.
    Immediate,
    /// Retry after the server-suggested duration.
    After(Duration),
    /// Retry with backoff (no server suggestion).
    Backoff,
    /// Retry after performing a prerequisite action.
    AfterAction(RetryAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    RefreshAuth,
    SwitchModel,
    ReduceContext,
}
