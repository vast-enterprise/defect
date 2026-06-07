//! Provider error types.

use std::time::Duration;

use thiserror::Error;

use crate::error::BoxError;

/// provider 操作失败的统一错误。
///
/// 顶层 struct 拆分原因：把 cross-cutting 的诊断信息（`request_id` 等）
/// 与分类信息（`kind`）分开，避免在每个 variant 中重复 `request_id` 字段。
#[derive(Debug, Error)]
#[error("{kind}")]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    /// 服务端返回的 request id（Anthropic `request-id` header /
    /// OpenAI `x-request-id` 等）。排障第一信号源，应尽力填充。
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

    /// 给出此错误的重试建议。
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

    /// 便捷判断：是否值得让 agent 自动重试。
    pub fn is_retryable(&self) -> bool {
        !matches!(self.retry_hint(), RetryHint::No)
    }
}

/// provider 错误的语义分类。
///
/// "兜底"原则：如果发现一类错误反复落入 [`ProviderErrorKind::Other`]，
/// 应当**优先把它提取为新的 variant**，而不是让 `Other` 变成默认值。
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ProviderErrorKind {
    // ---------- 认证 ----------
    /// 没配凭证。
    #[error("missing credential{}", var_hint.as_deref().map(|h| format!(" (hint: {h})")).unwrap_or_default())]
    AuthMissing { var_hint: Option<String> },

    /// 凭证格式错误。
    #[error("malformed credential{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    AuthMalformed { hint: Option<String> },

    /// 凭证被服务端拒绝（401）。
    #[error("credential rejected by server{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    AuthRejected { hint: Option<String> },

    /// OAuth/STS token 过期。
    #[error("auth token expired")]
    AuthExpired,

    // ---------- 配额 ----------
    /// 请求级速率限制。
    #[error("rate limit hit ({scope:?}){}", retry_after.map(|d| format!(", retry after {}s", d.as_secs())).unwrap_or_default())]
    RateLimit {
        retry_after: Option<Duration>,
        scope: RateLimitScope,
    },

    /// 余额不足 / 月度配额耗尽。
    #[error("quota exceeded{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    QuotaExceeded { hint: Option<String> },

    // ---------- 输入 ----------
    /// context window 撑爆。
    #[error("context overflow{}", match (used, limit) {
        (Some(u), Some(l)) => format!(" ({u} > {l})"),
        _ => String::new(),
    })]
    ContextOverflow {
        used: Option<u64>,
        limit: Option<u64>,
    },

    /// 单次 max_tokens 超过模型上限或被服务端拒。
    #[error("max_tokens invalid{}", match (requested, limit) {
        (Some(r), Some(l)) => format!(" ({r} > {l})"),
        _ => String::new(),
    })]
    MaxTokensInvalid {
        requested: Option<u64>,
        limit: Option<u64>,
    },

    /// 模型 ID 不存在 / 不可用。
    #[error("model not found: {model}")]
    ModelNotFound { model: String },

    /// 请求体被 wire 服务校验拒绝（schema 错误、互斥字段冲突）。
    #[error("bad request{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    BadRequest { hint: Option<String> },

    /// 请求里引用的工具 schema 自身被服务端拒绝。
    #[error("invalid tool schema for {tool}{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    InvalidToolSchema { tool: String, hint: Option<String> },

    // ---------- 安全/合规 ----------
    /// 输入触发安全过滤器。
    #[error("input blocked{}", policy.as_deref().map(|p| format!(" by {p}")).unwrap_or_default())]
    InputBlocked { policy: Option<String> },

    /// 模型生成被安全过滤器中断。
    #[error("output blocked{}", policy.as_deref().map(|p| format!(" by {p}")).unwrap_or_default())]
    OutputBlocked { policy: Option<String> },

    // ---------- 协议/服务端故障 ----------
    /// 5xx 或服务端报告的内部错误。
    #[error("server error{}{}",
        status.map(|s| format!(" ({s})")).unwrap_or_default(),
        hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    ServerError {
        status: Option<u16>,
        hint: Option<String>,
    },

    /// 服务端在生成中切流。
    #[error("server aborted stream{}", hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default())]
    ServerStreamAborted { hint: Option<String> },

    /// wire JSON / SSE 解析失败。
    #[error("malformed wire response: {0}")]
    Malformed(#[source] BoxError),

    /// 服务端响应了未在协议规范内的 wire 类型/字段。
    #[error("protocol violation: {hint}")]
    ProtocolViolation { hint: String },

    // ---------- 传输 ----------
    /// DNS / TCP / TLS / HTTP 层错误。
    #[error("transport error: {0}")]
    Transport(#[source] BoxError),

    /// 请求超时。
    #[error("request timeout at {phase:?}")]
    Timeout { phase: TimeoutPhase },

    // ---------- 控制流 ----------
    /// 用户/上层主动取消。
    #[error("canceled")]
    Canceled,

    // ---------- 兜底 ----------
    /// 未归类。实现新增分类时优先把此处的 case 提取出去。
    #[error("other provider error: {0}")]
    Other(#[source] BoxError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitScope {
    /// 每分钟请求数。
    Rpm,
    /// 每分钟 token 数。
    Tpm,
    /// 服务端报告但未细分。
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

/// 错误的重试建议。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryHint {
    /// 不可重试。
    No,
    /// 立刻重试一次。
    Immediate,
    /// 等服务端建议时长后重试。
    After(Duration),
    /// 退避重试（无服务端建议）。
    Backoff,
    /// 需先做某事再重试。
    AfterAction(RetryAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    RefreshAuth,
    SwitchModel,
    ReduceContext,
}
