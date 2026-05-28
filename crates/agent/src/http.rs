//! HTTP fetch backend抽象。
//!
//! [`HttpClient`] 是 `fetch` 工具与底层 HTTP 栈之间的 trait 边界。具体
//! 实现来自 [`defect-http`]——session 装配时由 CLI 把 `Arc<dyn HttpClient>`
//! 注入给 [`crate::session::AgentCore`]，再透过 [`crate::tool::ToolContext`]
//! 传给工具。
//!
//! 设计详见 `docs/internal/tools-fetch.md` §6.1。
//!
//! 与 [`crate::fs::FsBackend`] / [`crate::shell::ShellBackend`] 不同——HTTP
//! 没有 per-client capability 协商，所以 [`HttpClient`] 是进程级共享而非
//! per-session 装配；只是借同一份 `Arc<dyn …>` 注入模式，避免引入新的
//! 注入路径。
//!
//! [`defect-http`]: ../../../crates/http

use std::time::Duration;

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// 一次 HTTP fetch 请求。
///
/// v0 仅 `GET`——`fetch` 工具的 schema 也只暴露读取语义，不暴露 method /
/// header / body / auth。
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// 绝对 `http://` / `https://` URL。其它 scheme 由 fetch 工具层提前拒绝。
    pub url: String,
    /// 单次请求总超时；`None` 让 backend 用栈层默认。
    pub timeout: Option<Duration>,
    /// 是否跟随 3xx Location。`false` 时把 3xx 当终态返回。
    pub follow_redirects: bool,
    /// 最多 follow 几跳；`follow_redirects = false` 时被忽略。
    pub max_redirects: u32,
    /// body 累积上限——超出即截断，`HttpResponse::truncated = true`。
    pub max_response_bytes: u64,
}

/// 一次成功获取的响应。
///
/// `status` 是 final response（follow 后）的状态码，`final_url` 同理。
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    /// `content-type` header 原文（去掉 boundary / charset 等参数前的主类型由
    /// 工具层自行解析）；`None` 表示 server 没设。
    pub content_type: Option<String>,
    /// 已截断到 `max_response_bytes` 之内的 body。
    pub body: Vec<u8>,
    /// server 实际下发的字节数（不含截断后丢掉的——backend 在截断时停止读，
    /// 不准确，作为提示用）。
    pub bytes_received: u64,
    /// `true` 表示 body 因为超过 `max_response_bytes` 被截断。
    pub truncated: bool,
    /// follow 的跳数。0 表示首次响应即终态。
    pub redirects: u32,
    /// follow 完后的最终 URL；不 follow 则与 `request.url` 相同。
    pub final_url: String,
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum HttpClientError {
    /// URL 无法解析（scheme 错、host 缺失等）。
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// 单次请求总超时触发。
    #[error("http request timed out")]
    Timeout,

    /// 超过 `max_redirects` 跳。携带实际尝试的跳数。
    #[error("too many redirects ({0})")]
    TooManyRedirects(u32),

    /// transport 层错误（DNS / connect / TLS / IO）；source 是底层 error。
    #[error("http transport error: {0}")]
    Transport(#[source] BoxError),
}

/// HTTP fetch 后端 trait。
///
/// 实现者必须满足以下契约：
/// - `fetch` 必须在内部实现 `req.timeout` 的总超时（含 connect / read body）；
///   超时返回 [`HttpClientError::Timeout`]。
/// - 当 `req.follow_redirects = true` 时按 RFC 7231 follow 3xx，最多
///   `req.max_redirects` 跳；超过则 [`HttpClientError::TooManyRedirects`]。
/// - 读 body 时累加到 `req.max_response_bytes` 即停止，并在响应里设
///   `truncated = true`。
/// - HTTP status 任何值（含 4xx/5xx）都视为成功（[`HttpResponse::status`]
///   照实带回），只有 transport / decode 失败才返回 `Err`。
pub trait HttpClient: Send + Sync {
    fn fetch(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, HttpClientError>>;
}

/// 测试 / `echo` provider 的占位实现。任何 `fetch` 调用都返回
/// [`HttpClientError::Transport`]——让需要 `Arc<dyn HttpClient>` 的装配
/// 路径能跳过真实 HTTP 栈构造。
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
