//! 跨模块共用的 HTTP 基础设施。
//!
//! 在 `client_util::build_https_client` 之上 stack 一层薄壳：超时、
//! transport 抖动重试、HTTP/HTTPS 代理、统一 `User-Agent`。
//! HTTP client abstraction for the agent.
//!
//! 当前消费者：`defect-llm`（各 LLM provider）；规划中：`defect-tools`
//! 的 fetch tool。把这层独立成 crate 是为了避免后者再次依赖 `defect-llm`
//! 这种倒挂。
//!
//! 公共入口仅 [`build_http_stack`]、[`HttpStackConfig`]、[`HttpStack`]、
//! [`HttpStackError`]。具体 layer 实现在子模块里 `pub(crate)`，不暴露
//! 到 crate 之外——让上层调用方只见 type-erased Service。

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

/// `build_http_stack` 输出的类型擦除 service。
///
/// 输入 `toac::Request`，输出 `http::Response<hyper::body::Incoming>`，
/// 错误类型统一为 [`HttpStackError`]。每家 provider 把它喂给
/// `toac::ApiClient::new`。
///
/// 选 [`BoxCloneSyncService`] 而非 `BoxService`：toac 的 `tower::Service`
/// impl 要求 inner `S: Clone` 才能在 `poll_ready` 后克隆出未持有锁的副本
/// 走 future——见 toac `lib.rs` 的 `mem::replace` 模式。
pub type HttpStack =
    BoxCloneSyncService<toac::Request, http::Response<hyper::body::Incoming>, HttpStackError>;

/// HTTP 栈配置。
///
/// `Default::default()` 给 v0 推荐值——`total_timeout = 600s`、
/// `transport_retries = 2`、`initial_backoff = 200ms`、`user_agent = None`
/// （用编译期默认）、`proxy = ProxyConfig::FromEnv`。
#[derive(Debug, Clone)]
pub struct HttpStackConfig {
    /// 单次请求总超时。`None` 表示不限。SSE 流式响应在第一字节到达后
    /// 继续计时直到流结束——v0 默认 600s 覆盖 Anthropic extended thinking
    /// 的最长合理时长。
    pub total_timeout: Option<Duration>,

    /// transport 错误重试上限（不含首次）。`0` 禁用 retry layer。
    /// 仅重试 transport 抖动（DNS / TCP / TLS / hyper IO），HTTP 状态
    /// 任意值都视作"成功"放行——业务级重试在 turn-loop §7。
    pub transport_retries: u8,

    /// 重试初始 backoff。每次乘以 2、加 ±25% jitter，封顶 30s。
    pub initial_backoff: Duration,

    /// `User-Agent` header 值。`None` 时使用编译期默认
    /// （`defect-http/{version} ({git_sha[..8]})`）。
    pub user_agent: Option<String>,

    /// 代理配置。
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

/// 代理配置。
#[derive(Debug, Clone, Default)]
pub enum ProxyConfig {
    /// 从 env 读取 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`。
    #[default]
    FromEnv,
    /// 显式给定。
    Explicit(ProxySettings),
    /// 强制不走代理（即使 env 配了）。
    Disabled,
}

/// 显式代理设置。`http_proxy` / `https_proxy` 各自可为 `None`，
/// `no_proxy` 是域名后缀列表（参考 GNU `NO_PROXY` 风格）。
#[derive(Debug, Clone, Default)]
pub struct ProxySettings {
    pub http_proxy: Option<http::Uri>,
    pub https_proxy: Option<http::Uri>,
    pub no_proxy: Vec<String>,
}

/// HTTP 栈层错误。
///
/// 与 `toac::CallError<E>` 中的 `E` 对位——provider 在
/// `call_error_to_provider` 里把这层错翻成 `ProviderErrorKind`
/// （详见 HTTP retry/error semantics）。
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HttpStackError {
    /// transport 错误（DNS / TCP / TLS / hyper IO 等）。
    #[error("HTTP transport error: {0}")]
    Transport(#[source] BoxError),

    /// 请求超时。`phase` 标识在哪一阶段超时——v0 仅 `Total`，
    /// Staged timeouts for HTTP requests.
    #[error("HTTP request timed out (phase = {phase:?})")]
    Timeout { phase: TimeoutPhase },

    /// HTTP 栈配置错误（代理 URL 解析失败等）。
    #[error("HTTP layer config invalid: {hint}")]
    Config { hint: String },

    /// 代理 CONNECT 阶段失败。
    #[error("proxy CONNECT failed: {hint}")]
    ProxyConnect { hint: String },
}

/// 超时阶段。与 [`defect_agent::llm::TimeoutPhase`] 对位，但本 crate
/// 内部不引用 agent 的类型，避免 layer 实现耦合到 LLM 错误模型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimeoutPhase {
    Connect,
    ReadHeaders,
    ReadBody,
    Idle,
    Total,
}

/// 构造完整 HTTP 栈，输出可直接喂给 `toac::ApiClient::new`。
///
/// 当前 layer 顺序（外→内，请求方向）：
/// `UserAgent → Trace → Timeout? → hyper-util Client`
///
/// `Timeout` 仅在 `config.total_timeout = Some(_)` 时插入——`None` 直接
/// 跳过整个 timeout layer，避免 `tower::timeout` 把 Error 包成
/// [`tower::BoxError`] 时跟 `Identity` 类型不一致（`option_layer` 在
/// `None` 路径上不修改 Error 类型）。
pub fn build_http_stack(config: HttpStackConfig) -> Result<HttpStack, HttpStackError> {
    // 连接器层一次性合并 TLS + 代理：`ProxyConnector` 在没挂任何 entry
    // 时透明放行，所以 `Disabled` 也走同一份连接器类型，不引入 `if`
    // 分叉的两份 `HyperClient` 类型。
    let connector = proxy::build_proxy_connector(&config.proxy)?;
    let inner =
        HyperClient::builder(TokioExecutor::default()).build::<_, toac::body::Body>(connector);

    // hyper-util Client 的 Error → HttpStackError::Transport
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

/// 把 [`tower::timeout`] 引入的 [`tower::BoxError`] 翻回
/// [`HttpStackError`]：
/// - [`tower::timeout::error::Elapsed`] → `Timeout { phase: Total }`
/// - 其余应是 inner [`HttpStackError`]——`tower::timeout` 把它 box 起来
///   传出，`downcast` 还原即可
/// - 极端兜底（不应发生）→ `Transport`，保留原始来源
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
