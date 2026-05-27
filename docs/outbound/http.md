# HTTP 客户端基础设施

`crates/http/`（独立 `defect-http` crate，详见 §7）封装一层 **跨 crate 共用的 HTTP transport**——超时、transport 重试、代理、`User-Agent`。当前消费者：`defect-llm` 的各家 provider（`AnthropicProvider` / `OpenAiProvider` / `DeepSeekProvider`）。规划中：`defect-tools` 的 fetch tool。所有消费者通过同一份 [`build_http_stack`] 拿到包好 layer 的 `tower::Service`，再交给 `toac::ApiClient::new(...)`。

设计前提：

- toac 的 transport 是 BYO `tower::Service<http::Request<toac::body::Body>>`，所以这层基础设施天然能用 [tower 0.5](https://docs.rs/tower/0.5) 的 `ServiceBuilder` 一层套一层（[`docs/outbound/llm-anthropic.md`](./llm-anthropic.md) §1 已写明）。
- 当前已有的 `client-util::build_https_client::<Body>()` 直接返回 `hyper_util::client::legacy::Client`，本身就是 `tower::Service`——基础设施的形态是"在它上面再 stack 几层"，不替换它。
- LLM 主循环 [`turn-loop.md`](../internal/turn-loop.md) §7 已有**业务级**重试（按 `ProviderError::retry_hint` 处理 5xx / 429 / `AuthExpired` 等）。这层 HTTP 基础设施**不重做**那部分——它只兜底**纯 transport 抖动**（DNS / TCP / TLS / hyper 底层 IO 错误），与 turn-loop 的语义重试两层互不重叠。

---

## 1. 范围

| in scope（v0） | out of scope |
| --- | --- |
| 单次请求总超时（`Total` phase） | 连接 / 读 header / 读 body 分阶段超时（v1 加 [`TimeoutPhase`](../internal/llm-trait.md#724-错误归类) 时再细分） |
| Transport 错误重试（DNS / connect / TLS / hyper IO；指数退避 + jitter） | 5xx / 429 / `AuthExpired` 等业务级错误重试（在 turn-loop §7） |
| HTTP/HTTPS 代理（`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` env） | 鉴权代理（带 user/pass 的 proxy URL）；SOCKS5；PAC 脚本 |
| 统一 `User-Agent`（含版本号 + git sha） | 多组 UA 轮换；`X-Forwarded-*` 链 |
| `tracing::trace` HTTP method/path/status/elapsed | 完整 wire-level dump（`x-request-id`、retry attempt 计数已由 provider 层 / turn-loop §7 各自承担） |

明确**不做**的：

- **rate limiter**——Anthropic / OpenAI 的限流是服务端 429 + `Retry-After` 反推，本地预限会让服务端的 burst 信号失效，得不偿失。
- **circuit breaker**——单 provider 故障时由 turn-loop §7 用尽 `max_llm_retries` 后 `Err(TurnError::Provider)` 自然兜底；CLI / ACP 层会把错误抛给用户，不需要本地 trip 状态。
- **请求/响应缓存**——LLM 流式响应不可缓存；prompt cache 是服务端 feature。

## 2. 形态：tower 栈

```text
┌────────────────────────────────────────────┐
│  AnthropicProvider / OpenAiProvider        │
│  ↓ 持有                                     │
│  toac::ApiClient<S>                         │
│  ↓ Service<http::Request<Body>>             │
│  ┌────────────────────────────────────┐     │
│  │ defect-llm http::stack(config)     │     │
│  │                                     │     │
│  │   UserAgentLayer       (header)    │     │
│  │   ↓                                │     │
│  │   TraceLayer           (tracing)   │     │
│  │   ↓                                │     │
│  │   TimeoutLayer         (per-req)   │     │
│  │   ↓                                │     │
│  │   TransportRetryLayer  (DNS/conn)  │     │
│  │   ↓                                │     │
│  │   HyperHttpsClient<Body>           │     │
│  │     - HttpsConnector(Proxy?)       │     │
│  └────────────────────────────────────┘     │
└────────────────────────────────────────────┘
```

公共入口（`crates/http/src/lib.rs`）：

```rust
pub struct HttpStackConfig {
    /// 单次请求总超时。`None` 表示不限（仅 transport 自身超时，
    /// 如 hyper 默认的 keep-alive idle，按 hyper 默认）。
    pub total_timeout: Option<Duration>,
    /// transport 重试上限（不含首次）。0 表示禁用 retry layer。
    pub transport_retries: u8,
    /// 重试初始 backoff。每次乘以 2，加 ±25% jitter，封顶 30s。
    pub initial_backoff: Duration,
    /// `User-Agent` header 值。`None` 时使用编译期默认。
    pub user_agent: Option<String>,
    /// 代理配置。`None` 时 `from_env()` 决定是否启用。
    pub proxy: ProxyConfig,
}

pub enum ProxyConfig {
    /// 从 env 读取 HTTP_PROXY / HTTPS_PROXY / NO_PROXY。
    FromEnv,
    /// 显式给定。
    Explicit(ProxySettings),
    /// 强制不走代理（即使 env 配了）。
    Disabled,
}

pub struct ProxySettings {
    pub http_proxy: Option<Url>,
    pub https_proxy: Option<Url>,
    pub no_proxy: Vec<String>,  // 域名后缀列表
}

/// 构造完整 HTTP 栈，输出可直接喂给 `toac::ApiClient::new`。
pub fn build_http_stack(
    config: HttpStackConfig,
) -> Result<HttpStack, HttpStackError>;

pub type HttpStack =
    Pin<Box<dyn tower::Service<
        http::Request<toac::body::Body>,
        Response = http::Response<toac::body::Incoming>,
        Error = HttpStackError,
        Future = BoxFuture<'static, Result<_, _>>,
    > + Clone + Send + Sync + 'static>>;
```

`HttpStack` 是类型擦除的 `tower::Service`——理由同 [`llm-trait.md`](../internal/llm-trait.md) §2.1：让多家 provider 都把它装进同一个槽，且不让 layer 栈类型签名污染 provider 字段（hyper / tower 的具体类型展开有 ~5 层泛型，写在 `AnthropicProvider` 字段里 review 时全是噪音）。

每次方法调用一次 `Box` 的开销在 LLM 网络 IO（数百 ms）下不可测（同 §2.1 已论证）。

## 3. Layer 详细设计

### 3.1 TransportRetryLayer

只重试 **transport 抖动**——服务端任何 HTTP 响应都视作"成功"放行，让上层 provider/turn-loop 解析。

```rust
struct TransportRetryLayer {
    max_retries: u8,
    initial_backoff: Duration,
}

struct TransportRetry<S> { inner: S, ... }

impl<S, B> tower::Service<http::Request<B>> for TransportRetry<S>
where
    S: tower::Service<http::Request<B>, Error = HttpStackError> + Clone,
    B: Clone,  // ←⚠ 见下文 "幂等性约束"
{ ... }
```

#### 重试触发条件

| `HttpStackError` 来源 | 是否重试 |
| --- | --- |
| `hyper_util::client::legacy::Error::Connect(_)` | ✅ |
| DNS 解析失败（`ConnectError::Dns`） | ✅ |
| TLS 握手失败（`rustls::Error`） | ✅ |
| `io::ErrorKind::ConnectionRefused / ConnectionReset / TimedOut` | ✅ |
| EOF before headers received（hyper `IncompleteMessage`） | ✅（仅请求**已发出但未读到 status line**时；headers 已收完则不动） |
| HTTP status 任意值（200 / 4xx / 5xx） | ❌（不算 transport 错误） |
| 请求体 stream 错误 | ❌（请求体可能已部分送出，重试会引发副作用） |

实现上判定逻辑放在 `is_transport_retryable(&HttpStackError) -> bool` 单测可达。

#### 退避策略

```text
attempt 1 失败 → wait initial_backoff      ± 25% jitter
attempt 2 失败 → wait initial_backoff * 2  ± 25% jitter
attempt 3 失败 → wait initial_backoff * 4  ± 25% jitter
...
封顶 30s。`max_retries` 用尽后向上抛最后一次错误。
```

默认 `max_retries=2`、`initial_backoff=200ms`——两次重试覆盖单点 DNS 抖动 + 单次 TLS 重协商，再多就该让 provider 看到错误（用户 ctrl+c / turn-loop 失败计数器更适合做长时段决策）。

#### 幂等性约束

LLM API 的请求体是 `serde_json` 序列化后的 `Bytes`——天然 `Clone`，没有副作用问题。tower 的 `Retry` Layer 标准做法用 `Clone` bound 表达。但 SSE 响应一旦开始流式回传，**不能**重试——一旦 hyper 读到 status line 后，后续错误都属于"流中断"而非"未送达"。`TransportRetry` 内部用 `oneshot::channel` 包装：仅在 inner future poll 出错且 status 还没出现时算 transport 错误。

### 3.2 TimeoutLayer

直接复用 `tower::timeout::TimeoutLayer`。封装目的：

- **统一超时错误形态**——把 `tower::timeout::error::Elapsed` 包成 `HttpStackError::Timeout { phase: Total }`，便于上层映射到 [`ProviderErrorKind::Timeout { phase: Total }`](../internal/llm-trait.md#724-错误归类)。
- **可选化**——`config.total_timeout = None` 时跳过此 layer（让 hyper 默认行为生效）。

`Total` 是相对**整次 HTTP 请求**的——SSE 流式响应在第一字节到达后**继续**计时直到流结束。这与 LLM 思考链长流不冲突：v0 默认 600s（10 分钟），覆盖 Anthropic extended thinking 的最长合理时长。`AnthropicConfig` / `OpenAiConfig` 后续可以暴露 `request_timeout` 字段覆盖默认值——v0 不开口子，按 600s 写死。

### 3.3 UserAgentLayer

```rust
struct UserAgent {
    value: HeaderValue,  // 固定值，构造时一次性算好
    inner: ...,
}
```

每次调用 `inner.call(req)` 之前，向 `req.headers_mut()` 写入 `User-Agent`（不覆盖，若 provider 已经写了就跳过——`headers.entry(USER_AGENT).or_insert(...)` 语义）。

默认值：`defect-llm/{CARGO_PKG_VERSION} ({git_sha[..8]})`，git sha 由 `build.rs` 注入；`build.rs` 拿不到时退化为 `defect-llm/{version}`。

### 3.4 TraceLayer

**v0 不写自己的 layer**——直接用 [`tower-http`](https://docs.rs/tower-http) 的 `trace::TraceLayer::new_for_http()`，把 `tower-http = { version = "0.6", features = ["trace"] }` 加进 workspace。`tracing` event 走我们已有的 EnvFilter（[`docs/outbound/tracing.md`](./tracing.md)），通过 `RUST_LOG=defect_llm::http=trace` 打开。

如果 `tower-http` 的字段集合（method / uri / status / latency）不够，再独立写一份——那时再决定是否提取出 layer。v0 优先少写代码。

### 3.5 ProxyConnector

代理装在 **connector 层**，不是 service 层（service 层已是 hyper-util `Client` 之外）。形态（实际实现见 `crates/http/src/proxy.rs::build_proxy_connector`）：

```rust
use hyper_http_proxy::{Proxy, ProxyConnector, Intercept};
use hyper_rustls::HttpsConnectorBuilder;

fn build_proxy_connector(
    config: &ProxyConfig,
) -> Result<HttpsConnector<ProxyConnector<HttpConnector>>, HttpStackError> {
    let entries = resolve_proxy(config)?;
    // ⚠ enforce_http(false)：见下文"两个必踩的坑"#1。
    let mut http_connector = HttpConnector::new();
    http_connector.enforce_http(false);
    // ⚠ unsecured：见下文"两个必踩的坑"#2。
    let mut proxy_connector = ProxyConnector::unsecured(http_connector);
    for entry in entries {
        proxy_connector.add_proxy(Proxy::new(entry.intercept, entry.uri));
    }
    Ok(HttpsConnectorBuilder::new()
        .with_native_roots()?
        .https_or_http()
        .enable_all_versions()
        .wrap_connector(proxy_connector))
}
```

`ProxyConfig::Disabled` 也走这条路径，但 `entries` 是空 `Vec`——`ProxyConnector` 在 `match_proxy` 找不到任何 entry 时透明放行（见上游 `Service<Uri>` impl），所以连接器类型保持一致 `HttpsConnector<ProxyConnector<HttpConnector>>`，避免 `build_http_stack` 出现两份不同的连接器类型。

#### 两个必踩的坑

1. **内层 `HttpConnector` 必须 `enforce_http(false)`。** 默认 `HttpConnector` 拒绝 `https` scheme，`hyper-rustls::HttpsConnectorBuilder::build()` 会自己改这个 flag，但 `wrap_connector(_)` 不会；走 `https://` 时内层会先 `Err(InvalidUri/scheme is not http)`，根本到不了 TLS 阶段。`crates/http/src/proxy.rs::build_proxy_connector_does_not_reject_https_when_no_proxy_match` 单测专门盯这条回归。
2. **`ProxyConnector::unsecured(_)`，不是 `ProxyConnector::new(_)`。** 一旦 `hyper-http-proxy` 的 `__rustls`（任何 `rustls-tls-*-roots` feature）被打开，`ProxyConnector::new` 会内置一份 `tokio_rustls::TlsConnector`，并在 CONNECT 隧道之上**自己**做一次 TLS 握手，返回 `ProxyStream::Secured`。我们外层 `HttpsConnector::wrap_connector(_)` 会把这条已加密流再包一次 TLS——TLS-in-TLS，外层握手永远读不到 ServerHello（trace 里只看到 `ALPN protocol is None`，~14s 后超时为 `client error (Connect)`）。`unsecured` 让 ProxyConnector 只负责 CONNECT + 原始 TCP（返回 `ProxyStream::Regular`），TLS / ALPN 全交给外层 `HttpsConnector`。所以 workspace 的 `hyper-http-proxy` 依赖**关掉**所有 `rustls-tls-*-roots` feature。

#### 选型：`hyper-http-proxy`

候选生态及其与 workspace 已锁版本的相容性：

| 选择 | 与 workspace 版本耦合 | 维护状态 | 备注 |
| --- | --- | --- | --- |
| **`hyper-http-proxy 1.1.0`（metalbear-co）** ✅ | `rustls-tls-native-roots` feature 拉 `hyper-rustls 0.27`——**正好等于** workspace 已锁的 0.27.9，不引入版本分裂 | 持续维护 | hyper 1.x 生态 |
| `hyper-proxy2 0.1.0`（siketyan） | `rustls` feature 拉 `hyper-rustls 0.26`——与 workspace 0.27.9 版本分裂，编译两份 | 活跃但 0.x 不稳 | 否决 |
| `reqwest::Proxy` | reqwest 已在 workspace（`rmcp` 透明依赖），但 `reqwest::Proxy` 非独立 connector——要用得把 toac 的 HTTP 客户端整体切到 reqwest | — | 架构级改动，超出本提案 |
| 手写 `tower::Service<Uri>` CONNECT tunnel | 无新依赖，但要自己维护 ~150 行（CONNECT 帧 + auth header + NO_PROXY） | — | 不值得 |

**v0 选 `hyper-http-proxy 1.1.0`**：

- 不引入版本分裂——`hyper-rustls 0.27` / `tokio-rustls 0.26` / `rustls 0.23` 三件全部对齐 workspace 现有 lock。
- 上游已经实现 CONNECT tunnel + `Intercept::Http` / `Https` / `All` / `Custom(Fn)` 路由 + headers 注入；NO_PROXY 由我们在 `resolve_proxy` 里把 host 翻译成 `Intercept::Custom(scheme_and_host_matcher)` 实现。
- 出现 SOCKS5 / PAC 等更复杂需求时换更专的 crate，本接口不暴露 `hyper-http-proxy` 类型（仅在 `crates/http/src/proxy.rs` 内部用），切换零外部影响。

#### NO_PROXY 匹配

`resolve_proxy` 把 `ProxyConfig::FromEnv` / `Explicit` 翻译成 `Vec<(Intercept, Uri)>`：把 `NO_PROXY` 列表写成 `Intercept::Custom(closure)`，闭包内匹配规则按 [GNU 风格](https://about.gitlab.com/blog/we-need-to-talk-no-proxy/)：逗号分隔，每项域名后缀（`api.openai.com` 匹配 `*.openai.com`），`*` 等价 disable，IP 段（CIDR）v0 不做。

匹配函数 `crates/http/src/proxy.rs::matches_no_proxy(host: &str, patterns: &[String]) -> bool` 单测覆盖；`hyper-http-proxy` 的 `Intercept::Custom` 接 `Fn(Option<&str>, Option<&str>, Option<u16>) -> bool`，闭包内同时校验 scheme + NO_PROXY。

## 4. 错误形态

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HttpStackError {
    #[error("HTTP transport error: {0}")]
    Transport(#[source] BoxError),

    #[error("HTTP request timed out (phase = {phase:?})")]
    Timeout { phase: TimeoutPhase },

    #[error("HTTP layer config invalid: {hint}")]
    Config { hint: String },

    #[error("proxy CONNECT failed: {hint}")]
    ProxyConnect { hint: String },
}
```

**与 provider 层错误映射的接缝**：toac 的 `CallError<E>` 中的泛型 `E` 即是 `HttpStackError`。各 provider 在已有的 `call_error_to_provider` 里追加分支：

```rust
match err {
    CallError::Transport(HttpStackError::Timeout { phase })
        => ProviderError::new(ProviderErrorKind::Timeout { phase }),
    CallError::Transport(HttpStackError::ProxyConnect { hint })
        => ProviderError::new(ProviderErrorKind::Transport(BoxError::new(...))),
    CallError::Transport(other)
        => ProviderError::new(ProviderErrorKind::Transport(BoxError::new(other))),
    // ... Encode / Auth / Decode 不变
}
```

`Timeout` 这条是关键——之前 [`llm-trait.md`](../internal/llm-trait.md) §7.2 已经定义 `Timeout { phase }`，但当前 provider 实现没有路径产出，因为 hyper 默认行为是无限等待。这层加上后 `ProviderErrorKind::Timeout { phase: Total }` 才真正能被生成。

## 5. 与 provider 的接入

每家 provider 的 `new` 减一行加两行：

```rust
// before
let http = client_util::client::build_https_client::<Body>()?;
let client = ApiClient::new(http, base_url).with_auth(auth);

// after
let http = defect_http::build_http_stack(config.http.clone())?;
let client = ApiClient::new(http, base_url).with_auth(auth);
```

`AnthropicConfig` / `OpenAiConfig` 各自加一个字段：

```rust
pub struct OpenAiConfig {
    // ... 原字段
    /// HTTP 栈配置。默认 `HttpStackConfig::default()`。
    pub http: HttpStackConfig,
}
```

`HttpStackConfig::default()` 给出"v0 推荐值"：`total_timeout = Some(600s)`、`transport_retries = 2`、`initial_backoff = 200ms`、`user_agent = None`（用编译期默认）、`proxy = ProxyConfig::FromEnv`。

构造期 env 读不到 proxy 时一切照旧，不报错——只有 env **有**且**parse 失败**时报 `HttpStackError::Config`。

## 6. 测试策略

| 测试 | 形态 | 位置 |
| --- | --- | --- |
| `is_transport_retryable` + retry 行为单测（含 backoff 范围） | 普通 `#[test]` / `#[tokio::test]` | `crates/http/src/retry.rs::tests` |
| `matches_no_proxy` 表驱动单测 + `Intercept::Custom` 闭包语义 | 普通 `#[test]` | `crates/http/src/proxy.rs::tests` |
| `default_user_agent` 形态校验 | 普通 `#[test]` | `crates/http/src/user_agent.rs::tests` |
| Retry e2e：wiremock + 自定义 connector 注入故障 | _v1 计划_，未落地 | `crates/http/tests/http_retry_e2e.rs` |
| Timeout e2e：服务端 sleep 超过 timeout | _v1 计划_，未落地 | `crates/http/tests/http_timeout_e2e.rs` |
| Proxy CONNECT round-trip e2e | _v1 计划_，未落地 | `crates/http/tests/http_proxy_e2e.rs` |

**不**做：

- 真打 OpenAI / Anthropic 验证 retry——examples/ 里的 smoke 已经打了；本基础设施在 wiremock 这边能跑通就够。
- 模糊测试 retry backoff 的 jitter 分布——`rand::thread_rng()` 行为不在 v0 测试范围。

## 7. 与 crate 边界的关系

**已落地于独立 `defect-http` crate**（`crates/http/`），不挂在 `defect-llm` 下。原因：

- 同时还规划一个 fetch tool（`defect-tools` crate 内）也走这套 stack；如果 HTTP 基础设施挂在 `defect-llm` 下，`defect-tools` 就得倒挂依赖 `defect-llm`。提前独立避免后续大重构。
- 公共 API 表面（`build_http_stack` / `HttpStackConfig` / `HttpStack` / `HttpStackError` / `ProxyConfig` / `ProxySettings` / `TimeoutPhase`）显式 `pub`；layer 实现（`UserAgentLayer` / `TraceLayer` / `TransportRetryLayer` / `proxy::*`）一律 `pub(crate)`，不暴露到 crate 之外。
- 依赖图代价可接受——`tower` / `hyper-util` / `hyper-rustls` / `hyper-http-proxy` 都已是 workspace 现成依赖，独立 crate 只是给它们一个共享 home，没新增 transitive。

⚠️ MCP HTTP transport（[`docs/outbound/mcp-client.md`](./mcp-client.md) §SSE 部分）目前仍用 `rmcp` 自带 transport，没切到 `defect-http`——切换是 v1 议程。

## 8. 落地步骤

1. ✅ **`defect-http` crate 骨架**——空入口、`HttpStackConfig::default()`、`build_http_stack` 仅装 hyper-util client。两家 provider（含 deepseek 通过 openai 复用）切到新入口。
2. ✅ **UserAgent + Trace + Timeout layer**——按 §3.2 / §3.3 / §3.4。`TraceLayer` 不复用 `tower-http`（会把 response body 包成新类型，破坏 `Response<hyper::body::Incoming>` 类型签名），改写 ~110 行自家 layer。
3. ✅ **TransportRetryLayer**——按 §3.1 实现，先 `body.collect()` 成 `Bytes` 再每次 attempt 重建 `Request`（LLM JSON 请求几 KB，buffer 代价可忽略）。9 条单测覆盖触发条件、上限、backoff cap。
4. ✅ **Proxy connector**——`hyper-http-proxy 1.1.0` + `Intercept::Custom` 闭包包 NO_PROXY；连接器统一是 `HttpsConnector<ProxyConnector<HttpConnector>>`，`ProxyConfig::Disabled` 用空 entry list 透明放行。15 条单测覆盖 GNU 风格 NO_PROXY 匹配。
5. ✅ **`HttpStackError::Timeout` → `ProviderErrorKind::Timeout`**——两家 provider 各自把 `call_error_to_provider` 从泛型 `<E>` 收紧为 `CallError<HttpStackError>`，新增 `Timeout { phase }` 分支并把 `defect_http::TimeoutPhase` 翻成 agent 层的 `TimeoutPhase`。

后续 v1 议程见 §6 表格里标 _未落地_ 的三项 e2e。

## 9. 待续

- **流式响应中段错误的 retry 策略**——SSE 流到一半 server 关连接，目前由 turn-loop §7 兜底（重新发整次请求）。基础设施层是否值得做"resume from last event id"v1 再考虑（取决于服务端是否给 `Last-Event-ID` 支持，Anthropic / OpenAI 都不给，所以这条短期不会实现）。
- **per-host 连接池调优**——`hyper_util::client::legacy::Client::builder().pool_max_idle_per_host(...)`。默认值（无上限）在长跑 ACP 会话里没观测到问题，先不动。
- **HTTP/2 设置**——`enable_all_versions` 已经开了 HTTP/2，但 `http2_max_concurrent_streams` 等高级参数没暴露口子。同上：默认够用，等观测到瓶颈再加。
