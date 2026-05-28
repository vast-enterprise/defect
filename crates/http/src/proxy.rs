//! HTTP/HTTPS 代理 connector 装配。
//!
//! 设计详见 [`docs/outbound/http.md`] §3.5。
//!
//! 形态：连接层装一份 [`hyper_http_proxy::ProxyConnector<HttpConnector>`]，
//! 外面再用 [`hyper_rustls::HttpsConnector`] 包出 TLS。`ProxyConnector`
//! 在 `proxies` 列表为空时透明放行（见上游 [`Service<Uri>`] impl 的
//! `match_proxy` 分支），所以无论用户是否启用代理，连接器类型保持一致——
//! `HttpsConnector<ProxyConnector<HttpConnector>>`——避免 [`build_http_stack`]
//! 出现两份不同的连接器类型。
//!
//! NO_PROXY：把每条代理 entry 的 [`Intercept`] 写成 [`Intercept::Custom`]
//! 闭包，闭包内匹配 scheme + host 并对照 `NO_PROXY` 后缀列表。匹配规则
//! 按 [GNU 风格](https://about.gitlab.com/blog/we-need-to-talk-no-proxy/)：
//! 逗号分隔、域名后缀（`api.openai.com` 命中 `*.openai.com`）、`*` 等价
//! 全禁、IP CIDR / 端口 v0 不做。
//!
//! [`build_http_stack`]: super::build_http_stack

use std::env;
use std::sync::Arc;

use http::Uri;
use hyper_http_proxy::{Intercept, Proxy, ProxyConnector};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::connect::HttpConnector;

use super::{HttpStackError, ProxyConfig, ProxySettings};

/// 完整连接器类型——上层用这个类型构造 [`hyper_util::client::legacy::Client`]。
pub(crate) type ProxyAwareConnector = hyper_rustls::HttpsConnector<ProxyConnector<HttpConnector>>;

/// 从 [`ProxyConfig`] 构造完整连接器。
///
/// - `Disabled` → 仍返回 `ProxyConnector`，但不挂任何 entry，`match_proxy`
///   始终 `None`，行为等价"无代理"。
/// - `FromEnv` → 读 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`（大小写
///   都接受，小写优先，对齐 curl 习惯）。
/// - `Explicit` → 直接用给定值。
pub(crate) fn build_proxy_connector(
    config: &ProxyConfig,
) -> Result<ProxyAwareConnector, HttpStackError> {
    let entries = resolve_proxy(config)?;

    // ⚠ 必须 `enforce_http(false)`：默认 `HttpConnector` 拒绝 https scheme，
    // 走 `https://` 时由外层 `HttpsConnector` 接管 TLS、内层 `HttpConnector`
    // 仅负责 TCP。`ProxyConnector` 在没有命中 proxy entry 时透传给内层
    // `HttpConnector`（见上游 `Service<Uri>` impl 的 fallthrough 分支），
    // 默认 `enforce_http=true` 会让所有 `https://` 直接 `Err(InvalidUri)`。
    // hyper-rustls 自家的 `HttpsConnectorBuilder::build()` 也是这么改的，
    // 但 `wrap_connector(_)` 不会替我们改自定义连接器，需要手动设置。
    let mut http_connector = HttpConnector::new();
    http_connector.enforce_http(false);

    // ⚠ 必须 `unsecured`：开启 `__rustls`（任何 `rustls-tls-*-roots` feature）
    // 时 `ProxyConnector::new` 会内置一份 `tokio_rustls::TlsConnector`，并在
    // CONNECT 隧道之上**自己**做一次 TLS 握手，返回 `ProxyStream::Secured`。
    // 我们外层 `HttpsConnector::wrap_connector(_)` 会把这条已经加密的流再包
    // 一次 TLS——TLS-in-TLS，外层握手永远读不到 ServerHello，~14s 后超时。
    // 用 `unsecured` 关闭 ProxyConnector 自己的 TLS，让它只负责 CONNECT 隧道
    // + 原始 TCP（返回 `ProxyStream::Regular`），TLS 完全由外层
    // `HttpsConnector` 统一负责（HTTP/2 ALPN 也在那一层完成）。
    // 因此 workspace 把 `hyper-http-proxy` 的 `rustls-*-roots` feature 全关。
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

/// 解析后的单条代理 entry。
struct ResolvedProxy {
    intercept: Intercept,
    uri: Uri,
}

/// 把 [`ProxyConfig`] 翻成 `(Intercept, Uri)` 列表。
///
/// 没有代理时返回空列表（合法状态）；URI 解析失败 → `HttpStackError::Config`。
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

/// 读取 env 变量并 parse 成 [`Uri`]。优先读小写、回退大写——这是
/// curl / Go / requests 等主流客户端的事实约定。
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

/// `Intercept::Custom`：scheme 命中且 host 不在 NO_PROXY 列表里时
/// 才走代理。
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

/// 解析逗号分隔的 NO_PROXY 字符串。空白裁剪、跳过空项。
fn parse_no_proxy(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// 列表里出现 `*` → 等价禁用所有代理。
fn no_proxy_disables_all(patterns: &[String]) -> bool {
    patterns.iter().any(|p| p == "*")
}

/// 判断 `host` 是否被 NO_PROXY 列表豁免。
///
/// GNU 风格：每条 pattern 是域名（前导/尾随 `.` 都被剥掉）；
/// `host` 命中条件之一即豁免：
/// - `host == pattern`（去前缀点后）
/// - `host` 以 `.<pattern>` 结尾
///
/// `*` 已在 [`no_proxy_disables_all`] 提前处理，这里不再特判。
/// 端口（`example.com:8080`）/ IP CIDR v0 不做——pattern 中带 `:` / 数字
/// 网段都按字面对比，匹配不到就是不豁免（行为安全：宁可走代理也不
/// 假装匹配）。
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
        // GNU 风格：pattern 不带前导点也按后缀匹配。
        assert!(matches_no_proxy("api.openai.com", &pats(&["openai.com"])));
    }

    #[test]
    fn substring_does_not_match() {
        // "openai" 不应匹配 "myopenai.com"——必须以 "." 边界结尾。
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
        // 来自 `,foo.com,` 这种边角输入——parse_no_proxy 已经过滤，但
        // matches_no_proxy 收到也得幂等。
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
        // 回归测试：之前 `wrap_connector(ProxyConnector::new(HttpConnector::new()))`
        // 漏掉了 `enforce_http(false)`，导致没命中 proxy entry 的 https 请求
        // 在 `HttpConnector::call` 阶段直接 `Err(InvalidUri/scheme is not http)`，
        // 还没走到 TLS。这里直接 poll 一次连接，断言我们**没有**拿到那条
        // 错误——真正的 DNS / 拒连失败是允许的（不联网）。
        use http::Uri;
        use tower::{Service, ServiceExt};

        let connector = build_proxy_connector(&ProxyConfig::Disabled).expect("build");
        let uri: Uri = "https://example.invalid/".parse().unwrap();
        let svc = connector.ready_oneshot().await;
        // ready_oneshot 失败说明连接器自身 ready 不出来——目前 hyper-rustls
        // / hyper-util 的 ready 都是恒 ready，所以这里不该 panic。
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
        // 直接验证闭包语义：scheme 不匹配 → false；scheme 匹配但 host
        // 在 NO_PROXY → false；scheme 匹配且 host 不在 NO_PROXY → true。
        let no_proxy = Arc::<[String]>::from(pats(&[".openai.com"]));
        let intercept = scheme_intercept_with_no_proxy("https", no_proxy);
        // intercept_closure 不能直接调用——通过 Intercept::matches 验证。
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
