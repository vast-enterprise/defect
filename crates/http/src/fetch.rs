//! `fetch` 工具用的 HTTP 客户端实现。
//!
//! 与 `build_http_stack`（LLM provider 用、输入是 `toac::Request`）并行存在：
//! provider 路径需要 toac 的 wire body；fetch 路径只发裸 GET，body 是
//! `Empty<Bytes>`，并且需要在客户端这一层就完成「URL 校验 / 重定向 / 超时 /
//! body 截断」——这些都不该塞进 toac 的 service 链。
//!
//! 共享的部分仍然在 [`super::proxy::build_proxy_connector`]——TLS / 代理 /
//! 连接池语义两条路径完全一致。
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

/// 内部使用的 hyper-util Client 类型别名——和 `build_http_stack` 用的
/// 同一个连接器、同一个 body 类型族（这里是 `Empty<Bytes>`，因为 fetch
/// 只发 GET）。
type FetchHyperClient = HyperClient<ProxyAwareConnector, Empty<Bytes>>;

/// `fetch` 工具背后的 HTTP 客户端。
///
/// 行为契约见 [`HttpClient`]：超时 / 重定向 / body 截断都在这一层实现。
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

/// 按 [`HttpStackConfig`] 构造 fetch 客户端。
///
/// 复用同一份 [`ProxyConfig`] 与 UA 默认值；`transport_retries` /
/// `total_timeout` / `initial_backoff` 在 fetch 路径下不直接生效——
/// fetch 的超时按"单次调用粒度"由 [`HttpRequest::timeout`] 决定，
/// 重试由调用方自行决定（[`HttpClient`] 不重试，避免对带副作用的
/// `GET` 隐式翻倍流量）。
///
/// # Errors
///
/// 连接器构造失败（TLS roots 加载、代理 URL 解析）时返回
/// [`HttpStackError`]。
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

/// `build_fetch_client` + `Arc::new` 的便利入口——CLI 装配点的常用路径。
///
/// # Errors
///
/// 同 [`build_fetch_client`]。
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
                    // 3xx 但缺 Location：把它当终态返回（很少见，对应大部分客户端行为）。
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

/// 解析 redirect target。
///
/// - 绝对 URI（含 scheme + authority）→ 解析后校验 scheme。
/// - 协议相对（`//host/path`）→ 沿用原 scheme。
/// - 路径绝对（`/path`）→ 沿用原 scheme + authority。
/// - 路径相对（`other.html`）→ v0 不支持，返回错误（实际场景罕见且语义易出错）。
fn resolve_redirect(base: &Uri, location: &str) -> Result<Uri, HttpClientError> {
    let trimmed = location.trim();
    if trimmed.is_empty() {
        return Err(HttpClientError::Transport(BoxError::new(
            std::io::Error::other("empty Location header"),
        )));
    }

    if trimmed.contains("://") {
        // 绝对 URI
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
        // protocol-relative
        format!("{base_scheme}://{rest}")
    } else if trimmed.starts_with('/') {
        // path-absolute
        format!("{base_scheme}://{base_authority}{trimmed}")
    } else {
        // path-relative：v0 不支持
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

/// 极简版本——给 CLI 装配在没有完整 [`HttpStackConfig`] 时也能拉起来。
///
/// 等价于 `build_fetch_client(&HttpStackConfig::default())`。
///
/// # Errors
///
/// 同 [`build_fetch_client`]。
pub fn build_default_fetch_client_arc() -> Result<Arc<dyn HttpClient>, HttpStackError> {
    build_fetch_client_arc(&HttpStackConfig {
        proxy: ProxyConfig::FromEnv,
        ..HttpStackConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_http_uri_accepts_http_https() {
        parse_http_uri("http://example.com/").unwrap();
        parse_http_uri("https://example.com/path?q=1").unwrap();
    }

    #[test]
    fn parse_http_uri_rejects_other_schemes() {
        assert!(matches!(
            parse_http_uri("file:///etc/passwd"),
            Err(HttpClientError::InvalidUrl(_))
        ));
        assert!(matches!(
            parse_http_uri("ftp://example.com/"),
            Err(HttpClientError::InvalidUrl(_))
        ));
        assert!(matches!(
            parse_http_uri("data:text/plain,hi"),
            Err(HttpClientError::InvalidUrl(_))
        ));
    }

    #[test]
    fn parse_http_uri_rejects_missing_host() {
        assert!(matches!(
            parse_http_uri("http:///path"),
            Err(HttpClientError::InvalidUrl(_))
        ));
    }

    #[test]
    fn resolve_redirect_absolute() {
        let base: Uri = "https://example.com/a".parse().unwrap();
        let r = resolve_redirect(&base, "https://other.test/x").unwrap();
        assert_eq!(r.to_string(), "https://other.test/x");
    }

    #[test]
    fn resolve_redirect_path_absolute() {
        let base: Uri = "https://example.com/a/b".parse().unwrap();
        let r = resolve_redirect(&base, "/c/d").unwrap();
        assert_eq!(r.to_string(), "https://example.com/c/d");
    }

    #[test]
    fn resolve_redirect_protocol_relative() {
        let base: Uri = "https://example.com/a".parse().unwrap();
        let r = resolve_redirect(&base, "//other.test/x").unwrap();
        assert_eq!(r.to_string(), "https://other.test/x");
    }

    #[test]
    fn resolve_redirect_relative_rejected() {
        let base: Uri = "https://example.com/a/".parse().unwrap();
        assert!(resolve_redirect(&base, "other.html").is_err());
    }

    #[test]
    fn resolve_redirect_rejects_non_http_scheme() {
        let base: Uri = "https://example.com/a".parse().unwrap();
        assert!(matches!(
            resolve_redirect(&base, "ftp://example.com/x"),
            Err(HttpClientError::InvalidUrl(_))
        ));
    }
}
