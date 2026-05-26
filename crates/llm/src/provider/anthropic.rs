//! Anthropic 官方 API provider。
//!
//! 对接 `https://api.anthropic.com/v1/messages`，bearer token + SSE。
//!
//! 设计与字段对应详见 `docs/outbound/llm-anthropic.md`。

use std::env;
use std::fmt;
use std::sync::Arc;

use client_util::client::{HyperHttpsClient, build_https_client};
use defect_agent::error::BoxError;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelCapabilityOverrides,
    ModelInfo, ProtocolId, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream,
    RateLimitScope, ThinkingEcho,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use http::HeaderValue;
use toac::body::Body;
use toac::{ApiClient, CallError, MakeRequest, Operation, Request as ToacRequest};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower::Service;

use crate::protocol::anthropic_messages;
use crate::wire::anthropic::{
    components as wire,
    operations::v1::{messages, models},
    security,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";
const ANTHROPIC_VERSION: &str = "2023-06-01";

type Http = HyperHttpsClient<Body>;
type Client = ApiClient<Http>;

/// Anthropic provider 配置。
///
/// `api_key` / `base_url` 可显式提供，否则从环境变量读取。
#[derive(Debug, Default, Clone)]
pub struct AnthropicConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl AnthropicConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            base_url: env::var(BASE_URL_ENV).ok(),
        }
    }

    fn resolve_api_key(&self) -> Result<String, ProviderError> {
        self.api_key
            .clone()
            .or_else(|| env::var(API_KEY_ENV).ok())
            .ok_or_else(|| {
                ProviderError::new(ProviderErrorKind::AuthMissing {
                    var_hint: Some(API_KEY_ENV.into()),
                })
            })
    }

    fn resolve_base_url(&self) -> String {
        self.base_url
            .clone()
            .or_else(|| env::var(BASE_URL_ENV).ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
    }
}

pub struct AnthropicProvider {
    client: Client,
    info: ProviderInfo,
    capabilities: Capabilities,
    models: Arc<RwLock<Option<Vec<ModelInfo>>>>,
}

impl fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("info", &self.info)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self, ProviderError> {
        let token = config.resolve_api_key()?;
        let base_url = config.resolve_base_url();

        let auth = security::AuthConfig::builder().api_key_auth(token).build();
        let http = build_https_client::<Body>()
            .map_err(|e| ProviderError::new(ProviderErrorKind::Transport(BoxError::new(e))))?;
        let client = ApiClient::new(http, base_url).with_auth(auth);

        Ok(Self {
            client,
            info: ProviderInfo {
                vendor: "anthropic".into(),
                protocol: ProtocolId::AnthropicMessages,
                display_name: "Anthropic Claude".into(),
            },
            capabilities: Capabilities {
                tool_calls: FeatureSupport::Supported,
                parallel_tool_calls: FeatureSupport::Supported,
                thinking: FeatureSupport::Supported,
                vision: FeatureSupport::Supported,
                prompt_cache: FeatureSupport::Supported,
                thinking_echo: ThinkingEcho::Required,
            },
            models: Arc::default(),
        })
    }
}

impl LlmProvider for AnthropicProvider {
    fn info(&self) -> ProviderInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        async move {
            if let Some(cached) = self.models.read().await.clone() {
                return Ok(cached);
            }

            let request = with_anthropic_headers(models::get::Request {
                before_id: None,
                after_id: None,
                limit: None,
            });
            let resp = self
                .client
                .clone()
                .call(request)
                .await
                .map_err(call_error_to_provider)?;
            let request_id = extract_request_id(&resp.headers);

            let list = match resp.body {
                models::get::ResponseBody::Status200(l) => l,
                models::get::ResponseBody::Status400(e) => {
                    return Err(error_response(400, &e).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status401(e) => {
                    return Err(error_response(401, &e).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status429(e) => {
                    return Err(error_response(429, &e).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status500(e) => {
                    return Err(error_response(500, &e).with_request_id_opt(request_id));
                }
            };

            let mapped: Vec<ModelInfo> = list
                .data
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.id,
                    display_name: Some(m.display_name),
                    context_window: None,
                    max_output_tokens: None,
                    deprecated: false,
                    capabilities_overrides: ModelCapabilityOverrides::default(),
                })
                .collect();

            *self.models.write().await = Some(mapped.clone());
            Ok(mapped)
        }
        .boxed()
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.models
            .try_read()
            .ok()
            .and_then(|g| g.as_ref()?.iter().find(|m| m.id == model_id).cloned())
    }

    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        async move {
            let body = anthropic_messages::encode_request(&req);
            let op = with_anthropic_headers(messages::post::Request { body })
                .with_accept(HeaderValue::from_static("text/event-stream"));

            let mut client = self.client.clone();
            let resp = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    return Err(ProviderError::new(ProviderErrorKind::Canceled));
                }
                r = client.call(op) => r.map_err(call_error_to_provider)?,
            };
            let request_id = extract_request_id(&resp.headers);

            let stream = match resp.body {
                messages::post::ResponseBody::Status200Sse(s) => s,
                messages::post::ResponseBody::Status200Json(_) => {
                    return Err(ProviderError::new(ProviderErrorKind::ProtocolViolation {
                        hint: "server returned application/json despite Accept: text/event-stream"
                            .into(),
                    })
                    .with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status400(e) => {
                    return Err(error_response(400, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status401(e) => {
                    return Err(error_response(401, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status403(e) => {
                    return Err(error_response(403, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status404(e) => {
                    return Err(error_response(404, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status413(e) => {
                    return Err(error_response(413, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status429(e) => {
                    return Err(error_response(429, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status500(e) => {
                    return Err(error_response(500, &e).with_request_id_opt(request_id));
                }
                messages::post::ResponseBody::Status529(e) => {
                    return Err(error_response(529, &e).with_request_id_opt(request_id));
                }
            };

            let decoded = anthropic_messages::decode_stream(stream, cancel);
            Ok(Box::pin(decoded) as ProviderStream)
        }
        .boxed()
    }
}

// ---------- header injection adapter ------------------------------------

/// 给 op 装上 Anthropic 必需的 `anthropic-version` 头。
///
/// toac 没有 generic `with_header`——参考 [`toac::WithAccept`] 的写法
/// 自起一个最小 wrapper。仅注入固定 header，不与 [`toac::WithAccept`]
/// 互斥（两者都改 [`http::Request`] 的不同字段）。
fn with_anthropic_headers<Op>(op: Op) -> WithAnthropicHeaders<Op> {
    WithAnthropicHeaders { op }
}

#[derive(Debug, Clone)]
struct WithAnthropicHeaders<Op> {
    op: Op,
}

impl<Op> MakeRequest for WithAnthropicHeaders<Op>
where
    Op: MakeRequest + Send,
{
    type Error = Op::Error;

    #[allow(clippy::manual_async_fn)]
    fn make_request(
        self,
    ) -> impl std::future::Future<Output = Result<ToacRequest, Self::Error>> + Send {
        async move {
            let mut req = self.op.make_request().await?;
            req.headers_mut().insert(
                http::HeaderName::from_static("anthropic-version"),
                HeaderValue::from_static(ANTHROPIC_VERSION),
            );
            Ok(req)
        }
    }
}

impl<Op> Operation for WithAnthropicHeaders<Op>
where
    Op: Operation + Send,
{
    type Response = Op::Response;
}

impl<Op> WithAnthropicHeaders<Op> {
    fn with_accept(self, accept: HeaderValue) -> toac::WithAccept<Self> {
        toac::WithAccept::new(self, accept)
    }
}

// ---------- response header helpers -------------------------------------

/// 抽 Anthropic / OpenAI 通用的 request-id header。Anthropic 用
/// `request-id`，HTTP/2 / curl 等也会出 `x-request-id`，两者都收。
fn extract_request_id(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("request-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

trait WithRequestIdOpt {
    fn with_request_id_opt(self, id: Option<String>) -> Self;
}

impl WithRequestIdOpt for ProviderError {
    fn with_request_id_opt(self, id: Option<String>) -> Self {
        match id {
            Some(s) => self.with_request_id(s),
            None => self,
        }
    }
}

// ---------- error mapping -----------------------------------------------

fn call_error_to_provider<E>(err: CallError<E>) -> ProviderError
where
    E: std::error::Error + Send + Sync + 'static,
{
    match err {
        CallError::Encode(e) => ProviderError::new(ProviderErrorKind::BadRequest {
            hint: Some(e.to_string()),
        }),
        CallError::Auth(e) => ProviderError::new(ProviderErrorKind::AuthMalformed {
            hint: Some(e.to_string()),
        }),
        CallError::Transport(e) => {
            ProviderError::new(ProviderErrorKind::Transport(BoxError::new(e)))
        }
        CallError::Decode(e) => ProviderError::new(ProviderErrorKind::Malformed(BoxError::new(e))),
    }
}

/// 把 wire `ErrorResponse` + HTTP status 翻成 [`ProviderError`]。
///
/// 映射表见 `docs/outbound/llm-anthropic.md` §7。
fn error_response(status: u16, e: &wire::ErrorResponse) -> ProviderError {
    let message = e.error.message.clone();
    let upstream_type = e.error.r#type.as_str();
    let kind = match (status, upstream_type) {
        (401, _) => ProviderErrorKind::AuthRejected {
            hint: Some(message),
        },
        (400, t) if t == "invalid_request_error" && contains_max_tokens(&message) => {
            ProviderErrorKind::MaxTokensInvalid {
                requested: None,
                limit: None,
            }
        }
        (400, _) => ProviderErrorKind::BadRequest {
            hint: Some(message),
        },
        (403, _) => ProviderErrorKind::AuthRejected {
            hint: Some(message),
        },
        (404, "not_found_error") => ProviderErrorKind::ModelNotFound {
            model: extract_model(&message).unwrap_or_else(|| "unknown".into()),
        },
        (404, _) => ProviderErrorKind::ServerError {
            status: Some(404),
            hint: Some(message),
        },
        (413, _) => ProviderErrorKind::BadRequest {
            hint: Some("payload too large".into()),
        },
        (429, _) => ProviderErrorKind::RateLimit {
            retry_after: None,
            scope: RateLimitScope::Unspecified,
        },
        (529, _) => ProviderErrorKind::ServerError {
            status: Some(529),
            hint: Some("overloaded".into()),
        },
        (s, "overloaded_error") => ProviderErrorKind::ServerError {
            status: Some(s),
            hint: Some("overloaded".into()),
        },
        (s, _) => ProviderErrorKind::ServerError {
            status: Some(s),
            hint: Some(message),
        },
    };
    ProviderError::new(kind)
}

fn contains_max_tokens(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("max_tokens") || lower.contains("max tokens")
}

/// 从形如 `model: claude-foo` 这类错误信息里抠 model id。
/// 抠不出就返回 None，调用方按 `"unknown"` 兜底。
fn extract_model(msg: &str) -> Option<String> {
    let lower = msg.to_ascii_lowercase();
    let idx = lower.find("model")?;
    let tail = &msg[idx + "model".len()..];
    let trimmed = tail.trim_start_matches(|c: char| {
        c.is_whitespace() || c == ':' || c == '=' || c == '"' || c == '\''
    });
    let end = trimmed
        .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',')
        .unwrap_or(trimmed.len());
    let candidate = &trimmed[..end];
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_owned())
    }
}
