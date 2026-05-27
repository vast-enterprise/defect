//! OpenAI 兼容接口 provider。
//!
//! 通过 `base_url` 参数对接 OpenAI 官方与所有遵循 Chat Completions
//! 协议的兼容服务（DeepSeek、Qwen、本地 vllm 等）。bearer token + SSE。
//!
//! 设计与字段对应详见 `docs/outbound/llm-openai.md`。

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;

use defect_agent::error::BoxError;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelCapabilityOverrides,
    ModelInfo, ProtocolId, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream,
    RateLimitScope, ThinkingEcho, TimeoutPhase,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use http::HeaderValue;
use toac::body::codec::sse::SseEventStream;
use toac::{ApiClient, CallError, MakeRequest, Operation, Request as ToacRequest};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower::Service;

use defect_http::{HttpStack, HttpStackConfig, HttpStackError, build_http_stack};
use crate::protocol::openai_chat;
use crate::wire::openai::{
    components as wire,
    operations::{chat::completions as chat_completions, models},
    security,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const API_KEY_ENV: &str = "OPENAI_API_KEY";
const BASE_URL_ENV: &str = "OPENAI_BASE_URL";
const ORG_ENV: &str = "OPENAI_ORG";
const PROJECT_ENV: &str = "OPENAI_PROJECT";

pub(crate) type Client = ApiClient<HttpStack>;

/// OpenAI provider 配置。
///
/// `api_key` / `base_url` / `organization` / `project` 可显式提供，否则
/// 从环境变量读取。`capabilities_override` 用于兼容厂商（如 DeepSeek
/// 把 `thinking` 翻成 `Supported`）。`http` 配置 transport 层，默认见
/// [`HttpStackConfig::default`].
#[derive(Debug, Default, Clone)]
pub struct OpenAiConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub capabilities_override: Option<Capabilities>,
    pub http: HttpStackConfig,
}

impl OpenAiConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            base_url: env::var(BASE_URL_ENV).ok(),
            organization: env::var(ORG_ENV).ok(),
            project: env::var(PROJECT_ENV).ok(),
            capabilities_override: None,
            http: HttpStackConfig::default(),
        }
    }

    pub fn with_capabilities_override(mut self, caps: Capabilities) -> Self {
        self.capabilities_override = Some(caps);
        self
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

    fn resolve_org(&self) -> Option<String> {
        self.organization.clone().or_else(|| env::var(ORG_ENV).ok())
    }

    fn resolve_project(&self) -> Option<String> {
        self.project.clone().or_else(|| env::var(PROJECT_ENV).ok())
    }
}

pub struct OpenAiProvider {
    client: Client,
    info: ProviderInfo,
    capabilities: Capabilities,
    organization: Option<String>,
    project: Option<String>,
    models: Arc<RwLock<Option<Vec<ModelInfo>>>>,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("info", &self.info)
            .field("capabilities", &self.capabilities)
            .field("organization", &self.organization)
            .field("project", &self.project)
            .finish_non_exhaustive()
    }
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self, ProviderError> {
        let token = config.resolve_api_key()?;
        let base_url = config.resolve_base_url();
        let organization = config.resolve_org();
        let project = config.resolve_project();
        let capabilities = config
            .capabilities_override
            .unwrap_or(default_openai_capabilities());

        let auth = security::AuthConfig::builder().api_key_auth(token).build();
        let http = build_http_stack(config.http)
            .map_err(|e| ProviderError::new(ProviderErrorKind::Transport(BoxError::new(e))))?;
        let client = ApiClient::new(http, base_url).with_auth(auth);

        Ok(Self {
            client,
            info: ProviderInfo {
                vendor: "openai".into(),
                protocol: ProtocolId::OpenAiChat,
                display_name: "OpenAI Chat Completions".into(),
            },
            capabilities,
            organization,
            project,
            models: Arc::default(),
        })
    }

    pub(crate) fn client(&self) -> Client {
        self.client.clone()
    }
}

fn default_openai_capabilities() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Supported,
        prompt_cache: FeatureSupport::Supported,
        // OpenAI 官方 o1 / o3 不通过 wire 暴露 thinking 文本，无可回放；
        // 兼容厂商（DeepSeek 等）单独覆盖。
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

impl LlmProvider for OpenAiProvider {
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

            let request = self.with_openai_headers(models::get::Request {});
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
                    return Err(error_response(400, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status401(e) => {
                    return Err(error_response(401, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status403(e) => {
                    return Err(error_response(403, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status404(e) => {
                    return Err(error_response(404, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status413(e) => {
                    return Err(error_response(413, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status429(e) => {
                    return Err(error_response(429, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status500(e) => {
                    return Err(error_response(500, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status502(e) => {
                    return Err(error_response(502, &e, None).with_request_id_opt(request_id));
                }
                models::get::ResponseBody::Status503(e) => {
                    return Err(error_response(503, &e, None).with_request_id_opt(request_id));
                }
            };

            let upstream: Vec<ModelInfo> = list
                .data
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.id,
                    display_name: None,
                    context_window: None,
                    max_output_tokens: None,
                    deprecated: false,
                    capabilities_overrides: ModelCapabilityOverrides::default(),
                })
                .collect();

            let merged = merge_with_hardcoded(upstream);

            *self.models.write().await = Some(merged.clone());
            Ok(merged)
        }
        .boxed()
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        if let Some(cached) = self
            .models
            .try_read()
            .ok()
            .and_then(|g| g.as_ref()?.iter().find(|m| m.id == model_id).cloned())
        {
            return Some(cached);
        }
        hardcoded_lookup(model_id)
    }

    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        async move {
            let stream = self
                .start_chat_completion_stream(req, cancel.clone())
                .await?;
            let decoded = openai_chat::decode_stream(stream, cancel);
            Ok(Box::pin(decoded) as ProviderStream)
        }
        .boxed()
    }
}

impl OpenAiProvider {
    fn with_openai_headers<Op>(&self, op: Op) -> WithOpenAiHeaders<Op> {
        WithOpenAiHeaders {
            op,
            organization: self.organization.clone(),
            project: self.project.clone(),
        }
    }

    /// 解析当前请求的 thinking 回放策略：先看 per-model override，再 fallback
    /// 到 provider-level capability。详见
    /// `docs/internal/thinking-roundtrip.md` §4.2。
    fn thinking_echo_for_model(&self, model_id: &str) -> ThinkingEcho {
        self.model_info(model_id)
            .and_then(|m| m.capabilities_overrides.thinking_echo)
            .unwrap_or(self.capabilities.thinking_echo)
    }

    /// 发送一个 Chat Completions SSE 请求，并返回原始事件流。
    ///
    /// # Errors
    ///
    /// 当请求被取消、transport 失败、服务端返回非 200 SSE、或返回已知
    /// OpenAI 错误体时返回错误。
    pub(crate) async fn start_chat_completion_stream(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> Result<SseEventStream, ProviderError> {
        let echo_mode = self.thinking_echo_for_model(&req.model);
        let body = openai_chat::encode_request_with_echo(&req, echo_mode);
        let op = self
            .with_openai_headers(chat_completions::post::Request { body })
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
        let retry_after = extract_retry_after(&resp.headers);

        match resp.body {
            chat_completions::post::ResponseBody::Status200Sse(s) => Ok(s),
            chat_completions::post::ResponseBody::Status200Json(_) => {
                Err(ProviderError::new(ProviderErrorKind::ProtocolViolation {
                    hint: "server returned application/json despite Accept: text/event-stream"
                        .into(),
                })
                .with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status400(e) => {
                Err(error_response(400, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status401(e) => {
                Err(error_response(401, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status403(e) => {
                Err(error_response(403, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status404(e) => {
                Err(error_response(404, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status413(e) => {
                Err(error_response(413, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status429(e) => {
                Err(error_response(429, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status500(e) => {
                Err(error_response(500, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status502(e) => {
                Err(error_response(502, &e, retry_after).with_request_id_opt(request_id))
            }
            chat_completions::post::ResponseBody::Status503(e) => {
                Err(error_response(503, &e, retry_after).with_request_id_opt(request_id))
            }
        }
    }
}

// ---------- header injection adapter ------------------------------------

/// 给 op 装上可选的 `OpenAI-Organization` / `OpenAI-Project` 头。
///
/// toac 的生成 op 不接受任意 header，参考 [`toac::WithAccept`] 自起一个
/// 最小 wrapper。空值不发，发出空字符串会被有些兼容厂商当成非法值。
#[derive(Debug, Clone)]
struct WithOpenAiHeaders<Op> {
    op: Op,
    organization: Option<String>,
    project: Option<String>,
}

impl<Op> MakeRequest for WithOpenAiHeaders<Op>
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
            if let Some(org) = self.organization.as_deref()
                && let Ok(v) = HeaderValue::from_str(org)
            {
                req.headers_mut()
                    .insert(http::HeaderName::from_static("openai-organization"), v);
            }
            if let Some(project) = self.project.as_deref()
                && let Ok(v) = HeaderValue::from_str(project)
            {
                req.headers_mut()
                    .insert(http::HeaderName::from_static("openai-project"), v);
            }
            Ok(req)
        }
    }
}

impl<Op> Operation for WithOpenAiHeaders<Op>
where
    Op: Operation + Send,
{
    type Response = Op::Response;
}

impl<Op> WithOpenAiHeaders<Op> {
    fn with_accept(self, accept: HeaderValue) -> toac::WithAccept<Self> {
        toac::WithAccept::new(self, accept)
    }
}

// ---------- response header helpers -------------------------------------

/// OpenAI 用 `x-request-id` header 传 request id；HTTP/2 / 部分代理还会
/// 顺便带 `request-id`，两者都收。
fn extract_request_id(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// 解析 RFC 7231 `Retry-After` header。OpenAI 在 429 上偶尔发整数秒，
/// 也偶尔发 HTTP-date；先按整数秒尝试，date 形态留给上层退避兜底。
fn extract_retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    let v = headers.get(http::header::RETRY_AFTER)?.to_str().ok()?;
    v.trim().parse::<u64>().ok().map(Duration::from_secs)
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

/// 把 [`CallError<HttpStackError>`] 翻成 [`ProviderError`]。
///
/// 关键分支：[`HttpStackError::Timeout`] 单独翻成
/// [`ProviderErrorKind::Timeout`] 并把 phase 透传给 turn-loop §7 做重试
/// 决策——这条之前缺失，详见 [`docs/outbound/http.md`] §4。
/// [`HttpStackError::ProxyConnect`] / [`HttpStackError::Config`] 都按
/// transport 错处理（结构化原因写进 `Display`，turn-loop 拿到的就是
/// transport-flavor 的 backoff 重试）。
pub(crate) fn call_error_to_provider(err: CallError<HttpStackError>) -> ProviderError {
    match err {
        CallError::Encode(e) => ProviderError::new(ProviderErrorKind::BadRequest {
            hint: Some(e.to_string()),
        }),
        CallError::Auth(e) => ProviderError::new(ProviderErrorKind::AuthMalformed {
            hint: Some(e.to_string()),
        }),
        CallError::Transport(HttpStackError::Timeout { phase }) => {
            ProviderError::new(ProviderErrorKind::Timeout {
                phase: map_timeout_phase(phase),
            })
        }
        CallError::Transport(e) => {
            ProviderError::new(ProviderErrorKind::Transport(BoxError::new(e)))
        }
        CallError::Decode(e) => ProviderError::new(ProviderErrorKind::Malformed(BoxError::new(e))),
    }
}

/// 把 [`HttpStackError`] 携带的 [`defect_http::TimeoutPhase`] 翻成
/// agent 层的 [`TimeoutPhase`]。两者形态一致（都是 `Connect / ReadHeaders /
/// ReadBody / Idle / Total`），但分属不同 crate，避免 layer 实现耦合到
/// LLM 错误模型。v0 实际只产 `Total`，其余 arm 是为后续分阶段超时占位。
fn map_timeout_phase(phase: defect_http::TimeoutPhase) -> TimeoutPhase {
    match phase {
        defect_http::TimeoutPhase::Connect => TimeoutPhase::Connect,
        defect_http::TimeoutPhase::ReadHeaders => TimeoutPhase::ReadHeaders,
        defect_http::TimeoutPhase::ReadBody => TimeoutPhase::ReadBody,
        defect_http::TimeoutPhase::Idle => TimeoutPhase::Idle,
        defect_http::TimeoutPhase::Total => TimeoutPhase::Total,
        // 上游 `#[non_exhaustive]`：未来新增 phase 时退到 Total——不爆栈、
        // 不丢信息（turn-loop §7 对所有 phase 走相同 Backoff 路径）。
        _ => TimeoutPhase::Total,
    }
}

/// 把 wire `OpenAiErrorResponse` + HTTP status 翻成 [`ProviderError`]。
///
/// 映射表见 `docs/outbound/llm-openai.md` §6。
fn error_response(
    status: u16,
    e: &wire::OpenAiErrorResponse,
    retry_after: Option<Duration>,
) -> ProviderError {
    let message = e.error.message.clone();
    let upstream_type = e.error.r#type.as_str();
    let upstream_code = e.error.code.as_deref();
    let upstream_param = e.error.param.as_deref();
    let kind = match (status, upstream_type, upstream_code) {
        (401, _, _) => ProviderErrorKind::AuthRejected {
            hint: Some(message),
        },
        (400, _, Some("context_length_exceeded")) => ProviderErrorKind::ContextOverflow {
            used: None,
            limit: None,
        },
        (400, "invalid_request_error", _)
            if upstream_param == Some("max_tokens")
                || upstream_param == Some("max_completion_tokens")
                || contains_max_tokens(&message) =>
        {
            ProviderErrorKind::MaxTokensInvalid {
                requested: None,
                limit: None,
            }
        }
        (400, _, _) => ProviderErrorKind::BadRequest {
            hint: Some(message),
        },
        (403, _, Some("insufficient_quota")) => ProviderErrorKind::QuotaExceeded {
            hint: Some(message),
        },
        (403, _, _) => ProviderErrorKind::AuthRejected {
            hint: Some(message),
        },
        (404, _, Some("model_not_found")) => ProviderErrorKind::ModelNotFound {
            model: extract_model(&message).unwrap_or_else(|| "unknown".into()),
        },
        (404, _, _) => ProviderErrorKind::ServerError {
            status: Some(404),
            hint: Some(message),
        },
        (413, _, _) => ProviderErrorKind::BadRequest {
            hint: Some("payload too large".into()),
        },
        (429, t, _) => ProviderErrorKind::RateLimit {
            retry_after,
            scope: rate_limit_scope_from(t, &message),
        },
        (s, _, _) if s >= 500 => ProviderErrorKind::ServerError {
            status: Some(s),
            hint: Some(message),
        },
        (s, _, _) => ProviderErrorKind::ServerError {
            status: Some(s),
            hint: Some(message),
        },
    };
    ProviderError::new(kind)
}

fn rate_limit_scope_from(upstream_type: &str, message: &str) -> RateLimitScope {
    let lower = message.to_ascii_lowercase();
    if upstream_type.contains("tokens_per_min")
        || lower.contains("tokens per min")
        || lower.contains("tpm")
    {
        RateLimitScope::Tpm
    } else if upstream_type.contains("requests_per_min")
        || lower.contains("requests per min")
        || lower.contains("rpm")
    {
        RateLimitScope::Rpm
    } else {
        RateLimitScope::Unspecified
    }
}

fn contains_max_tokens(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("max_tokens")
        || lower.contains("max tokens")
        || lower.contains("max_completion_tokens")
}

/// 从形如 `model: gpt-foo` 这类错误信息里抠 model id。
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

// ---------- hardcoded model table ---------------------------------------

/// v0 硬编码模型表。
///
/// `/v1/models` 在不同兼容厂商间 schema 差异大（OpenAI 只发 `id` /
/// `created` / `owned_by`，DeepSeek 不发 `context_window`，Together 字段
/// 又不一样），但 `id` 全家共享。这里维护一份小表，给已知 OpenAI 系列
/// 模型补 `context_window` / `max_output_tokens` / `capabilities_overrides`
/// 等常用字段；上游列表里没有的 id 直接保持 `None`。
///
/// 表项数据来源：OpenAI 官方文档 + DeepSeek 官方文档（截至 2025）。
fn hardcoded_models() -> &'static [HardcodedModel] {
    &[
        HardcodedModel {
            id: "gpt-4o-mini",
            display_name: Some("GPT-4o mini"),
            context_window: Some(128_000),
            max_output_tokens: Some(16_384),
            overrides: None,
        },
        HardcodedModel {
            id: "gpt-4o",
            display_name: Some("GPT-4o"),
            context_window: Some(128_000),
            max_output_tokens: Some(16_384),
            overrides: None,
        },
        HardcodedModel {
            id: "o1-mini",
            display_name: Some("o1-mini"),
            context_window: Some(128_000),
            max_output_tokens: Some(65_536),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: Some(FeatureSupport::Unsupported),
                thinking_echo: None,
            }),
        },
        HardcodedModel {
            id: "o1",
            display_name: Some("o1"),
            context_window: Some(200_000),
            max_output_tokens: Some(100_000),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: Some(FeatureSupport::Unsupported),
                thinking_echo: None,
            }),
        },
        HardcodedModel {
            id: "o3-mini",
            display_name: Some("o3-mini"),
            context_window: Some(200_000),
            max_output_tokens: Some(100_000),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: Some(FeatureSupport::Unsupported),
                thinking_echo: None,
            }),
        },
        HardcodedModel {
            id: "o3",
            display_name: Some("o3"),
            context_window: Some(200_000),
            max_output_tokens: Some(100_000),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: Some(FeatureSupport::Unsupported),
                thinking_echo: None,
            }),
        },
        HardcodedModel {
            id: "deepseek-v4-flash",
            display_name: Some("DeepSeek v4 Flash"),
            context_window: Some(1_000_000),
            max_output_tokens: Some(384_000),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: None,
                // v4-flash thinking 模式同 v4-pro：上一轮 reasoning_content
                // 必须回放，否则 400 "must be passed back to the API"。
                thinking_echo: Some(ThinkingEcho::Required),
            }),
        },
        HardcodedModel {
            id: "deepseek-v4-pro",
            display_name: Some("DeepSeek v4 Pro"),
            context_window: Some(1_000_000),
            max_output_tokens: Some(384_000),
            overrides: Some(ModelCapabilityOverrides {
                thinking: Some(FeatureSupport::Supported),
                vision: None,
                prompt_cache: None,
                parallel_tool_calls: None,
                // v4-pro 走 https://api.deepseek.com/anthropic 端点（Anthropic
                // 协议），不走 /v1/chat/completions——这条 echo 配置仅在用户
                // 把 v4-pro 接到 OpenAI 兼容路径时生效。
                thinking_echo: Some(ThinkingEcho::Required),
            }),
        },
    ]
}

#[derive(Debug, Clone, Copy)]
struct HardcodedModel {
    id: &'static str,
    display_name: Option<&'static str>,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
    overrides: Option<ModelCapabilityOverrides>,
}

fn hardcoded_lookup(model_id: &str) -> Option<ModelInfo> {
    hardcoded_models()
        .iter()
        .find(|m| m.id == model_id)
        .map(|m| ModelInfo {
            id: m.id.to_owned(),
            display_name: m.display_name.map(str::to_owned),
            context_window: m.context_window,
            max_output_tokens: m.max_output_tokens,
            deprecated: false,
            capabilities_overrides: m.overrides.unwrap_or_default(),
        })
}

/// 用上游列表为骨干，硬编码表为补丁：上游有的就拿上游 id，硬编码表
/// 命中就把元信息往里补；硬编码表里有但上游没列出的 id 也合进来
/// （兼容厂商 `/v1/models` 缺失主流模型时的兜底）。
fn merge_with_hardcoded(upstream: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let mut by_id: HashMap<String, ModelInfo> =
        upstream.into_iter().map(|m| (m.id.clone(), m)).collect();
    for hc in hardcoded_models() {
        let entry = by_id.entry(hc.id.to_owned()).or_insert_with(|| ModelInfo {
            id: hc.id.to_owned(),
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: ModelCapabilityOverrides::default(),
        });
        if entry.display_name.is_none() {
            entry.display_name = hc.display_name.map(str::to_owned);
        }
        if entry.context_window.is_none() {
            entry.context_window = hc.context_window;
        }
        if entry.max_output_tokens.is_none() {
            entry.max_output_tokens = hc.max_output_tokens;
        }
        if let Some(overrides) = hc.overrides {
            let cur = entry.capabilities_overrides;
            entry.capabilities_overrides = ModelCapabilityOverrides {
                thinking: cur.thinking.or(overrides.thinking),
                vision: cur.vision.or(overrides.vision),
                prompt_cache: cur.prompt_cache.or(overrides.prompt_cache),
                parallel_tool_calls: cur.parallel_tool_calls.or(overrides.parallel_tool_calls),
                thinking_echo: cur.thinking_echo.or(overrides.thinking_echo),
            };
        }
    }
    let mut merged: Vec<ModelInfo> = by_id.into_values().collect();
    merged.sort_by(|a, b| a.id.cmp(&b.id));
    merged
}
