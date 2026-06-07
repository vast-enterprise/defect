//! Anthropic official API provider.
//!
//! Connects to `https://api.anthropic.com/v1/messages` using bearer token + SSE.
//!
//! Anthropic provider implementation — field mapping and request building.

use std::env;
use std::fmt;
use std::sync::Arc;

use defect_agent::error::BoxError;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelCapabilityOverrides,
    ModelInfo, ProtocolId, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream,
    RateLimitScope, ThinkingEcho, TimeoutPhase,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use http::HeaderValue;
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
use defect_http::{HttpStack, HttpStackConfig, HttpStackError, build_http_stack};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";
const ANTHROPIC_VERSION: &str = "2023-06-01";

type Client = ApiClient<HttpStack>;

/// Anthropic provider configuration.
///
/// `api_key` / `base_url` can be provided explicitly, otherwise they are read from
/// environment variables.
/// `api_key_env` overrides the default `ANTHROPIC_API_KEY` name — renaming the env var
/// for the same endpoint is common when using a secret manager (mirrors the OpenAI side;
/// see `OpenAiConfig::api_key_env`). `http` configures the transport layer (timeout /
/// retry / proxy / user-agent); defaults are in [`HttpStackConfig::default`].
#[derive(Debug, Default, Clone)]
pub struct AnthropicConfig {
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub http: HttpStackConfig,
}

impl AnthropicConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            api_key_env: None,
            base_url: env::var(BASE_URL_ENV).ok(),
            http: HttpStackConfig::default(),
        }
    }

    fn resolve_api_key(&self) -> Result<String, ProviderError> {
        if let Some(api_key) = self.api_key.clone() {
            return Ok(api_key);
        }
        let env_name = self.api_key_env.as_deref().unwrap_or(API_KEY_ENV);
        env::var(env_name).map_err(|_| {
            ProviderError::new(ProviderErrorKind::AuthMissing {
                var_hint: Some(env_name.into()),
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
        let http = build_http_stack(config.http)
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

// Header injection adapter

/// Attach the required `anthropic-version` header to `op`.
///
/// `toac` has no generic `with_header` — following the pattern of [`toac::WithAccept`],
/// build a minimal wrapper. It only injects a fixed header and does not conflict with
/// [`toac::WithAccept`] (both modify different fields of [`http::Request`]).
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

/// Extract the common `request-id` header used by both Anthropic and OpenAI. Anthropic
/// uses `request-id`, while HTTP/2, curl, etc. may produce `x-request-id`; accept both.
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

// // ---------- error mapping -----------------------------------------------

/// Maps [`CallError<HttpStackError>`] to [`ProviderError`].
///
/// Key branch: [`HttpStackError::Timeout`] is mapped to
/// [`ProviderErrorKind::Timeout`] and the `phase` is forwarded to the turn loop for
/// retry decision — this was previously missing; see HTTP retry semantics.
fn call_error_to_provider(err: CallError<HttpStackError>) -> ProviderError {
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

/// See the counterpart documentation in [`super::openai::call_error_to_provider`]. Each
/// provider keeps its own independent function to avoid mutual dependency; this one
/// reuses the phase mapping function from the OpenAI provider.
fn map_timeout_phase(phase: defect_http::TimeoutPhase) -> TimeoutPhase {
    match phase {
        defect_http::TimeoutPhase::Connect => TimeoutPhase::Connect,
        defect_http::TimeoutPhase::ReadHeaders => TimeoutPhase::ReadHeaders,
        defect_http::TimeoutPhase::ReadBody => TimeoutPhase::ReadBody,
        defect_http::TimeoutPhase::Idle => TimeoutPhase::Idle,
        defect_http::TimeoutPhase::Total => TimeoutPhase::Total,
        _ => TimeoutPhase::Total,
    }
}

/// Translate a wire `ErrorResponse` + HTTP status into a [`ProviderError`].
///
/// Mapping table — see Anthropic provider design.
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

/// Extracts the model id from error messages like `model: claude-foo`.
/// Returns `None` if extraction fails; callers should fall back to `"unknown"`.
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
