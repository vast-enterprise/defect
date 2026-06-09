//! DeepSeek provider.
//!
//! DeepSeek uses the OpenAI Chat Completions compatible protocol, so we reuse
//! [`super::openai::OpenAiProvider`] for chat, transport, and protocol decoding.
//! The only thing we override is `GET /models`: DeepSeek's model schema lacks the
//! `created` field required by the OpenAI OAS, so we cannot directly reuse the
//! wire types generated for OpenAI.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderError, ProviderInfo, ProviderStream, ThinkingEcho,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use http::HeaderValue;
use http::Request;
use serde::Deserialize;
use toac::{MakeRequest, ParseResponse};
use tokio_util::sync::CancellationToken;
use tower::Service;

use super::openai::{OpenAiConfig, OpenAiProvider};
use crate::protocol::deepseek_chat;
use crate::protocol::openai_chat::ChatDialect;
use defect_agent::llm::ReasoningEffort;
use defect_http::HttpStackConfig;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
const BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";

/// DeepSeek provider configuration.
#[derive(Debug, Default, Clone)]
pub struct DeepSeekConfig {
    pub api_key: Option<String>,
    /// Overrides the default `DEEPSEEK_API_KEY` environment variable name. Semantics are
    /// the same as [`super::openai::OpenAiConfig::api_key_env`].
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    /// `reasoning_effort` override; semantics are the same as
    /// [`super::openai::OpenAiConfig::reasoning_effort`].
    pub reasoning_effort: Option<ReasoningEffort>,
    pub http: HttpStackConfig,
}

impl DeepSeekConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            api_key_env: None,
            base_url: env::var(BASE_URL_ENV).ok(),
            reasoning_effort: None,
            http: HttpStackConfig::default(),
        }
    }

    fn resolve_api_key(&self) -> Option<String> {
        if let Some(api_key) = self.api_key.clone() {
            return Some(api_key);
        }
        if let Some(env_name) = self.api_key_env.as_deref()
            && let Ok(v) = env::var(env_name)
        {
            return Some(v);
        }
        // Only DeepSeek's own key. Do NOT fall back to OPENAI_API_KEY: even though
        // DeepSeek speaks the OpenAI wire protocol, silently sending an OpenAI key to
        // `api.deepseek.com` is a surprising cross-provider credential leak — a user with
        // only OPENAI_API_KEY set who selects DeepSeek should get a clear "missing key"
        // error, not have their OpenAI key shipped to a different vendor.
        env::var(API_KEY_ENV).ok()
    }

    fn resolve_base_url(&self) -> String {
        self.base_url
            .clone()
            .or_else(|| env::var(BASE_URL_ENV).ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
    }
}

/// DeepSeek provider.
#[derive(Debug)]
pub struct DeepSeekProvider {
    inner: Arc<OpenAiProvider>,
    info: ProviderInfo,
}

impl DeepSeekProvider {
    /// # Errors
    ///
    /// Returns an error if credentials are missing or the underlying HTTP client fails to
    /// initialize.
    pub fn new(config: DeepSeekConfig) -> Result<Self, ProviderError> {
        let openai_cfg = OpenAiConfig {
            api_key: config.resolve_api_key(),
            api_key_env: None,
            base_url: Some(config.resolve_base_url()),
            organization: None,
            project: None,
            vendor: "deepseek".into(),
            display_name: "DeepSeek Chat".into(),
            headers: HashMap::new(),
            capabilities_override: Some(default_deepseek_capabilities()),
            reasoning_effort: config.reasoning_effort,
            chat_dialect: ChatDialect::DeepSeek,
            http: config.http,
        };
        let inner = Arc::new(OpenAiProvider::new(openai_cfg)?);
        Ok(Self {
            inner,
            info: ProviderInfo {
                vendor: "deepseek".into(),
                protocol: ProtocolId::OpenAiChat,
                display_name: "DeepSeek Chat".into(),
            },
        })
    }
}

fn default_deepseek_capabilities() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Supported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Supported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

impl LlmProvider for DeepSeekProvider {
    fn info(&self) -> ProviderInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        async move {
            let request = DeepSeekListModelsRequest {};
            let mut client = self.inner.client();
            let response = client
                .call(request)
                .await
                .map_err(super::openai::call_error_to_provider)?;

            let models = response
                .data
                .into_iter()
                .map(|model| {
                    self.inner
                        .model_info(&model.id)
                        .unwrap_or_else(|| model.into())
                })
                .collect();
            Ok(models)
        }
        .boxed()
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.inner.model_info(model_id)
    }

    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        async move {
            let stream = self
                .inner
                .start_chat_completion_stream(req, cancel.clone())
                .await?;
            let decoded = deepseek_chat::decode_stream(stream, cancel);
            Ok(Box::pin(decoded) as ProviderStream)
        }
        .boxed()
    }
}

#[derive(Debug, Clone)]
struct DeepSeekListModelsRequest {}

impl MakeRequest for DeepSeekListModelsRequest {
    type Error = std::convert::Infallible;

    /// # Errors
    ///
    /// This implementation does not construct request errors.
    async fn make_request(self) -> Result<Request<toac::body::Body>, Self::Error> {
        let mut builder = Request::builder().method(http::Method::GET).uri("/models");
        builder = builder.header(
            http::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        let mut request = builder
            .body(toac::body::Body::empty())
            .expect("valid DeepSeek /models request");
        request
            .extensions_mut()
            .insert(toac::OperationSecurity(&[&["ApiKeyAuth"]]));
        Ok(request)
    }
}

impl toac::Operation for DeepSeekListModelsRequest {
    type Response = DeepSeekModelsResponse;
}

#[derive(Debug, Clone, Deserialize)]
struct DeepSeekModelsResponse {
    data: Vec<DeepSeekModel>,
}

impl ParseResponse for DeepSeekModelsResponse {
    type Error = toac::DecodeError;

    /// # Errors
    ///
    /// Returns an error if the response body is not valid JSON or does not match the
    /// structure returned by the DeepSeek `/models` endpoint.
    async fn parse_response<B>(response: http::Response<B>) -> Result<Self, Self::Error>
    where
        B: http_body::Body<Data = bytes::Bytes> + Send + Sync + 'static,
        B::Error: Into<toac::BoxError>,
    {
        let (_parts, body) = response.into_parts();
        let decoder = toac::body::codec::json::JsonDecoder;
        toac::body::codec::decode_body(&decoder, body)
            .await
            .map_err(|error| toac::DecodeError::Codec(error.into()))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct DeepSeekModel {
    id: String,
}

impl From<DeepSeekModel> for ModelInfo {
    fn from(value: DeepSeekModel) -> Self {
        Self {
            id: value.id,
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: Default::default(),
        }
    }
}
