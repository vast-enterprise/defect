//! DeepSeek provider。
//!
//! DeepSeek 走 OpenAI Chat Completions 兼容协议，所以这里复用
//! [`super::openai::OpenAiProvider`] 处理 chat / transport / 协议解码。
//! 唯一单独 override 的是 `GET /models`：DeepSeek 返回的 model schema
//! 缺少 OpenAI OAS 中要求的 `created` 字段，不能直接复用 OpenAI 生成的
//! wire 类型。

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
use defect_http::HttpStackConfig;
use crate::protocol::deepseek_chat;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
const BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";

/// DeepSeek provider 配置。
#[derive(Debug, Default, Clone)]
pub struct DeepSeekConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub http: HttpStackConfig,
}

impl DeepSeekConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            base_url: env::var(BASE_URL_ENV).ok(),
            http: HttpStackConfig::default(),
        }
    }

    fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| env::var(API_KEY_ENV).ok())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
    }

    fn resolve_base_url(&self) -> String {
        self.base_url
            .clone()
            .or_else(|| env::var(BASE_URL_ENV).ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
    }
}

/// DeepSeek provider。
#[derive(Debug)]
pub struct DeepSeekProvider {
    inner: Arc<OpenAiProvider>,
    info: ProviderInfo,
}

impl DeepSeekProvider {
    /// # Errors
    ///
    /// 缺凭证或底层 HTTP 客户端初始化失败时返回错误。
    pub fn new(config: DeepSeekConfig) -> Result<Self, ProviderError> {
        let openai_cfg = OpenAiConfig {
            api_key: config.resolve_api_key(),
            base_url: Some(config.resolve_base_url()),
            organization: None,
            project: None,
            capabilities_override: Some(default_deepseek_capabilities()),
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
    /// 此实现不会构造请求错误。
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
    /// 响应体不是合法 JSON，或与 DeepSeek `/models` 返回结构不匹配时失败。
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
