//! DeepSeek provider。
//!
//! DeepSeek 走 OpenAI Chat Completions 兼容协议（`POST /chat/completions`，
//! `GET /models`），所以这里**复用** [`super::openai::OpenAiProvider`] 跑
//! transport / wire / 协议解码，外面只换：
//!
//! - **默认 base_url**：`https://api.deepseek.com/v1`，可由 `DEEPSEEK_BASE_URL`
//!   或显式 config 覆盖
//! - **默认凭证 env**：`DEEPSEEK_API_KEY`（兼容写法 `OPENAI_API_KEY` 也认）
//! - **vendor / display_name**：`info()` 报 `"deepseek"` / `"DeepSeek Chat"`
//! - **capabilities.thinking**：`Supported`（DeepSeek `deepseek-reasoner`
//!   原生发 `delta.reasoning_content`，协议层已经把它翻成 `ThinkingDelta`）
//!
//! 模型表（`deepseek-chat` / `deepseek-reasoner` 的 context_window 等）已经
//! 维护在 `provider/openai.rs` 的硬编码表里，无需在这里重复。

use std::env;
use std::sync::Arc;

use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderError, ProviderInfo, ProviderStream, ThinkingEcho,
};
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

use super::openai::{OpenAiConfig, OpenAiProvider};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";
const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
const BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";

/// DeepSeek provider 配置。
///
/// 字段语义同 [`OpenAiConfig`] 的相应字段；缺省时按 `DEEPSEEK_*` env 解析，
/// `api_key` 还会兜底到 `OPENAI_API_KEY`，方便共享凭证的开发环境。
#[derive(Debug, Default, Clone)]
pub struct DeepSeekConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl DeepSeekConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var(API_KEY_ENV).ok(),
            base_url: env::var(BASE_URL_ENV).ok(),
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
///
/// 形态上是个 thin wrapper：对外实现 [`LlmProvider`]，对内全部委托给一个
/// 配好 base_url + capabilities 的 [`OpenAiProvider`]，再把 `info()` 换成
/// `"deepseek"`。这样保证：
///
/// 1. wire/protocol 改动两边自动同步，不会漂
/// 2. DeepSeek 的 `reasoning_content` 已经在协议层处理，无需 provider 介入
#[derive(Debug)]
pub struct DeepSeekProvider {
    inner: Arc<OpenAiProvider>,
    info: ProviderInfo,
}

impl DeepSeekProvider {
    /// 装配 DeepSeek provider。
    ///
    /// # Errors
    ///
    /// 缺凭证 / TLS 客户端构建失败时返回 [`ProviderError`]。
    pub fn new(config: DeepSeekConfig) -> Result<Self, ProviderError> {
        let openai_cfg = OpenAiConfig {
            api_key: config.resolve_api_key(),
            base_url: Some(config.resolve_base_url()),
            organization: None,
            project: None,
            capabilities_override: Some(default_deepseek_capabilities()),
        };
        let inner = OpenAiProvider::new(openai_cfg)?;
        Ok(Self {
            inner: Arc::new(inner),
            info: ProviderInfo {
                vendor: "deepseek".into(),
                protocol: ProtocolId::OpenAiChat,
                display_name: "DeepSeek Chat".into(),
            },
        })
    }
}

/// DeepSeek provider 默认能力矩阵。
///
/// 与 OpenAI 默认差在 `thinking`：DeepSeek `deepseek-reasoner` 原生流式发
/// reasoning，协议层已支持 → 标 `Supported`。`vision` 标 `Unsupported`：
/// 截至 2025 DeepSeek 公开 API 还没多模态输入；上层裁图前会查能力。
fn default_deepseek_capabilities() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Supported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Supported,
        // DeepSeek 各模型行为不一致：R1 系列禁止回放 reasoning_content，
        // v4-pro 必须回放。provider 默认走保守 Forbidden，模型级覆盖走
        // [`ModelCapabilityOverrides::thinking_echo`]（见 provider/openai.rs
        // 硬编码表）。
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
        self.inner.list_models()
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.inner.model_info(model_id)
    }

    fn complete(
        &self,
        req: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        self.inner.complete(req, cancel)
    }
}
