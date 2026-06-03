//! 装配 [`ProviderRegistry`] 与单个 provider 实例。
//!
//! - [`build_registry`]：装配期入口，给定一份 [`LoadedConfig`] 返回
//!   `(ProviderRegistry, TurnConfig)`，用于直接 attach 到
//!   `DefaultAgentCore::builder().registry(...)`。
//! - [`build_single_llm_provider`]：按 [`ProviderKind`] 构造一个 provider
//!   实例；外部如果要"换掉某家 provider"可以独立调用此函数后自己组装
//!   `ProviderEntry`。
//! - [`build_provider_entries`]：为 `ProviderRegistry::new` 准备的 entries
//!   列表——默认 entry + 用户在 `[providers.*]` 配过的其他 entry。
//!
//! [`ProviderKind`]: defect_config::ProviderKind

// BTreeMap/HashMap + http header 类型仅 provider_headers（openai）用。
#[cfg(feature = "provider-openai")]
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use defect_acp::EchoProvider;
use defect_agent::llm::{
    LlmProvider, ModelCapabilityOverrides, ModelInfo, ProviderEntry, ProviderRegistry,
};
use defect_agent::session::{SessionCapabilitiesConfig, TurnConfig};
use defect_config::{
    LoadedConfig, ProviderConfigFile, ProviderConfigs, ProviderKind as ConfigProviderKind,
    ProviderProtocol,
};
// 仅 reasoning-effort 映射用，随 openai/deepseek 编入。
#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
use defect_config::ReasoningEffort as ConfigReasoningEffort;
#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
use defect_agent::llm::ReasoningEffort as LlmReasoningEffort;
#[cfg(feature = "provider-anthropic")]
use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
#[cfg(feature = "provider-bedrock")]
use defect_llm::provider::bedrock::{BedrockConfig, BedrockProvider};
#[cfg(feature = "provider-deepseek")]
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
#[cfg(feature = "provider-openai")]
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
#[cfg(feature = "provider-openai")]
use http::{HeaderName, HeaderValue};

use crate::http_stack::build_http_stack_config;

pub(crate) const BEDROCK_PROVIDER: &str = "bedrock";
// LiteLLM 走 OpenAI provider，相关常量随 provider-openai 编入。
#[cfg(feature = "provider-openai")]
pub(crate) const LITELLM_API_KEY_ENV: &str = "LITELLM_API_KEY";
#[cfg(feature = "provider-openai")]
pub(crate) const LITELLM_DEFAULT_BASE_URL: &str = "http://localhost:4000/v1";
#[cfg(feature = "provider-openai")]
const CUSTOM_OPENAI_DISPLAY_NAME: &str = "Custom OpenAI-compatible";
#[cfg(feature = "provider-bedrock")]
const CUSTOM_BEDROCK_DISPLAY_NAME: &str = "Amazon Bedrock";
#[cfg(feature = "provider-openai")]
const LITELLM_DISPLAY_NAME: &str = "LiteLLM Gateway";

/// 装配 provider registry 与默认 turn config。
///
/// 入口给主 binary：
/// ```ignore
/// let (registry, turn_config) = defect_cli::providers::build_registry(&config).await?;
/// DefaultAgentCore::builder().registry(registry).config(turn_config)...
/// ```
pub async fn build_registry(
    config: &LoadedConfig,
) -> anyhow::Result<(Arc<ProviderRegistry>, TurnConfig)> {
    let http_config = build_http_stack_config(&config.effective.http)?;
    let entries = build_provider_entries(config, http_config).await?;
    let turn_config = config.effective.turn.clone();
    let registry = ProviderRegistry::new(entries, &turn_config.model)
        .map_err(|e| anyhow::anyhow!("provider registry init failed: {e}"))?;
    Ok((Arc::new(registry), turn_config))
}

/// 按 `[providers]` 段为每个有效 [`ProviderKind`] 装配一个
/// [`ProviderEntry`]——默认 provider 必在；其他 entry 仅在它们声明了
/// `default_model` / `models` 时才装配。
pub async fn build_provider_entries(
    config: &LoadedConfig,
    http_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Vec<ProviderEntry>> {
    let default_kind = config.effective.cli.provider.clone();
    let default_provider =
        build_single_llm_provider(&default_kind, config, http_config.clone()).await?;
    let mut entries = vec![ProviderEntry::new(
        default_provider,
        entry_models(
            provider_config_for_kind(&config.effective.providers, &default_kind),
            Some(config.effective.turn.model.as_str()),
        ),
        provider_session_capabilities(config, &default_kind),
    )];

    for provider_kind in configured_entry_kinds(config) {
        if provider_kind == default_kind {
            continue;
        }
        let models = entry_models(
            provider_config_for_kind(&config.effective.providers, &provider_kind),
            None,
        );
        if models.is_empty() {
            continue;
        }
        let provider =
            build_single_llm_provider(&provider_kind, config, http_config.clone()).await?;
        entries.push(ProviderEntry::new(
            provider,
            models,
            provider_session_capabilities(config, &provider_kind),
        ));
    }

    Ok(entries)
}

/// 按 [`ProviderKind`] 实例化一个 provider。
///
/// 下游二次开发想"自己换 OpenAI 实现"时——独立调用此函数构造默认
/// 那家，再 push 一份自定义 entry 进 [`ProviderRegistry::new`]。
// http_config 只被 anthropic/openai/deepseek 走 hyper 的 provider 用；bedrock 走
// AWS SDK 自己的 transport、echo 无 transport。这些组合下参数未用，按需放行。
#[cfg_attr(
    not(any(
        feature = "provider-anthropic",
        feature = "provider-openai",
        feature = "provider-deepseek"
    )),
    allow(unused_variables)
)]
pub async fn build_single_llm_provider(
    provider_kind: &ConfigProviderKind,
    config: &LoadedConfig,
    http_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    match provider_kind {
        ConfigProviderKind::Echo => Ok(Arc::new(EchoProvider::new()) as Arc<dyn LlmProvider>),
        #[cfg(feature = "provider-anthropic")]
        ConfigProviderKind::Anthropic => Ok(Arc::new(
            AnthropicProvider::new(AnthropicConfig {
                api_key: None,
                api_key_env: config.effective.providers.anthropic.api_key_env.clone(),
                base_url: config.effective.providers.anthropic.base_url.clone(),
                http: http_config,
            })
            .map_err(|e| anyhow::anyhow!("anthropic provider init failed: {e}"))?,
        ) as Arc<dyn LlmProvider>),
        #[cfg(feature = "provider-openai")]
        ConfigProviderKind::Openai => build_openai_provider(
            "openai",
            "OpenAI Chat Completions",
            config.effective.providers.openai.clone(),
            http_config,
        ),
        #[cfg(feature = "provider-deepseek")]
        ConfigProviderKind::Deepseek => Ok(Arc::new(
            DeepSeekProvider::new(DeepSeekConfig {
                api_key: None,
                api_key_env: config.effective.providers.deepseek.api_key_env.clone(),
                base_url: config.effective.providers.deepseek.base_url.clone(),
                reasoning_effort: config
                    .effective
                    .providers
                    .deepseek
                    .reasoning_effort
                    .map(map_reasoning_effort),
                http: http_config,
            })
            .map_err(|e| anyhow::anyhow!("deepseek provider init failed: {e}"))?,
        ) as Arc<dyn LlmProvider>),
        // LiteLLM 复用 OpenAI provider 实现，因此跟随 provider-openai。
        #[cfg(feature = "provider-openai")]
        ConfigProviderKind::Litellm => {
            build_litellm_provider(config.effective.providers.litellm.clone(), http_config)
        }
        // 被配置选中但未编译进本 build 的 provider：hard fail，给出可操作提示。
        // echo 永远可用、不进这条；custom 单独在下方处理。
        #[cfg(not(feature = "provider-anthropic"))]
        ConfigProviderKind::Anthropic => Err(provider_not_compiled("anthropic")),
        #[cfg(not(feature = "provider-openai"))]
        ConfigProviderKind::Openai => Err(provider_not_compiled("openai")),
        #[cfg(not(feature = "provider-deepseek"))]
        ConfigProviderKind::Deepseek => Err(provider_not_compiled("deepseek")),
        #[cfg(not(feature = "provider-openai"))]
        ConfigProviderKind::Litellm => Err(provider_not_compiled("openai")),
        ConfigProviderKind::Custom(name) => {
            let Some(provider) = config
                .effective
                .providers
                .get(&ConfigProviderKind::Custom(name.clone()))
            else {
                return Err(anyhow::anyhow!("missing [providers.{name}] configuration"));
            };
            // 协议默认值：bedrock / aws 段存在 → anthropic-messages；
            // 否则按 OpenAI Chat。这条派遣前没有兜底——`bedrock` 习惯写
            // `[providers.bedrock] aws = { ... }` 不显式标 protocol，被默认
            // 路由到 OpenAI builder 后报 "missing OPENAI_API_KEY"，与实际
            // 配置完全不沾边。
            let protocol = provider.protocol.unwrap_or_else(|| {
                if name == BEDROCK_PROVIDER || provider.aws.is_some() {
                    ProviderProtocol::AnthropicMessages
                } else {
                    ProviderProtocol::OpenaiChat
                }
            });
            match protocol {
                #[cfg(feature = "provider-openai")]
                ProviderProtocol::OpenaiChat => build_openai_provider(
                    name,
                    provider
                        .display_name
                        .as_deref()
                        .unwrap_or(CUSTOM_OPENAI_DISPLAY_NAME),
                    provider.clone(),
                    http_config,
                ),
                #[cfg(not(feature = "provider-openai"))]
                ProviderProtocol::OpenaiChat => Err(provider_not_compiled("openai")),
                ProviderProtocol::AnthropicMessages => {
                    if name == BEDROCK_PROVIDER || provider.aws.is_some() {
                        #[cfg(feature = "provider-bedrock")]
                        {
                            build_bedrock_provider(name, provider.clone()).await
                        }
                        #[cfg(not(feature = "provider-bedrock"))]
                        {
                            Err(provider_not_compiled("bedrock"))
                        }
                    } else {
                        Err(anyhow::anyhow!(
                            "custom provider `{name}` uses protocol `anthropic-messages`, \
                             but only AWS Bedrock transport is implemented for custom providers"
                        ))
                    }
                }
            }
        }
    }
}

/// 被配置选中、但未通过 `provider-*` feature 编译进本 build 的 provider —— hard
/// fail 并提示要开哪个 feature（遵循 fail-loud：不静默回退到 echo）。
///
/// 全 provider 开启时没有任何调用点，故仅在至少裁掉一家时编入。
#[cfg(not(all(
    feature = "provider-anthropic",
    feature = "provider-bedrock",
    feature = "provider-openai",
    feature = "provider-deepseek"
)))]
fn provider_not_compiled(feature_suffix: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "provider was selected but not compiled into this build; \
         rebuild with `--features provider-{feature_suffix}` \
         (or use the default feature set)"
    )
}

/// 把全局 [`capabilities`] 与 `providers.<p>.capabilities` 合并，再投影成
/// agent 侧的 [`SessionCapabilitiesConfig`]。供每个 entry 自带——这样
/// session 跨 provider 切 model 时也能拿到正确的 capability 配置。
///
/// [`capabilities`]: defect_config::CapabilitiesConfig
fn provider_session_capabilities(
    config: &LoadedConfig,
    provider: &ConfigProviderKind,
) -> SessionCapabilitiesConfig {
    match provider {
        ConfigProviderKind::Anthropic => config
            .effective
            .providers
            .anthropic
            .capabilities
            .merge_into(config.effective.capabilities),
        ConfigProviderKind::Openai => config
            .effective
            .providers
            .openai
            .capabilities
            .merge_into(config.effective.capabilities),
        ConfigProviderKind::Deepseek => config
            .effective
            .providers
            .deepseek
            .capabilities
            .merge_into(config.effective.capabilities),
        ConfigProviderKind::Litellm => config
            .effective
            .providers
            .litellm
            .capabilities
            .merge_into(config.effective.capabilities),
        ConfigProviderKind::Echo => config.effective.capabilities,
        ConfigProviderKind::Custom(name) => config
            .effective
            .providers
            .get(&ConfigProviderKind::Custom(name.clone()))
            .map(|provider| {
                provider
                    .capabilities
                    .merge_into(config.effective.capabilities)
            })
            .unwrap_or(config.effective.capabilities),
    }
    .to_session_capabilities()
}

fn configured_entry_kinds(config: &LoadedConfig) -> Vec<ConfigProviderKind> {
    let mut kinds = vec![
        ConfigProviderKind::Anthropic,
        ConfigProviderKind::Openai,
        ConfigProviderKind::Deepseek,
        ConfigProviderKind::Litellm,
    ];
    kinds.extend(
        config
            .effective
            .providers
            .custom
            .keys()
            .cloned()
            .map(ConfigProviderKind::Custom),
    );
    kinds
}

fn provider_config_for_kind<'a>(
    providers: &'a ProviderConfigs,
    kind: &ConfigProviderKind,
) -> Option<&'a ProviderConfigFile> {
    providers.get(kind)
}

fn entry_models(
    provider: Option<&ProviderConfigFile>,
    fallback_model: Option<&str>,
) -> Vec<ModelInfo> {
    let mut models: Vec<ModelInfo> = Vec::new();
    if let Some(provider) = provider {
        // `default_model` 只有 id（裸字符串），无展示名。
        if let Some(default_model) = &provider.default_model {
            push_unique_model(&mut models, default_model, None);
        }
        if let Some(entries) = &provider.models {
            for entry in entries {
                push_unique_model(&mut models, entry.id(), entry.name());
            }
        }
    }
    if models.is_empty()
        && let Some(fallback_model) = fallback_model
    {
        push_unique_model(&mut models, fallback_model, None);
    }
    models
}

/// 按 id 去重地追加一个 [`ModelInfo`]。已存在同 id 时：若新声明带展示名而旧的
/// 没有，用新的补上（让 `[[models]]` 里的 `name` 覆盖来自 `default_model` 的
/// 裸 id 项）；否则保持原样。
fn push_unique_model(models: &mut Vec<ModelInfo>, id: &str, name: Option<&str>) {
    if let Some(existing) = models.iter_mut().find(|m| m.id == id) {
        if existing.display_name.is_none() {
            existing.display_name = name.map(str::to_string);
        }
        return;
    }
    models.push(ModelInfo {
        id: id.to_string(),
        display_name: name.map(str::to_string),
        context_window: None,
        max_output_tokens: None,
        deprecated: false,
        capabilities_overrides: ModelCapabilityOverrides::default(),
    });
}

#[cfg(feature = "provider-openai")]
fn build_litellm_provider(
    provider: ProviderConfigFile,
    http_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    let provider = ProviderDefaults {
        base_url: LITELLM_DEFAULT_BASE_URL,
        api_key_env: LITELLM_API_KEY_ENV,
    }
    .apply(provider);
    build_openai_provider("litellm", LITELLM_DISPLAY_NAME, provider, http_config)
}

#[cfg(feature = "provider-bedrock")]
async fn build_bedrock_provider(
    vendor: &str,
    provider: ProviderConfigFile,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    let aws = provider.aws.unwrap_or_default();
    let provider = BedrockProvider::new(BedrockConfig {
        vendor: Some(vendor.to_string()),
        display_name: Some(
            provider
                .display_name
                .unwrap_or_else(|| CUSTOM_BEDROCK_DISPLAY_NAME.to_string()),
        ),
        base_url: provider.base_url,
        default_model: provider.default_model,
        // Bedrock provider 的 model 列表只需 id；展示名在 entry_models 链路另取。
        models: provider
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.id().to_string())
            .collect(),
        aws_profile: aws.profile,
        aws_region: aws.region,
    })
    .await
    .map_err(|e| anyhow::anyhow!("{vendor} provider init failed: {e}"))?;
    Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
}

#[cfg(feature = "provider-openai")]
fn build_openai_provider(
    vendor: &str,
    display_name: &str,
    provider: ProviderConfigFile,
    http_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    let provider = OpenAiProvider::new(OpenAiConfig {
        api_key: provider
            .api_key_env
            .as_deref()
            .and_then(|env| std::env::var(env).ok()),
        base_url: provider.base_url,
        organization: provider.organization,
        project: provider.project,
        vendor: vendor.to_string(),
        display_name: display_name.to_string(),
        api_key_env: provider.api_key_env,
        headers: provider_headers(provider.headers)?,
        capabilities_override: None,
        reasoning_effort: provider.reasoning_effort.map(map_reasoning_effort),
        chat_dialect: defect_llm::protocol::openai_chat::ChatDialect::OpenAi,
        http: http_config,
    })
    .map_err(|e| anyhow::anyhow!("{vendor} provider init failed: {e}"))?;
    Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
}

/// 给 OpenAI-兼容 provider 填默认 base_url / api_key_env。
///
/// `pub(crate)` 暴露给 unit test——LiteLLM 装配走这条路径。
#[cfg(feature = "provider-openai")]
pub(crate) struct ProviderDefaults {
    pub(crate) base_url: &'static str,
    pub(crate) api_key_env: &'static str,
}

#[cfg(feature = "provider-openai")]
impl ProviderDefaults {
    pub(crate) fn apply(self, mut provider: ProviderConfigFile) -> ProviderConfigFile {
        provider
            .base_url
            .get_or_insert_with(|| self.base_url.to_string());
        provider
            .api_key_env
            .get_or_insert_with(|| self.api_key_env.to_string());
        provider
    }
}

#[cfg(feature = "provider-openai")]
fn provider_headers(
    headers: BTreeMap<String, String>,
) -> anyhow::Result<HashMap<HeaderName, HeaderValue>> {
    let mut parsed = HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| anyhow::anyhow!("invalid provider header name `{name}`: {e}"))?;
        let header_value = HeaderValue::from_str(&value)
            .map_err(|e| anyhow::anyhow!("invalid provider header value for `{name}`: {e}"))?;
        parsed.insert(header_name, header_value);
    }
    Ok(parsed)
}

#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
pub(crate) fn map_reasoning_effort(value: ConfigReasoningEffort) -> LlmReasoningEffort {
    match value {
        ConfigReasoningEffort::None => LlmReasoningEffort::None,
        ConfigReasoningEffort::Minimal => LlmReasoningEffort::Minimal,
        ConfigReasoningEffort::Low => LlmReasoningEffort::Low,
        ConfigReasoningEffort::Medium => LlmReasoningEffort::Medium,
        ConfigReasoningEffort::High => LlmReasoningEffort::High,
        ConfigReasoningEffort::Xhigh => LlmReasoningEffort::Xhigh,
    }
}
