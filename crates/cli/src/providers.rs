//! Assembles a [`ProviderRegistry`] and individual provider instances.
//!
//! - [`build_registry`]: entry point for assembly; given a [`LoadedConfig`], returns
//!   `(ProviderRegistry, TurnConfig)` for direct attachment to
//!   `DefaultAgentCore::builder().registry(...)`.
//! - [`build_single_llm_provider`]: constructs a provider instance by [`ProviderKind`];
//!   callers that want to "swap out a provider" can call this function independently
//!   and assemble their own `ProviderEntry`.
//! - [`build_provider_entries`]: the list of entries for `ProviderRegistry::new` —
//!   the default entry plus any additional entries the user configured under
//!   `[providers.*]`.
//!
//! [`ProviderKind`]: defect_config::ProviderKind

// BTreeMap/HashMap and http header types are only used by provider_headers, which both
// the openai and anthropic providers feed their custom-header maps through.
#[cfg(any(feature = "provider-openai", feature = "provider-anthropic"))]
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
// Only used for reasoning-effort mapping, included alongside openai/deepseek.
#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
use defect_agent::llm::ReasoningEffort as LlmReasoningEffort;
#[cfg(any(feature = "provider-openai", feature = "provider-deepseek"))]
use defect_config::ReasoningEffort as ConfigReasoningEffort;
#[cfg(feature = "provider-anthropic")]
use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
#[cfg(feature = "provider-bedrock")]
use defect_llm::provider::bedrock::{BedrockConfig, BedrockProvider};
#[cfg(feature = "provider-deepseek")]
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
#[cfg(feature = "provider-openai")]
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
#[cfg(any(feature = "provider-openai", feature = "provider-anthropic"))]
use http::{HeaderName, HeaderValue};

use crate::http_stack::build_http_stack_config;

pub(crate) const BEDROCK_PROVIDER: &str = "bedrock";
// LiteLLM uses the OpenAI provider; related constants are compiled in under
// `provider-openai`.
#[cfg(feature = "provider-openai")]
pub(crate) const LITELLM_API_KEY_ENV: &str = "LITELLM_API_KEY";
#[cfg(feature = "provider-openai")]
pub(crate) const LITELLM_DEFAULT_BASE_URL: &str = "http://localhost:4000/v1";
#[cfg(feature = "provider-openai")]
const CUSTOM_OPENAI_DISPLAY_NAME: &str = "Custom OpenAI-compatible";
#[cfg(feature = "provider-anthropic")]
const CUSTOM_ANTHROPIC_DISPLAY_NAME: &str = "Custom Anthropic-compatible";
#[cfg(feature = "provider-bedrock")]
const CUSTOM_BEDROCK_DISPLAY_NAME: &str = "Amazon Bedrock";
#[cfg(feature = "provider-openai")]
const LITELLM_DISPLAY_NAME: &str = "LiteLLM Gateway";

/// Assembles the provider registry and default turn config.
///
/// Entry point for the main binary:
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
    let registry = ProviderRegistry::new(entries, &turn_config.provider, &turn_config.model)
        .map_err(|e| anyhow::anyhow!("provider registry init failed: {e}"))?;
    Ok((Arc::new(registry), turn_config))
}

/// For each valid `ProviderKind` in the `[providers]` section, assemble a
/// [`ProviderEntry`] — the default provider is always included; other entries are only
/// included if they declare `default_model` or `models`.
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

/// Instantiate a provider based on [`ProviderKind`](defect_config::ProviderKind).
///
/// When downstream developers want to swap in their own OpenAI implementation, call this
/// function independently to construct the default provider, then push a custom entry
/// into [`ProviderRegistry::new`].
// `http_config` is only used by the anthropic, openai, and deepseek providers (which use
// hyper); bedrock uses the AWS SDK's own transport, and echo has no transport. For these
// combinations the parameter is unused and is allowed accordingly.
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
        ConfigProviderKind::Defect => Ok(Arc::new(EchoProvider::new()) as Arc<dyn LlmProvider>),
        #[cfg(feature = "provider-anthropic")]
        ConfigProviderKind::Anthropic => build_anthropic_provider(
            "anthropic",
            None,
            config.effective.providers.anthropic.clone(),
            http_config,
        ),
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
        // LiteLLM reuses the OpenAI provider implementation, so it follows
        // `provider-openai`.
        #[cfg(feature = "provider-openai")]
        ConfigProviderKind::Litellm => {
            build_litellm_provider(config.effective.providers.litellm.clone(), http_config)
        }
        // Providers selected by config but not compiled into this build: hard fail with
        // actionable hint.
        // Echo is always available and never reaches this branch; custom is handled
        // separately below.
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
            // Protocol default: if the provider is `bedrock` or has an `aws` section, use
            // `AnthropicMessages`; otherwise fall back to `OpenaiChat`. Previously there
            // was no fallback before dispatch — users writing `[providers.bedrock] aws =
            // { ... }` without an explicit `protocol` would be routed to the OpenAI
            // builder, producing a misleading "missing OPENAI_API_KEY" error unrelated to
            // their actual configuration.
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
                        // Custom HTTP endpoint speaking the Anthropic Messages protocol.
                        // Reuses `AnthropicProvider`; `auth_header` lets the gateway's
                        // credential header differ from the official `x-api-key`.
                        #[cfg(feature = "provider-anthropic")]
                        {
                            let display_name = provider
                                .display_name
                                .clone()
                                .unwrap_or_else(|| CUSTOM_ANTHROPIC_DISPLAY_NAME.to_string());
                            build_anthropic_provider(
                                name,
                                Some(display_name),
                                provider.clone(),
                                http_config,
                            )
                        }
                        #[cfg(not(feature = "provider-anthropic"))]
                        {
                            Err(provider_not_compiled("anthropic"))
                        }
                    }
                }
            }
        }
    }
}

/// A provider that was selected by configuration but not compiled into this build via a
/// `provider-*` feature — hard fail with a message indicating which feature to enable
/// (following the fail-loud principle: no silent fallback to echo).
///
/// This function has no call sites when all providers are enabled, so it is only compiled
/// when at least one provider is excluded.
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

/// Merge the global [`capabilities`] with `providers.<p>.capabilities` and project the
/// result into the agent-side [`SessionCapabilitiesConfig`]. Each entry carries its own
/// copy so that the session can obtain the correct capability configuration when
/// switching models across providers.
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
        ConfigProviderKind::Defect => config.effective.capabilities,
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
        // `default_model` is just an ID (a bare string) with no display name or limits.
        if let Some(default_model) = &provider.default_model {
            push_unique_model(&mut models, default_model, None, None, None);
        }
        if let Some(entries) = &provider.models {
            for entry in entries {
                push_unique_model(
                    &mut models,
                    entry.id(),
                    entry.name(),
                    entry.context_window(),
                    entry.max_output_tokens(),
                );
            }
        }
    }
    if models.is_empty()
        && let Some(fallback_model) = fallback_model
    {
        push_unique_model(&mut models, fallback_model, None, None, None);
    }
    models
}

/// Append a [`ModelInfo`] deduplicated by `id`. If an entry with the same `id` already
/// exists, fill in any field the existing entry is missing (so a `[[models]]` table form
/// can enrich a bare id contributed by `default_model`); otherwise leave it unchanged.
fn push_unique_model(
    models: &mut Vec<ModelInfo>,
    id: &str,
    name: Option<&str>,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
) {
    if let Some(existing) = models.iter_mut().find(|m| m.id == id) {
        if existing.display_name.is_none() {
            existing.display_name = name.map(str::to_string);
        }
        existing.context_window = existing.context_window.or(context_window);
        existing.max_output_tokens = existing.max_output_tokens.or(max_output_tokens);
        return;
    }
    models.push(ModelInfo {
        id: id.to_string(),
        display_name: name.map(str::to_string),
        context_window,
        max_output_tokens,
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
        // Display names are fetched separately in the `entry_models` pipeline, but the
        // limits (`context_window` / `max_output_tokens`) must be carried into the provider
        // here — the Bedrock SDK cannot discover them, and compaction reads them back via
        // `provider.model_info()`.
        models: provider
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| defect_llm::provider::bedrock::BedrockModel {
                id: m.id().to_string(),
                context_window: m.context_window(),
                max_output_tokens: m.max_output_tokens(),
            })
            .collect(),
        aws_profile: aws.profile,
        aws_region: aws.region,
        anthropic_beta: aws.anthropic_beta,
    })
    .await
    .map_err(|e| anyhow::anyhow!("{vendor} provider init failed: {e}"))?;
    Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
}

/// Build an `AnthropicProvider` for either the built-in `anthropic` kind or a custom
/// `anthropic-messages` HTTP endpoint (a gateway fronting the protocol). `vendor` is the
/// registry key; `display_name`, when `None`, falls back to the provider's own default.
#[cfg(feature = "provider-anthropic")]
fn build_anthropic_provider(
    vendor: &str,
    display_name: Option<String>,
    provider: ProviderConfigFile,
    http_config: defect_http::HttpStackConfig,
) -> anyhow::Result<Arc<dyn LlmProvider>> {
    let provider = AnthropicProvider::new(AnthropicConfig {
        api_key: provider
            .api_key_env
            .as_deref()
            .and_then(|env| std::env::var(env).ok()),
        api_key_env: provider.api_key_env,
        base_url: provider.base_url,
        vendor: Some(vendor.to_string()),
        display_name,
        auth_header: provider.auth_header,
        headers: provider_headers(provider.headers)?,
        http: http_config,
    })
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

/// Fill default `base_url` / `api_key_env` for OpenAI-compatible providers.
///
/// `pub(crate)` is exposed for unit tests — LiteLLM assembly uses this path.
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

#[cfg(any(feature = "provider-openai", feature = "provider-anthropic"))]
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
