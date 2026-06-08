//! Configuration loading and merging.
//!
//! This module consolidates user configuration, project configuration, local project
//! overrides, and CLI overrides into a single strongly-typed configuration object ready
//! for startup.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

mod frontmatter;
mod hooks;
mod loader;
mod mcp;
mod mcp_json;
mod overrides;
mod profiles;
mod skills;
mod types;

pub use loader::{find_repo_root, load_config, load_dotenv_compat, user_config_path};
pub use overrides::parse_cli_override;
pub use profiles::{ProfileSpec, discover_profiles};
pub use skills::{SkillSpec, discover_skills};
pub use types::{
    AnthropicConfigFile, BasePromptConfigFile, BashToolConfig, CapabilitiesConfig, CliConfig,
    CliOverrides, ConfigError, ConfigLayerEntry, ConfigLayerStack, ConfigSource, ConfigWarning,
    DeepSeekConfigFile, EffectiveConfig, FetchFormat, FetchToolConfig, FsToolConfig,
    HookCommandSpec, HookEntry, HookHandlerSpec, HookMatcher, HookPromptRender, HookPromptSpec,
    HookShellKind, HooksConfig, HttpClientConfig, HttpProxyConfig, HttpProxyMode,
    HttpProxySettings, LangfuseConfig, LiteLlmConfigFile, LoadConfigOptions, LoadedConfig,
    McpConfig, McpRemoteServerConfig, McpServerConfig, McpStdioServerConfig, ModelEntry,
    OpenAiConfigFile, OtlpTracingConfig, PromptConfigFile, ProviderAwsConfigFile,
    ProviderCapabilityOverrides, ProviderConfigFile, ProviderConfigs, ProviderKind,
    ProviderProtocol, ReasoningEffort, SandboxConfig, SandboxMode, SearchToolConfig, ToolsConfig,
    TracingConfig,
};
