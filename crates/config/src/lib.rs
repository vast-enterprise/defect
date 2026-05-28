//! 配置加载与合并。
//!
//! P1 负责把用户配置、项目配置、本地项目覆盖与 CLI override 收敛成一份
//! 可直接用于启动的强类型配置对象。

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]

mod hooks;
mod loader;
mod mcp;
mod overrides;
mod types;

pub use loader::{load_config, load_dotenv_compat};
pub use overrides::parse_cli_override;
pub use types::{
    AnthropicConfigFile, BasePromptConfigFile, BashToolConfig, CapabilitiesConfig, CliConfig,
    CliOverrides, ConfigError, ConfigLayerEntry, ConfigLayerStack, ConfigSource, ConfigWarning,
    DeepSeekConfigFile, EffectiveConfig, FetchFormat, FetchToolConfig, FsToolConfig,
    HookCommandSpec, HookEntry, HookHandlerSpec, HookMatcher, HookPromptRender, HookPromptSpec,
    HookShellKind, HooksConfig, HttpClientConfig, HttpProxyConfig, HttpProxyMode,
    HttpProxySettings, LoadConfigOptions, LoadedConfig, McpConfig, McpRemoteServerConfig,
    McpServerConfig, McpStdioServerConfig, OpenAiConfigFile, OtlpTracingConfig, PromptConfigFile,
    ProviderCapabilityOverrides, ProviderConfigs, ProviderKind, SandboxConfig, SandboxMode,
    ToolsConfig, TracingConfig,
};
