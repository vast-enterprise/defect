use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use defect_agent::error::BoxError;
use defect_agent::session::{SearchCapabilityConfig, SessionCapabilitiesConfig, TurnConfig};
use serde::{Deserialize, Serialize};
use toml::Value as TomlValue;

pub(crate) const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";
pub(crate) const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
pub(crate) const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";
pub(crate) const DEFAULT_ECHO_MODEL: &str = "echo";
pub(crate) const DEFAULT_BASH_TIMEOUT_MS: u64 = 30_000;
pub(crate) const DEFAULT_BASH_MAX_TIMEOUT_MS: u64 = 600_000;
pub(crate) const DEFAULT_FS_READ_LIMIT: u32 = 2_000;
pub(crate) const DEFAULT_FS_READ_MAX_LIMIT: u32 = 5_000;

pub(crate) const USER_CONFIG_RELATIVE: &str = "defect/config.toml";
pub(crate) const PROJECT_CONFIG_RELATIVE: &str = ".defect/config.toml";
pub(crate) const PROJECT_LOCAL_CONFIG_RELATIVE: &str = ".defect/config.local.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[default]
    Echo,
    Anthropic,
    Openai,
    Deepseek,
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Echo => "echo",
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Deepseek => "deepseek",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    Defaults,
    User,
    Project,
    ProjectLocal,
    Cli,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigLayerEntry {
    pub source: ConfigSource,
    pub path: Option<PathBuf>,
    pub raw_toml: Option<String>,
    pub value: TomlValue,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConfigLayerStack {
    pub layers: Vec<ConfigLayerEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    IgnoredProjectKey {
        path: PathBuf,
        key: String,
        reason: &'static str,
    },
    UnknownKey {
        path: PathBuf,
        key: String,
    },
    DeprecatedKey {
        path: PathBuf,
        old: String,
        new: String,
    },
    /// 配置文件里出现了某段，但在当前 mode 下不会生效。
    ///
    /// 详见 `docs/proposals/config-capabilities-and-tools.md` §6.2。
    /// 典型场景：`capabilities.search.mode = "delegate"` 时仍写了
    /// `[tools.search]`——本地 search 不会注册，该段实际不生效。
    InactiveSection {
        path: PathBuf,
        section: String,
        reason: String,
    },
    /// 撞名的 MCP 工具在 session 启动期被重命名为 `mcp.<server>.<name>`。
    ///
    /// 详见 `docs/proposals/search-capability-and-fetch-tool.md` §7.3 与
    /// `config-capabilities-and-tools.md` §14。`search` / `fetch` 一律
    /// 重命名（无视 capability mode 与 tool enabled），避免 MCP 旁路占名。
    McpToolRenamed {
        server: String,
        original: String,
        renamed: String,
    },
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: BoxError,
    },

    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: BoxError,
    },

    #[error("invalid config at {path}: {message}")]
    Invalid { path: PathBuf, message: String },

    #[error(transparent)]
    Source(#[from] BoxError),
}

#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub provider: Option<ProviderKind>,
    pub model: Option<String>,
    pub config_overrides: Vec<(String, TomlValue)>,
}

#[derive(Debug, Clone, Default)]
pub struct LoadConfigOptions {
    pub cwd: PathBuf,
    pub cli: CliOverrides,
    pub xdg_config_home: Option<PathBuf>,
    pub home_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub layers: ConfigLayerStack,
    pub effective: EffectiveConfig,
    pub warnings: Vec<ConfigWarning>,
}

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub cli: CliConfig,
    pub turn: TurnConfig,
    pub base_prompt: BasePromptConfigFile,
    pub prompt: PromptConfigFile,
    /// 全局 capability 来源选择。`providers.<p>.capabilities` 覆写在
    /// session 启动期叠加。详见
    /// `docs/proposals/config-capabilities-and-tools.md` §3 / §13。
    pub capabilities: CapabilitiesConfig,
    pub providers: ProviderConfigs,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfig,
    pub tracing: TracingConfig,
    pub mcp: McpConfig,
    pub http: HttpClientConfig,
}

/// 全局 capability 配置入口。
///
/// 与 [`SessionCapabilitiesConfig`] 形态等价；在 `EffectiveConfig` 上保
/// 留独立类型是为了未来追加非 session 级 capability 时不动 agent crate。
/// 当前 P1 仅有 `search`，直接复用 agent 侧的 `SearchCapabilityConfig`。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapabilitiesConfig {
    pub search: SearchCapabilityConfig,
}

impl CapabilitiesConfig {
    /// 用单条 [`SearchCapabilityConfig`] 构造。跨 crate 调用方需要这个
    /// 入口，因为本结构体 `#[non_exhaustive]` 后不能直接 struct literal。
    #[must_use]
    pub const fn with_search(search: SearchCapabilityConfig) -> Self {
        Self { search }
    }

    /// 转成 agent 侧的 [`SessionCapabilitiesConfig`]，供
    /// `DefaultAgentCoreBuilder::capabilities` 直接消费。
    #[must_use]
    pub fn to_session_capabilities(self) -> SessionCapabilitiesConfig {
        SessionCapabilitiesConfig::with_search(self.search)
    }
}

/// 单个 provider 下对全局 capability 的覆写。
///
/// `None` 字段意味着「跟随全局」。详见 §5 / §13.2。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderCapabilityOverrides {
    pub search: Option<SearchCapabilityConfig>,
}

impl ProviderCapabilityOverrides {
    /// 用单条 search 覆写构造。`None` = 跟随全局。
    #[must_use]
    pub const fn with_search(search: Option<SearchCapabilityConfig>) -> Self {
        Self { search }
    }

    /// 把全局 [`CapabilitiesConfig`] 与本 provider 的覆写合并。
    /// 未覆写字段沿用全局值。
    #[must_use]
    pub fn merge_into(&self, base: CapabilitiesConfig) -> CapabilitiesConfig {
        CapabilitiesConfig::with_search(self.search.unwrap_or(base.search))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub provider: ProviderKind,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BasePromptConfigFile {
    pub file: Option<PathBuf>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptConfigFile {
    pub file: String,
    pub text: Option<String>,
    pub provider_overlays: BTreeMap<String, String>,
    pub model_overlays: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProviderConfigs {
    pub anthropic: AnthropicConfigFile,
    pub openai: OpenAiConfigFile,
    pub deepseek: DeepSeekConfigFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolsConfig {
    pub bash: BashToolConfig,
    pub fs: FsToolConfig,
    /// `[tools.fetch]` 段。预留出来——P1 仅 schema 落地，工具实现在后续 PR。
    pub fetch: FetchToolConfig,
}

/// 本地 `fetch` 工具的配置。详见
/// `docs/proposals/config-capabilities-and-tools.md` §7.2。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchToolConfig {
    pub enabled: bool,
    pub default_timeout_secs: u32,
    pub max_timeout_secs: u32,
    pub max_response_bytes: u64,
    pub default_format: FetchFormat,
    pub html_to_markdown: bool,
    pub follow_redirects: bool,
}

impl Default for FetchToolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_timeout_secs: 30,
            max_timeout_secs: 120,
            max_response_bytes: 5 * 1024 * 1024,
            default_format: FetchFormat::Markdown,
            html_to_markdown: true,
            follow_redirects: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FetchFormat {
    #[default]
    Markdown,
    Html,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashToolConfig {
    pub default_timeout_ms: u64,
    pub max_timeout_ms: u64,
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: DEFAULT_BASH_TIMEOUT_MS,
            max_timeout_ms: DEFAULT_BASH_MAX_TIMEOUT_MS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsToolConfig {
    pub read_default_limit: u32,
    pub read_max_limit: u32,
}

impl Default for FsToolConfig {
    fn default() -> Self {
        Self {
            read_default_limit: DEFAULT_FS_READ_LIMIT,
            read_max_limit: DEFAULT_FS_READ_MAX_LIMIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxConfig {
    pub mode: SandboxMode,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            mode: SandboxMode::AskWrites,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    ReadOnly,
    #[default]
    AskWrites,
    Open,
    DenyAll,
}

impl SandboxMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::AskWrites => "ask-writes",
            Self::Open => "open",
            Self::DenyAll => "deny-all",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnthropicConfigFile {
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<String>>,
    pub capabilities: ProviderCapabilityOverrides,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpenAiConfigFile {
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<String>>,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub capabilities: ProviderCapabilityOverrides,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeepSeekConfigFile {
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<String>>,
    pub capabilities: ProviderCapabilityOverrides,
}

/// HTTP 客户端栈的 typed 配置。
///
/// 仅描述用户意图（`None` 一律按"用 HTTP 栈层默认值"理解）；CLI 入口在
/// 装配 provider 时把它翻成 [`defect_http::HttpStackConfig`]。
///
/// 和 [`defect_http::HttpStackConfig`] 不直接共享类型是为了保持 crate
/// 单向依赖：`defect-config` 不引 `defect-http`，避免 fetch tool 这类
/// 后续消费者再次倒挂。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HttpClientConfig {
    /// 单次请求总超时；`None` = 用 HTTP 栈层默认（600s）。
    pub total_timeout_ms: Option<u64>,
    /// transport 错误重试上限（不含首次）；`None` = 默认 2，`Some(0)` 禁用。
    pub transport_retries: Option<u8>,
    /// 重试初始 backoff；`None` = 默认 200ms。
    pub initial_backoff_ms: Option<u64>,
    /// `User-Agent` header 覆盖；`None` = 用编译期默认
    /// `defect-http/{version} ({git_sha})`。
    pub user_agent: Option<String>,
    /// 代理子配置。`mode` 默认 `FromEnv`（读取 `HTTP_PROXY` 等）。
    pub proxy: HttpProxyConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpProxyConfig {
    pub mode: HttpProxyMode,
    /// 显式代理；仅在 `mode = Explicit` 时生效。
    pub explicit: HttpProxySettings,
}

impl Default for HttpProxyConfig {
    fn default() -> Self {
        Self {
            mode: HttpProxyMode::FromEnv,
            explicit: HttpProxySettings::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum HttpProxyMode {
    #[default]
    FromEnv,
    Disabled,
    Explicit,
}

impl HttpProxyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FromEnv => "from-env",
            Self::Disabled => "disabled",
            Self::Explicit => "explicit",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HttpProxySettings {
    pub http_proxy: Option<String>,
    pub https_proxy: Option<String>,
    pub no_proxy: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TracingConfig {
    pub filter: Option<String>,
    pub otlp: Option<OtlpTracingConfig>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OtlpTracingConfig {
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct McpConfig {
    pub enabled_servers: Vec<String>,
    pub servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerConfig {
    Stdio(McpStdioServerConfig),
    Http(McpRemoteServerConfig),
    Sse(McpRemoteServerConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStdioServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpRemoteServerConfig {
    pub url: String,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum McpTransportKind {
    Stdio,
    Http,
    Sse,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ConfigToml {
    #[serde(default)]
    pub(crate) default: DefaultSection,
    #[serde(default)]
    pub(crate) base_prompt: BasePromptSection,
    #[serde(default)]
    pub(crate) prompt: PromptSection,
    #[serde(default)]
    pub(crate) turn: TurnSection,
    #[serde(default)]
    pub(crate) capabilities: CapabilitiesSection,
    #[serde(default)]
    pub(crate) providers: ProvidersSection,
    #[serde(default)]
    pub(crate) tools: ToolsSection,
    #[serde(default)]
    pub(crate) sandbox: SandboxSection,
    #[serde(default)]
    pub(crate) tracing: TracingSection,
    #[serde(default)]
    pub(crate) mcp: McpSection,
    #[serde(default)]
    pub(crate) http: HttpSection,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct CapabilitiesSection {
    pub(crate) search: Option<SearchCapabilitySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct SearchCapabilitySection {
    pub(crate) mode: Option<defect_agent::session::SearchCapabilityMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ProviderCapabilitiesSection {
    pub(crate) search: Option<SearchCapabilitySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DefaultSection {
    pub(crate) provider: Option<ProviderKind>,
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct BasePromptSection {
    pub(crate) file: Option<String>,
    pub(crate) text: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TurnSection {
    pub(crate) system_prompt: Option<String>,
    pub(crate) request_limit: Option<u32>,
    pub(crate) compact_threshold_tokens: Option<u64>,
    pub(crate) max_llm_retries: Option<u32>,
    pub(crate) max_concurrent_tools: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PromptSection {
    pub(crate) file: Option<String>,
    pub(crate) text: Option<String>,
    pub(crate) providers: Option<BTreeMap<String, PromptOverlaySection>>,
    pub(crate) models: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PromptOverlaySection {
    pub(crate) text: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ProvidersSection {
    pub(crate) anthropic: Option<AnthropicProviderSection>,
    pub(crate) openai: Option<OpenAiProviderSection>,
    pub(crate) deepseek: Option<DeepSeekProviderSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AnthropicProviderSection {
    pub(crate) base_url: Option<String>,
    pub(crate) default_model: Option<String>,
    pub(crate) models: Option<Vec<String>>,
    pub(crate) capabilities: Option<ProviderCapabilitiesSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OpenAiProviderSection {
    pub(crate) base_url: Option<String>,
    pub(crate) default_model: Option<String>,
    pub(crate) models: Option<Vec<String>>,
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) capabilities: Option<ProviderCapabilitiesSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DeepSeekProviderSection {
    pub(crate) base_url: Option<String>,
    pub(crate) default_model: Option<String>,
    pub(crate) models: Option<Vec<String>>,
    pub(crate) capabilities: Option<ProviderCapabilitiesSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ToolsSection {
    pub(crate) bash: Option<BashToolSection>,
    pub(crate) fs: Option<FsToolSection>,
    pub(crate) fetch: Option<FetchToolSection>,
    /// `[tools.search]`：mode = local 时本地实现的参数。P1 仅识别段，
    /// 不强 schema——具体字段在 search tool 落地 PR 里再细化。是否生效
    /// 由 `loader::collect_inactive_section_warnings` 在合并后从 raw
    /// TomlValue 上判断；这个字段只是为了让 `serde::deserialize` 把段读掉，
    /// 避免 `UnknownKey` warning，它本身不需要被消费。
    #[allow(dead_code)]
    pub(crate) search: Option<TomlValue>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FetchToolSection {
    pub(crate) enabled: Option<bool>,
    pub(crate) default_timeout_secs: Option<u32>,
    pub(crate) max_timeout_secs: Option<u32>,
    pub(crate) max_response_bytes: Option<u64>,
    pub(crate) default_format: Option<FetchFormat>,
    pub(crate) html_to_markdown: Option<bool>,
    pub(crate) follow_redirects: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct BashToolSection {
    pub(crate) default_timeout_ms: Option<u64>,
    pub(crate) max_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FsToolSection {
    pub(crate) read_default_limit: Option<u32>,
    pub(crate) read_max_limit: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct SandboxSection {
    pub(crate) mode: Option<SandboxMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TracingSection {
    pub(crate) filter: Option<String>,
    pub(crate) otlp: Option<OtlpTracingSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OtlpTracingSection {
    pub(crate) endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct McpSection {
    pub(crate) enabled_servers: Option<Vec<String>>,
    pub(crate) servers: Option<BTreeMap<String, McpServerSection>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct HttpSection {
    pub(crate) total_timeout_ms: Option<u64>,
    pub(crate) transport_retries: Option<u8>,
    pub(crate) initial_backoff_ms: Option<u64>,
    pub(crate) user_agent: Option<String>,
    pub(crate) proxy: Option<HttpProxySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct HttpProxySection {
    pub(crate) mode: Option<HttpProxyMode>,
    pub(crate) http_proxy: Option<String>,
    pub(crate) https_proxy: Option<String>,
    pub(crate) no_proxy: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct McpServerSection {
    pub(crate) transport: Option<McpTransportKind>,
    pub(crate) command: Option<String>,
    pub(crate) args: Option<Vec<String>>,
    pub(crate) env: Option<BTreeMap<String, String>>,
    pub(crate) url: Option<String>,
    pub(crate) headers: Option<BTreeMap<String, String>>,
}
