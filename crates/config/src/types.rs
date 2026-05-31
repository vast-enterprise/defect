use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use defect_agent::error::BoxError;
use defect_agent::session::{SessionCapabilitiesConfig, TurnConfig, WebSearchCapabilityConfig};
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

const PROVIDER_ECHO: &str = "echo";
const PROVIDER_ANTHROPIC: &str = "anthropic";
const PROVIDER_OPENAI: &str = "openai";
const PROVIDER_DEEPSEEK: &str = "deepseek";
const PROVIDER_LITELLM: &str = "litellm";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ProviderKind {
    #[default]
    Echo,
    Anthropic,
    Openai,
    Deepseek,
    Litellm,
    Custom(String),
}

impl ProviderKind {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Echo => PROVIDER_ECHO,
            Self::Anthropic => PROVIDER_ANTHROPIC,
            Self::Openai => PROVIDER_OPENAI,
            Self::Deepseek => PROVIDER_DEEPSEEK,
            Self::Litellm => PROVIDER_LITELLM,
            Self::Custom(value) => value,
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<ProviderKind> for String {
    fn from(value: ProviderKind) -> Self {
        value.to_string()
    }
}

impl From<String> for ProviderKind {
    fn from(value: String) -> Self {
        match value.as_str() {
            PROVIDER_ECHO => Self::Echo,
            PROVIDER_ANTHROPIC => Self::Anthropic,
            PROVIDER_OPENAI => Self::Openai,
            PROVIDER_DEEPSEEK => Self::Deepseek,
            PROVIDER_LITELLM => Self::Litellm,
            _ => Self::Custom(value),
        }
    }
}

impl From<&str> for ProviderKind {
    fn from(value: &str) -> Self {
        match value {
            PROVIDER_ECHO => Self::Echo,
            PROVIDER_ANTHROPIC => Self::Anthropic,
            PROVIDER_OPENAI => Self::Openai,
            PROVIDER_DEEPSEEK => Self::Deepseek,
            PROVIDER_LITELLM => Self::Litellm,
            other => Self::Custom(other.to_string()),
        }
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
    DeprecatedKey {
        path: PathBuf,
        old: String,
        new: String,
    },
    /// 配置文件里出现了某段，但在当前 mode 下不会生效。
    ///
    /// 详见 `docs/internal/capabilities.md` §3 的语义对照表。
    /// 典型场景：写了某段配置但相应能力被关闭（例如旧版 `capabilities.search`
    /// 段落已被废弃为 `[capabilities.web_search]`，旧键不再生效）。
    InactiveSection {
        path: PathBuf,
        section: String,
        reason: String,
    },
    /// 撞名的 MCP 工具在 session 启动期被重命名为 `mcp.<server>.<name>`。
    ///
    /// 详见 `docs/internal/capabilities.md` §6.2。所有 MCP 工具一律
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
    /// session 启动期叠加。详见 `docs/internal/capabilities.md` §3 / §5。
    pub capabilities: CapabilitiesConfig,
    pub providers: ProviderConfigs,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfig,
    pub tracing: TracingConfig,
    pub mcp: McpConfig,
    pub http: HttpClientConfig,
    /// 解析后的 hook 配置。详见 `docs/internal/hooks.md`。
    pub hooks: HooksConfig,
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// hook 系统的有效配置：按 step `event_name` 分桶，组内按声明顺序执行 pipeline。
///
/// 桶的键是挂载点的 `event_name`（snake_case，如 `before_turn_end`）——与
/// `defect_agent::hooks::step::ALL_EVENT_NAMES` 同一套名字。用 map 而非固定字段：新增挂载点时
/// 配置层零改动。详见 `docs/internal/hooks.md`。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HooksConfig {
    /// `event_name` → 该事件下声明的条目（按声明顺序）。
    pub buckets: std::collections::BTreeMap<String, Vec<HookEntry>>,
}

impl HooksConfig {
    /// 该配置上是否声明过任何 hook。`false` 时 CLI 装配可直接走 noop 引擎。
    pub fn is_empty(&self) -> bool {
        self.buckets.values().all(Vec::is_empty)
    }

    /// 取某事件下的条目（无则空切片）。
    pub fn get(&self, event_name: &str) -> &[HookEntry] {
        self.buckets.get(event_name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// 在某事件下追加一条。
    pub fn push(&mut self, event_name: impl Into<String>, entry: HookEntry) {
        self.buckets.entry(event_name.into()).or_default().push(entry);
    }
}

/// 单条 hook 配置：matcher + handler + 来源层。
#[derive(Debug, Clone, PartialEq)]
pub struct HookEntry {
    pub matcher: HookMatcher,
    pub handler: HookHandlerSpec,
    /// 该 hook 的来源层。Phase G 的 trust gating 用它判断是否需要显式信任。
    pub source: ConfigSource,
}

/// 事件 matcher。空字段 = 匹配该事件下所有触发；详见
/// `docs/internal/hooks.md` §5.3。
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct HookMatcher {
    /// 工具名精确匹配（仅 `*ToolUse*` 事件）。
    pub tool: Option<String>,
    /// 工具名 glob 匹配（仅 `*ToolUse*` 事件）。
    pub tool_glob: Option<String>,
    /// `SafetyClass` 过滤（仅 `PreToolUse`）；任一匹配即命中。空 vec = 不过滤。
    pub safety: Vec<defect_agent::tool::SafetyClass>,
}

/// Handler 规格——v0 三种形态。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookHandlerSpec {
    /// 进程内 Rust handler，按名字引用 `crate::hooks::builtin::registry()`。
    Builtin { name: String },
    /// 外部命令。详见 `docs/internal/hooks.md` §4.2。
    Command(HookCommandSpec),
    /// 调用 LLM。详见 `docs/internal/hooks.md` §4.3。
    Prompt(HookPromptSpec),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookCommandSpec {
    /// 直接 spawn argv，不经任何 shell。
    Argv {
        argv: Vec<String>,
        /// Windows 平台覆盖；`None` 时 fall back 到 `argv`。
        argv_windows: Option<Vec<String>>,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
    /// 显式 shell。`shell` 字段必须存在，引擎不再"自动选 sh"。
    Shell {
        shell: HookShellKind,
        command: String,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookShellKind {
    Sh,
    Bash,
    Pwsh,
    Cmd,
    /// `program` + 透传 `args`（不含 command 本身）。
    Custom {
        program: String,
        args: Vec<String>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HookPromptSpec {
    /// `None` = 用 session 默认 model。
    pub model: Option<String>,
    pub system: String,
    pub render: HookPromptRender,
    pub timeout_sec: Option<u64>,
}

impl HookPromptSpec {
    /// 跨 crate 构造入口——`#[non_exhaustive]` 后 struct literal 不能用。
    #[must_use]
    pub fn new(
        model: Option<String>,
        system: String,
        render: HookPromptRender,
        timeout_sec: Option<u64>,
    ) -> Self {
        Self {
            model,
            system,
            render,
            timeout_sec,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookPromptRender {
    /// 直接喂 `HookEvent` 的 JSON 序列化结果。
    Json,
    /// 用 handlebars 模板从 event 字段取值。
    Template { template: String },
}

/// 全局 capability 配置入口。
///
/// 与 [`SessionCapabilitiesConfig`] 形态等价；在 `EffectiveConfig` 上保
/// 留独立类型是为了未来追加非 session 级 capability 时不动 agent crate。
/// 当前 P1 仅有 `web_search`（hosted-only），本地 grep/glob 工具不属于
/// capability 层，由 `[tools.search]` 单独管理。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapabilitiesConfig {
    pub web_search: WebSearchCapabilityConfig,
}

impl CapabilitiesConfig {
    /// 用单条 [`WebSearchCapabilityConfig`] 构造。跨 crate 调用方需要这个
    /// 入口，因为本结构体 `#[non_exhaustive]` 后不能直接 struct literal。
    #[must_use]
    pub const fn with_web_search(web_search: WebSearchCapabilityConfig) -> Self {
        Self { web_search }
    }

    /// 转成 agent 侧的 [`SessionCapabilitiesConfig`]，供
    /// `DefaultAgentCoreBuilder::capabilities` 直接消费。
    #[must_use]
    pub fn to_session_capabilities(self) -> SessionCapabilitiesConfig {
        SessionCapabilitiesConfig::with_web_search(self.web_search)
    }
}

/// 单个 provider 下对全局 capability 的覆写。
///
/// `None` 字段意味着「跟随全局」。详见 §5 / §13.2。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderCapabilityOverrides {
    pub web_search: Option<WebSearchCapabilityConfig>,
}

impl ProviderCapabilityOverrides {
    /// 用单条 web_search 覆写构造。`None` = 跟随全局。
    #[must_use]
    pub const fn with_web_search(web_search: Option<WebSearchCapabilityConfig>) -> Self {
        Self { web_search }
    }

    /// 把全局 [`CapabilitiesConfig`] 与本 provider 的覆写合并。
    /// 未覆写字段沿用全局值。
    #[must_use]
    pub fn merge_into(&self, base: CapabilitiesConfig) -> CapabilitiesConfig {
        CapabilitiesConfig::with_web_search(self.web_search.unwrap_or(base.web_search))
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
    pub anthropic: ProviderConfigFile,
    pub openai: ProviderConfigFile,
    pub deepseek: ProviderConfigFile,
    pub litellm: ProviderConfigFile,
    pub custom: BTreeMap<String, ProviderConfigFile>,
}

impl ProviderConfigs {
    #[must_use]
    pub fn get(&self, provider: &ProviderKind) -> Option<&ProviderConfigFile> {
        match provider {
            ProviderKind::Echo => None,
            ProviderKind::Anthropic => Some(&self.anthropic),
            ProviderKind::Openai => Some(&self.openai),
            ProviderKind::Deepseek => Some(&self.deepseek),
            ProviderKind::Litellm => Some(&self.litellm),
            ProviderKind::Custom(name) => self.custom.get(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolsConfig {
    pub bash: BashToolConfig,
    pub fs: FsToolConfig,
    /// `[tools.fetch]` 段。预留出来——P1 仅 schema 落地，工具实现在后续 PR。
    pub fetch: FetchToolConfig,
    /// `[tools.search]` 段。本地 `search` tool（grep/glob）的参数。
    /// 与 `[capabilities.web_search]` 相互独立，由 `enabled` 单独决定是否注册。
    /// 详见 `docs/internal/tools-search.md`。
    pub search: SearchToolConfig,
}

/// 本地 `fetch` 工具的配置。详见 `docs/internal/tools-fetch.md` §7。
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

/// 本地 `search` 工具的配置。详见 `docs/internal/tools-search.md` §7。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchToolConfig {
    pub enabled: bool,
    pub default_head_limit: u32,
    pub max_head_limit: u32,
    pub max_file_size_bytes: u64,
    pub max_result_bytes: u64,
    pub max_walk_files: u64,
    pub respect_gitignore_default: bool,
}

impl Default for SearchToolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_head_limit: 100,
            max_head_limit: 1000,
            max_file_size_bytes: 16 * 1024 * 1024,
            max_result_bytes: 256 * 1024,
            max_walk_files: 100_000,
            respect_gitignore_default: true,
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

pub type AnthropicConfigFile = ProviderConfigFile;
pub type OpenAiConfigFile = ProviderConfigFile;
pub type DeepSeekConfigFile = ProviderConfigFile;
pub type LiteLlmConfigFile = ProviderConfigFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderProtocol {
    AnthropicMessages,
    OpenaiChat,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProviderConfigFile {
    pub protocol: Option<ProviderProtocol>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<String>>,
    pub display_name: Option<String>,
    pub api_key_env: Option<String>,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub aws: Option<ProviderAwsConfigFile>,
    pub headers: BTreeMap<String, String>,
    pub capabilities: ProviderCapabilityOverrides,
    /// `reasoning_effort` wire 参数。`None` = 不发送，沿用 provider 默认。
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderAwsConfigFile {
    pub profile: Option<String>,
    pub region: Option<String>,
}

/// OpenAI 兼容协议的 `reasoning_effort` 取值。
///
/// 与 OpenAI 官方 wire 枚举 1:1 对齐：`xhigh` 仅 `gpt-5.1-codex-max` 之后
/// 支持，`none` 仅 `gpt-5.1` 之后支持；配置层不区分模型，原样下发由上游
/// 校验。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
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
    pub langfuse: Option<LangfuseConfig>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OtlpTracingConfig {
    pub endpoint: Option<String>,
}

/// Langfuse 上报配置。详见 `docs/internal/observability-langfuse.md` §6。
///
/// 默认关闭；`enabled = true` 但缺 key 时由装配层告警并禁用（不静默成功）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LangfuseConfig {
    pub enabled: bool,
    /// Langfuse host，如 `https://cloud.langfuse.com`。`None` 用装配层默认。
    pub host: Option<String>,
    pub public_key: Option<String>,
    pub secret_key: Option<String>,
    /// 周期冲刷间隔（毫秒）。`None` 用装配层默认。
    pub flush_interval_ms: Option<u64>,
    /// 单批最大事件数。`None` 用装配层默认。
    pub max_batch: Option<usize>,
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
#[serde(deny_unknown_fields)]
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
    /// `[hooks]` 段不走 `ConfigToml::try_into`（数组语义是 append+dedupe，详见
    /// `crate::hooks`）。这里用 `toml::Value` 吸收它，避免 `deny_unknown_fields`
    /// 把 `[[hooks.*]]` 误判为未知段；hooks 自己的解析器做 schema 校验。
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) hooks: Option<TomlValue>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilitiesSection {
    pub(crate) web_search: Option<WebSearchCapabilitySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebSearchCapabilitySection {
    pub(crate) mode: Option<defect_agent::session::WebSearchCapabilityMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderCapabilitiesSection {
    pub(crate) web_search: Option<WebSearchCapabilitySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DefaultSection {
    pub(crate) provider: Option<ProviderKind>,
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BasePromptSection {
    pub(crate) file: Option<String>,
    pub(crate) text: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TurnSection {
    pub(crate) system_prompt: Option<String>,
    pub(crate) request_limit: Option<u32>,
    pub(crate) compact_threshold_tokens: Option<u64>,
    pub(crate) compact_ratio: Option<f64>,
    pub(crate) max_llm_retries: Option<u32>,
    pub(crate) max_concurrent_tools: Option<usize>,
    /// `before turn-end` hook 强制续命的硬上限。`None` ⇒ 用 agent 侧默认（3）。
    pub(crate) max_hook_continues: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptSection {
    pub(crate) file: Option<String>,
    pub(crate) text: Option<String>,
    pub(crate) providers: Option<BTreeMap<String, PromptOverlaySection>>,
    pub(crate) models: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptOverlaySection {
    pub(crate) text: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ProvidersSection {
    pub(crate) anthropic: Option<AnthropicProviderSection>,
    pub(crate) openai: Option<OpenAiProviderSection>,
    pub(crate) deepseek: Option<DeepSeekProviderSection>,
    pub(crate) litellm: Option<LiteLlmProviderSection>,
    #[serde(flatten)]
    pub(crate) custom: BTreeMap<String, ProviderSection>,
}

pub(crate) type AnthropicProviderSection = ProviderSection;
pub(crate) type OpenAiProviderSection = ProviderSection;
pub(crate) type DeepSeekProviderSection = ProviderSection;
pub(crate) type LiteLlmProviderSection = ProviderSection;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderSection {
    pub(crate) protocol: Option<ProviderProtocol>,
    pub(crate) base_url: Option<String>,
    pub(crate) default_model: Option<String>,
    pub(crate) models: Option<Vec<String>>,
    pub(crate) display_name: Option<String>,
    pub(crate) api_key_env: Option<String>,
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) aws: Option<ProviderAwsConfigFile>,
    pub(crate) headers: Option<BTreeMap<String, String>>,
    pub(crate) capabilities: Option<ProviderCapabilitiesSection>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolsSection {
    pub(crate) bash: Option<BashToolSection>,
    pub(crate) fs: Option<FsToolSection>,
    pub(crate) fetch: Option<FetchToolSection>,
    /// `[tools.search]`：本地 `search` tool（grep/glob）参数。是否注册仅
    /// 取决于 `enabled`，与 `[capabilities.web_search]` 完全独立。
    pub(crate) search: Option<SearchToolSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchToolSection {
    pub(crate) enabled: Option<bool>,
    pub(crate) default_head_limit: Option<u32>,
    pub(crate) max_head_limit: Option<u32>,
    pub(crate) max_file_size_bytes: Option<u64>,
    pub(crate) max_result_bytes: Option<u64>,
    pub(crate) max_walk_files: Option<u64>,
    pub(crate) respect_gitignore_default: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub(crate) struct BashToolSection {
    pub(crate) default_timeout_ms: Option<u64>,
    pub(crate) max_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FsToolSection {
    pub(crate) read_default_limit: Option<u32>,
    pub(crate) read_max_limit: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SandboxSection {
    pub(crate) mode: Option<SandboxMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TracingSection {
    pub(crate) filter: Option<String>,
    pub(crate) otlp: Option<OtlpTracingSection>,
    pub(crate) langfuse: Option<LangfuseSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OtlpTracingSection {
    pub(crate) endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) struct LangfuseSection {
    pub(crate) enabled: Option<bool>,
    pub(crate) host: Option<String>,
    pub(crate) public_key: Option<String>,
    pub(crate) secret_key: Option<String>,
    pub(crate) flush_interval_ms: Option<u64>,
    pub(crate) max_batch: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpSection {
    pub(crate) enabled_servers: Option<Vec<String>>,
    pub(crate) servers: Option<BTreeMap<String, McpServerSection>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HttpSection {
    pub(crate) total_timeout_ms: Option<u64>,
    pub(crate) transport_retries: Option<u8>,
    pub(crate) initial_backoff_ms: Option<u64>,
    pub(crate) user_agent: Option<String>,
    pub(crate) proxy: Option<HttpProxySection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HttpProxySection {
    pub(crate) mode: Option<HttpProxyMode>,
    pub(crate) http_proxy: Option<String>,
    pub(crate) https_proxy: Option<String>,
    pub(crate) no_proxy: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpServerSection {
    pub(crate) transport: Option<McpTransportKind>,
    pub(crate) command: Option<String>,
    pub(crate) args: Option<Vec<String>>,
    pub(crate) env: Option<BTreeMap<String, String>>,
    pub(crate) url: Option<String>,
    pub(crate) headers: Option<BTreeMap<String, String>>,
}
