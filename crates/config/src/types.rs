use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use defect_agent::error::BoxError;
use defect_agent::session::{
    BackgroundProgressConfig, SessionCapabilitiesConfig, TurnConfig, WebSearchCapabilityConfig,
};
use serde::{Deserialize, Serialize};
use toml::Value as TomlValue;

pub(crate) const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";
pub(crate) const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
pub(crate) const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";
pub(crate) const DEFAULT_ECHO_MODEL: &str = "echo";
pub(crate) const DEFAULT_BASH_TIMEOUT_MS: u64 = 30_000;
pub(crate) const DEFAULT_BASH_MAX_TIMEOUT_MS: u64 = 600_000;
pub(crate) const DEFAULT_BASH_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
pub(crate) const DEFAULT_FS_READ_LIMIT: u32 = 2_000;
pub(crate) const DEFAULT_FS_READ_MAX_LIMIT: u32 = 5_000;

pub(crate) const USER_CONFIG_RELATIVE: &str = "defect/config.toml";
pub(crate) const PROJECT_CONFIG_RELATIVE: &str = ".defect/config.toml";
pub(crate) const PROJECT_LOCAL_CONFIG_RELATIVE: &str = ".defect/config.local.toml";

const PROVIDER_DEFECT: &str = "defect";
const PROVIDER_ANTHROPIC: &str = "anthropic";
const PROVIDER_OPENAI: &str = "openai";
const PROVIDER_DEEPSEEK: &str = "deepseek";
const PROVIDER_LITELLM: &str = "litellm";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ProviderKind {
    /// Built-in placeholder provider: echoes the user's most recent message back as-is
    /// (model id `echo`).
    /// Requires no external credentials; serves as the fallback default for
    /// `default.provider`.
    #[default]
    Defect,
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
            Self::Defect => PROVIDER_DEFECT,
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
            PROVIDER_DEFECT => Self::Defect,
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
            PROVIDER_DEFECT => Self::Defect,
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
    /// A configuration section exists in the file but is inactive under the current mode.
    ///
    /// See capabilities semantic mapping.
    /// Typical scenario: a configuration section is written but the corresponding
    /// capability is disabled (e.g., the legacy `capabilities.search` section has been
    /// deprecated in favor of `[capabilities.web_search]`, and the old key no longer
    /// takes effect).
    InactiveSection {
        path: PathBuf,
        section: String,
        reason: String,
    },
    /// Conflicting MCP tools are renamed to `mcp__<server>__<name>` during session
    /// startup.
    ///
    /// See capabilities for MCP tool classification. All MCP tools are renamed
    /// (regardless of capability mode or tool enabled) to prevent MCP bypass name
    /// squatting.
    McpToolRenamed {
        server: String,
        original: String,
        renamed: String,
    },
    /// A `.mcp.json` server was shadowed by a same-named TOML `[mcp.servers.<name>]`
    /// entry. The TOML entry (more explicit, layered source) wins; the `.mcp.json`
    /// definition is ignored.
    McpJsonOverridden { path: PathBuf, server: String },
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
    pub sandbox: Option<SandboxMode>,
    pub config_overrides: Vec<(String, TomlValue)>,
}

#[derive(Debug, Clone, Default)]
pub struct LoadConfigOptions {
    pub cwd: PathBuf,
    pub cli: CliOverrides,
    pub xdg_config_home: Option<PathBuf>,
    pub home_dir: Option<PathBuf>,
    /// `--local` sandbox mode: ignores global/user-level config and user-level
    /// agents/skills directories,
    /// only uses the project root `.defect/`. When `true`, user layers are always absent
    /// (see
    /// `resolve_user_config_path` / `resolve_user_agents_dir` /
    /// `resolve_user_skills_dir`).
    pub local: bool,
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
    /// Global capability source selection. Overridden by `providers.<p>.capabilities`
    /// during session startup.
    pub capabilities: CapabilitiesConfig,
    pub providers: ProviderConfigs,
    pub tools: ToolsConfig,
    pub sandbox: SandboxConfig,
    pub tracing: TracingConfig,
    pub mcp: McpConfig,
    pub http: HttpClientConfig,
    /// Resolved hook configuration.
    pub hooks: HooksConfig,
}

// Hooks

/// Valid configuration for the hook system: pipelines are grouped by step `event_name`
/// and executed in declaration order within each group.
///
/// Bucket keys are the mount point's `event_name` (snake_case, e.g. `before_turn_end`) —
/// the same set of names as `defect_agent::hooks::step::ALL_EVENT_NAMES`. A map is used
/// instead of fixed fields so that adding a new mount point requires no changes at the
/// config layer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HooksConfig {
    /// `event_name` → entries declared under that event, in declaration order.
    pub buckets: std::collections::BTreeMap<String, Vec<HookEntry>>,
}

impl HooksConfig {
    /// Whether any hooks have been declared on this config. When `false`, CLI assembly
    /// can use the noop engine directly.
    pub fn is_empty(&self) -> bool {
        self.buckets.values().all(Vec::is_empty)
    }

    /// Returns the entries for a given event, or an empty slice if none exist.
    pub fn get(&self, event_name: &str) -> &[HookEntry] {
        self.buckets
            .get(event_name)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Appends a hook entry under the given event name.
    pub fn push(&mut self, event_name: impl Into<String>, entry: HookEntry) {
        self.buckets
            .entry(event_name.into())
            .or_default()
            .push(entry);
    }
}

/// A single hook configuration: matcher + handler + source layer.
#[derive(Debug, Clone, PartialEq)]
pub struct HookEntry {
    /// Optional human-readable name, used only for tracing/observability to identify this
    /// hook.
    /// `None` ⇒ falls back to an anonymous label at assembly time (see `defect-cli`'s
    /// hook assembly).
    /// Does not participate in deduplication or disable matching (that only uses matcher
    /// + handler) — purely for display.
    pub name: Option<String>,
    pub matcher: HookMatcher,
    pub handler: HookHandlerSpec,
    /// The source layer of this hook. Phase G's trust gating uses it to decide whether
    /// explicit trust is required.
    pub source: ConfigSource,
}

/// Event matcher. Empty fields match all triggers under that event; see the hooks trust
/// model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct HookMatcher {
    /// Exact tool name match (only `*ToolUse*` events).
    pub tool: Option<String>,
    /// Tool name glob match (only `*ToolUse*` events).
    pub tool_glob: Option<String>,
    /// `SafetyClass` filter (only for `PreToolUse`); any match triggers. Empty vec = no
    /// filtering.
    pub safety: Vec<defect_agent::tool::SafetyClass>,
}

/// Handler spec — three variants.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookHandlerSpec {
    /// In-process Rust handler, referenced by name via
    /// `crate::hooks::builtin::registry()`.
    Builtin { name: String },
    /// External command (see hooks command handler).
    Command(HookCommandSpec),
    /// Calls an LLM (see hooks prompt handler).
    Prompt(HookPromptSpec),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookCommandSpec {
    /// Spawn `argv` directly, without any shell.
    Argv {
        argv: Vec<String>,
        /// Windows override; `None` falls back to `argv`.
        argv_windows: Option<Vec<String>>,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
    /// Explicit shell. The `shell` field is required; the engine no longer auto-selects
    /// `sh`.
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
    /// `program` plus passthrough `args` (excluding the command itself).
    Custom {
        program: String,
        args: Vec<String>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HookPromptSpec {
    /// `None` = use the session's default model.
    pub model: Option<String>,
    pub system: String,
    pub render: HookPromptRender,
    pub timeout_sec: Option<u64>,
}

impl HookPromptSpec {
    /// Cross-crate construction entry point — struct literals are unavailable after
    /// `#[non_exhaustive]`.
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
    /// Feed the JSON-serialized `HookEvent` directly.
    Json,
    /// Renders a handlebars template using fields from the event.
    Template { template: String },
}

/// Top-level capability configuration entry point.
///
/// Structurally equivalent to [`SessionCapabilitiesConfig`]; a separate type on
/// `EffectiveConfig` is kept so that future non-session-level capabilities can be
/// added without touching the agent crate. Currently P1 only has `web_search`
/// (hosted-only); local grep/glob tools are not part of the capability layer and
/// are managed separately by `[tools.search]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapabilitiesConfig {
    pub web_search: WebSearchCapabilityConfig,
}

impl CapabilitiesConfig {
    /// Construct with a single [`WebSearchCapabilityConfig`]. Cross-crate callers need
    /// this entry point because the struct is `#[non_exhaustive]` and cannot be built
    /// with a struct literal directly.
    #[must_use]
    pub const fn with_web_search(web_search: WebSearchCapabilityConfig) -> Self {
        Self { web_search }
    }

    /// Converts to the agent-side [`SessionCapabilitiesConfig`], for direct consumption
    /// by
    /// `DefaultAgentCoreBuilder::capabilities`.
    #[must_use]
    pub fn to_session_capabilities(self) -> SessionCapabilitiesConfig {
        SessionCapabilitiesConfig::with_web_search(self.web_search)
    }
}

/// Overrides for global capabilities under a single provider.
///
/// A `None` field means "follow the global setting".
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderCapabilityOverrides {
    pub web_search: Option<WebSearchCapabilityConfig>,
}

impl ProviderCapabilityOverrides {
    /// Construct with a single `web_search` override. `None` means follow the global
    /// setting.
    #[must_use]
    pub const fn with_web_search(web_search: Option<WebSearchCapabilityConfig>) -> Self {
        Self { web_search }
    }

    /// Merges the global [`CapabilitiesConfig`] with this provider's overrides.
    /// Unset fields fall back to the global values.
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
            ProviderKind::Defect => None,
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
    /// The `[tools.fetch]` section. Reserved — P1 only lands the schema; the tool
    /// implementation will follow in a later PR.
    pub fetch: FetchToolConfig,
    /// The `[tools.search]` section. Parameters for the local `search` tool (grep/glob).
    /// Independent of `[capabilities.web_search]`; registration is controlled separately
    /// by `enabled`.
    /// See search tool config.
    pub search: SearchToolConfig,
    /// `[tools.background]` section. Configuration for the background subagent progress
    /// view (progress ring capacity / per-block text character limit) — the "last N
    /// blocks" that the main agent sees via `inspect_background_task`. The source of
    /// truth is on the agent side ([`BackgroundProgressConfig`]); this is a direct reuse.
    pub background: BackgroundProgressConfig,
}

/// Configuration for the local fetch tool.
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
    /// Maximum bytes of merged stdout/stderr captured per command; output beyond this is
    /// dropped and reported as truncated. Applies to the local shell backend.
    pub output_max_bytes: usize,
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: DEFAULT_BASH_TIMEOUT_MS,
            max_timeout_ms: DEFAULT_BASH_MAX_TIMEOUT_MS,
            output_max_bytes: DEFAULT_BASH_OUTPUT_MAX_BYTES,
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

/// Configuration for the local search tool.
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

/// A model candidate configuration declaration.
///
/// TOML supports two forms (accepted via [`Deserialize`] with `untagged`):
/// - Plain string `"gpt-5.5"`: only the id is given; the UI display name falls back to
///   the id.
/// - Table `{ id = "...", name = "Opus 4.8" }`: pairs a long id with a short display
///   name.
///
/// `name` is mapped to [`defect_agent::llm::ModelInfo::display_name`]; the ACP uses it as
/// the label for model selector options. When `None`, the wire layer falls back to the
/// id.
///
/// `context_window` / `max_output_tokens` let the user declare model metadata that the
/// provider cannot discover at runtime — most importantly for **Bedrock**, whose SDK does
/// not return model limits, so without this the compaction watermarks have no window to key
/// off and the context can grow unbounded. Both are optional; when absent the provider's
/// own value (if any) or the compaction fallback applies.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ModelEntry {
    /// A plain ID (legacy format).
    Id(String),
    /// A detailed entry with a display name and optional limits.
    Detailed {
        id: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        context_window: Option<u64>,
        #[serde(default)]
        max_output_tokens: Option<u64>,
    },
}

impl ModelEntry {
    /// The model ID (present in both variants).
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Detailed { id, .. } => id,
        }
    }

    /// Optional display name (only available in table form).
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Id(_) => None,
            Self::Detailed { name, .. } => name.as_deref(),
        }
    }

    /// Optional context-window size (tokens), only from the table form.
    #[must_use]
    pub fn context_window(&self) -> Option<u64> {
        match self {
            Self::Id(_) => None,
            Self::Detailed { context_window, .. } => *context_window,
        }
    }

    /// Optional max output tokens, only from the table form.
    #[must_use]
    pub fn max_output_tokens(&self) -> Option<u64> {
        match self {
            Self::Id(_) => None,
            Self::Detailed {
                max_output_tokens, ..
            } => *max_output_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProviderConfigFile {
    pub protocol: Option<ProviderProtocol>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub models: Option<Vec<ModelEntry>>,
    pub display_name: Option<String>,
    pub api_key_env: Option<String>,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub aws: Option<ProviderAwsConfigFile>,
    pub headers: BTreeMap<String, String>,
    pub capabilities: ProviderCapabilityOverrides,
    /// `reasoning_effort` wire parameter. `None` = do not send, use provider default.
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderAwsConfigFile {
    pub profile: Option<String>,
    pub region: Option<String>,
    /// `anthropic_beta` flags injected into the Bedrock request body (Anthropic Messages
    /// `anthropic_beta` array). Default (absent) sends nothing. Some newer models (e.g. Opus
    /// 4.8) reject the default data retention mode and require `["no-data-retention-v1"]`;
    /// set it explicitly. The flag is shared by every model under this provider, and Bedrock
    /// 400s a model that does not support a given flag — so only enable it on a
    /// provider whose models all accept the flag.
    pub anthropic_beta: Option<Vec<String>>,
}

/// Values for the `reasoning_effort` parameter in the OpenAI-compatible protocol.
///
/// 1:1 mapping with the official OpenAI wire enum: `xhigh` is only supported after
/// `gpt-5.1-codex-max`, and `none` is only supported after `gpt-5.1`. The configuration
/// layer does not distinguish between models; values are passed through as-is and
/// validated upstream.
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

/// Typed configuration for the HTTP client stack.
///
/// Only captures user intent (`None` always means "use the HTTP stack layer default");
/// the CLI entry point converts it to `defect_http::HttpStackConfig` when assembling the
/// provider.
///
/// Not sharing a type directly with `defect_http::HttpStackConfig` preserves a one-way
/// crate dependency: `defect-config` does not depend on `defect-http`, preventing
/// downstream consumers like fetch tool from creating a reverse dependency.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HttpClientConfig {
    /// Total timeout for a single request; `None` means use the HTTP stack default
    /// (600s).
    pub total_timeout_ms: Option<u64>,
    /// Maximum number of transport error retries (excluding the first attempt); `None`
    /// means default 2, `Some(0)` disables retries.
    pub transport_retries: Option<u8>,
    /// Initial backoff for retries; `None` = default 200ms.
    pub initial_backoff_ms: Option<u64>,
    /// Override the `User-Agent` header; `None` uses the compile-time default
    /// `defect-http/{version} ({git_sha})`.
    pub user_agent: Option<String>,
    /// Proxy sub-configuration. `mode` defaults to `FromEnv` (reads `HTTP_PROXY` etc.).
    pub proxy: HttpProxyConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpProxyConfig {
    pub mode: HttpProxyMode,
    /// Explicit proxy; only effective when `mode = Explicit`.
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

/// Langfuse upload configuration.
///
/// Disabled by default; if `enabled = true` but keys are missing, the assembly layer
/// warns and disables it (no silent success).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LangfuseConfig {
    pub enabled: bool,
    /// Langfuse host, e.g. `https://cloud.langfuse.com`. `None` uses the assembly layer
    /// default.
    pub host: Option<String>,
    pub public_key: Option<String>,
    pub secret_key: Option<String>,
    /// Flush interval in milliseconds. `None` uses the assembly layer default.
    pub flush_interval_ms: Option<u64>,
    /// Maximum number of events per batch. `None` uses the assembly layer default.
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
    /// The `[hooks]` section is not processed by `ConfigToml::try_into` (its array
    /// semantics are append+dedupe; see `crate::hooks`). We absorb it here as
    /// `toml::Value` to prevent `deny_unknown_fields` from misidentifying `[[hooks.*]]`
    /// as an unknown section; hooks' own parser performs schema validation.
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

/// How `[turn] request_limit` is interpreted. The numeric `request_limit` supplies `N`;
/// this key selects the strategy. Defaults to `Adaptive` when omitted, so a bare
/// `request_limit = N` keeps its historical self-expanding behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestLimitMode {
    /// Hard cap at `N` LLM calls; never expands.
    Fixed,
    /// Start at `N`; each executed tool raises the cap by one (the default).
    Adaptive,
    /// No cap at all; `N` is ignored.
    Unbounded,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TurnSection {
    pub(crate) system_prompt: Option<String>,
    /// `[turn.sampling]` — per-call generation parameters for the **main agent**
    /// (`max_tokens` / `temperature` / `top_p` / `top_k`). Mirrors the subagent profile's
    /// `[sampling]` section. Omitted fields fall back to the provider default; in
    /// particular, a missing `max_tokens` lets the Anthropic protocol layer apply its own
    /// fallback. `reasoning_effort` is *not* here — it is a provider-level concern wired in
    /// `cli/providers.rs` and switchable per-session via ACP.
    pub(crate) sampling: Option<SamplingSection>,
    pub(crate) request_limit: Option<u32>,
    /// Strategy for `request_limit`. `None` ⇒ `Adaptive` (back-compatible).
    pub(crate) request_limit_mode: Option<RequestLimitMode>,
    pub(crate) compact_threshold_tokens: Option<u64>,
    pub(crate) compact_ratio: Option<f64>,
    /// Enables background full compaction (asynchronous summarization when the soft
    /// watermark is exceeded, without blocking the current turn).
    pub(crate) background_compact_enabled: Option<bool>,
    /// Background compaction soft watermark as a fraction of `context_window` (default
    /// 0.7).
    pub(crate) compact_soft_ratio: Option<f64>,
    /// Enables micro-compaction: cleans oversized `tool_result` entries from older turns
    /// without invoking the LLM.
    pub(crate) microcompact_enabled: Option<bool>,
    /// Micro‑compact watermark as a fraction of `context_window` (default 0.6).
    pub(crate) microcompact_ratio: Option<f64>,
    pub(crate) max_llm_retries: Option<u32>,
    pub(crate) max_concurrent_tools: Option<usize>,
    /// Hard upper limit on forced continues from the `before turn-end` hook. `None` ⇒ use
    /// the agent-side default (3).
    pub(crate) max_hook_continues: Option<u32>,
    /// Maximum subagent vertical recursion depth. `None` ⇒ use the agent-side default
    /// (4). `0` ⇒ disallow dispatching any subagent (the top-level tool set does not
    /// contain `spawn_agent`).
    pub(crate) subagent_max_depth: Option<u32>,
}

/// `[turn.sampling]` — main-agent generation parameters. Each field is independently
/// optional and overrides the corresponding [`SamplingParams`](defect_agent::llm::SamplingParams)
/// default only when present. Mirrors the subagent profile's `[sampling]` section.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SamplingSection {
    /// Maximum tokens the model may generate in a single response. Omitted ⇒ the protocol
    /// layer's own fallback applies (e.g. Anthropic's `DEFAULT_MAX_TOKENS`).
    pub(crate) max_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
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
    pub(crate) models: Option<Vec<ModelEntry>>,
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
    /// `[tools.search]`: parameters for the local `search` tool (grep/glob). Registration
    /// depends solely on `enabled` and is completely independent of
    /// `[capabilities.web_search]`.
    pub(crate) search: Option<SearchToolSection>,
    /// `[tools.background]`: background subagent progress view (ring capacity / text
    /// limit).
    pub(crate) background: Option<BackgroundToolSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackgroundToolSection {
    /// Default number of recent message blocks returned when `inspect` is called without
    /// `recent_blocks`. Defaults to 10.
    pub(crate) default_recent_blocks: Option<usize>,
    /// Character limit for free-form body text in a single block (assistant/thought/tool
    /// result). Default 0 = keep only summary/metadata.
    pub(crate) block_text_limit: Option<usize>,
    /// How many finished background-task entries to retain in the task table before the
    /// oldest are evicted. Default 64. Bounds memory for long sessions with many tasks.
    pub(crate) finished_tasks_cap: Option<usize>,
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
    pub(crate) output_max_bytes: Option<usize>,
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
