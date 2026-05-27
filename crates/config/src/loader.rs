use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::llm::SamplingParams;
use defect_agent::session::{BasePromptConfig, PromptConfig, TurnConfig, TurnRequestLimit};
use toml::Value as TomlValue;

use crate::mcp::{is_known_mcp_key, is_known_mcp_prefix, resolve_mcp_config};
use crate::overrides::{
    build_cli_layer, merge_toml_values, remove_toml_path, remove_toml_table_key,
};
use crate::types::{
    AnthropicConfigFile, BasePromptConfigFile, BashToolConfig, CapabilitiesConfig, CliConfig,
    ConfigError, ConfigLayerEntry, ConfigLayerStack, ConfigSource, ConfigToml, ConfigWarning,
    DEFAULT_ANTHROPIC_MODEL, DEFAULT_BASH_MAX_TIMEOUT_MS, DEFAULT_BASH_TIMEOUT_MS,
    DEFAULT_DEEPSEEK_MODEL, DEFAULT_ECHO_MODEL, DEFAULT_FS_READ_LIMIT, DEFAULT_FS_READ_MAX_LIMIT,
    DEFAULT_OPENAI_MODEL, DeepSeekConfigFile, EffectiveConfig, FetchToolConfig, FsToolConfig,
    HttpClientConfig, HttpProxyConfig, HttpProxySettings, LoadConfigOptions, LoadedConfig,
    OpenAiConfigFile, OtlpTracingConfig, PROJECT_CONFIG_RELATIVE, PROJECT_LOCAL_CONFIG_RELATIVE,
    PromptConfigFile, ProviderCapabilityOverrides, ProviderConfigs, ProviderKind, SandboxConfig,
    SandboxMode, ToolsConfig, TracingConfig, USER_CONFIG_RELATIVE,
};
use defect_agent::session::SearchCapabilityConfig;

/// 加载并合并 `defect` 的有效配置。
///
/// precedence 为：`default < user < project < project-local < CLI`。
///
/// # Errors
///
/// 当用户配置路径无法解析、任一配置文件读盘失败、TOML 解析失败，或合并后的
/// 配置无法反序列化为强类型结构时返回 [`ConfigError`]。
pub fn load_config(opts: LoadConfigOptions) -> Result<LoadedConfig, ConfigError> {
    let cwd = canonicalize_or_original(&opts.cwd);
    let user_path = resolve_user_config_path(&opts)?;
    let repo_root = find_repo_root(&cwd);
    let project_path = repo_root
        .as_ref()
        .map(|root| root.join(PROJECT_CONFIG_RELATIVE));
    let project_local_path = repo_root
        .as_ref()
        .map(|root| root.join(PROJECT_LOCAL_CONFIG_RELATIVE));

    let mut layers = Vec::new();
    let mut warnings = Vec::new();

    let defaults = TomlValue::Table(Default::default());
    layers.push(ConfigLayerEntry {
        source: ConfigSource::Defaults,
        path: None,
        raw_toml: None,
        value: defaults.clone(),
    });

    let mut merged = defaults;
    let mut base_prompt: Option<BasePromptConfigFile> = None;

    if let Some((user_layer, layer_warnings)) = load_optional_layer(ConfigSource::User, user_path)?
    {
        warnings.extend(layer_warnings);
        if let Some(candidate) = extract_base_prompt(&user_layer.value, user_layer.path.as_ref()) {
            base_prompt = Some(candidate);
        }
        merge_toml_values(&mut merged, &user_layer.value);
        layers.push(user_layer);
    }

    if let Some((project_layer, layer_warnings)) =
        load_optional_layer_opt(ConfigSource::Project, project_path)?
    {
        warnings.extend(layer_warnings);
        if let Some(candidate) =
            extract_base_prompt(&project_layer.value, project_layer.path.as_ref())
        {
            base_prompt = Some(candidate);
        }
        let (value, layer_warnings) =
            sanitize_shared_project_layer(project_layer.path.as_ref(), &project_layer.value);
        warnings.extend(layer_warnings);
        merge_toml_values(&mut merged, &value);
        layers.push(ConfigLayerEntry {
            value,
            ..project_layer
        });
    }

    if let Some((project_local_layer, layer_warnings)) =
        load_optional_layer_opt(ConfigSource::ProjectLocal, project_local_path)?
    {
        warnings.extend(layer_warnings);
        if let Some(candidate) = extract_base_prompt(
            &project_local_layer.value,
            project_local_layer.path.as_ref(),
        ) {
            base_prompt = Some(candidate);
        }
        merge_toml_values(&mut merged, &project_local_layer.value);
        layers.push(project_local_layer);
    }

    if let Some(cli_layer) = build_cli_layer(&opts.cli)? {
        if let Some(candidate) = extract_base_prompt(&cli_layer.value, cli_layer.path.as_ref()) {
            base_prompt = Some(candidate);
        }
        merge_toml_values(&mut merged, &cli_layer.value);
        layers.push(cli_layer);
    }

    let parsed: ConfigToml = merged
        .clone()
        .try_into()
        .map_err(|err| ConfigError::Invalid {
            path: PathBuf::from("<merged>"),
            message: err.to_string(),
        })?;
    let effective = build_effective_config(
        Path::new("<merged>"),
        parsed,
        base_prompt.unwrap_or_default(),
    )?;
    collect_inactive_section_warnings(&merged, &effective.capabilities, &mut warnings);

    Ok(LoadedConfig {
        layers: ConfigLayerStack { layers },
        effective,
        warnings,
    })
}

/// 兼容读取 `cwd/.env`，仅为缺失的环境变量补值。
///
/// # Errors
///
/// 当 `.env` 文件存在但读取失败时返回 [`ConfigError::Io`]。
pub fn load_dotenv_compat(cwd: &Path) -> Result<(), ConfigError> {
    let path = cwd.join(".env");
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ConfigError::Io {
                path,
                source: BoxError::new(err),
            });
        }
    };

    let existing = raw_env_keys();
    for (key, value) in dotenv_updates_from_str(&raw, &existing) {
        // SAFETY: CLI 在任何并发任务启动前调用；这里仅做进程启动阶段的 env 补值。
        unsafe {
            env::set_var(key, value);
        }
    }
    Ok(())
}

fn build_effective_config(
    path: &Path,
    config: ConfigToml,
    base_prompt: BasePromptConfigFile,
) -> Result<EffectiveConfig, ConfigError> {
    // `base_prompt` 的最终选择在 `load_config()` 中完成，这里只保留 typed decode
    // 对 schema 的约束，并显式消费字段避免它与 raw-layer 解析脱节。
    let _ = config.base_prompt.file.as_deref();
    let _ = config.base_prompt.text.as_deref();
    let provider = config.default.provider.unwrap_or_default();
    let provider_model = match provider {
        ProviderKind::Echo => Some(DEFAULT_ECHO_MODEL.to_string()),
        ProviderKind::Anthropic => config
            .providers
            .anthropic
            .as_ref()
            .and_then(|cfg| cfg.default_model.clone())
            .or_else(|| Some(DEFAULT_ANTHROPIC_MODEL.to_string())),
        ProviderKind::Openai => config
            .providers
            .openai
            .as_ref()
            .and_then(|cfg| cfg.default_model.clone())
            .or_else(|| Some(DEFAULT_OPENAI_MODEL.to_string())),
        ProviderKind::Deepseek => config
            .providers
            .deepseek
            .as_ref()
            .and_then(|cfg| cfg.default_model.clone())
            .or_else(|| Some(DEFAULT_DEEPSEEK_MODEL.to_string())),
    };
    let allowed_models = match provider {
        ProviderKind::Echo => None,
        ProviderKind::Anthropic => config
            .providers
            .anthropic
            .as_ref()
            .and_then(|cfg| cfg.models.clone()),
        ProviderKind::Openai => config
            .providers
            .openai
            .as_ref()
            .and_then(|cfg| cfg.models.clone()),
        ProviderKind::Deepseek => config
            .providers
            .deepseek
            .as_ref()
            .and_then(|cfg| cfg.models.clone()),
    };
    let model = config
        .default
        .model
        .or(provider_model)
        .unwrap_or_else(|| DEFAULT_ECHO_MODEL.to_string());

    let prompt = PromptConfigFile {
        file: config.prompt.file.unwrap_or_else(|| "AGENTS.md".to_owned()),
        text: config.prompt.text,
        provider_overlays: config
            .prompt
            .providers
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(provider, overlay)| overlay.text.map(|text| (provider, text)))
            .collect(),
        model_overlays: config.prompt.models.unwrap_or_default(),
    };

    let mut turn = TurnConfig {
        model: model.clone(),
        allowed_models,
        base_prompt: BasePromptConfig {
            file: base_prompt.file.clone(),
            text: base_prompt.text.clone(),
        },
        prompt: PromptConfig {
            file: prompt.file.clone(),
            text: prompt.text.clone(),
            provider_overlays: prompt.provider_overlays.clone(),
            model_overlays: prompt.model_overlays.clone(),
        },
        ..TurnConfig::default()
    };
    turn.system_prompt = config.turn.system_prompt;
    if let Some(request_limit) = config.turn.request_limit {
        turn.request_limit = TurnRequestLimit::Adaptive {
            initial: request_limit,
            expand_on_progress: true,
        };
    }
    if let Some(compact_threshold_tokens) = config.turn.compact_threshold_tokens {
        turn.compact_threshold_tokens = Some(compact_threshold_tokens);
    }
    if let Some(max_llm_retries) = config.turn.max_llm_retries {
        turn.max_llm_retries = max_llm_retries;
    }
    if let Some(max_concurrent_tools) = config.turn.max_concurrent_tools {
        turn.max_concurrent_tools = max_concurrent_tools;
    }
    if turn.sampling == SamplingParams::default() {
        // 保持 default sampling 显式落在 effective config 中，方便后续扩字段。
    }

    let capabilities = CapabilitiesConfig::with_search(SearchCapabilityConfig::new(
        config
            .capabilities
            .search
            .as_ref()
            .and_then(|s| s.mode)
            .unwrap_or_default(),
    ));
    let fetch_default = FetchToolConfig::default();
    let fetch = config
        .tools
        .fetch
        .map(|cfg| FetchToolConfig {
            enabled: cfg.enabled.unwrap_or(fetch_default.enabled),
            default_timeout_secs: cfg
                .default_timeout_secs
                .unwrap_or(fetch_default.default_timeout_secs),
            max_timeout_secs: cfg
                .max_timeout_secs
                .unwrap_or(fetch_default.max_timeout_secs),
            max_response_bytes: cfg
                .max_response_bytes
                .unwrap_or(fetch_default.max_response_bytes),
            default_format: cfg.default_format.unwrap_or(fetch_default.default_format),
            html_to_markdown: cfg
                .html_to_markdown
                .unwrap_or(fetch_default.html_to_markdown),
            follow_redirects: cfg
                .follow_redirects
                .unwrap_or(fetch_default.follow_redirects),
        })
        .unwrap_or(fetch_default);

    Ok(EffectiveConfig {
        cli: CliConfig { provider, model },
        turn,
        base_prompt,
        prompt,
        capabilities,
        providers: ProviderConfigs {
            anthropic: config
                .providers
                .anthropic
                .map(|cfg| AnthropicConfigFile {
                    base_url: cfg.base_url,
                    default_model: cfg.default_model,
                    models: cfg.models,
                    capabilities: provider_capability_overrides(cfg.capabilities.as_ref()),
                })
                .unwrap_or_default(),
            openai: config
                .providers
                .openai
                .map(|cfg| OpenAiConfigFile {
                    base_url: cfg.base_url,
                    default_model: cfg.default_model,
                    models: cfg.models,
                    organization: cfg.organization,
                    project: cfg.project,
                    capabilities: provider_capability_overrides(cfg.capabilities.as_ref()),
                })
                .unwrap_or_default(),
            deepseek: config
                .providers
                .deepseek
                .map(|cfg| DeepSeekConfigFile {
                    base_url: cfg.base_url,
                    default_model: cfg.default_model,
                    models: cfg.models,
                    capabilities: provider_capability_overrides(cfg.capabilities.as_ref()),
                })
                .unwrap_or_default(),
        },
        tools: ToolsConfig {
            bash: config
                .tools
                .bash
                .map(|cfg| BashToolConfig {
                    default_timeout_ms: cfg.default_timeout_ms.unwrap_or(DEFAULT_BASH_TIMEOUT_MS),
                    max_timeout_ms: cfg.max_timeout_ms.unwrap_or(DEFAULT_BASH_MAX_TIMEOUT_MS),
                })
                .unwrap_or_default(),
            fs: config
                .tools
                .fs
                .map(|cfg| FsToolConfig {
                    read_default_limit: cfg.read_default_limit.unwrap_or(DEFAULT_FS_READ_LIMIT),
                    read_max_limit: cfg.read_max_limit.unwrap_or(DEFAULT_FS_READ_MAX_LIMIT),
                })
                .unwrap_or_default(),
            fetch,
        },
        sandbox: SandboxConfig {
            mode: config.sandbox.mode.unwrap_or(SandboxMode::AskWrites),
        },
        tracing: TracingConfig {
            filter: config.tracing.filter,
            otlp: config.tracing.otlp.map(|otlp| OtlpTracingConfig {
                endpoint: otlp.endpoint,
            }),
        },
        mcp: resolve_mcp_config(path, config.mcp).map_err(|message| ConfigError::Invalid {
            path: path.to_path_buf(),
            message,
        })?,
        http: HttpClientConfig {
            total_timeout_ms: config.http.total_timeout_ms,
            transport_retries: config.http.transport_retries,
            initial_backoff_ms: config.http.initial_backoff_ms,
            user_agent: config.http.user_agent,
            proxy: config
                .http
                .proxy
                .map(|cfg| HttpProxyConfig {
                    mode: cfg.mode.unwrap_or_default(),
                    explicit: HttpProxySettings {
                        http_proxy: cfg.http_proxy,
                        https_proxy: cfg.https_proxy,
                        no_proxy: cfg.no_proxy.unwrap_or_default(),
                    },
                })
                .unwrap_or_default(),
        },
    })
}

fn provider_capability_overrides(
    section: Option<&crate::types::ProviderCapabilitiesSection>,
) -> ProviderCapabilityOverrides {
    let Some(section) = section else {
        return ProviderCapabilityOverrides::default();
    };
    ProviderCapabilityOverrides::with_search(
        section
            .search
            .as_ref()
            .and_then(|s| s.mode)
            .map(SearchCapabilityConfig::new),
    )
}

/// 在 `[capabilities.search]` 与 `[tools.search]` 段共存时按
/// `docs/proposals/config-capabilities-and-tools.md` §6.2 的表发
/// `ConfigWarning::InactiveSection`。注意：仅在 mode = `delegate` /
/// `disabled` 时发；mode = `local` 时 `[tools.search]` 是正常的本地实现
/// 参数。
fn collect_inactive_section_warnings(
    merged: &TomlValue,
    capabilities: &CapabilitiesConfig,
    warnings: &mut Vec<ConfigWarning>,
) {
    use defect_agent::session::SearchCapabilityMode;

    let has_tools_search = merged
        .get("tools")
        .and_then(TomlValue::as_table)
        .map(|t| t.contains_key("search"))
        .unwrap_or(false);
    if !has_tools_search {
        return;
    }
    let mode = capabilities.search.mode;
    // `#[non_exhaustive]` 上来的兜底：未来追加 mode 时默认按 inactive 提示，
    // 让用户至少看到一条 warning，再按需细化。
    let mode_label = match mode {
        SearchCapabilityMode::Local => return,
        SearchCapabilityMode::Delegate => "delegate",
        SearchCapabilityMode::Disabled => "disabled",
        _ => "unknown",
    };
    warnings.push(ConfigWarning::InactiveSection {
        path: PathBuf::from("<merged>"),
        section: "tools.search".into(),
        reason: format!("capabilities.search.mode = \"{mode_label}\""),
    });
}

fn sanitize_shared_project_layer(
    path: Option<&PathBuf>,
    value: &TomlValue,
) -> (TomlValue, Vec<ConfigWarning>) {
    let mut sanitized = value.clone();
    let mut warnings = Vec::new();
    let Some(path) = path.cloned() else {
        return (sanitized, warnings);
    };

    if remove_toml_path(&mut sanitized, &["default", "provider"]) {
        warnings.push(ConfigWarning::IgnoredProjectKey {
            path: path.clone(),
            key: "default.provider".into(),
            reason: "shared project config must not redirect model traffic",
        });
    }

    if let Some(providers) = sanitized
        .get_mut("providers")
        .and_then(TomlValue::as_table_mut)
    {
        for (provider_name, provider_value) in providers.iter_mut() {
            for key in ["base_url", "organization", "project", "api_key", "token"] {
                if remove_toml_table_key(provider_value, key) {
                    warnings.push(ConfigWarning::IgnoredProjectKey {
                        path: path.clone(),
                        key: format!("providers.{provider_name}.{key}"),
                        reason: "shared project config must not redirect endpoints or credentials",
                    });
                }
            }
        }
    }

    if remove_toml_path(&mut sanitized, &["tracing", "otlp"]) {
        warnings.push(ConfigWarning::IgnoredProjectKey {
            path: path.clone(),
            key: "tracing.otlp".into(),
            reason: "shared project config must not redirect observability sinks",
        });
    }

    // 仓库内共享配置不能静默把出站流量改到第三方代理；timeout / retries /
    // user_agent 等不会改变流量目的地，仍然允许仓库统一调优。
    if remove_toml_path(&mut sanitized, &["http", "proxy"]) {
        warnings.push(ConfigWarning::IgnoredProjectKey {
            path,
            key: "http.proxy".into(),
            reason: "shared project config must not redirect outbound HTTP traffic",
        });
    }

    (sanitized, warnings)
}

fn load_optional_layer(
    source: ConfigSource,
    path: PathBuf,
) -> Result<Option<(ConfigLayerEntry, Vec<ConfigWarning>)>, ConfigError> {
    load_optional_layer_opt(source, Some(path))
}

fn load_optional_layer_opt(
    source: ConfigSource,
    path: Option<PathBuf>,
) -> Result<Option<(ConfigLayerEntry, Vec<ConfigWarning>)>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(ConfigError::Io {
                path,
                source: BoxError::new(err),
            });
        }
    };
    let value: TomlValue = raw.parse::<TomlValue>().map_err(|err| ConfigError::Parse {
        path: path.clone(),
        source: BoxError::new(err),
    })?;
    let mut warnings = Vec::new();
    collect_unknown_keys(&path, None, &value, &mut warnings);
    Ok(Some((
        ConfigLayerEntry {
            source,
            path: Some(path),
            raw_toml: Some(raw),
            value,
        },
        warnings,
    )))
}

fn collect_unknown_keys(
    path: &Path,
    prefix: Option<&str>,
    value: &TomlValue,
    warnings: &mut Vec<ConfigWarning>,
) {
    let Some(table) = value.as_table() else {
        return;
    };
    for (key, nested) in table {
        let full_key = prefix
            .map(|prefix| format!("{prefix}.{key}"))
            .unwrap_or_else(|| key.clone());
        if is_known_config_key(&full_key) {
            collect_unknown_keys(path, Some(&full_key), nested, warnings);
            continue;
        }
        if nested.is_table() && is_known_config_prefix(&full_key) {
            collect_unknown_keys(path, Some(&full_key), nested, warnings);
            continue;
        }
        warnings.push(ConfigWarning::UnknownKey {
            path: path.to_path_buf(),
            key: full_key,
        });
    }
}

pub(crate) fn dotenv_updates_from_str(
    raw: &str,
    existing_keys: &[impl AsRef<str>],
) -> Vec<(String, String)> {
    raw.lines()
        .filter_map(|line| parse_dotenv_line(line.trim()))
        .filter(|(key, _)| {
            !existing_keys
                .iter()
                .any(|existing| existing.as_ref() == key.as_str())
        })
        .collect()
}

fn raw_env_keys() -> Vec<String> {
    env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .collect()
}

fn parse_dotenv_line(line: &str) -> Option<(String, String)> {
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), strip_quotes(value.trim()).to_string()))
}

fn strip_quotes(s: &str) -> &str {
    let bytes = s.as_bytes();
    if let [first @ (b'"' | b'\''), .., last] = bytes
        && first == last
    {
        return &s[1..s.len() - 1];
    }
    s
}

fn is_known_config_key(key: &str) -> bool {
    matches!(
        key,
        "base_prompt.file"
            | "base_prompt.text"
            | "default.provider"
            | "default.model"
            | "prompt.file"
            | "prompt.text"
            | "prompt.providers"
            | "prompt.models"
            | "turn.system_prompt"
            | "turn.request_limit"
            | "turn.compact_threshold_tokens"
            | "turn.max_llm_retries"
            | "turn.max_concurrent_tools"
            | "capabilities.search.mode"
            | "providers.anthropic.base_url"
            | "providers.anthropic.default_model"
            | "providers.anthropic.models"
            | "providers.anthropic.capabilities.search.mode"
            | "providers.openai.base_url"
            | "providers.openai.default_model"
            | "providers.openai.models"
            | "providers.openai.organization"
            | "providers.openai.project"
            | "providers.openai.capabilities.search.mode"
            | "providers.deepseek.base_url"
            | "providers.deepseek.default_model"
            | "providers.deepseek.models"
            | "providers.deepseek.capabilities.search.mode"
            | "tools.bash.default_timeout_ms"
            | "tools.bash.max_timeout_ms"
            | "tools.fs.read_default_limit"
            | "tools.fs.read_max_limit"
            | "tools.fetch.enabled"
            | "tools.fetch.default_timeout_secs"
            | "tools.fetch.max_timeout_secs"
            | "tools.fetch.max_response_bytes"
            | "tools.fetch.default_format"
            | "tools.fetch.html_to_markdown"
            | "tools.fetch.follow_redirects"
            | "sandbox.mode"
            | "tracing.filter"
            | "tracing.otlp.endpoint"
            | "mcp.enabled_servers"
            | "http.total_timeout_ms"
            | "http.transport_retries"
            | "http.initial_backoff_ms"
            | "http.user_agent"
            | "http.proxy.mode"
            | "http.proxy.http_proxy"
            | "http.proxy.https_proxy"
            | "http.proxy.no_proxy"
    ) || is_known_mcp_key(key)
        || is_known_tools_search_key(key)
}

/// `[tools.search]` 段的 schema 在 P1 还没敲定（详见
/// `docs/proposals/config-capabilities-and-tools.md` §9）；这里把整段
/// 视为已知，避免每个未来字段都触发 `UnknownKey`。当 mode != `local`
/// 时由 `InactiveSection` warning 提示用户该段实际不会生效。
fn is_known_tools_search_key(key: &str) -> bool {
    key == "tools.search" || key.starts_with("tools.search.")
}

fn is_known_config_prefix(key: &str) -> bool {
    matches!(
        key,
        "default"
            | "base_prompt"
            | "prompt"
            | "turn"
            | "capabilities"
            | "capabilities.search"
            | "providers"
            | "providers.anthropic"
            | "providers.anthropic.capabilities"
            | "providers.anthropic.capabilities.search"
            | "providers.openai"
            | "providers.openai.capabilities"
            | "providers.openai.capabilities.search"
            | "providers.deepseek"
            | "providers.deepseek.capabilities"
            | "providers.deepseek.capabilities.search"
            | "tools"
            | "tools.bash"
            | "tools.fs"
            | "tools.fetch"
            | "tools.search"
            | "sandbox"
            | "tracing"
            | "tracing.otlp"
            | "mcp"
            | "http"
            | "http.proxy"
    ) || is_known_mcp_prefix(key)
}

fn resolve_user_config_path(opts: &LoadConfigOptions) -> Result<PathBuf, ConfigError> {
    if let Some(xdg) = &opts.xdg_config_home {
        return Ok(xdg.join(USER_CONFIG_RELATIVE));
    }
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg).join(USER_CONFIG_RELATIVE));
    }
    if let Some(home) = &opts.home_dir {
        return Ok(home.join(".config/defect/config.toml"));
    }
    if let Ok(home) = env::var("HOME") {
        return Ok(PathBuf::from(home).join(".config/defect/config.toml"));
    }

    Err(ConfigError::Invalid {
        path: PathBuf::from("<env>"),
        message: "neither XDG_CONFIG_HOME nor HOME is set".into(),
    })
}

fn find_repo_root(cwd: &Path) -> Option<PathBuf> {
    for dir in cwd.ancestors() {
        let git_dir = dir.join(".git");
        if git_dir.exists() {
            return Some(dir.to_path_buf());
        }
    }
    None
}

fn canonicalize_or_original(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn extract_base_prompt(
    config: &TomlValue,
    source_path: Option<&PathBuf>,
) -> Option<BasePromptConfigFile> {
    let base = config.get("base_prompt")?.as_table()?;
    let file = base
        .get("file")
        .and_then(TomlValue::as_str)
        .map(PathBuf::from);
    let text = base
        .get("text")
        .and_then(TomlValue::as_str)
        .map(str::to_owned);
    if file.is_none() && text.is_none() {
        None
    } else {
        let file = file.map(|path| match source_path {
            Some(path_root) if path.is_relative() => {
                path_root.parent().unwrap_or(path_root).join(path)
            }
            _ => path,
        });
        Some(BasePromptConfigFile { file, text })
    }
}

#[cfg(test)]
#[path = "loader/test.rs"]
mod test;
