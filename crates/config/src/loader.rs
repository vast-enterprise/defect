use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::llm::SamplingParams;
use defect_agent::session::{TurnConfig, TurnRequestLimit};
use toml::Value as TomlValue;

use crate::overrides::{
    build_cli_layer, merge_toml_values, remove_toml_path, remove_toml_table_key,
};
use crate::types::{
    AnthropicConfigFile, BashToolConfig, CliConfig, ConfigError, ConfigLayerEntry,
    ConfigLayerStack, ConfigSource, ConfigToml, ConfigWarning, DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_BASH_MAX_TIMEOUT_MS, DEFAULT_BASH_TIMEOUT_MS, DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_ECHO_MODEL, DEFAULT_FS_READ_LIMIT, DEFAULT_FS_READ_MAX_LIMIT, DEFAULT_OPENAI_MODEL,
    DeepSeekConfigFile, EffectiveConfig, FsToolConfig, LoadConfigOptions, LoadedConfig,
    OpenAiConfigFile, OtlpTracingConfig, PROJECT_CONFIG_RELATIVE, PROJECT_LOCAL_CONFIG_RELATIVE,
    ProviderConfigs, ProviderKind, SandboxConfig, SandboxMode, ToolsConfig, TracingConfig,
    USER_CONFIG_RELATIVE,
};

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

    if let Some((user_layer, layer_warnings)) = load_optional_layer(ConfigSource::User, user_path)?
    {
        warnings.extend(layer_warnings);
        merge_toml_values(&mut merged, &user_layer.value);
        layers.push(user_layer);
    }

    if let Some((project_layer, layer_warnings)) =
        load_optional_layer_opt(ConfigSource::Project, project_path)?
    {
        warnings.extend(layer_warnings);
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
        merge_toml_values(&mut merged, &project_local_layer.value);
        layers.push(project_local_layer);
    }

    if let Some(cli_layer) = build_cli_layer(&opts.cli)? {
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
    let effective = build_effective_config(parsed);

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
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
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

fn build_effective_config(config: ConfigToml) -> EffectiveConfig {
    let provider = config.default.provider.unwrap_or_default();
    let provider_model = match provider {
        ProviderKind::Echo => Some(DEFAULT_ECHO_MODEL.to_string()),
        ProviderKind::Anthropic => config
            .providers
            .anthropic
            .as_ref()
            .and_then(|cfg| cfg.model.clone())
            .or_else(|| Some(DEFAULT_ANTHROPIC_MODEL.to_string())),
        ProviderKind::Openai => config
            .providers
            .openai
            .as_ref()
            .and_then(|cfg| cfg.model.clone())
            .or_else(|| Some(DEFAULT_OPENAI_MODEL.to_string())),
        ProviderKind::Deepseek => config
            .providers
            .deepseek
            .as_ref()
            .and_then(|cfg| cfg.model.clone())
            .or_else(|| Some(DEFAULT_DEEPSEEK_MODEL.to_string())),
    };
    let model = config
        .default
        .model
        .or(provider_model)
        .unwrap_or_else(|| DEFAULT_ECHO_MODEL.to_string());

    let mut turn = TurnConfig {
        model: model.clone(),
        ..TurnConfig::default()
    };
    if let Some(system_prompt) = config.turn.system_prompt {
        turn.system_prompt = Some(system_prompt);
    }
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

    EffectiveConfig {
        cli: CliConfig { provider, model },
        turn,
        providers: ProviderConfigs {
            anthropic: config
                .providers
                .anthropic
                .map(|cfg| AnthropicConfigFile {
                    base_url: cfg.base_url,
                    model: cfg.model,
                })
                .unwrap_or_default(),
            openai: config
                .providers
                .openai
                .map(|cfg| OpenAiConfigFile {
                    base_url: cfg.base_url,
                    model: cfg.model,
                    organization: cfg.organization,
                    project: cfg.project,
                })
                .unwrap_or_default(),
            deepseek: config
                .providers
                .deepseek
                .map(|cfg| DeepSeekConfigFile {
                    base_url: cfg.base_url,
                    model: cfg.model,
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
    }
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
            path,
            key: "tracing.otlp".into(),
            reason: "shared project config must not redirect observability sinks",
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
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
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
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

fn is_known_config_key(key: &str) -> bool {
    matches!(
        key,
        "default.provider"
            | "default.model"
            | "turn.system_prompt"
            | "turn.request_limit"
            | "turn.compact_threshold_tokens"
            | "turn.max_llm_retries"
            | "turn.max_concurrent_tools"
            | "providers.anthropic.base_url"
            | "providers.anthropic.model"
            | "providers.openai.base_url"
            | "providers.openai.model"
            | "providers.openai.organization"
            | "providers.openai.project"
            | "providers.deepseek.base_url"
            | "providers.deepseek.model"
            | "tools.bash.default_timeout_ms"
            | "tools.bash.max_timeout_ms"
            | "tools.fs.read_default_limit"
            | "tools.fs.read_max_limit"
            | "sandbox.mode"
            | "tracing.filter"
            | "tracing.otlp.endpoint"
    )
}

fn is_known_config_prefix(key: &str) -> bool {
    matches!(
        key,
        "default"
            | "turn"
            | "providers"
            | "providers.anthropic"
            | "providers.openai"
            | "providers.deepseek"
            | "tools"
            | "tools.bash"
            | "tools.fs"
            | "sandbox"
            | "tracing"
            | "tracing.otlp"
    )
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

#[cfg(test)]
#[path = "loader/test.rs"]
mod test;
