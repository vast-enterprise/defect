use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::llm::SamplingParams;
use defect_agent::session::{BasePromptConfig, PromptConfig, TurnConfig, TurnRequestLimit};
use toml::Value as TomlValue;

use crate::hooks::{LayerHooks, merge_layer_hooks, parse_layer_hooks};
use crate::mcp::resolve_mcp_config;
use crate::overrides::{build_cli_layer, merge_toml_values};
use crate::types::{
    BasePromptConfigFile, BashToolConfig, CapabilitiesConfig, CliConfig, ConfigError,
    ConfigLayerEntry, ConfigLayerStack, ConfigSource, ConfigToml, ConfigWarning,
    DEFAULT_ANTHROPIC_MODEL, DEFAULT_BASH_MAX_TIMEOUT_MS, DEFAULT_BASH_TIMEOUT_MS,
    DEFAULT_DEEPSEEK_MODEL, DEFAULT_ECHO_MODEL, DEFAULT_FS_READ_LIMIT, DEFAULT_FS_READ_MAX_LIMIT,
    DEFAULT_OPENAI_MODEL, EffectiveConfig, FetchToolConfig, FsToolConfig, HooksConfig,
    HttpClientConfig, HttpProxyConfig, HttpProxySettings, LangfuseConfig, LoadConfigOptions,
    LoadedConfig, OtlpTracingConfig, PROJECT_CONFIG_RELATIVE, PROJECT_LOCAL_CONFIG_RELATIVE,
    PromptConfigFile, ProviderCapabilityOverrides, ProviderConfigFile, ProviderConfigs,
    ProviderKind, ProviderSection, SandboxConfig, SandboxMode, SearchToolConfig, ToolsConfig,
    TracingConfig, USER_CONFIG_RELATIVE,
};
use defect_agent::session::{BackgroundProgressConfig, WebSearchCapabilityConfig};

/// Loads and merges the effective configuration for `defect`.
///
/// Precedence is: `default < user < project < project-local < CLI`.
///
/// # Errors
///
/// Returns [`ConfigError`] when the user config path cannot be resolved, any config file
/// fails to read from disk, TOML parsing fails, or the merged configuration cannot be
/// deserialized into a strongly-typed structure.
pub fn load_config(opts: LoadConfigOptions) -> Result<LoadedConfig, ConfigError> {
    let cwd = canonicalize_or_original(&opts.cwd);
    let user_path = resolve_user_config_path(&opts);
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
    // Hooks cannot use "merge then decode" — array merge semantics are append+dedupe; see
    // the comment at the top of `crates/config/src/hooks.rs`. Extract each layer
    // separately, then call `merge_layer_hooks` at the end.
    let mut hook_layers: Vec<LayerHooks> = Vec::new();

    if let Some((user_layer, layer_warnings)) =
        load_optional_layer_opt(ConfigSource::User, user_path)?
    {
        warnings.extend(layer_warnings);
        if let Some(candidate) = extract_base_prompt(&user_layer.value, user_layer.path.as_ref()) {
            base_prompt = Some(candidate);
        }
        if let Some(path) = user_layer.path.clone() {
            hook_layers.push(parse_layer_hooks(
                path,
                ConfigSource::User,
                &user_layer.value,
            )?);
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
        if let Some(path) = project_layer.path.clone() {
            hook_layers.push(parse_layer_hooks(
                path,
                ConfigSource::Project,
                &project_layer.value,
            )?);
        }
        merge_toml_values(&mut merged, &project_layer.value);
        layers.push(project_layer);
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
        if let Some(path) = project_local_layer.path.clone() {
            hook_layers.push(parse_layer_hooks(
                path,
                ConfigSource::ProjectLocal,
                &project_local_layer.value,
            )?);
        }
        merge_toml_values(&mut merged, &project_local_layer.value);
        layers.push(project_local_layer);
    }

    if let Some(cli_layer) = build_cli_layer(&opts.cli)? {
        if let Some(candidate) = extract_base_prompt(&cli_layer.value, cli_layer.path.as_ref()) {
            base_prompt = Some(candidate);
        }
        // CLI overrides use dotted-key syntax, which cannot express `[[hooks.*]]` arrays
        // — hooks cannot be constructed from the command line. `parse_layer_hooks` is not
        // called here to avoid implying that the CLI layer could contain hooks.
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
    let hooks = merge_layer_hooks(hook_layers);
    let effective = build_effective_config(
        Path::new("<merged>"),
        parsed,
        base_prompt.unwrap_or_default(),
        hooks,
    )?;

    Ok(LoadedConfig {
        layers: ConfigLayerStack { layers },
        effective,
        warnings,
    })
}

/// Reads `cwd/.env` compatibly, only filling in missing environment variables.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] if the `.env` file exists but cannot be read.
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
        // SAFETY: called before any concurrent tasks are spawned; this only sets env vars
        // during process startup.
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
    hooks: HooksConfig,
) -> Result<EffectiveConfig, ConfigError> {
    // The final selection of `base_prompt` is done in `load_config()`; here we only keep
    // the typed decode constraint on the schema and explicitly consume the fields to
    // avoid them falling out of sync with the raw-layer parsing.
    let _ = config.base_prompt.file.as_deref();
    let _ = config.base_prompt.text.as_deref();
    let provider = config.default.provider.unwrap_or_default();
    let provider_config = raw_provider_config(&config.providers, &provider);
    if matches!(provider, ProviderKind::Custom(_)) && provider_config.is_none() {
        return Err(ConfigError::Invalid {
            path: path.to_path_buf(),
            message: format!(
                "default.provider `{provider}` has no matching [providers.{provider}] section"
            ),
        });
    }
    let provider_model = provider_default_model(&provider, provider_config);
    // The allowed_models allowlist only needs the id; the display name is taken
    // separately in the entry_models path.
    let provider_allowed_models: Option<Vec<String>> = provider_config.and_then(|cfg| {
        cfg.models
            .as_ref()
            .map(|models| models.iter().map(|m| m.id().to_string()).collect())
    });
    let model = match config.default.model.or(provider_model) {
        Some(model) => model,
        None => {
            return Err(ConfigError::Invalid {
                path: path.to_path_buf(),
                message: format!(
                    "default.model or providers.{provider}.default_model is required for provider `{provider}`"
                ),
            });
        }
    };
    let allowed_models = merged_allowed_models(
        provider_allowed_models,
        configured_provider_models(&config.providers),
        &model,
    );

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
        // `ProviderKind::as_str()` returns the runtime vendor string (matching
        // `ProviderInfo.vendor` set for each provider in `cli/providers.rs`),
        // so it selects the correct provider half.
        provider: provider.as_str().to_string(),
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
    if let Some(compact_ratio) = config.turn.compact_ratio {
        turn.compact_ratio = Some(compact_ratio);
    }
    if let Some(background_compact_enabled) = config.turn.background_compact_enabled {
        turn.background_compact_enabled = background_compact_enabled;
    }
    if let Some(compact_soft_ratio) = config.turn.compact_soft_ratio {
        turn.compact_soft_ratio = Some(compact_soft_ratio);
    }
    if let Some(microcompact_enabled) = config.turn.microcompact_enabled {
        turn.microcompact_enabled = microcompact_enabled;
    }
    if let Some(microcompact_ratio) = config.turn.microcompact_ratio {
        turn.microcompact_ratio = Some(microcompact_ratio);
    }
    if let Some(max_llm_retries) = config.turn.max_llm_retries {
        turn.max_llm_retries = max_llm_retries;
    }
    if let Some(max_concurrent_tools) = config.turn.max_concurrent_tools {
        turn.max_concurrent_tools = max_concurrent_tools;
    }
    if let Some(max_hook_continues) = config.turn.max_hook_continues {
        turn.max_hook_continues = max_hook_continues;
    }
    if let Some(subagent_max_depth) = config.turn.subagent_max_depth {
        turn.subagent_max_depth = subagent_max_depth;
    }
    validate_compact_ratios(path, &turn)?;
    if turn.sampling == SamplingParams::default() {
        // Keep the default sampling explicitly in the effective config so that adding
        // fields later is easier.
    }

    let capabilities = CapabilitiesConfig::with_web_search(WebSearchCapabilityConfig::new(
        config
            .capabilities
            .web_search
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

    let search_default = SearchToolConfig::default();
    let search = config
        .tools
        .search
        .map(|cfg| SearchToolConfig {
            enabled: cfg.enabled.unwrap_or(search_default.enabled),
            default_head_limit: cfg
                .default_head_limit
                .unwrap_or(search_default.default_head_limit),
            max_head_limit: cfg.max_head_limit.unwrap_or(search_default.max_head_limit),
            max_file_size_bytes: cfg
                .max_file_size_bytes
                .unwrap_or(search_default.max_file_size_bytes),
            max_result_bytes: cfg
                .max_result_bytes
                .unwrap_or(search_default.max_result_bytes),
            max_walk_files: cfg.max_walk_files.unwrap_or(search_default.max_walk_files),
            respect_gitignore_default: cfg
                .respect_gitignore_default
                .unwrap_or(search_default.respect_gitignore_default),
        })
        .unwrap_or(search_default);

    let background_default = BackgroundProgressConfig::default();
    let background = config
        .tools
        .background
        .map(|cfg| BackgroundProgressConfig {
            default_recent_blocks: cfg
                .default_recent_blocks
                .unwrap_or(background_default.default_recent_blocks),
            block_text_limit: cfg
                .block_text_limit
                .unwrap_or(background_default.block_text_limit),
        })
        .unwrap_or(background_default);

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
                .map(provider_config_file)
                .unwrap_or_default(),
            openai: config
                .providers
                .openai
                .map(provider_config_file)
                .unwrap_or_default(),
            deepseek: config
                .providers
                .deepseek
                .map(provider_config_file)
                .unwrap_or_default(),
            litellm: config
                .providers
                .litellm
                .map(provider_config_file)
                .unwrap_or_default(),
            custom: config
                .providers
                .custom
                .into_iter()
                .map(|(name, cfg)| (name, provider_config_file(cfg)))
                .collect(),
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
            search,
            background,
        },
        sandbox: SandboxConfig {
            mode: config.sandbox.mode.unwrap_or(SandboxMode::AskWrites),
        },
        tracing: TracingConfig {
            filter: config.tracing.filter,
            otlp: config.tracing.otlp.map(|otlp| OtlpTracingConfig {
                endpoint: otlp.endpoint,
            }),
            langfuse: config.tracing.langfuse.map(|lf| LangfuseConfig {
                enabled: lf.enabled.unwrap_or(false),
                host: lf.host,
                public_key: lf.public_key,
                secret_key: lf.secret_key,
                flush_interval_ms: lf.flush_interval_ms,
                max_batch: lf.max_batch,
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
        hooks,
    })
}

/// Validate the three-tier compaction watermark ratios: each ratio must be in `(0, 1]`,
/// and `micro ≤ soft < hard` (only enforced between tiers that are actually set). Any
/// inversion or out-of-bounds value is a hard [`ConfigError::Invalid`] fail — no silent
/// correction, to avoid "running with awkward watermarks".
fn validate_compact_ratios(path: &Path, turn: &TurnConfig) -> Result<(), ConfigError> {
    let invalid = |message: String| ConfigError::Invalid {
        path: path.to_path_buf(),
        message,
    };
    // Validate the domain of each ratio.
    for (name, ratio) in [
        ("microcompact_ratio", turn.microcompact_ratio),
        ("compact_soft_ratio", turn.compact_soft_ratio),
        ("compact_ratio", turn.compact_ratio),
    ] {
        if let Some(r) = ratio
            && !(r > 0.0 && r <= 1.0)
        {
            return Err(invalid(format!("[turn].{name} must be in (0, 1], got {r}")));
        }
    }
    // Only enforce ordering constraints for enabled tiers: micro ≤ soft < hard.
    let micro = turn
        .microcompact_enabled
        .then_some(turn.microcompact_ratio)
        .flatten();
    let soft = turn
        .background_compact_enabled
        .then_some(turn.compact_soft_ratio)
        .flatten();
    let hard = turn.compact_ratio;
    if let (Some(soft), Some(hard)) = (soft, hard)
        && soft >= hard
    {
        return Err(invalid(format!(
            "[turn].compact_soft_ratio ({soft}) must be < compact_ratio ({hard}); \
             the soft watermark must be strictly below the hard one to leave room for background compaction"
        )));
    }
    if let (Some(micro), Some(soft)) = (micro, soft)
        && micro > soft
    {
        return Err(invalid(format!(
            "[turn].microcompact_ratio ({micro}) must be ≤ compact_soft_ratio ({soft})"
        )));
    }
    if let (Some(micro), Some(hard)) = (micro, hard)
        && micro >= hard
    {
        return Err(invalid(format!(
            "[turn].microcompact_ratio ({micro}) must be < compact_ratio ({hard})"
        )));
    }
    Ok(())
}

fn raw_provider_config<'a>(
    providers: &'a crate::types::ProvidersSection,
    provider: &ProviderKind,
) -> Option<&'a ProviderSection> {
    match provider {
        ProviderKind::Defect => None,
        ProviderKind::Anthropic => providers.anthropic.as_ref(),
        ProviderKind::Openai => providers.openai.as_ref(),
        ProviderKind::Deepseek => providers.deepseek.as_ref(),
        ProviderKind::Litellm => providers.litellm.as_ref(),
        ProviderKind::Custom(name) => providers.custom.get(name),
    }
}

fn merged_allowed_models(
    provider_allowed_models: Option<Vec<String>>,
    configured_models: Vec<String>,
    current_model: &str,
) -> Option<Vec<String>> {
    let mut models = provider_allowed_models.unwrap_or_default();
    append_unique_models(&mut models, configured_models);
    if models.is_empty() {
        return None;
    }
    if !models.iter().any(|model| model == current_model) {
        models.insert(0, current_model.to_string());
    }
    Some(models)
}

fn configured_provider_models(providers: &crate::types::ProvidersSection) -> Vec<String> {
    let mut models = Vec::new();
    for section in [
        providers.anthropic.as_ref(),
        providers.openai.as_ref(),
        providers.deepseek.as_ref(),
        providers.litellm.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        append_unique_models(&mut models, provider_declared_models(section));
    }
    for section in providers.custom.values() {
        append_unique_models(&mut models, provider_declared_models(section));
    }
    models
}

fn provider_declared_models(section: &ProviderSection) -> Vec<String> {
    let mut models = Vec::new();
    if let Some(default_model) = &section.default_model {
        models.push(default_model.clone());
    }
    if let Some(section_models) = &section.models {
        // The allowed_models allowlist only cares about the id — discard the display
        // name.
        append_unique_models(
            &mut models,
            section_models.iter().map(|m| m.id().to_string()).collect(),
        );
    }
    models
}

fn append_unique_models(target: &mut Vec<String>, source: Vec<String>) {
    for model in source {
        if !target.iter().any(|existing| existing == &model) {
            target.push(model);
        }
    }
}

fn provider_default_model(
    provider: &ProviderKind,
    config: Option<&ProviderSection>,
) -> Option<String> {
    if let Some(default_model) = config.and_then(|cfg| cfg.default_model.clone()) {
        return Some(default_model);
    }
    match provider {
        ProviderKind::Defect => Some(DEFAULT_ECHO_MODEL.to_string()),
        ProviderKind::Anthropic => Some(DEFAULT_ANTHROPIC_MODEL.to_string()),
        ProviderKind::Openai => Some(DEFAULT_OPENAI_MODEL.to_string()),
        ProviderKind::Deepseek => Some(DEFAULT_DEEPSEEK_MODEL.to_string()),
        ProviderKind::Litellm => None,
        ProviderKind::Custom(_) => None,
    }
}

fn provider_config_file(cfg: ProviderSection) -> ProviderConfigFile {
    ProviderConfigFile {
        protocol: cfg.protocol,
        base_url: cfg.base_url,
        default_model: cfg.default_model,
        models: cfg.models,
        display_name: cfg.display_name,
        api_key_env: cfg.api_key_env,
        organization: cfg.organization,
        project: cfg.project,
        aws: cfg.aws,
        headers: cfg.headers.unwrap_or_default(),
        capabilities: provider_capability_overrides(cfg.capabilities.as_ref()),
        reasoning_effort: cfg.reasoning_effort,
    }
}

fn provider_capability_overrides(
    section: Option<&crate::types::ProviderCapabilitiesSection>,
) -> ProviderCapabilityOverrides {
    let Some(section) = section else {
        return ProviderCapabilityOverrides::default();
    };
    ProviderCapabilityOverrides::with_web_search(
        section
            .web_search
            .as_ref()
            .and_then(|s| s.mode)
            .map(WebSearchCapabilityConfig::new),
    )
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
    // Unknown key validation runs per-layer here: `deny_unknown_fields` errors from serde
    // during decoding include the file path for that layer (decoding after merging would
    // only report `<merged>`). See Config merge behavior.
    reject_unknown_keys(&path, &value)?;
    let warnings = Vec::new();
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

/// Performs layer-by-layer typed-decode validation: when an unknown key is encountered,
/// serde immediately errors, which is converted to [`ConfigError::Invalid`] with the file
/// path for that layer. The `[hooks]` section is absorbed by `ConfigToml::hooks` to allow
/// unknown fields through, as its own parser handles schema validation.
fn reject_unknown_keys(path: &Path, value: &TomlValue) -> Result<(), ConfigError> {
    value
        .clone()
        .try_into::<ConfigToml>()
        .map(|_| ())
        .map_err(|err| ConfigError::Invalid {
            path: path.to_path_buf(),
            message: err.to_string(),
        })
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

/// Resolve the user-level `config.toml` path. Shares the same priority logic as
/// [`crate::profiles`]'s `resolve_user_agents_dir` and [`crate::skills`]; returns
/// `None` when not found — if the user has no XDG or HOME set, the user-level
/// configuration is simply absent and should not prevent the program from starting.
fn resolve_user_config_path(opts: &LoadConfigOptions) -> Option<PathBuf> {
    // `--local`: ignore global/user-level config — sandbox only recognizes the project
    // root `.defect/`.
    if opts.local {
        return None;
    }
    if let Some(xdg) = &opts.xdg_config_home {
        return Some(xdg.join(USER_CONFIG_RELATIVE));
    }
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join(USER_CONFIG_RELATIVE));
    }
    if let Some(home) = &opts.home_dir {
        return Some(home.join(".config/defect/config.toml"));
    }
    if let Ok(home) = env::var("HOME") {
        return Some(PathBuf::from(home).join(".config/defect/config.toml"));
    }

    None
}

pub fn find_repo_root(cwd: &Path) -> Option<PathBuf> {
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
mod tests;
