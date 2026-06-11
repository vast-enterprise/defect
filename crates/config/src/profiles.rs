//! Subagent profile discovery and parsing.
//!
//! Profiles are selected by name via the `spawn_agent` tool, allowing a parent agent to
//! delegate tasks to a fresh, context-isolated child agent. See the design note
//! `project-subagent-design`.
//!
//! ## Two formats (same fields, choose one)
//!
//! - **Directory variant**: `agents/<name>/`, containing a `config.toml` (TOML
//!   configuration) plus a system prompt file (default `system.md`, overridden by
//!   `[prompt] file`). Suitable when the prompt is long or you want to keep it in a
//!   separate file.
//! - **Single-file variant**: `agents/<name>.md`, where frontmatter (`+++` ⇒ **TOML**,
//!   `---` ⇒ **YAML**, community standard) is followed by the system prompt body. The
//!   field schema is identical to `config.toml`. Good for a one-file solution. This
//!   variant carries no extra resource files, so the `[prompt]` table is **illegal**
//!   here. YAML requires the `yaml` feature (enabled by default); without it, `---`
//!   headers hard-fail with an actionable error, while `+++` still works.
//!
//! If both variants exist with the same name in the same directory (e.g., `reviewer/` and
//! `reviewer.md`), it's a hard error — one name must have a single source of truth.
//!
//! ## Layered discovery
//!
//! Same structure as the main configuration ([`crate::loader`]):
//! - User layer: `<XDG_CONFIG_HOME>/defect/agents/` (or `~/.config/defect/agents/`)
//! - Project layer: `<repo_root>/.defect/agents/`
//!
//! When the same name exists in both layers, **the project layer overrides the user
//! layer**.
//!
//! ## Sandbox
//!
//! File references in `config.toml` (e.g., `[prompt] file`) are resolved relative to the
//! profile directory, using [`defect_agent::fs::resolve_workspace_path`] with the root
//! pinned to the profile directory. This blocks `../` traversal and symlink escapes —
//! this sandbox protects **the profile's own resource files**, which is separate from the
//! workspace sandbox used by the child agent during execution.

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::fs::resolve_workspace_path;
use defect_agent::llm::SamplingParams;
use defect_agent::session::TurnRequestLimit;
use serde::Deserialize;

use crate::frontmatter::{parse_frontmatter, split_frontmatter};
use crate::hooks::profile_hooks_from_raw;
use crate::loader::{find_repo_root, resolve_request_limit};
use crate::types::{ConfigError, ConfigSource, HooksConfig, LoadConfigOptions, RequestLimitMode};

/// Profile-level agent directory (relative to repo root). Mirrors [`crate::types`]'s
/// `PROJECT_CONFIG_RELATIVE` (`.defect/config.toml`).
const PROJECT_AGENTS_RELATIVE: &str = ".defect/agents";
/// User-level profile directory (relative to `XDG_CONFIG_HOME`). Corresponds to
/// `USER_CONFIG_RELATIVE` (`defect/config.toml`).
const USER_AGENTS_RELATIVE: &str = "defect/agents";
/// Default value for `[prompt] file` — `system.md` under the profile directory.
const DEFAULT_PROMPT_FILE: &str = "system.md";
/// Default `[tools] allow`: read-only set. Omitting `allow` yields a sub-agent that can
/// only read and search; safety is ensured by the absence of mutating tools (the tool
/// allowlist is the primary defense).
const DEFAULT_TOOL_ALLOW: &[&str] = &["read_file", "search"];

/// A parsed subagent profile.
///
/// Produced by [`discover_profiles`]; consumed by the `spawn_agent` tool and the
/// top-level CLI `--profile` flag.
#[derive(Debug, Clone)]
pub struct ProfileSpec {
    /// Profile name (directory name). Corresponds to the `profile` enum variant in
    /// `spawn_agent`.
    pub name: String,
    /// Absolute path to the profile directory.
    pub dir: PathBuf,
    /// Selection description — `spawn_agent` uses this to let the LLM decide which
    /// profile to pick; it also goes into the tool description's catalog. Required.
    pub description: String,
    /// Optional model override; omitted ⇒ the sub-agent falls back to the parent
    /// session's currently selected model.
    ///
    /// There is no separate `provider` field: the model ID already uniquely determines
    /// the provider via the provider registry's `entry_for_model`, so adding a provider
    /// field would create a second source of truth. To use a specific provider, simply
    /// write a model ID that belongs to that provider.
    pub model: Option<String>,
    /// The pre-resolved system prompt text (from `[prompt] file`).
    pub system_prompt_text: String,
    /// Tool allowlist — sub-agents can only see these tools. Omitted ⇒
    /// `DEFAULT_TOOL_ALLOW`.
    pub tool_allow: Vec<String>,
    /// Optional sampling parameter overrides.
    pub sampling: Option<SamplingParams>,
    /// When `true`, the subagent's system prompt is prefixed with the project instruction
    /// layer (`AGENTS.md`), so it inherits project world-knowledge (build/test/arch
    /// conventions) without inheriting the parent's identity. Default `false` (isolation +
    /// token economy). Configured via `inherit_project_prompt`.
    pub inherit_project_prompt: bool,
    /// Optional per-turn LLM-call cap. Omitted ⇒ the subagent uses a fixed anti-runaway
    /// default. Configured via `request_limit` (+ `request_limit_mode`), the same keys and
    /// semantics as the top-level `[turn]` config.
    pub request_limit: Option<TurnRequestLimit>,
    /// The `[hooks]` declared by this profile — hooks attached when a sub-agent runs a
    /// turn.
    ///
    /// Consistent with the "inherit world, not identity" principle: a profile's hooks are
    /// part of its identity, declared in the profile's own `config.toml` / frontmatter,
    /// and are **not** inherited from the parent session. Each entry carries the
    /// [`ConfigSource`] of the profile's layer (replaced when a project layer overrides a
    /// user layer, since the entire [`ProfileSpec`] is overridden). Omitted ⇒ empty
    /// (sub-agent has no hooks).
    pub hooks: HooksConfig,
}

/// Raw deserialization shape of `config.toml`. `deny_unknown_fields` matches the main
/// config — unknown keys hard-fail ([[feedback-minimize-no-paternalistic-guards]]).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileConfigToml {
    /// Required — if missing, serde reports "missing field `description`", which
    /// [`discover_profiles`] wraps into a hard error that includes the file path.
    description: String,
    #[serde(default)]
    model: Option<String>,
    /// `[default]` table — accepts `model` like the top-level config. Equivalent to the
    /// root-level `model`; setting both is a hard error.
    #[serde(default)]
    default: Option<ProfileDefaultToml>,
    #[serde(default)]
    prompt: Option<ProfilePromptToml>,
    #[serde(default)]
    tools: Option<ProfileToolsToml>,
    #[serde(default)]
    sampling: Option<ProfileSamplingToml>,
    /// When `true`, prefix the subagent's system prompt with the project `AGENTS.md` layer.
    #[serde(default)]
    inherit_project_prompt: bool,
    /// Per-turn LLM-call cap (same keys/semantics as top-level `[turn] request_limit`).
    #[serde(default)]
    request_limit: Option<u32>,
    /// Strategy for `request_limit` (`fixed` / `adaptive` / `unbounded`); `None` ⇒
    /// `adaptive` when a number is given, matching the top-level default.
    #[serde(default)]
    request_limit_mode: Option<RequestLimitMode>,
    /// The `[hooks]` table: event name → array of hook entries for that event. Its shape
    /// is identical to the top-level `[hooks]` (reuses `HookEntryRaw`). A profile is a
    /// single closed truth source and does not support cross-layer `disable` — a
    /// `disable` key causes a hard fail as if the event name were unknown.
    #[serde(default)]
    hooks: BTreeMap<String, toml::Value>,
}

/// Profile system-prompt source. Mirrors the top-level `[prompt]` shape: either an inline
/// `text` or a `file` path (folder profiles only). At most one may be set.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilePromptToml {
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

/// Optional `[default]` table, accepted so a profile can be written with the same
/// `[default] model` key the top-level config uses (in addition to the root-level `model`
/// shorthand). At most one of the two model sources may be set.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileDefaultToml {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileToolsToml {
    #[serde(default)]
    allow: Option<Vec<String>>,
}

/// Subset of sampling overrides – only expose the scalars currently needed; when
/// mapping to [`SamplingParams`], merge on top of `default()` and leave other fields
/// (`thinking` / `stop_sequences`) at their defaults.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileSamplingToml {
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
}

impl ProfileSamplingToml {
    fn into_params(self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            ..SamplingParams::default()
        }
    }
}

/// Discover and parse all available profiles.
///
/// Scans user-level first, then project-level; for profiles with the same name, the
/// project-level one overrides the user-level one. If any profile's `config.toml` fails
/// to parse or its `system.md` is out of bounds or unreadable, it is a hard error (fail
/// loud, do not silently skip bad profiles). Non-profile items in the directory
/// (subdirectories without `config.toml`, non-directory entries) are silently skipped.
///
/// # Errors
/// - [`ConfigError::Io`]: reading `config.toml` / `system.md` failed
/// - [`ConfigError::Invalid`]: `config.toml` parsing failed, missing `description`, or
///   `system.md` path out of bounds
pub fn discover_profiles(
    opts: &LoadConfigOptions,
) -> Result<BTreeMap<String, ProfileSpec>, ConfigError> {
    let mut profiles = BTreeMap::new();

    // User layer first, project layer second — later writes override earlier ones, so
    // project overrides user. `source` is tagged per layer for the profile's `[hooks]`
    // entries to record provenance (trust gating).
    if let Some(user_dir) = resolve_user_agents_dir(opts) {
        scan_agents_dir(&user_dir, ConfigSource::User, &mut profiles)?;
    }
    if let Some(repo_root) = find_repo_root(&opts.cwd) {
        scan_agents_dir(
            &repo_root.join(PROJECT_AGENTS_RELATIVE),
            ConfigSource::Project,
            &mut profiles,
        )?;
    }

    Ok(profiles)
}

/// Scan an `agents/` directory, parse each profile into a [`ProfileSpec`], and write it
/// into `out` (when names collide across layers, the current layer overwrites the
/// previous one — the caller passes layers in user→project order to implement "project
/// overrides user"). If the directory does not exist, this is a no-op.
///
/// Two profile forms coexist:
/// - **Folder**: a subdirectory containing `config.toml`; the name is the directory name,
///   and the system prompt comes from `[prompt] file` (default `system.md`).
/// - **Single file**: `<name>.md`; the name is the filename without extension, TOML
///   frontmatter is between `+++` delimiters, and the system prompt follows.
///
/// **Within the same layer**, two profiles with the same name (e.g. `reviewer/` and
/// `reviewer.md`) cause a hard error — avoid having two sources of truth for one name.
fn scan_agents_dir(
    agents_dir: &Path,
    source: ConfigSource,
    out: &mut BTreeMap<String, ProfileSpec>,
) -> Result<(), ConfigError> {
    let entries = match std::fs::read_dir(agents_dir) {
        Ok(entries) => entries,
        // The directory not existing is normal (the user hasn't created any profiles) —
        // not an error.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ConfigError::Io {
                path: agents_dir.to_path_buf(),
                source: BoxError::new(err),
            });
        }
    };

    // First collect into a local map for this layer to detect name collisions within it,
    // then merge the whole layer into `out`.
    let mut layer: BTreeMap<String, ProfileSpec> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|err| ConfigError::Io {
            path: agents_dir.to_path_buf(),
            source: BoxError::new(err),
        })?;
        let path = entry.path();

        let parsed = if path.is_dir() {
            let config_path = path.join("config.toml");
            if !config_path.is_file() {
                // Subdirectories without a config.toml are not profiles — skip silently.
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
                continue;
            };
            Some((name, parse_profile_folder(&path, &config_path, source)?))
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let Some(name) = path.file_stem().and_then(|n| n.to_str()).map(str::to_owned) else {
                continue;
            };
            Some((name, parse_profile_file(agents_dir, &path, source)?))
        } else {
            // Not a directory or `.md` file — skip.
            None
        };

        if let Some((name, mut spec)) = parsed {
            spec.name = name.clone();
            if layer.insert(name.clone(), spec).is_some() {
                return Err(ConfigError::Invalid {
                    path: agents_dir.to_path_buf(),
                    message: format!(
                        "duplicate subagent profile `{name}` in the same layer \
                         (a folder and a `.md` file cannot share a name)"
                    ),
                });
            }
        }
    }

    out.extend(layer);
    Ok(())
}

/// Assembles the parsed frontmatter/config and the already-obtained system prompt text
/// into a [`ProfileSpec`]. Shared by both the directory-based and single-file variants —
/// they differ only in where the system prompt text comes from; all other field mappings
/// are identical. The `name` is filled in uniformly by the caller in `scan_agents_dir`.
fn spec_from_cfg(
    dir: &Path,
    cfg: ProfileConfigToml,
    system_prompt_text: String,
    source: ConfigSource,
    config_path: &Path,
) -> Result<ProfileSpec, ConfigError> {
    let tool_allow = cfg
        .tools
        .and_then(|t| t.allow)
        .unwrap_or_else(|| DEFAULT_TOOL_ALLOW.iter().map(|s| s.to_string()).collect());
    // Converts the `[hooks]` section of a profile into a `HooksConfig`, where each hook
    // carries the `source` of the profile's layer. Misspelled event names or invalid
    // handler shapes hard-fail here, with errors pointing to the config file path.
    let hooks = profile_hooks_from_raw(cfg.hooks, source, config_path)?;
    let request_limit =
        resolve_request_limit(config_path, cfg.request_limit, cfg.request_limit_mode)?;
    // Model may be given at the root (`model = "…"`) or under `[default] model` (matching
    // the top-level config). Accept either, but not both.
    let default_model = cfg.default.and_then(|d| d.model);
    let model = match (cfg.model, default_model) {
        (Some(_), Some(_)) => {
            return Err(ConfigError::Invalid {
                path: config_path.to_path_buf(),
                message: "set the model either as root `model` or as `[default] model`, not both"
                    .into(),
            });
        }
        (root, default) => root.or(default),
    };
    Ok(ProfileSpec {
        name: String::new(), // Filled in by `scan_agents_dir`
        dir: dir.to_path_buf(),
        description: cfg.description,
        model,
        system_prompt_text,
        tool_allow,
        sampling: cfg.sampling.map(ProfileSamplingToml::into_params),
        inherit_project_prompt: cfg.inherit_project_prompt,
        request_limit,
        hooks,
    })
}

/// Parse a folder-based profile: read `config.toml`, then read the system prompt from the
/// file specified by `[prompt] file` (defaults to `system.md`, resolved relative to the
/// profile directory with sandbox confinement).
fn parse_profile_folder(
    dir: &Path,
    config_path: &Path,
    source: ConfigSource,
) -> Result<ProfileSpec, ConfigError> {
    let raw = std::fs::read_to_string(config_path).map_err(|err| ConfigError::Io {
        path: config_path.to_path_buf(),
        source: BoxError::new(err),
    })?;
    let cfg: ProfileConfigToml = toml::from_str(&raw).map_err(|err| ConfigError::Invalid {
        path: config_path.to_path_buf(),
        message: err.to_string(),
    })?;

    // System prompt source: inline `[prompt] text`, or `[prompt] file` (default
    // `system.md`). At most one may be set; `text` wins when present, `file` is read from
    // disk otherwise.
    let (inline_text, prompt_file) = match cfg.prompt.as_ref() {
        Some(p) => {
            if p.text.is_some() && p.file.is_some() {
                return Err(ConfigError::Invalid {
                    path: config_path.to_path_buf(),
                    message: "set `[prompt] text` or `[prompt] file`, not both".into(),
                });
            }
            (p.text.clone(), p.file.clone())
        }
        None => (None, None),
    };
    let system_prompt_text = if let Some(text) = inline_text {
        text
    } else {
        let prompt_file = prompt_file.unwrap_or_else(|| DEFAULT_PROMPT_FILE.to_string());
        let prompt_path = resolve_workspace_path(dir, Path::new(&prompt_file)).map_err(|err| {
            ConfigError::Invalid {
                path: config_path.to_path_buf(),
                message: format!("invalid `prompt.file` `{prompt_file}`: {err}"),
            }
        })?;
        std::fs::read_to_string(&prompt_path).map_err(|err| ConfigError::Io {
            path: prompt_path.clone(),
            source: BoxError::new(err),
        })?
    };

    spec_from_cfg(dir, cfg, system_prompt_text, source, config_path)
}

/// Parse a single-file profile: `<name>.md` with frontmatter (`+++` TOML or `---` YAML)
/// followed by the system prompt body. `dir` is the `agents/` directory containing the
/// `.md` file (a single-file profile has no extra resource files, so `[prompt] file` is
/// meaningless and causes a conflict if specified).
fn parse_profile_file(
    dir: &Path,
    file_path: &Path,
    source: ConfigSource,
) -> Result<ProfileSpec, ConfigError> {
    let raw = std::fs::read_to_string(file_path).map_err(|err| ConfigError::Io {
        path: file_path.to_path_buf(),
        source: BoxError::new(err),
    })?;
    let (kind, frontmatter, body) =
        split_frontmatter(&raw).ok_or_else(|| ConfigError::Invalid {
            path: file_path.to_path_buf(),
            message: "single-file profile must start with frontmatter delimited by `+++` (TOML) \
                      or `---` (YAML)"
                .into(),
        })?;

    let cfg: ProfileConfigToml =
        parse_frontmatter(kind, frontmatter).map_err(|message| ConfigError::Invalid {
            path: file_path.to_path_buf(),
            message,
        })?;

    if cfg.prompt.is_some() {
        return Err(ConfigError::Invalid {
            path: file_path.to_path_buf(),
            message: "single-file profile takes its system prompt from the body after the \
                      frontmatter; remove the `[prompt]` table"
                .into(),
        });
    }
    spec_from_cfg(dir, cfg, body.to_string(), source, file_path)
}

/// Parse the user-level `agents/` directory. Uses the same priority order as
/// [`crate::loader`]'s `resolve_user_config_path`, but returns `None` when not found (if
/// neither XDG nor HOME is set, the user-level profile is simply absent, unlike the main
/// config which would hard error).
fn resolve_user_agents_dir(opts: &LoadConfigOptions) -> Option<PathBuf> {
    // `--local`: ignore the user-level agents directory.
    if opts.local {
        return None;
    }
    if let Some(xdg) = &opts.xdg_config_home {
        return Some(xdg.join(USER_AGENTS_RELATIVE));
    }
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join(USER_AGENTS_RELATIVE));
    }
    if let Some(home) = &opts.home_dir {
        return Some(home.join(".config/defect/agents"));
    }
    if let Ok(home) = env::var("HOME") {
        return Some(PathBuf::from(home).join(".config/defect/agents"));
    }
    None
}

#[cfg(test)]
mod tests;
