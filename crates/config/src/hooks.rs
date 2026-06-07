//! Parsing of the `[hooks]` section.
//!
//! Hook configuration types.
//!
//! ## Shape
//!
//! `[hooks]` is a table whose keys are mount-point `event_name`s (snake_case, e.g.
//! `before_turn_end`) and whose values are arrays of hook entries for that event. The set
//! of valid event names is `defect_agent::hooks::step::ALL_EVENT_NAMES` — misspelled or
//! unknown event keys **hard fail** (they are not silently dropped). The special key
//! `disable` is not an event; it is an array of disable directives.
//!
//! ## Why hooks do not go through `ConfigToml::try_into`
//!
//! All other sections first flatten every layer into a single TOML via
//! `merge_toml_values`, then decode. But the merge semantics for hook arrays are **append
//! + dedupe** — TOML's default array overwrite would let a project-local layer
//! silently remove upstream hooks, mirroring claude-code issue #106. Therefore hooks must
//! be merged by appending arrays at the layer stage, and each hook must retain its source
//! [`ConfigSource`] for trust gating.
//!
//! The entry point of this module is [`parse_layer_hooks`]: it extracts the hooks
//! declared in a single layer from the raw [`toml::Value`] and returns them tagged with a
//! source label. Cross-layer merging (append + dedupe + apply disable) happens in
//! [`crate::loader`].

use std::collections::BTreeMap;
use std::path::PathBuf;

use defect_agent::hooks::step::is_known_event;
use defect_agent::tool::SafetyClass;
use serde::Deserialize;
use toml::Value as TomlValue;

use crate::types::{
    ConfigError, ConfigSource, HookCommandSpec, HookEntry, HookHandlerSpec, HookMatcher,
    HookPromptRender, HookPromptSpec, HookShellKind, HooksConfig,
};

/// Maps each event bucket to the entries declared under it.
#[derive(Debug, Clone, Default)]
pub(crate) struct LayerHooks {
    pub(crate) entries: HooksConfig,
    /// Disable entries declared by this layer. Disables are not restricted to the
    /// declaring layer; during merging, entries matching the (event, matcher, handler)
    /// triple are removed from the accumulated result.
    pub(crate) disables: Vec<HookDisable>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HookDisable {
    pub(crate) event: String,
    pub(crate) matcher: HookMatcher,
    pub(crate) handler: HookHandlerSpec,
}

/// Extract the hook entries and disable directives declared in a single layer from its
/// raw TOML.
///
/// The `value` parameter must be the top-level [`TomlValue::Table`] for that layer (user
/// / project / project-local / cli) — **not** the result of merging layers.
pub(crate) fn parse_layer_hooks(
    path: PathBuf,
    source: ConfigSource,
    value: &TomlValue,
) -> Result<LayerHooks, ConfigError> {
    let Some(hooks_value) = value.get("hooks") else {
        return Ok(LayerHooks::default());
    };
    let table = hooks_value.as_table().ok_or_else(|| ConfigError::Invalid {
        path: path.clone(),
        message: "[hooks] must be a table of event-name → entries".to_string(),
    })?;

    let mut entries = HooksConfig::default();
    let mut disables = Vec::new();

    for (key, raw) in table {
        if key == "disable" {
            let raw_list: Vec<HookDisableRaw> =
                raw.clone()
                    .try_into()
                    .map_err(|err: toml::de::Error| ConfigError::Invalid {
                        path: path.clone(),
                        message: format!("invalid [[hooks.disable]]: {err}"),
                    })?;
            for d in raw_list {
                if !is_known_event(&d.event) {
                    return Err(ConfigError::Invalid {
                        path: path.clone(),
                        message: format!(
                            "[[hooks.disable]].event = {:?} is not a known hook event name",
                            d.event
                        ),
                    });
                }
                disables.push(HookDisable {
                    event: d.event,
                    matcher: d.matcher.into_typed(),
                    handler: d.handler.into_typed(&path)?,
                });
            }
            continue;
        }

        // The event key must be a known hook point; otherwise hard-fail (do not silently
        // drop misspelled keys).
        if !is_known_event(key) {
            return Err(ConfigError::Invalid {
                path: path.clone(),
                message: format!("[[hooks.{key}]] is not a known hook event name"),
            });
        }
        let raw_list: Vec<HookEntryRaw> =
            raw.clone()
                .try_into()
                .map_err(|err: toml::de::Error| ConfigError::Invalid {
                    path: path.clone(),
                    message: format!("invalid [[hooks.{key}]]: {err}"),
                })?;
        for r in raw_list {
            entries.push(
                key.clone(),
                HookEntry {
                    name: r.name,
                    matcher: r.matcher.into_typed(),
                    handler: r.handler.into_typed(&path)?,
                    source,
                },
            );
        }
    }

    Ok(LayerHooks { entries, disables })
}

/// Converts a subagent profile's `[hooks]` table (already parsed by serde into event-name
/// → raw-entry arrays) into a [`HooksConfig`], attaching the profile's layer `source` to
/// each entry.
///
/// Unlike [`parse_layer_hooks`], a profile is a **single closed truth source** — there is
/// no upstream to append to, deduplicate against, or disable. Therefore the `disable` key
/// is unsupported and no cross-layer merging occurs. Event names are still validated
/// against `ALL_EVENT_NAMES`; a misspelled name is a hard failure (not silently dropped).
/// `path` is used only for error reporting.
pub(crate) fn profile_hooks_from_raw(
    raw: BTreeMap<String, Vec<HookEntryRaw>>,
    source: ConfigSource,
    path: &std::path::Path,
) -> Result<HooksConfig, ConfigError> {
    let mut entries = HooksConfig::default();
    for (key, list) in raw {
        if !is_known_event(&key) {
            return Err(ConfigError::Invalid {
                path: path.to_path_buf(),
                message: format!("[[hooks.{key}]] is not a known hook event name"),
            });
        }
        for r in list {
            entries.push(
                key.clone(),
                HookEntry {
                    name: r.name,
                    matcher: r.matcher.into_typed(),
                    handler: r.handler.into_typed(path)?,
                    source,
                },
            );
        }
    }
    Ok(entries)
}

/// Merge multiple [`LayerHooks`] into a final [`HooksConfig`]:
/// - Append event buckets in declaration order (user → project → project-local → cli)
/// - Deduplicate consecutive entries with identical (matcher, handler) pairs, keeping the
///   first occurrence
/// - Apply all [`HookDisable`] entries: remove any (matcher, handler) match from the
///   corresponding bucket, regardless of which layer it came from
pub(crate) fn merge_layer_hooks(layers: Vec<LayerHooks>) -> HooksConfig {
    let mut merged = HooksConfig::default();
    let mut disables: Vec<HookDisable> = Vec::new();

    for layer in layers {
        for (event, list) in layer.entries.buckets {
            merged.buckets.entry(event).or_default().extend(list);
        }
        disables.extend(layer.disables);
    }

    for bucket in merged.buckets.values_mut() {
        dedupe_in_place(bucket);
    }

    for disable in disables {
        if let Some(bucket) = merged.buckets.get_mut(&disable.event) {
            bucket.retain(|entry| {
                !(entry.matcher == disable.matcher && entry.handler == disable.handler)
            });
        }
    }

    merged
}

fn dedupe_in_place(entries: &mut Vec<HookEntry>) {
    let mut i = 0;
    while i < entries.len() {
        let mut j = i + 1;
        while j < entries.len() {
            let dup = match (entries.get(i), entries.get(j)) {
                (Some(a), Some(b)) => a.matcher == b.matcher && a.handler == b.handler,
                _ => false,
            };
            if dup {
                entries.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

// Raw deserialization shapes

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HookEntryRaw {
    /// Optional display name (for tracing / observability).
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    #[serde(rename = "match")]
    matcher: HookMatcherRaw,
    handler: HookHandlerRaw,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HookDisableRaw {
    event: String,
    #[serde(default)]
    #[serde(rename = "match")]
    matcher: HookMatcherRaw,
    handler: HookHandlerRaw,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct HookMatcherRaw {
    tool: Option<String>,
    tool_glob: Option<String>,
    safety: Option<Vec<SafetyClass>>,
}

impl HookMatcherRaw {
    fn into_typed(self) -> HookMatcher {
        HookMatcher {
            tool: self.tool,
            tool_glob: self.tool_glob,
            safety: self.safety.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HookHandlerRaw {
    Builtin {
        name: String,
    },
    Command {
        argv: Option<Vec<String>>,
        argv_windows: Option<Vec<String>>,
        shell: Option<HookShellRaw>,
        command: Option<String>,
        cwd: Option<PathBuf>,
        env: Option<BTreeMap<String, String>>,
        timeout_sec: Option<u64>,
    },
    Prompt {
        model: Option<String>,
        system: String,
        render: HookPromptRenderRaw,
        timeout_sec: Option<u64>,
    },
}

impl HookHandlerRaw {
    fn into_typed(self, path: &std::path::Path) -> Result<HookHandlerSpec, ConfigError> {
        let invalid = |message: String| ConfigError::Invalid {
            path: path.to_path_buf(),
            message,
        };
        match self {
            Self::Builtin { name } => Ok(HookHandlerSpec::Builtin { name }),
            Self::Command {
                argv,
                argv_windows,
                shell,
                command,
                cwd,
                env,
                timeout_sec,
            } => {
                let env = env.unwrap_or_default();
                let spec = match (argv, shell, command) {
                    (Some(argv), None, None) => {
                        if argv.is_empty() {
                            return Err(invalid("command handler `argv` must not be empty".into()));
                        }
                        HookCommandSpec::Argv {
                            argv,
                            argv_windows,
                            cwd,
                            env,
                            timeout_sec,
                        }
                    }
                    (None, Some(shell), Some(command)) => {
                        if argv_windows.is_some() {
                            return Err(invalid(
                                "`argv_windows` is only valid for argv-form command handlers"
                                    .into(),
                            ));
                        }
                        HookCommandSpec::Shell {
                            shell: shell.into_typed(),
                            command,
                            cwd,
                            env,
                            timeout_sec,
                        }
                    }
                    (None, Some(_), None) => {
                        return Err(invalid(
                            "command handler with `shell` set requires `command`".into(),
                        ));
                    }
                    (None, None, _) => {
                        return Err(invalid(
                            "command handler requires either `argv` or (`shell` + `command`)"
                                .into(),
                        ));
                    }
                    (Some(_), Some(_), _) | (Some(_), None, Some(_)) => {
                        return Err(invalid(
                            "command handler must not mix `argv` with `shell`/`command`".into(),
                        ));
                    }
                };
                Ok(HookHandlerSpec::Command(spec))
            }
            Self::Prompt {
                model,
                system,
                render,
                timeout_sec,
            } => Ok(HookHandlerSpec::Prompt(HookPromptSpec {
                model,
                system,
                render: render.into_typed(),
                timeout_sec,
            })),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HookShellRaw {
    Sh,
    Bash,
    Pwsh,
    Cmd,
    /// `{ type = "custom", program = "...", args = [...] }`
    #[serde(untagged)]
    Custom(HookShellCustomRaw),
}

#[derive(Debug, Deserialize)]
struct HookShellCustomRaw {
    program: String,
    #[serde(default)]
    args: Vec<String>,
}

impl HookShellRaw {
    fn into_typed(self) -> HookShellKind {
        match self {
            Self::Sh => HookShellKind::Sh,
            Self::Bash => HookShellKind::Bash,
            Self::Pwsh => HookShellKind::Pwsh,
            Self::Cmd => HookShellKind::Cmd,
            Self::Custom(raw) => HookShellKind::Custom {
                program: raw.program,
                args: raw.args,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HookPromptRenderRaw {
    Json,
    Template { template: String },
}

impl HookPromptRenderRaw {
    fn into_typed(self) -> HookPromptRender {
        match self {
            Self::Json => HookPromptRender::Json,
            Self::Template { template } => HookPromptRender::Template { template },
        }
    }
}
