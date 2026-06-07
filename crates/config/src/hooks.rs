//! `[hooks]` 段解析。
//!
//! Hook configuration types.
//!
//! ## 形态
//!
//! `[hooks]` 是一张表，键是挂载点的 `event_name`（snake_case，如 `before_turn_end`），值是该事件下
//! 的 hook 条目数组。事件名的合法集合是 `defect_agent::hooks::step::ALL_EVENT_NAMES`——拼错/未知的
//! 事件键 **hard fail**（不静默丢弃）。特殊键 `disable` 不是事件，是禁用指令数组。
//!
//! ## 为什么 hooks 不走 `ConfigToml::try_into`
//!
//! 其它段一律先 `merge_toml_values` 把所有 layer 拍平成一份 TOML，再 decode。
//! 但 hooks 数组的合并语义是 **append + dedupe**（§5.4）——TOML 默认数组覆盖
//! 会让 project-local 静默移除上游 hook，等同 claude-code 的 issue #106。
//! 因此 hooks 在 layer 阶段就要按数组 append 合并，且每条 hook 要保留来源
//! [`ConfigSource`] 供 trust gating 使用。
//!
//! 此模块的入口是 [`parse_layer_hooks`]：从一个 layer 的原始 [`toml::Value`] 抽出当前层声明的
//! hooks，附上 source 标签返回。跨层合并（append + dedupe + apply disable）在 [`crate::loader`]。

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

/// 每个事件桶 -> 该桶下声明的条目。
#[derive(Debug, Clone, Default)]
pub(crate) struct LayerHooks {
    pub(crate) entries: HooksConfig,
    /// 该层声明的 disable 条目。disable 不限定来源层，按 (event, matcher,
    /// handler) 三元组在合并阶段从累积结果里移除。
    pub(crate) disables: Vec<HookDisable>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HookDisable {
    pub(crate) event: String,
    pub(crate) matcher: HookMatcher,
    pub(crate) handler: HookHandlerSpec,
}

/// 从一层的原始 TOML 抽出该层声明的 hook 条目与 disable 指令。
///
/// 入参 `value` 应当是该层（user / project / project-local / cli）单独的
/// 顶层 [`TomlValue::Table`]——**不是** layer 间已合并的结果。
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

        // 事件键：必须是已知挂载点，否则 hard fail（不静默丢弃拼错的键）。
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

/// 把一个 subagent profile 的 `[hooks]` 表（已被 serde 解析成事件名 → 原始条目
/// 数组）转成 [`HooksConfig`]，每条带上 profile 所在层的 `source`。
///
/// 与 [`parse_layer_hooks`] 的差别：profile 是**单一闭合真相源**——没有上游可
/// append / dedupe / disable，故不支持 `disable` 键，也不跨层合并。事件名仍按
/// `ALL_EVENT_NAMES` 校验，拼错 hard fail（不静默丢弃）。`path` 仅供报错定位。
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

/// 把多层 [`LayerHooks`] 合并成最终 [`HooksConfig`]：
/// - 各事件桶按声明顺序 append（user → project → project-local → cli）
/// - (matcher, handler) 完全相同的连续条目去重——保留首次出现
/// - 应用所有 [`HookDisable`]：从对应桶里删掉所有 (matcher, handler) 匹配的条目，无论来自哪层
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

// ---------------------------------------------------------------------------
// Raw deserialization shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HookEntryRaw {
    /// 可选展示名（tracing / 可观测性用）。
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
