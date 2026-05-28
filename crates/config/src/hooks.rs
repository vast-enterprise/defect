//! `[hooks]` 段解析。
//!
//! 详见 `docs/internal/hooks.md` §5。
//!
//! ## 为什么 hooks 不走 `ConfigToml::try_into`
//!
//! 其它段一律先 `merge_toml_values` 把所有 layer 拍平成一份 TOML，再 decode。
//! 但 hooks 数组的合并语义是 **append + dedupe**（§5.4）——TOML 默认数组覆盖
//! 会让 project-local 静默移除上游 hook，等同 claude-code 的 issue #106。
//! 因此 hooks 在 layer 阶段就要按数组 append 合并，且每条 hook 要保留来源
//! [`ConfigSource`] 供 Phase G 的 trust gating 使用。
//!
//! 此模块的入口是 [`parse_layer_hooks`]：从一个 layer 的原始
//! [`toml::Value`] 抽出当前层声明的 hooks，附上 source 标签返回。
//! 跨层合并逻辑（append + dedupe + apply disable）在 [`crate::loader`] 里。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    pub(crate) event: HookEventTag,
    pub(crate) matcher: HookMatcher,
    pub(crate) handler: HookHandlerSpec,
}

/// 事件标签——内部用，1:1 对应 `HooksConfig` 五个字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HookEventTag {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
}

impl HookEventTag {
    pub(crate) fn from_key(key: &str) -> Option<Self> {
        Some(match key {
            "session_start" => Self::SessionStart,
            "user_prompt_submit" => Self::UserPromptSubmit,
            "pre_tool_use" => Self::PreToolUse,
            "post_tool_use" => Self::PostToolUse,
            "post_tool_use_failure" => Self::PostToolUseFailure,
            _ => return None,
        })
    }
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
    let raw: HooksSection = hooks_value
        .clone()
        .try_into()
        .map_err(|err: toml::de::Error| ConfigError::Invalid {
            path: path.clone(),
            message: format!("invalid [hooks] section: {err}"),
        })?;

    let mut entries = HooksConfig::default();
    let attach = |dst: &mut Vec<HookEntry>,
                  raw_list: Vec<HookEntryRaw>,
                  bucket: HookEventTag,
                  path: &PathBuf|
     -> Result<(), ConfigError> {
        for raw in raw_list {
            dst.push(HookEntry {
                matcher: raw.matcher.into_typed(),
                handler: raw.handler.into_typed(bucket, path)?,
                source,
            });
        }
        Ok(())
    };

    attach(
        &mut entries.session_start,
        raw.session_start.unwrap_or_default(),
        HookEventTag::SessionStart,
        &path,
    )?;
    attach(
        &mut entries.user_prompt_submit,
        raw.user_prompt_submit.unwrap_or_default(),
        HookEventTag::UserPromptSubmit,
        &path,
    )?;
    attach(
        &mut entries.pre_tool_use,
        raw.pre_tool_use.unwrap_or_default(),
        HookEventTag::PreToolUse,
        &path,
    )?;
    attach(
        &mut entries.post_tool_use,
        raw.post_tool_use.unwrap_or_default(),
        HookEventTag::PostToolUse,
        &path,
    )?;
    attach(
        &mut entries.post_tool_use_failure,
        raw.post_tool_use_failure.unwrap_or_default(),
        HookEventTag::PostToolUseFailure,
        &path,
    )?;

    let disables = raw
        .disable
        .unwrap_or_default()
        .into_iter()
        .map(|raw| {
            let event = HookEventTag::from_key(&raw.event).ok_or_else(|| ConfigError::Invalid {
                path: path.clone(),
                message: format!(
                    "[[hooks.disable]].event = {:?} is not a known event name",
                    raw.event
                ),
            })?;
            Ok(HookDisable {
                event,
                matcher: raw.matcher.into_typed(),
                handler: raw.handler.into_typed(event, &path)?,
            })
        })
        .collect::<Result<Vec<_>, ConfigError>>()?;

    Ok(LayerHooks { entries, disables })
}

/// 把多层 [`LayerHooks`] 合并成最终 [`HooksConfig`]：
/// - 各层数组按声明顺序 append（user → project → project-local → cli）
/// - (matcher, handler) 完全相同的连续条目去重——保留首次出现
/// - 应用所有 [`HookDisable`]：从对应桶里删掉所有 (matcher, handler) 三元组
///   匹配的条目，无论这条 hook 来自哪一层
pub(crate) fn merge_layer_hooks(layers: Vec<LayerHooks>) -> HooksConfig {
    let mut merged = HooksConfig::default();
    let mut disables: Vec<HookDisable> = Vec::new();

    for layer in layers {
        merged.session_start.extend(layer.entries.session_start);
        merged
            .user_prompt_submit
            .extend(layer.entries.user_prompt_submit);
        merged.pre_tool_use.extend(layer.entries.pre_tool_use);
        merged.post_tool_use.extend(layer.entries.post_tool_use);
        merged
            .post_tool_use_failure
            .extend(layer.entries.post_tool_use_failure);
        disables.extend(layer.disables);
    }

    dedupe_in_place(&mut merged.session_start);
    dedupe_in_place(&mut merged.user_prompt_submit);
    dedupe_in_place(&mut merged.pre_tool_use);
    dedupe_in_place(&mut merged.post_tool_use);
    dedupe_in_place(&mut merged.post_tool_use_failure);

    for disable in disables {
        let bucket = match disable.event {
            HookEventTag::SessionStart => &mut merged.session_start,
            HookEventTag::UserPromptSubmit => &mut merged.user_prompt_submit,
            HookEventTag::PreToolUse => &mut merged.pre_tool_use,
            HookEventTag::PostToolUse => &mut merged.post_tool_use,
            HookEventTag::PostToolUseFailure => &mut merged.post_tool_use_failure,
        };
        bucket.retain(|entry| {
            !(entry.matcher == disable.matcher && entry.handler == disable.handler)
        });
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

#[derive(Debug, Default, Deserialize)]
struct HooksSection {
    session_start: Option<Vec<HookEntryRaw>>,
    user_prompt_submit: Option<Vec<HookEntryRaw>>,
    pre_tool_use: Option<Vec<HookEntryRaw>>,
    post_tool_use: Option<Vec<HookEntryRaw>>,
    post_tool_use_failure: Option<Vec<HookEntryRaw>>,
    disable: Option<Vec<HookDisableRaw>>,
}

#[derive(Debug, Deserialize)]
struct HookEntryRaw {
    #[serde(default)]
    #[serde(rename = "match")]
    matcher: HookMatcherRaw,
    handler: HookHandlerRaw,
}

#[derive(Debug, Deserialize)]
struct HookDisableRaw {
    event: String,
    #[serde(default)]
    #[serde(rename = "match")]
    matcher: HookMatcherRaw,
    handler: HookHandlerRaw,
}

#[derive(Debug, Default, Deserialize)]
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
    fn into_typed(self, event: HookEventTag, path: &Path) -> Result<HookHandlerSpec, ConfigError> {
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
            } => {
                // §4.3.2：所有 5 件套都允许 prompt handler；引擎不在配置层做
                // 「不要把 prompt 挂到 PreToolUse」之类的策略校验。
                let _ = event;
                Ok(HookHandlerSpec::Prompt(HookPromptSpec {
                    model,
                    system,
                    render: render.into_typed(),
                    timeout_sec,
                }))
            }
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
