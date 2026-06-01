//! Subagent profile 发现与解析。
//!
//! profile 被 `spawn_agent` 工具按名挑选，让父 agent 把任务委派给一个 fresh、
//! 隔离上下文的子 agent。设计见记忆 `project-subagent-design`。
//!
//! ## 两种格式（同一套字段，二选一）
//!
//! - **文件夹版**：`agents/<name>/`，含 `config.toml`（TOML 配置）+ 一个
//!   system prompt 文件（默认 `system.md`，由 `[prompt] file` 指定）。适合
//!   prompt 较长、想拆成独立文件的场景。
//! - **单文件版**：`agents/<name>.md`，frontmatter（`+++` ⇒ **TOML**，
//!   `---` ⇒ **YAML**，社区标准）之后正文即 system prompt。字段 schema 与
//!   `config.toml` 完全一致。适合一个文件搞定。单文件版不带额外资源文件，
//!   故 `[prompt]` 表在此**非法**。YAML 需 `yaml` feature（默认开）；关闭后
//!   `---` 头会以可操作错误 hard fail，`+++` 仍可用。
//!
//! 同一层内两种形态同名（`reviewer/` 与 `reviewer.md`）⇒ hard error——
//! 一个名字不允许两份真相源。
//!
//! ## 分层发现
//!
//! 与主配置同构（[`crate::loader`]）：
//! - 用户层 `<XDG_CONFIG_HOME>/defect/agents/`（或 `~/.config/defect/agents/`）
//! - 项目层 `<repo_root>/.defect/agents/`
//!
//! 跨层同名时**项目层覆盖用户层**。
//!
//! ## 沙箱
//!
//! `config.toml` 里 `[prompt] file` 等文件引用一律相对 profile 目录解析，
//! 并复用 [`defect_agent::fs::resolve_workspace_path`]（root 钉在 profile
//! 目录）阻断 `../` 越界与 symlink 越狱——这道沙箱守的是 **profile 自身的
//! 资源文件**，与子 agent 干活的工作区沙箱是两回事。

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::fs::resolve_workspace_path;
use defect_agent::llm::SamplingParams;
use serde::Deserialize;

use crate::frontmatter::{parse_frontmatter, split_frontmatter};
use crate::hooks::{HookEntryRaw, profile_hooks_from_raw};
use crate::loader::find_repo_root;
use crate::types::{ConfigError, ConfigSource, HooksConfig, LoadConfigOptions};

/// profile 项目层目录（相对 repo root）。对位 [`crate::types`] 的
/// `PROJECT_CONFIG_RELATIVE`（`.defect/config.toml`）。
const PROJECT_AGENTS_RELATIVE: &str = ".defect/agents";
/// profile 用户层目录（相对 XDG_CONFIG_HOME）。对位 `USER_CONFIG_RELATIVE`
/// （`defect/config.toml`）。
const USER_AGENTS_RELATIVE: &str = "defect/agents";
/// `[prompt] file` 缺省值——profile 目录下的 `system.md`。
const DEFAULT_PROMPT_FILE: &str = "system.md";
/// `[tools] allow` 缺省值：只读集。省略 allow 即得到一个只能读/搜的子 agent，
/// 靠"没有 mutating 工具"保证安全（工具白名单是主防线）。
const DEFAULT_TOOL_ALLOW: &[&str] = &["read_file", "search"];

/// 一个解析好的 subagent profile。
///
/// 由 [`discover_profiles`] 产出；`spawn_agent` 工具与 CLI 顶层 `--profile`
/// 都消费它。
#[derive(Debug, Clone)]
pub struct ProfileSpec {
    /// profile 名（= 目录名）。`spawn_agent` 的 `profile` enum 取值。
    pub name: String,
    /// profile 文件夹的绝对路径。
    pub dir: PathBuf,
    /// 选择期描述——`spawn_agent` 据此让 LLM 决定挑哪个 profile，也进工具
    /// description 的 catalog。必填。
    pub description: String,
    /// 可选 model 覆盖；省略 ⇒ 子 agent 回落到父会话当前选中的 model。
    /// 不单设 `provider`：model id 经 provider registry 的 `entry_for_model`
    /// 已唯一确定 provider，再加 provider 字段就是第二份真相源。需要指定某家
    /// provider 时，写一个该 provider 名下的 model id 即可。
    pub model: Option<String>,
    /// 已读好的 system prompt 文本（来自 `[prompt] file`）。
    pub system_prompt_text: String,
    /// 工具白名单——子 agent 只看得到这些工具。省略 ⇒ [`DEFAULT_TOOL_ALLOW`]。
    pub tool_allow: Vec<String>,
    /// 可选采样参数覆盖。
    pub sampling: Option<SamplingParams>,
    /// 该 profile 自己声明的 `[hooks]`——子 agent 跑 turn 时挂的钩子。
    ///
    /// 与"继承世界、不继承身份"原则一致：profile 的钩子是它身份的一部分，由
    /// profile 自己的 `config.toml` / frontmatter 声明，**不**从父会话继承。
    /// 每条带上 profile 所在层的 [`ConfigSource`]（项目层覆盖用户层时也随之
    /// 替换，因为整个 [`ProfileSpec`] 被覆盖）。省略 ⇒ 空（子 agent 无钩子）。
    pub hooks: HooksConfig,
}

/// `config.toml` 的原始反序列化形态。`deny_unknown_fields` 与主配置一致——
/// 未知键 hard fail（[[feedback-minimize-no-paternalistic-guards]]）。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileConfigToml {
    /// 必填——缺失即 serde 报 "missing field `description`"，被 [`discover_profiles`]
    /// 包成带文件路径的 hard error。
    description: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<ProfilePromptToml>,
    #[serde(default)]
    tools: Option<ProfileToolsToml>,
    #[serde(default)]
    sampling: Option<ProfileSamplingToml>,
    /// `[hooks]` 表：事件名 → 该事件下的 hook 条目数组。形态与主配置
    /// `[hooks]` 完全一致（复用 [`HookEntryRaw`]）。profile 是单一闭合真相源，
    /// 不支持跨层 `disable`——出现 `disable` 键会按未知事件名 hard fail。
    #[serde(default)]
    hooks: BTreeMap<String, Vec<HookEntryRaw>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilePromptToml {
    file: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileToolsToml {
    #[serde(default)]
    allow: Option<Vec<String>>,
}

/// 采样覆盖的子集——只暴露 v0 用得到的几个标量；映射到
/// [`SamplingParams`] 时叠在 `default()` 上，其余字段（thinking /
/// stop_sequences）保持默认。
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

/// 发现并解析所有可用 profile。
///
/// 先扫用户层、再扫项目层；同名 profile 项目层覆盖用户层。任一 profile 的
/// `config.toml` 解析失败 / `system.md` 越界或读不到，都是 hard error（fail
/// loud，不静默跳过坏 profile）。目录里非 profile 的杂项（无 `config.toml`
/// 的子目录、非目录项）静默跳过。
///
/// # Errors
/// - [`ConfigError::Io`]：读 `config.toml` / `system.md` 失败
/// - [`ConfigError::Invalid`]：`config.toml` 解析失败、缺 `description`、
///   或 `system.md` 路径越界
pub fn discover_profiles(
    opts: &LoadConfigOptions,
) -> Result<BTreeMap<String, ProfileSpec>, ConfigError> {
    let mut profiles = BTreeMap::new();

    // 用户层先，项目层后——后写覆盖先写，实现"项目覆盖用户"。source 随层标注，
    // 供 profile 的 `[hooks]` 条目记录来源（trust gating）。
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

/// 扫一个 `agents/` 目录，把其中每个 profile 解析成 [`ProfileSpec`] 写入
/// `out`（跨层同名时本层覆盖先前层——调用方按 用户→项目 顺序传入实现
/// "项目覆盖用户"）。目录不存在 ⇒ no-op。
///
/// 两种 profile 形态并存：
/// - **文件夹**：含 `config.toml` 的子目录，名 = 目录名，system prompt 来自
///   `[prompt] file`（默认 `system.md`）。
/// - **单文件**：`<name>.md`，名 = 文件名去扩展名，`+++` 之间是 TOML
///   frontmatter，其后是 system prompt 正文。
///
/// **同一层内**两个 profile 同名（如 `reviewer/` 与 `reviewer.md`）⇒ hard
/// error——避免一个名字两份真相源。
fn scan_agents_dir(
    agents_dir: &Path,
    source: ConfigSource,
    out: &mut BTreeMap<String, ProfileSpec>,
) -> Result<(), ConfigError> {
    let entries = match std::fs::read_dir(agents_dir) {
        Ok(entries) => entries,
        // 目录不存在是常态（用户没建任何 profile）——不是错误。
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ConfigError::Io {
                path: agents_dir.to_path_buf(),
                source: BoxError::new(err),
            });
        }
    };

    // 先收进本层局部 map，以便检测层内同名冲突；再整体并入 `out`。
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
                // 没有 config.toml 的子目录不是 profile——跳过，不报错。
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
            // 非目录、非 .md 文件——跳过。
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

/// 把解析好的 frontmatter/config + 已得到的 system prompt 文本组装成
/// [`ProfileSpec`]。文件夹版与单文件版共用——两种形态只在"system prompt
/// 文本从哪来"上不同，其余字段映射一致。`name` 由调用方在 `scan_agents_dir`
/// 统一回填。
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
    // profile 的 `[hooks]` → HooksConfig，每条带上 profile 所在层的 source。
    // 事件名拼错 / 非法 handler 形态在此 hard fail，路径定位到 config 文件。
    let hooks = profile_hooks_from_raw(cfg.hooks, source, config_path)?;
    Ok(ProfileSpec {
        name: String::new(), // 由 scan_agents_dir 回填
        dir: dir.to_path_buf(),
        description: cfg.description,
        model: cfg.model,
        system_prompt_text,
        tool_allow,
        sampling: cfg.sampling.map(ProfileSamplingToml::into_params),
        hooks,
    })
}

/// 解析文件夹版 profile：读 `config.toml`，再按 `[prompt] file`（默认
/// `system.md`，相对 profile 目录 + 沙箱守界）读 system prompt。
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

    let prompt_file = cfg
        .prompt
        .as_ref()
        .map(|p| p.file.clone())
        .unwrap_or_else(|| DEFAULT_PROMPT_FILE.to_string());
    let prompt_path = resolve_workspace_path(dir, Path::new(&prompt_file)).map_err(|err| {
        ConfigError::Invalid {
            path: config_path.to_path_buf(),
            message: format!("invalid `prompt.file` `{prompt_file}`: {err}"),
        }
    })?;
    let system_prompt_text =
        std::fs::read_to_string(&prompt_path).map_err(|err| ConfigError::Io {
            path: prompt_path.clone(),
            source: BoxError::new(err),
        })?;

    spec_from_cfg(dir, cfg, system_prompt_text, source, config_path)
}

/// 解析单文件版 profile：`<name>.md`，frontmatter（`+++` TOML 或 `---` YAML）
/// 之后正文即 system prompt。`dir` 取 `.md` 所在的 `agents/` 目录（profile 不带
/// 额外资源文件，故 `[prompt] file` 在单文件版**无意义**，写了即冲突报错）。
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

/// 解析用户层 `agents/` 目录。与 [`crate::loader`] 的
/// `resolve_user_config_path` 同源优先级，但**找不到时返回 `None`**（用户
/// 没设 XDG/HOME 时用户层 profile 直接缺席，不像主配置那样 hard error）。
fn resolve_user_agents_dir(opts: &LoadConfigOptions) -> Option<PathBuf> {
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
#[path = "profiles/test.rs"]
mod test;
