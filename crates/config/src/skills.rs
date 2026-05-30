//! Skill 发现与解析。
//!
//! Skill 是用户为 agent 配置的可复用提示片段——一段 markdown body，加上同目录
//! 下可选的 `scripts/` / `refs/` 资源文件。模型在需要时通过 `skill` 工具按名
//! 拉取 body 进上下文（progressive disclosure 的 L2）。设计见
//! `docs/internal/skills.md`。
//!
//! ## 文件形态（对齐 Anthropic / Codex 的 Agent Skills open standard）
//!
//! `<agents-or-skills-dir>/skills/<name>/SKILL.md`，frontmatter（`+++` ⇒ TOML，
//! `---` ⇒ YAML）之后正文即 skill body。skill 名 = 目录名。目录内可同级放
//! `scripts/` / `refs/` 子目录，模型用普通 `bash` / `read_file` 工具按需读取
//! （L3）——本模块只解析 `SKILL.md`，不扫描资源文件。
//!
//! 与 subagent profile（[`crate::profiles`]）共用 frontmatter 解析
//! （[`crate::frontmatter`]）与分层发现骨架，但**语义不同**：
//! - profile 是"派生一个隔离子 agent 执行任务"（`spawn_agent` 的 `task`）；
//! - skill 是"把一段说明注入当前对话"（`skill` 工具的 `name`）。
//!
//! ## 分层发现
//!
//! 与主配置 / profile 同构：
//! - 用户层 `<XDG_CONFIG_HOME>/defect/skills/`（或 `~/.config/defect/skills/`）
//! - 项目层 `<repo_root>/.defect/skills/`
//!
//! 跨层同名时**项目层覆盖用户层**（整体替换，不 merge——body 是不可分的
//! markdown，按字段合并没有自然语义）。

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use serde::Deserialize;

use crate::frontmatter::{parse_frontmatter, split_frontmatter};
use crate::loader::find_repo_root;
use crate::types::{ConfigError, LoadConfigOptions};

/// skill 项目层目录（相对 repo root）。对位 [`crate::profiles`] 的
/// `PROJECT_AGENTS_RELATIVE`（`.defect/agents`）。
const PROJECT_SKILLS_RELATIVE: &str = ".defect/skills";
/// skill 用户层目录（相对 XDG_CONFIG_HOME）。
const USER_SKILLS_RELATIVE: &str = "defect/skills";
/// 每个 skill 目录里必有的清单文件名（对齐 Anthropic / Codex open standard）。
const SKILL_MANIFEST_FILE: &str = "SKILL.md";
/// skill `description` 建议长度上限——超出仅 warn，不截断（进 L1 清单的成本
/// 控制，参考 Anthropic 的实践）。
const DESCRIPTION_SOFT_LIMIT: usize = 200;

/// 一个解析好的 skill。
///
/// 由 [`discover_skills`] 产出；`skill` 工具消费它——`name` / `description` 进
/// 工具 schema 的清单，`body` 在模型按名拉取时作为 tool result，`dir` 让模型
/// 知道资源文件（`scripts/` / `refs/`）的绝对路径根。
#[derive(Debug, Clone)]
pub struct SkillSpec {
    /// skill 名（= 目录名）。`skill` 工具的 `name` enum 取值。
    pub name: String,
    /// skill 目录的绝对路径，供 `skill` 工具回填给模型拼资源文件路径。
    pub dir: PathBuf,
    /// 选择期描述——进 L1 清单让模型决定是否加载。必填。
    pub description: String,
    /// `SKILL.md` 去 frontmatter 后的 body 全文（L2 加载时的内容）。
    pub body: String,
}

/// `SKILL.md` frontmatter 的原始反序列化形态。
///
/// 保留 `deny_unknown_fields`（与 [`crate::profiles`] 一致）抓必填项拼写错
/// （`naem` / `desciption` 这类 typo 不会被静默放过）；同时把 Agent Skills
/// open-standard 的 `always` / `triggers` / `allowed_tools` 几个字段**显式占位**
/// ——v0 解析但不消费（见 `docs/internal/skills.md` §3.2 / §4.3）。
///
/// 占位（而非 deny / 而非完全放开）的取舍：deny 会把"用户已写好的
/// Anthropic / Codex 格式 skill 扔进来就能用"（§2.1）这个卖点废掉；完全放开
/// 又丢了 typo 保护。显式列出文档已承诺的字段两头兼顾——v1 接入时这些字段从
/// "被忽略"变"被消费"，对用户文件向后兼容。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillManifestToml {
    /// 必填，且必须与目录名一致——清单展示与 `skill` 工具入参用同一个名字，
    /// 不一致会让模型按清单里的名字调用却找不到。
    name: String,
    /// 必填——进 L1 清单。缺失即 serde 报 "missing field `description`"，被
    /// [`discover_skills`] 包成带文件路径的 hard error。
    description: String,
    /// 占位（§5.1）：v1 接入后 `true` 表示该 skill 的 body 直接拼进 system
    /// prompt（always-on），v0 解析后不消费。
    #[serde(default)]
    #[allow(dead_code, reason = "open-standard 占位字段，v0 解析但不消费")]
    always: Option<bool>,
    /// 占位（§4.3）：按文件 glob / prompt 关键字自动激活，v0 解析后不消费。
    #[serde(default)]
    #[allow(dead_code, reason = "open-standard 占位字段，v0 解析但不消费")]
    triggers: Option<SkillTriggersToml>,
    /// 占位：v1 用于让 ACP 客户端做 tool gating（参考 Anthropic `allowed-tools`，
    /// 故同时接受连字符写法）。v0 解析后不消费。
    #[serde(default, alias = "allowed-tools")]
    #[allow(dead_code, reason = "open-standard 占位字段，v0 解析但不消费")]
    allowed_tools: Option<Vec<String>>,
}

/// `[triggers]` 子表的占位形态——v0 解析但不消费（见 [`SkillManifestToml`]）。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillTriggersToml {
    #[serde(default)]
    #[allow(dead_code, reason = "open-standard 占位字段，v0 解析但不消费")]
    globs: Vec<String>,
    #[serde(default)]
    #[allow(dead_code, reason = "open-standard 占位字段，v0 解析但不消费")]
    keywords: Vec<String>,
}

/// 发现并解析所有可用 skill。
///
/// 先扫用户层、再扫项目层；同名 skill 项目层覆盖用户层。任一 skill 的
/// `SKILL.md` 解析失败 / frontmatter 缺失 / `name` 与目录名不符，都是 hard
/// error（fail loud，不静默跳过坏 skill——与 [`crate::profiles`] 同款，区别于
/// 旧设计稿的 warn-and-skip）。目录里非 skill 的杂项（无 `SKILL.md` 的子目录、
/// 非目录项）静默跳过。
///
/// # Errors
/// - [`ConfigError::Io`]：读 `SKILL.md` 失败
/// - [`ConfigError::Invalid`]：`SKILL.md` 缺 frontmatter、解析失败、缺
///   `name` / `description`、或 `name` ≠ 目录名
pub fn discover_skills(
    opts: &LoadConfigOptions,
) -> Result<BTreeMap<String, SkillSpec>, ConfigError> {
    let mut skills = BTreeMap::new();

    // 用户层先，项目层后——后写覆盖先写，实现"项目覆盖用户"。
    if let Some(user_dir) = resolve_user_skills_dir(opts) {
        scan_skills_dir(&user_dir, &mut skills)?;
    }
    if let Some(repo_root) = find_repo_root(&opts.cwd) {
        scan_skills_dir(&repo_root.join(PROJECT_SKILLS_RELATIVE), &mut skills)?;
    }

    Ok(skills)
}

/// 扫一个 `skills/` 目录，把其中每个 skill 解析成 [`SkillSpec`] 写入 `out`
/// （跨层同名时本层覆盖先前层——调用方按 用户→项目 顺序传入实现"项目覆盖
/// 用户"）。目录不存在 ⇒ no-op。
fn scan_skills_dir(
    skills_dir: &Path,
    out: &mut BTreeMap<String, SkillSpec>,
) -> Result<(), ConfigError> {
    let entries = match std::fs::read_dir(skills_dir) {
        Ok(entries) => entries,
        // 目录不存在是常态（用户没建任何 skill）——不是错误。
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(ConfigError::Io {
                path: skills_dir.to_path_buf(),
                source: BoxError::new(err),
            });
        }
    };

    for entry in entries {
        let entry = entry.map_err(|err| ConfigError::Io {
            path: skills_dir.to_path_buf(),
            source: BoxError::new(err),
        })?;
        let path = entry.path();
        if !path.is_dir() {
            // skill 只走 dir-per-skill 形态——非目录项跳过。
            continue;
        }
        let manifest_path = path.join(SKILL_MANIFEST_FILE);
        if !manifest_path.is_file() {
            // 没有 SKILL.md 的子目录不是 skill——跳过，不报错。
            continue;
        }
        let Some(dir_name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        let spec = parse_skill(&path, &manifest_path, &dir_name)?;
        out.insert(dir_name, spec);
    }

    Ok(())
}

/// 解析一个 skill 目录：读 `SKILL.md`，切 frontmatter，校验 `name` 与目录名
/// 一致，body 即 frontmatter 之后的正文。
fn parse_skill(dir: &Path, manifest_path: &Path, dir_name: &str) -> Result<SkillSpec, ConfigError> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|err| ConfigError::Io {
        path: manifest_path.to_path_buf(),
        source: BoxError::new(err),
    })?;
    let (kind, frontmatter, body) =
        split_frontmatter(&raw).ok_or_else(|| ConfigError::Invalid {
            path: manifest_path.to_path_buf(),
            message: "SKILL.md must start with frontmatter delimited by `+++` (TOML) or `---` \
                      (YAML)"
                .into(),
        })?;

    let manifest: SkillManifestToml =
        parse_frontmatter(kind, frontmatter).map_err(|message| ConfigError::Invalid {
            path: manifest_path.to_path_buf(),
            message,
        })?;

    if manifest.name != dir_name {
        return Err(ConfigError::Invalid {
            path: manifest_path.to_path_buf(),
            message: format!(
                "skill `name` (`{}`) must match its directory name (`{dir_name}`)",
                manifest.name
            ),
        });
    }

    if manifest.description.len() > DESCRIPTION_SOFT_LIMIT {
        tracing::warn!(
            skill = %dir_name,
            len = manifest.description.len(),
            limit = DESCRIPTION_SOFT_LIMIT,
            "skill description exceeds the soft length limit; it inflates the L1 manifest budget",
        );
    }

    Ok(SkillSpec {
        name: manifest.name,
        dir: dir.to_path_buf(),
        description: manifest.description,
        body: body.to_string(),
    })
}

/// 解析用户层 `skills/` 目录。与 [`crate::profiles`] 的 `resolve_user_agents_dir`
/// 同源优先级（XDG_CONFIG_HOME → HOME/.config）；找不到时返回 `None`（用户没设
/// XDG/HOME 时用户层 skill 直接缺席，不 hard error）。
fn resolve_user_skills_dir(opts: &LoadConfigOptions) -> Option<PathBuf> {
    if let Some(xdg) = &opts.xdg_config_home {
        return Some(xdg.join(USER_SKILLS_RELATIVE));
    }
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join(USER_SKILLS_RELATIVE));
    }
    if let Some(home) = &opts.home_dir {
        return Some(home.join(".config/defect/skills"));
    }
    if let Ok(home) = env::var("HOME") {
        return Some(PathBuf::from(home).join(".config/defect/skills"));
    }
    None
}

#[cfg(test)]
#[path = "skills/test.rs"]
mod test;
