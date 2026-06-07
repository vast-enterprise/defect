//! Skill discovery and parsing.
//!
//! Skills are reusable prompt fragments that users configure for an agent: a markdown
//! body, plus optional `scripts/` / `refs/` resource files in the same directory. When
//! needed, the model pulls a skill's body into context by name via the `skill` tool
//! (progressive disclosure L2). See Skill configuration types for the design.
//!
//! ## File layout (aligned with the Anthropic / Codex Agent Skills open standard)
//!
//! `<agents-or-skills-dir>/skills/<name>/SKILL.md` — the skill body follows the
//! frontmatter (`+++` ⇒ TOML, `---` ⇒ YAML). The skill name is the directory name. The
//! directory may contain sibling `scripts/` / `refs/` subdirectories; the model reads
//! them on demand using ordinary `bash` / `read_file` tools (L3). This module only parses
//! `SKILL.md` and does not scan resource files.
//!
//! Shares frontmatter parsing ([`crate::frontmatter`]) and the layered discovery skeleton
//! with subagent profiles ([`crate::profiles`]), but the **semantics differ**:
//! - A profile "spawns an isolated sub-agent to execute a task" (`spawn_agent`'s `task`);
//! - A skill "injects instructions into the current conversation" (`skill` tool's
//!   `name`).
//!
//! ## Layered discovery
//!
//! Same structure as the main config / profiles:
//! - User layer: `<XDG_CONFIG_HOME>/defect/skills/` (or `~/.config/defect/skills/`)
//! - Project layer: `<repo_root>/.defect/skills/`
//!
//! When the same name exists in both layers, the **project layer overrides the user
//! layer** (full replacement, not merged — the body is an indivisible markdown block, so
//! field-level merging has no natural semantics).

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use defect_agent::error::BoxError;
use defect_agent::tool::SkillTriggers;
use serde::Deserialize;

use crate::frontmatter::{parse_frontmatter, split_frontmatter};
use crate::loader::find_repo_root;
use crate::types::{ConfigError, LoadConfigOptions};

/// Project-level skill directory (relative to repo root). Mirrors [`crate::profiles`]'s
/// `PROJECT_AGENTS_RELATIVE` (`.defect/agents`).
const PROJECT_SKILLS_RELATIVE: &str = ".defect/skills";
/// User-level skill directory (relative to `XDG_CONFIG_HOME`).
const USER_SKILLS_RELATIVE: &str = "defect/skills";
/// Mandatory manifest filename inside every skill directory (aligned with the Anthropic /
/// Codex open standard).
const SKILL_MANIFEST_FILE: &str = "SKILL.md";
/// Soft length limit for skill `description` — exceeding it only warns, does not truncate
/// (cost control for inclusion in the L1 manifest, following Anthropic's practice).
const DESCRIPTION_SOFT_LIMIT: usize = 200;

/// A parsed skill.
///
/// Produced by [`discover_skills`]; consumed by the `skill` tool — `name` / `description`
/// go into the tool schema's manifest, `body` is returned as the tool result when the
/// model fetches by name, and `dir` gives the model the absolute root for resource files
/// (`scripts/` / `refs/`). `always` / `triggers` drive automatic activation (see
/// [`SkillTriggers`]), and are projected into the agent-side `SkillEntry` during CLI
/// assembly.
#[derive(Debug, Clone)]
pub struct SkillSpec {
    /// Skill name (directory name). Value of the `name` enum in the `skill` tool.
    pub name: String,
    /// Absolute path to the skill directory, used by the `skill` tool to backfill
    /// resource file paths for the model.
    pub dir: PathBuf,
    /// Selection-phase description – included in the L1 manifest so the model can decide
    /// whether to load it. Required.
    pub description: String,
    /// The full body of `SKILL.md` after stripping the frontmatter (content loaded at
    /// L2).
    pub body: String,
    /// `always: true` ⇒ body is injected directly into the system prompt at session start
    /// (always-on).
    pub always: bool,
    /// Auto-activation trigger conditions (globs are compiled into `GlobSet`/keywords
    /// during parsing). Reuses the agent-side type; CLI projection clones directly.
    pub triggers: SkillTriggers,
}

/// Raw deserialization form of `SKILL.md` frontmatter.
///
/// Keeps `deny_unknown_fields` (consistent with [`crate::profiles`]) to catch
/// misspellings of required fields (typos like `naem` / `desciption` are not silently
/// ignored). The `always` / `triggers` fields from the Agent Skills open standard are now
/// consumed (auto-activation), while `allowed_tools` remains an explicit
/// placeholder reserved for tool gating.
///
/// Trade-off of explicit listing (vs. deny vs. fully open): deny would break the selling
/// point that "users can drop in an existing Anthropic / Codex-format skill and it just
/// works"; fully open loses typo protection. Explicitly listing the documented
/// fields balances both — consumed fields go from "ignored" to "consumed", with backward
/// compatibility for user files.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillManifestToml {
    /// Required, and must match the directory name — the manifest display and the `skill`
    /// tool argument use the same name; a mismatch would cause the model to look up the
    /// skill by the manifest name and fail to find it.
    name: String,
    /// Required – goes into the L1 manifest. If missing, serde reports "missing field
    /// `description`", which [`discover_skills`] wraps into a hard error with the file
    /// path.
    description: String,
    /// `true` means this skill's body is directly appended to the system prompt at
    /// session start (always-on).
    #[serde(default)]
    always: Option<bool>,
    /// Automatic activation trigger conditions (by file glob or prompt keyword).
    #[serde(default)]
    triggers: Option<SkillTriggersToml>,
    /// Placeholder for ACP client tool gating (inspired by Anthropic's
    /// `allowed-tools`, so the hyphenated form is also accepted). Currently parsed but
    /// not consumed; reserved for tool gating.
    #[serde(default, alias = "allowed-tools")]
    #[allow(
        dead_code,
        reason = "open-standard placeholder field; currently parsed but not consumed"
    )]
    allowed_tools: Option<Vec<String>>,
}

/// `[triggers]` sub-table: auto-activation conditions. `globs` is compiled into a
/// [`globset::GlobSet`] in [`parse_skill`] (bad globs fail fast).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillTriggersToml {
    #[serde(default)]
    globs: Vec<String>,
    #[serde(default)]
    keywords: Vec<String>,
}

/// Discover and parse all available skills.
///
/// User-level skills are scanned first, then project-level skills; a project-level skill
/// with the same name overrides the user-level one. For any skill, a failed `SKILL.md`
/// parse, missing frontmatter, or a `name` that does not match the directory name is a
/// hard error (fail loud, do not silently skip bad skills — same as the `profiles`
/// module, unlike the old design's warn-and-skip). Non-skill items in the directory
/// (subdirectories without `SKILL.md`, non-directory entries) are silently skipped.
///
/// # Errors
/// - [`ConfigError::Io`]: reading `SKILL.md` failed
/// - [`ConfigError::Invalid`]: `SKILL.md` missing frontmatter, parse failure, missing
///   `name` / `description`, or `name` ≠ directory name
pub fn discover_skills(
    opts: &LoadConfigOptions,
) -> Result<BTreeMap<String, SkillSpec>, ConfigError> {
    let mut skills = BTreeMap::new();

    // User-layer first, project-layer second — later writes overwrite earlier ones, so
    // project settings override user settings.
    if let Some(user_dir) = resolve_user_skills_dir(opts) {
        scan_skills_dir(&user_dir, &mut skills)?;
    }
    if let Some(repo_root) = find_repo_root(&opts.cwd) {
        scan_skills_dir(&repo_root.join(PROJECT_SKILLS_RELATIVE), &mut skills)?;
    }

    Ok(skills)
}

/// Scan a `skills/` directory, parse each skill into a [`SkillSpec`], and write it into
/// `out` (when the same name appears across layers, the current layer overwrites previous
/// ones — the caller passes directories in user→project order to implement "project
/// overrides user"). If the directory does not exist, this is a no-op.
fn scan_skills_dir(
    skills_dir: &Path,
    out: &mut BTreeMap<String, SkillSpec>,
) -> Result<(), ConfigError> {
    let entries = match std::fs::read_dir(skills_dir) {
        Ok(entries) => entries,
        // It is normal for the directory to not exist (the user has not created any
        // skills) — this is not an error.
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
            // Skills only use the dir-per-skill layout — skip non-directory entries.
            continue;
        }
        let manifest_path = path.join(SKILL_MANIFEST_FILE);
        if !manifest_path.is_file() {
            // Subdirectories without a SKILL.md are not skills — skip silently.
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

/// Parse a skill directory: read `SKILL.md`, split the frontmatter, verify that `name`
/// matches the directory name, and treat the body as the content after the frontmatter.
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

    // Process triggers: compile `globs` into a `GlobSet` (invalid globs hard-fail
    // immediately, with the skill path), and keep `keywords` as-is. No `[triggers]` table
    // means default empty triggers.
    let triggers = match manifest.triggers {
        Some(t) => SkillTriggers {
            globs: compile_globs(&t.globs, manifest_path)?,
            keywords: t.keywords,
        },
        None => SkillTriggers::default(),
    };

    Ok(SkillSpec {
        name: manifest.name,
        dir: dir.to_path_buf(),
        description: manifest.description,
        body: body.to_string(),
        always: manifest.always.unwrap_or(false),
        triggers,
    })
}

/// Compiles `triggers.globs` into a [`globset::GlobSet`]. Empty input ⇒ `None` (no glob
/// triggers).
/// Any invalid glob ⇒ [`ConfigError::Invalid`] with the `SKILL.md` path and the globset
/// error
/// (fails loudly, does not silently swallow bad globs).
fn compile_globs(
    globs: &[String],
    manifest_path: &Path,
) -> Result<Option<globset::GlobSet>, ConfigError> {
    if globs.is_empty() {
        return Ok(None);
    }
    let mut builder = globset::GlobSetBuilder::new();
    for pat in globs {
        let glob = globset::Glob::new(pat).map_err(|err| ConfigError::Invalid {
            path: manifest_path.to_path_buf(),
            message: format!("invalid trigger glob `{pat}`: {err}"),
        })?;
        builder.add(glob);
    }
    let set = builder.build().map_err(|err| ConfigError::Invalid {
        path: manifest_path.to_path_buf(),
        message: format!("failed to build trigger glob set: {err}"),
    })?;
    Ok(Some(set))
}

/// Resolves the user-level `skills/` directory. Follows the same priority as
/// [`crate::profiles`]'s `resolve_user_agents_dir` (`XDG_CONFIG_HOME` → `HOME/.config`);
/// returns `None` when not found (if neither `XDG_CONFIG_HOME` nor `HOME` is set, user
/// skills are simply absent, not a hard error).
fn resolve_user_skills_dir(opts: &LoadConfigOptions) -> Option<PathBuf> {
    // `--local`: ignore the user-level skills directory.
    if opts.local {
        return None;
    }
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
mod tests;
