//! `skill`: loads the full specification of a skill into the current conversation.
//!
//! A skill is a user-configurable reusable prompt fragment — a markdown body plus
//! optional
//! `scripts/` / `refs/` resources in the same directory. The model sees skill contents in
//! three
//! layers via progressive disclosure:
//!
//! - **L1 manifest**: all skills' `name + description`. This tool embeds them into its
//!   own
//!   `description` (same pattern as `spawn_agent` embedding the profile catalog), so the
//!   model
//!   knows which skills are available from startup. Optionally also injected into the
//!   system
//!   prompt by `crate::hooks::builtin::SkillManifestHook`.
//! - **L2 body**: the full `SKILL.md` fetched by name when the model calls this tool,
//!   arriving
//!   as a tool result — the model then works according to the instructions **within the
//!   current
//!   conversation** (unlike `spawn_agent` which spawns an isolated sub-session).
//! - **L3 attachments**: `scripts/*.sh` / `refs/*.md` referenced in the body, read on
//!   demand by
//!   the model via ordinary `bash` / `read_file` tools — the tool result includes the
//!   absolute
//!   skill directory path for constructing paths.
//!
//! This tool is a pure [`Tool`] implementation with `safety_hint = ReadOnly` (only
//! queries the
//! in-memory loaded skill index, no disk writes, no network access), treated identically
//! to other
//! built-in tools.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::error::BoxError;
use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};

/// The name of the `skill` tool.
pub(crate) const SKILL_TOOL_NAME: &str = "skill";

/// Auto-activation triggers for a skill (the `triggers` sub-table of the Agent Skills
/// open-standard).
///
/// Defined on the agent side, populated and reused by `defect-config` during parsing
/// (dependency direction: config → agent, not reversible). `globs` is compiled into a
/// [`globset::GlobSet`] at config parse time—invalid globs fail fast immediately, no
/// runtime parsing; `None` when no globs are configured. See
/// `crate::hooks::builtin::SkillTriggersHook` for matching logic.
#[derive(Debug, Clone, Default)]
pub struct SkillTriggers {
    /// Compiled file-path glob set; `None` means no globs were configured.
    pub globs: Option<globset::GlobSet>,
    /// Prompt keywords (case-insensitive substring matching).
    pub keywords: Vec<String>,
}

/// A skill that can be loaded by the `skill` tool (agent-side representation).
///
/// `SkillSpec` in `defect-config` is the configuration-side source of truth; during CLI
/// assembly it is projected into this struct before being handed to the tool. The two are
/// kept separate because `defect-config` depends on `defect-agent` — a reverse dependency
/// would create a cycle (same boundary as [`crate::tool::SubagentProfile`] /
/// `ProfileSpec`).
#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// Description shown in the selection phase, included in the L1 manifest (the catalog
    /// of tool descriptions).
    pub description: String,
    /// The full body of `SKILL.md` after stripping frontmatter — returned to the model
    /// during L2 loading.
    pub body: String,
    /// Absolute path to the skill directory, backfilled in L2 tool results so the model
    /// can construct absolute paths to resources like `scripts/` / `refs/` for `bash` /
    /// `read_file`.
    pub dir: PathBuf,
    /// `always: true` ⇒ body is directly appended to the system prompt at session start
    /// (always-on; see `crate::hooks::builtin::SkillManifestHook`).
    pub always: bool,
    /// Automatic activation triggers (by file glob or prompt keyword); see
    /// [`SkillTriggers`].
    pub triggers: SkillTriggers,
}

/// The `skill` tool. It is registered on `StaticToolRegistry` and shared across sessions
/// of the owning `AgentCore` via `process_tools` (it is **not** a process-global
/// singleton—a single process may host multiple `AgentCore` instances, each with its own
/// skill index).
pub struct SkillTool {
    schema: ToolSchema,
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillTool {
    /// Constructs a `skill` tool. When `skills` is empty, callers **should not** register
    /// this tool
    /// (the schema's `name` enum will be empty, so it will always fail) — see
    /// [`Self::has_skills`].
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        let schema = build_schema(&skills);
        Self { schema, skills }
    }

    /// Whether any skills were discovered. The assembler uses this to decide whether to
    /// register this tool.
    pub fn has_skills(skills: &BTreeMap<String, SkillEntry>) -> bool {
        !skills.is_empty()
    }
}

/// Dynamically builds the schema: `name` is an enum of discovered skill names (hard
/// constraint), and the tool description embeds a catalog of `- <name>: <description>`
/// entries (soft guidance, i.e. an L1 manifest). Both are required: the enum alone gives
/// the model no context for usage, while the catalog alone risks the model misspelling
/// names (same rationale as [`crate::tool::SpawnAgentTool`]'s `build_schema`).
fn build_schema(skills: &BTreeMap<String, SkillEntry>) -> ToolSchema {
    let names: Vec<&str> = skills.keys().map(String::as_str).collect();
    let catalog = skills
        .iter()
        .map(|(name, s)| format!("- {name}: {}", s.description))
        .collect::<Vec<_>>()
        .join("\n");
    let description = format!(
        "Load the full instructions for a specialized skill into the current conversation. \
         Use this when the task at hand matches one of the skills below; the loaded content may \
         contain detailed workflow guidance plus references to scripts / files in the skill's \
         directory that you can then read with `bash` / `read_file`. After loading, carry out the \
         task in this same conversation.\n\n\
         Available skills:\n{catalog}"
    );
    ToolSchema {
        name: SKILL_TOOL_NAME.to_string(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": names,
                    "description": "Which skill to load. See the tool description for what each skill does."
                }
            },
            "required": ["name"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct SkillArgs {
    name: String,
}

impl Tool for SkillTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // Only queries the in-memory skill index and feeds the body text back to the
        // model — no disk writes, no network access.
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(format!("Load skill `{name}`"));
            fields.kind = Some(ToolKind::Think);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        let skills = self.skills.clone();
        let fut = async move {
            let parsed: SkillArgs = match serde_json::from_value(args) {
                Ok(v) => v,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };

            let Some(skill) = skills.get(&parsed.name) else {
                return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(format!(
                    "unknown skill `{}`; available: {}",
                    parsed.name,
                    skills.keys().cloned().collect::<Vec<_>>().join(", ")
                )))));
            };

            let output = render_skill(&parsed.name, skill);
            let mut fields = ToolCallUpdateFields::default();
            fields.content = Some(vec![ToolCallContent::Content(Content::new(
                ContentBlock::Text(TextContent::new(output.clone())),
            ))]);
            // raw_output is for telemetry (the langfuse projector reads only raw_output
            // as the observation output).
            fields.raw_output = Some(serde_json::Value::String(output));
            ToolEvent::Completed(fields)
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

/// Compose the tool result text for an L2-loaded skill: title + directory hint + body.
/// The directory hint tells the model the absolute root for resources like `scripts/` /
/// `refs/` (analogous to opencode's "Base directory" line).
fn render_skill(name: &str, skill: &SkillEntry) -> String {
    format!(
        "# Skill: {name}\n\n{body}\n\n\
         Skill directory: {dir}\n\
         Relative paths in this skill (e.g. scripts/, refs/) are relative to that directory; \
         read them with `read_file` / `bash` as needed.",
        body = skill.body,
        dir = skill.dir.display(),
    )
}

fn io_err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

#[cfg(test)]
mod tests;
