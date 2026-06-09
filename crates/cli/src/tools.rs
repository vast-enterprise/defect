//! Assembles the `process_tools` registry.
//!
//! The tools grouped here (bash / fs / fetch / search / skill / spawn_agent, etc.) are
//! mounted once on a [`StaticToolRegistry`] as the `process_tools` of an `AgentCore`
//! instance, shared across all sessions of that core — **not a process-global singleton**
//! (when using defect as a library, a single process may have multiple `AgentCore`
//! instances, each with its own copy). MCP tools go through the session-level
//! [`McpToolFactory`](defect_mcp::McpToolFactory) assembled in the `mcp_servers` module.

use std::collections::BTreeMap;
use std::sync::Arc;

use defect_agent::hooks::HookEngine;
use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::llm::ProviderRegistry;
use defect_agent::policy::SandboxPolicy;
use defect_agent::session::{CompositeRegistry, StaticToolRegistry, ToolRegistry};
use defect_agent::tool::{
    CancelBackgroundTaskTool, InspectBackgroundTaskTool, SkillEntry, SkillTool, SpawnAgentTool,
    SubagentProfile,
};
use defect_config::{LoadedConfig, ProfileSpec, SkillSpec};
use defect_tools::{BashTool, EditFileTool, FetchTool, ReadFileTool, SearchTool, WriteFileTool};

use crate::hooks::{HookEngineBuildError, HookEngineCtx, build_engine_arc};

/// Assembles the `process_tools` tool set from the `[tools]` section (shared across
/// sessions for a given `AgentCore` instance).
///
/// `fetch` / `search` are individually controlled via the `enabled` field; the local
/// `search` tool is completely independent from the hosted `web_search` capability — both
/// can be enabled simultaneously.
pub fn build_process_tools(config: &LoadedConfig) -> Arc<dyn ToolRegistry> {
    let mut builder = StaticToolRegistry::builder()
        .insert(Arc::new(BashTool::from_config(
            &config.effective.tools.bash,
        )))
        .insert(Arc::new(ReadFileTool::from_config(
            &config.effective.tools.fs,
        )))
        .insert(Arc::new(WriteFileTool::new()))
        .insert(Arc::new(EditFileTool::new()));
    if config.effective.tools.fetch.enabled {
        builder = builder.insert(Arc::new(FetchTool::from_config(
            &config.effective.tools.fetch,
        )));
    }
    if config.effective.tools.search.enabled {
        builder = builder.insert(Arc::new(SearchTool::from_config(
            &config.effective.tools.search,
        )));
    }
    Arc::new(builder.build())
}

// NOTE: the top-level `--profile` tool allowlist is no longer filtered here at assembly
// time. It is enforced per-session by `DefaultAgentCore::apply_tool_allow` (which reuses
// `defect_agent::session::filter_registry_by_allowlist`) AFTER MCP tools join the pool, so
// profiles may allow `mcp__*` tools. See assembly.rs `build_default_process_tools`.

/// Projects [`ProfileSpec`] from `defect-config` into the agent-side [`SubagentProfile`],
/// and compiles each profile's declared `[hooks]` into a hook engine injection.
///
/// The split exists because `defect-config` depends on `defect-agent` (a reverse
/// dependency would create a cycle); the CLI performs this projection at the assembly
/// boundary. The hook engine is assembled here because it needs the builtin registry and
/// provider registry (same origin as the main session's hook assembly, see
/// [`crate::hooks`]).
///
/// An empty `[hooks]` in a profile ⇒ `hooks: None` (the sub-agent has no hooks, matching
/// pre-change behavior).
///
/// # Errors
/// Hard-fails if hook engine assembly fails for any profile (unknown builtin, prompt hook
/// reference to an unregistered model, etc.). The error includes the profile name for
/// identification.
fn project_profiles(
    specs: &BTreeMap<String, ProfileSpec>,
    builtins: &BuiltinRegistry,
    hook_rt: &HookEngineCtx<'_>,
) -> Result<BTreeMap<String, SubagentProfile>, ProfileHookBuildError> {
    specs
        .iter()
        .map(|(name, spec)| {
            let hooks = if spec.hooks.is_empty() {
                None
            } else {
                let engine = build_engine_arc(&spec.hooks, builtins, hook_rt).map_err(|err| {
                    ProfileHookBuildError {
                        profile: name.clone(),
                        source: err,
                    }
                })?;
                Some(engine as Arc<dyn HookEngine>)
            };
            Ok((
                name.clone(),
                SubagentProfile {
                    description: spec.description.clone(),
                    model: spec.model.clone(),
                    system_prompt: spec.system_prompt_text.clone(),
                    tool_allow: spec.tool_allow.clone(),
                    sampling: spec.sampling.clone(),
                    inherit_project_prompt: spec.inherit_project_prompt,
                    request_limit: spec.request_limit,
                    hooks,
                },
            ))
        })
        .collect()
}

/// Hook engine build failed for a subagent profile; include the profile name for
/// identification.
#[derive(Debug, thiserror::Error)]
#[error("subagent profile `{profile}` hook engine build failed: {source}")]
pub struct ProfileHookBuildError {
    pub profile: String,
    #[source]
    pub source: HookEngineBuildError,
}

/// Project [`SkillSpec`] from `defect-config` into the agent-side [`SkillEntry`],
/// mirroring the cross-crate assembly-boundary projection pattern used in
/// `project_profiles`.
pub fn project_skills(specs: &BTreeMap<String, SkillSpec>) -> BTreeMap<String, SkillEntry> {
    specs
        .iter()
        .map(|(name, spec)| {
            (
                name.clone(),
                SkillEntry {
                    description: spec.description.clone(),
                    body: spec.body.clone(),
                    dir: spec.dir.clone(),
                    always: spec.always,
                    triggers: spec.triggers.clone(),
                },
            )
        })
        .collect()
}

/// Assembles the process tool set, overlaying `spawn_agent` and `skill` tools when
/// profiles or skills are present.
///
/// Composition: first build the base tool set (bash/fs/fetch/search), then place
/// `spawn_agent` (when any profile is found) and `skill` (when any skill is found) into
/// an overlay registry, and combine them with [`CompositeRegistry`] on top of the base.
///
/// - `spawn_agent`'s "child tool source" is the **base tool set** (without these overlay
///   tools), so child agents structurally cannot access `spawn_agent`—preventing
///   recursion; they also cannot access `skill` (skill is a top-level agent capability;
///   child agents use their own profile prompt); similarly they cannot access
///   `inspect_background_task` / `cancel_background_task` (the background task table
///   belongs to the top-level session, and child agents' nested turns have
///   `ToolContext::background` as `None`).
/// - When both profiles and skills are empty, no overlay is applied and the pure base is
///   returned.
///
/// `base_prompt` is inherited by child agents (the "you are an agent that uses tools"
/// base prompt); the profile's role prompt is appended separately.
///
/// `builtins` / `hook_rt` are used to compile each profile's `[hooks]` into a hook engine
/// (see `project_profiles`)—a child agent's hooks are part of its identity and are not
/// inherited from the parent.
///
/// # Errors
/// If any profile's hook engine fails to build, it is a hard failure
/// ([`ProfileHookBuildError`]).
// This is a boundary assembly function: its parameters are the individual components of
// `AgentCore`; extracting them into a struct would fragment the call site (in `cli.rs`,
// they are passed one by one), so two extra hook-assembly dependencies are kept inline.
#[allow(clippy::too_many_arguments)]
pub fn build_process_tools_with_subagents(
    config: &LoadedConfig,
    profiles: &BTreeMap<String, ProfileSpec>,
    skills: &BTreeMap<String, SkillEntry>,
    registry: &Arc<ProviderRegistry>,
    policy: &Arc<dyn SandboxPolicy>,
    base_prompt: Option<String>,
    builtins: &BuiltinRegistry,
    hook_rt: &HookEngineCtx<'_>,
) -> Result<Arc<dyn ToolRegistry>, ProfileHookBuildError> {
    let base = build_process_tools(config);
    let projected = project_profiles(profiles, builtins, hook_rt)?;
    let has_profiles = SpawnAgentTool::has_profiles(&projected);
    let has_skills = SkillTool::has_skills(skills);
    if !has_profiles && !has_skills {
        return Ok(base);
    }

    let mut overlay = StaticToolRegistry::builder();
    if has_profiles {
        let spawn = SpawnAgentTool::new(
            Arc::new(projected),
            registry.clone(),
            policy.clone(),
            base.clone(),
            base_prompt,
        );
        overlay = overlay.insert(Arc::new(spawn));
        // Background task control surface: query progress / early cancellation. Same tier
        // as `spawn_agent` — only meaningful when the agent can spawn background
        // subagents (`has_profiles`), and likewise only inserted into the overlay, not
        // into the subagent's tool subset source, so subagents structurally cannot reach
        // it (same reasoning as disabling recursion).
        overlay = overlay.insert(Arc::new(InspectBackgroundTaskTool::new()));
        overlay = overlay.insert(Arc::new(CancelBackgroundTaskTool::new()));
    }
    if has_skills {
        let skill = SkillTool::new(Arc::new(skills.clone()));
        overlay = overlay.insert(Arc::new(skill));
    }
    let overlay_reg: Arc<dyn ToolRegistry> = Arc::new(overlay.build());
    Ok(Arc::new(CompositeRegistry::new(overlay_reg, base)))
}
