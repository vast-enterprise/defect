//! 装配 `process_tools` registry。
//!
//! 这里组的工具（bash / fs / fetch / search / skill / spawn_agent 等）一次性挂在
//! [`StaticToolRegistry`] 上，作为某个 `AgentCore` 实例的 `process_tools`、被该
//! core 的各 session 共享一份——**不是进程全局单例**（把 defect 当库引用时一个
//! 进程可装配多个 `AgentCore`，各持自己的一份）。MCP 工具走 session-level
//! [`McpToolFactory`] 在 `mcp_servers` 模块里组装。

use std::collections::BTreeMap;
use std::sync::Arc;

use defect_agent::hooks::HookEngine;
use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::llm::ProviderRegistry;
use defect_agent::policy::SandboxPolicy;
use defect_agent::session::{CompositeRegistry, StaticToolRegistry, ToolRegistry};
use defect_agent::tool::{SkillEntry, SkillTool, SpawnAgentTool, SubagentProfile};
use defect_config::{LoadedConfig, ProfileSpec, SkillSpec};
use defect_tools::{BashTool, EditFileTool, FetchTool, ReadFileTool, SearchTool, WriteFileTool};

use crate::hooks::{HookEngineBuildError, HookEngineCtx, build_engine_arc};

/// 按 `[tools]` 段装配 `process_tools` 工具集合（随 `AgentCore` 实例、跨该 core
/// 的 session 共享）。
///
/// `fetch` / `search` 通过 `enabled` 字段单独控制；本地 `search` 工具
/// 与 hosted `web_search` capability 完全独立——两者可同时启用。
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

/// 按白名单从 base 工具集裁子集，用于顶层 `--profile`（把整个会话跑成某个
/// profile）。unknown 工具名 hard error（fail loud）。`spawn_agent` 即便在
/// 白名单里也会被排除——顶层 profile 是叶子 agent，不再派生子 agent。
///
/// # Errors
/// profile 的 `allow` 含 base 工具集里不存在的名字时返回 `Err(name)`。
pub fn filter_tools_by_allowlist(
    base: &Arc<dyn ToolRegistry>,
    allow: &[String],
) -> Result<Arc<dyn ToolRegistry>, String> {
    let mut builder = StaticToolRegistry::builder();
    for name in allow {
        if name == "spawn_agent" {
            continue;
        }
        match base.get(name) {
            Some(tool) => builder = builder.insert(tool),
            None => return Err(name.clone()),
        }
    }
    Ok(Arc::new(builder.build()))
}

/// 把 `defect-config` 的 [`ProfileSpec`] 投影成 agent 侧 [`SubagentProfile`]，
/// 顺带把每个 profile 自己声明的 `[hooks]` 编译成 hook 引擎注入。
///
/// 两边分开是因为 `defect-config` 依赖 `defect-agent`（不能反向依赖成环）；
/// CLI 在装配边界做这一次投影。hook 引擎在此装配是因为它需要 builtin 注册表
/// 与 provider registry（与主 session 的 hook 装配同源，见 [`crate::hooks`]）。
///
/// profile 的 `[hooks]` 为空 ⇒ `hooks: None`（子 agent 无钩子，行为同改动前）。
///
/// # Errors
/// 任一 profile 的 hook 引擎装配失败（未知 builtin / prompt hook 引用未注册
/// model 等）即 hard fail，错误里带上 profile 名定位。
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
                    hooks,
                },
            ))
        })
        .collect()
}

/// 某个 subagent profile 的 hook 引擎装配失败。带上 profile 名定位。
#[derive(Debug, thiserror::Error)]
#[error("subagent profile `{profile}` hook engine build failed: {source}")]
pub struct ProfileHookBuildError {
    pub profile: String,
    #[source]
    pub source: HookEngineBuildError,
}

/// 把 `defect-config` 的 [`SkillSpec`] 投影成 agent 侧 [`SkillEntry`]——与
/// [`project_profiles`] 同款跨 crate 装配边界投影。
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

/// 装配进程工具集，并在发现到 profile / skill 时分别叠上 `spawn_agent` /
/// `skill` 工具。
///
/// 组合方式：先建 base 工具集（bash/fs/fetch/search），把 `spawn_agent`
/// （发现到任意 profile 时）与 `skill`（发现到任意 skill 时）放进一份 overlay
/// registry，用 [`CompositeRegistry`] 叠在 base 之上。
///
/// - `spawn_agent` 持有的"裁子集来源"是 **base 工具集**（不含这两个 overlay
///   工具），所以子 agent 结构性拿不到 spawn_agent——禁递归；也拿不到 skill
///   （skill 是顶层 agent 的能力，子 agent 走自己的 profile prompt）。
/// - profile 与 skill 都空时不叠 overlay，返回纯 base。
///
/// `base_prompt` 继承给子 agent（"你是会用工具的 agent"那段底座）；profile
/// 的角色 prompt 另外叠在其后。
///
/// `builtins` / `hook_rt` 供把每个 profile 自己的 `[hooks]` 编译成 hook 引擎
/// （见 [`project_profiles`]）——子 agent 的钩子是它身份的一部分，不从父继承。
///
/// # Errors
/// 任一 profile 的 hook 引擎装配失败即 hard fail（[`ProfileHookBuildError`]）。
// 装配边界函数：参数都是 AgentCore 各独立组件，拆成 struct 反而割裂调用方
// （cli.rs 那一处把它们逐个传入），故就地多带两个 hook 装配依赖。
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
    }
    if has_skills {
        let skill = SkillTool::new(Arc::new(skills.clone()));
        overlay = overlay.insert(Arc::new(skill));
    }
    let overlay_reg: Arc<dyn ToolRegistry> = Arc::new(overlay.build());
    Ok(Arc::new(CompositeRegistry::new(overlay_reg, base)))
}
