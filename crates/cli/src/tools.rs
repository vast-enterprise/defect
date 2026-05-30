//! 装配进程内 tool registry。
//!
//! 跑在 agent 进程里的工具（bash / fs / fetch / search 等）一次性挂在
//! [`StaticToolRegistry`] 上；MCP 工具走 session-level [`McpToolFactory`]
//! 在 `mcp_servers` 模块里组装。

use std::collections::BTreeMap;
use std::sync::Arc;

use defect_agent::llm::ProviderRegistry;
use defect_agent::policy::SandboxPolicy;
use defect_agent::session::{CompositeRegistry, StaticToolRegistry, ToolRegistry};
use defect_agent::tool::{SpawnAgentTool, SubagentProfile};
use defect_config::{LoadedConfig, ProfileSpec};
use defect_tools::{BashTool, EditFileTool, FetchTool, ReadFileTool, SearchTool, WriteFileTool};

/// 按 `[tools]` 段装配进程内工具集合。
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

/// 把 `defect-config` 的 [`ProfileSpec`] 投影成 agent 侧 [`SubagentProfile`]。
///
/// 两边分开是因为 `defect-config` 依赖 `defect-agent`（不能反向依赖成环）；
/// CLI 在装配边界做这一次投影。
fn project_profiles(specs: &BTreeMap<String, ProfileSpec>) -> BTreeMap<String, SubagentProfile> {
    specs
        .iter()
        .map(|(name, spec)| {
            (
                name.clone(),
                SubagentProfile {
                    description: spec.description.clone(),
                    model: spec.model.clone(),
                    system_prompt: spec.system_prompt_text.clone(),
                    tool_allow: spec.tool_allow.clone(),
                    sampling: spec.sampling.clone(),
                },
            )
        })
        .collect()
}

/// 装配进程工具集，并在发现到任意 profile 时挂上 `spawn_agent` 工具。
///
/// 组合方式：先建 base 工具集（bash/fs/fetch/search），再把 `spawn_agent`
/// 单独放进一份 registry，用 [`CompositeRegistry`] 叠在 base 之上。`spawn_agent`
/// 自己持有的"裁子集来源"是 **base 工具集**（不含 spawn_agent），所以子 agent
/// 结构性拿不到 spawn_agent——禁递归。profile 为空时不挂，返回纯 base。
///
/// `base_prompt` 继承给子 agent（"你是会用工具的 agent"那段底座）；profile
/// 的角色 prompt 另外叠在其后。
pub fn build_process_tools_with_subagents(
    config: &LoadedConfig,
    profiles: &BTreeMap<String, ProfileSpec>,
    registry: &Arc<ProviderRegistry>,
    policy: &Arc<dyn SandboxPolicy>,
    base_prompt: Option<String>,
) -> Arc<dyn ToolRegistry> {
    let base = build_process_tools(config);
    let projected = project_profiles(profiles);
    if !SpawnAgentTool::has_profiles(&projected) {
        return base;
    }
    let spawn = SpawnAgentTool::new(
        Arc::new(projected),
        registry.clone(),
        policy.clone(),
        base.clone(),
        base_prompt,
    );
    let spawn_reg: Arc<dyn ToolRegistry> =
        Arc::new(StaticToolRegistry::builder().insert(Arc::new(spawn)).build());
    Arc::new(CompositeRegistry::new(spawn_reg, base))
}
