//! `defect` 二进制入口——只做拼装，所有 helper 在 [`defect_cli`] lib 里。
//!
//! 想做二次开发？把这个文件复制到你自己的 crate，按需替换某一步即可：
//! - 换 provider：替换 `providers::build_registry` 调用，自己组 [`ProviderRegistry`]
//! - 换工具：替换 `tools::build_process_tools`
//! - 换 hook：传一份自己的 `Arc<dyn HookEngine>`
//! - 换 storage：自己实现 `StorageObserver` 等价物
//!
//! [`ProviderRegistry`]: defect_agent::llm::ProviderRegistry

use std::env;
use std::sync::Arc;

use clap::Parser;
use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::session::{AgentCore, DefaultAgentCore};
use defect_config::{LoadConfigOptions, load_dotenv_compat};
use defect_mcp::McpToolFactory;
use defect_storage::StorageObserver;

use defect_cli::{
    args::CliArgs,
    hooks::{self, HookEngineCtx},
    http_stack::build_http_stack_config,
    mcp_servers::build_default_mcp_servers,
    observability,
    paths::default_sessions_root,
    policy::build_policy,
    providers::build_registry,
    tools::{
        build_process_tools, build_process_tools_with_subagents, filter_tools_by_allowlist,
        project_skills,
    },
};
use defect_obs::init_tracing;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) 读 .env / 解析 CLI / 加载分层配置 / 装 tracing
    let cwd = env::current_dir()?;
    load_dotenv_compat(&cwd).map_err(|e| anyhow::anyhow!("dotenv load failed: {e}"))?;

    let cli = CliArgs::parse();
    let load_opts = LoadConfigOptions {
        cwd,
        cli: cli.to_overrides()?,
        ..LoadConfigOptions::default()
    };
    let config = defect_config::load_config(load_opts.clone())
        .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;
    init_tracing(config.effective.tracing.filter.as_deref())?;

    for warning in &config.warnings {
        tracing::warn!("{warning:?}");
    }

    // subagent profile 发现（与主配置同源分层；同名项目层覆盖用户层）。
    let profiles = defect_config::discover_profiles(&load_opts)
        .map_err(|e| anyhow::anyhow!("profile discovery failed: {e}"))?;

    // skill 发现（与 profile 同源分层）。投影成 agent 侧索引，供 `skill` 工具
    // 与 `skill-manifest` hook 共用——单一真相源。
    let skill_specs = defect_config::discover_skills(&load_opts)
        .map_err(|e| anyhow::anyhow!("skill discovery failed: {e}"))?;
    let skills = project_skills(&skill_specs);

    // 2) 装 provider registry / HTTP 栈 / 工具集合 / 存储 / hook 引擎
    let (registry, mut turn_config) = build_registry(&config).await?;
    let policy = build_policy(config.effective.sandbox.mode);

    // 继承给子 agent 的 base_prompt 文本（"你是会用工具的 agent"那段底座）。
    let base_prompt_text = resolve_base_prompt_text(&config)?;

    // 顶层 --profile：把整个会话跑成某个 profile（叶子 agent，不派生子 agent）。
    // 与嵌套 subagent 不同，这里有人在场（ACP 客户端），故保留正常 policy。
    if let Some(profile_name) = cli.profile.as_deref() {
        let spec = profiles.get(profile_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown --profile `{profile_name}`; available: {}",
                profiles.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        if let Some(model) = &spec.model {
            turn_config.model = model.clone();
        }
        // profile 角色 prompt 作为 session overlay 叠加。
        turn_config.system_prompt = Some(spec.system_prompt_text.clone());
    }

    tracing::info!(
        provider = %registry.default_entry().provider().info().vendor,
        model = %turn_config.model,
        profile = cli.profile.as_deref().unwrap_or("<none>"),
        "starting defect ACP server on stdio"
    );

    let http_client =
        defect_http::build_fetch_client_arc(&build_http_stack_config(&config.effective.http)?)
            .map_err(|e| anyhow::anyhow!("fetch http client init failed: {e}"))?;

    // 工具集：顶层 --profile 时按其白名单裁；否则 base + (有 profile 则) spawn_agent。
    let tools = if let Some(profile_name) = cli.profile.as_deref() {
        let spec = profiles
            .get(profile_name)
            .expect("profile existence checked above");
        let base = build_process_tools(&config);
        filter_tools_by_allowlist(&base, &spec.tool_allow).map_err(|name| {
            anyhow::anyhow!("profile `{profile_name}` allows unknown tool `{name}`")
        })?
    } else {
        build_process_tools_with_subagents(
            &config,
            &profiles,
            &skills,
            &registry,
            &policy,
            base_prompt_text,
        )
    };
    let storage = Arc::new(StorageObserver::new(default_sessions_root()?));

    // 可观测性：langfuse 上报（默认关闭，需 [tracing.langfuse].enabled + key）。
    let langfuse = observability::build_langfuse_observer(
        config.effective.tracing.langfuse.as_ref(),
        build_http_stack_config(&config.effective.http)?,
    )?
    .map(Arc::new);
    if langfuse.is_some() {
        tracing::info!("langfuse reporting enabled");
    }

    // `skill-manifest` builtin 持有 skill 索引，无参工厂构造不出来——这里用
    // 捕获索引的闭包注册进 registry，用户可在 `[[hooks.session_start]]` 里按名
    // 挂上（与 `skill` 工具 description 内嵌的 catalog 同源）。
    let mut builtin_registry = BuiltinRegistry::defaults();
    {
        let skills_for_hook = Arc::new(skills.clone());
        builtin_registry.register_step("skill-manifest", move || {
            Arc::new(defect_agent::hooks::builtin::SkillManifestHook::new(
                skills_for_hook.clone(),
            ))
        });
    }
    let hook_engine = hooks::build_engine_arc(
        &config.effective.hooks,
        &builtin_registry,
        &HookEngineCtx {
            registry: &registry,
            default_model: turn_config.model.as_str(),
        },
    )
    .map_err(|e| anyhow::anyhow!("hook engine build failed: {e}"))?;

    // 3) 拼装 AgentCore，启 stdio ACP server
    let mut builder = DefaultAgentCore::builder()
        .registry(registry)
        .process_tools(tools)
        .policy(policy)
        .observe_session(storage.clone())
        .session_loader(storage)
        .session_tool_factory(Arc::new(McpToolFactory::with_default_servers(
            build_default_mcp_servers(&config),
        )))
        .config(turn_config)
        .http(http_client)
        .hook_engine(hook_engine);
    if let Some(langfuse) = langfuse {
        builder = builder.observe_session(langfuse);
    }
    let agent = builder.build();

    defect_acp::serve(Arc::new(agent) as Arc<dyn AgentCore>).await?;
    Ok(())
}

/// 解析继承给 subagent 的 base_prompt 文本：`[base_prompt] file` 读文件 +
/// `text` 内联，按"文件在前、内联在后"拼接（与
/// `defect_agent::session::resolve_system_prompt` 的 base 段顺序一致）。
/// 两者都没配 ⇒ `None`。
fn resolve_base_prompt_text(
    config: &defect_config::LoadedConfig,
) -> anyhow::Result<Option<String>> {
    let bp = &config.effective.base_prompt;
    let mut sections = Vec::new();
    if let Some(file) = bp.file.as_deref() {
        let text = std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("base_prompt file {} read failed: {e}", file.display()))?;
        sections.push(text);
    }
    if let Some(text) = bp.text.as_deref() {
        sections.push(text.to_owned());
    }
    Ok((!sections.is_empty()).then(|| sections.join("\n\n")))
}
