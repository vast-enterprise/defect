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
    policy::{build_mode_catalog, build_policy},
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
    // REPL 全放行：REPL 是开发期手搓 prompt 的便捷入口，没有 ACP 客户端那样的
    // 权限弹窗 UI，也不在终端做交互式确认。默认 AskWrites 会让 Mutating 工具
    // （spawn_agent / write_file / bash 等）停在 PermissionGate 等一个永远不会
    // 来的应答 → turn 卡死。故 REPL 强制 Open，让 policy 直接放行。安全裁剪请走
    // ACP（有权限弹窗）或显式 sandbox 配置。
    let sandbox_mode = if cli.repl {
        defect_config::SandboxMode::Open
    } else {
        config.effective.sandbox.mode
    };
    let policy = build_policy(sandbox_mode);

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
        sandbox = ?sandbox_mode,
        "starting defect {}",
        if cli.repl { "repl on stdio" } else { "ACP server on stdio" }
    );

    let http_client =
        defect_http::build_fetch_client_arc(&build_http_stack_config(&config.effective.http)?)
            .map_err(|e| anyhow::anyhow!("fetch http client init failed: {e}"))?;

    // 共享一份 skill 索引（Arc）：既给 builtin 注册表的闭包，也给主 session 的
    // 自动挂载（见 build_main_session_engine）。单一真相源。
    let skills_arc = Arc::new(skills.clone());

    // `skill-manifest` / `skill-triggers` builtin 持有 skill 索引，无参工厂构造
    // 不出来——这里用捕获索引的闭包注册进 registry（subagent profile 的
    // `[hooks]` 装配也用这份注册表；用户也可在 `[[hooks.*]]` 里按名引用）。主
    // session 则由 build_main_session_engine 自动挂载，无需用户配置。装在工具集
    // 之前，因为 subagent profile 装配要用这份 builtin 注册表。
    let mut builtin_registry = BuiltinRegistry::defaults();
    {
        let skills_for_hook = skills_arc.clone();
        builtin_registry.register_step("skill-manifest", move || {
            Arc::new(defect_agent::hooks::builtin::SkillManifestHook::new(
                skills_for_hook.clone(),
            ))
        });
        let skills_for_trig = skills_arc.clone();
        builtin_registry.register_step("skill-triggers", move || {
            Arc::new(defect_agent::hooks::builtin::SkillTriggersHook::new(
                skills_for_trig.clone(),
            ))
        });
    }
    let hook_rt = HookEngineCtx {
        registry: &registry,
        default_model: turn_config.model.as_str(),
    };

    // 工具集：顶层 --profile 时按其白名单裁；否则 base + (有 profile 则) spawn_agent。
    // subagent profile 各自的 `[hooks]` 在 build_process_tools_with_subagents 里
    // 编译成 hook 引擎注入对应 SubagentProfile。
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
            &builtin_registry,
            &hook_rt,
        )
        .map_err(|e| anyhow::anyhow!("subagent hook engine build failed: {e}"))?
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

    let hook_engine = hooks::build_main_session_engine(
        &config.effective.hooks,
        &builtin_registry,
        &hook_rt,
        &skills_arc,
    )
    .map_err(|e| anyhow::anyhow!("hook engine build failed: {e}"))?;

    // 3) 拼装 AgentCore，启 stdio ACP server
    // 权限模式目录：暴露全部 4 个 SandboxMode 给 ACP 客户端，当前选中
    // = 启动期解析出的 `sandbox_mode`。支持 `session/set_mode` 运行时切换。
    let mode_catalog = build_mode_catalog(sandbox_mode);

    let mut builder = DefaultAgentCore::builder()
        .registry(registry)
        .process_tools(tools)
        .policy(policy)
        .modes(mode_catalog)
        .observe_session(storage.clone())
        .session_loader(storage)
        .session_tool_factory(Arc::new(McpToolFactory::with_default_servers(
            build_default_mcp_servers(&config),
        )))
        .config(turn_config)
        .background_progress(config.effective.tools.background)
        .http(http_client)
        .hook_engine(hook_engine);
    if let Some(langfuse) = langfuse {
        builder = builder.observe_session(langfuse);
    }
    let agent = builder.build();
    let agent = Arc::new(agent) as Arc<dyn AgentCore>;

    // `--repl`：进程内最小交互 REPL（feature-gated）。否则走 stdio ACP server。
    if cli.repl {
        run_repl(agent).await?;
    } else {
        defect_acp::serve(agent).await?;
    }
    Ok(())
}

/// 启动 REPL。`repl` feature 开启时跑真正的 REPL；裁掉时这个 flag 仍能
/// 解析，但运行期 hard fail 提示重新带 feature 编译——不静默退化成 ACP。
#[cfg(feature = "repl")]
async fn run_repl(agent: Arc<dyn AgentCore>) -> anyhow::Result<()> {
    let cwd = env::current_dir()?;
    defect_cli::repl::run(agent, cwd).await
}

#[cfg(not(feature = "repl"))]
async fn run_repl(_agent: Arc<dyn AgentCore>) -> anyhow::Result<()> {
    anyhow::bail!(
        "this binary was built without the `repl` feature; \
         rebuild with `--features repl` (on by default) to use --repl"
    )
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
