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
    tools::build_process_tools,
};
use defect_obs::init_tracing;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) 读 .env / 解析 CLI / 加载分层配置 / 装 tracing
    let cwd = env::current_dir()?;
    load_dotenv_compat(&cwd).map_err(|e| anyhow::anyhow!("dotenv load failed: {e}"))?;

    let cli = CliArgs::parse();
    let config = defect_config::load_config(LoadConfigOptions {
        cwd,
        cli: cli.to_overrides()?,
        ..LoadConfigOptions::default()
    })
    .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;
    init_tracing(config.effective.tracing.filter.as_deref())?;

    for warning in &config.warnings {
        tracing::warn!("{warning:?}");
    }

    // 2) 装 provider registry / HTTP 栈 / 工具集合 / 存储 / hook 引擎
    let (registry, turn_config) = build_registry(&config).await?;

    tracing::info!(
        provider = %registry.default_entry().provider().info().vendor,
        model = %turn_config.model,
        "starting defect ACP server on stdio"
    );

    let http_client = defect_http::build_fetch_client_arc(&build_http_stack_config(
        &config.effective.http,
    )?)
    .map_err(|e| anyhow::anyhow!("fetch http client init failed: {e}"))?;

    let tools = build_process_tools(&config);
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

    let builtin_registry = BuiltinRegistry::defaults();
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
        .policy(build_policy(config.effective.sandbox.mode))
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
