//! `defect` 二进制入口——只做最薄的 CLI 生命周期编排。
//!
//! 默认 AgentCore 拼装已收束到 [`defect_cli::assembly::CliAgentBuilder`]：
//! 下游二次开发可以从那层 builder 上裁剪默认 feature set，并叠加自己的
//! provider / tool / hook / observer / MCP factory。

use std::env;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use clap::Parser;
use defect_agent::session::AgentCore;
use defect_cli::args::CliArgs;
use defect_cli::assembly::{CliAgentBuilder, ReplMode};
use defect_config::{LoadConfigOptions, load_dotenv_compat};
use defect_obs::init_tracing;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cwd = env::current_dir()?;
    load_dotenv_compat(&cwd).map_err(|e| anyhow::anyhow!("dotenv load failed: {e}"))?;

    let cli = CliArgs::parse();
    let load_opts = LoadConfigOptions {
        cwd: cwd.clone(),
        cli: cli.to_overrides()?,
        local: cli.local,
        ..LoadConfigOptions::default()
    };
    let config = defect_config::load_config(load_opts.clone())
        .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;
    init_tracing(config.effective.tracing.filter.as_deref())?;

    for warning in &config.warnings {
        tracing::warn!("{warning:?}");
    }

    let mut builder = CliAgentBuilder::new(cwd, load_opts, config).repl(if cli.repl {
        ReplMode::Enabled
    } else {
        ReplMode::Disabled
    });
    if cli.local {
        builder = builder.local_sessions();
    }
    if let Some(profile) = cli.profile {
        builder = builder.profile(profile);
    }
    if let Some(resume) = cli.resume {
        builder = builder.resume(resume);
    }

    let built = builder.build().await?;
    if let Some(id) = &built.resume_session_id {
        tracing::info!(session_id = %id.0, "resuming session");
    }
    tracing::info!(
        model = %built.turn_config.model,
        sandbox = ?built.sandbox_mode,
        "starting defect {}",
        if cli.repl { "repl on stdio" } else { "ACP server on stdio" }
    );

    if cli.repl {
        run_repl(built.agent, built.resume_session_id).await?;
    } else {
        defect_acp::serve_with_resume(built.agent, built.resume_session_id).await?;
    }
    Ok(())
}

/// 启动 REPL。`repl` feature 开启时跑真正的 REPL；裁掉时这个 flag 仍能
/// 解析，但运行期 hard fail 提示重新带 feature 编译——不静默退化成 ACP。
#[cfg(feature = "repl")]
async fn run_repl(agent: Arc<dyn AgentCore>, resume: Option<SessionId>) -> anyhow::Result<()> {
    let cwd = env::current_dir()?;
    defect_cli::repl::run(agent, cwd, resume).await
}

#[cfg(not(feature = "repl"))]
async fn run_repl(_agent: Arc<dyn AgentCore>, _resume: Option<SessionId>) -> anyhow::Result<()> {
    anyhow::bail!(
        "this binary was built without the `repl` feature; \
         rebuild with `--features repl` (on by default) to use --repl"
    )
}
