//! `defect` 二进制入口——只做最薄的 CLI 生命周期编排。
//!
//! 默认 AgentCore 拼装已收束到 [`defect_cli::assembly::CliAgentBuilder`]：
//! 下游二次开发可以从那层 builder 上裁剪默认 feature set，并叠加自己的
//! provider / tool / hook / observer / MCP factory。

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use clap::Parser;
use defect_agent::session::AgentCore;
use defect_cli::args::{CliArgs, OutputFormat};
use defect_cli::assembly::{CliAgentBuilder, ReplMode};
use defect_config::{LoadConfigOptions, load_dotenv_compat};
use defect_obs::init_tracing;

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
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

    // --message（单轮）与 --goal（目标驱动多轮循环）都是无人值守 headless 模式。
    let headless = cli.message.is_some() || cli.goal.is_some();
    let mut builder = CliAgentBuilder::new(cwd.clone(), load_opts, config).repl(if cli.repl {
        ReplMode::Enabled
    } else {
        ReplMode::Disabled
    });
    // 无人值守模式：包 NonInteractivePolicy，避免无 TTY 挂死在权限确认上。
    if headless {
        builder = builder.non_interactive();
    }
    if let Some(goal) = &cli.goal {
        builder = builder.goal(goal.clone());
        if let Some(max_turns) = cli.max_turns {
            builder = builder.max_turns(max_turns);
        }
    }
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
        if cli.goal.is_some() {
            "goal-driven loop"
        } else if cli.message.is_some() {
            "one-shot --message"
        } else if cli.repl {
            "repl on stdio"
        } else {
            "ACP server on stdio"
        }
    );

    // 出口优先级：--goal > --message > --repl > ACP server。
    // --goal 与 --message 都复用 oneshot runner：单次 run_turn + 消费事件 + 退出码。
    // 区别仅在 agent 装配——goal 模式挂了 goal-gate hook，turn 会在内部被续命多轮，
    // 对 CLI 层透明（仍是一次 run_turn 调用）。
    if let Some(prompt) = cli.goal.or(cli.message) {
        // ask-writes 下 Ask 被降级为 Deny 才算"无人值守缺口"；open/deny-all/read-only
        // 的 Deny 是用户预期的，不参与 denied 退出码。
        let track_denied = matches!(built.sandbox_mode, defect_config::SandboxMode::AskWrites);
        return run_oneshot(
            built.agent,
            cwd,
            prompt,
            cli.format,
            built.resume_session_id,
            track_denied,
            built.goal,
        )
        .await;
    } else if cli.repl {
        run_repl(built.agent, built.resume_session_id).await?;
    } else {
        defect_acp::serve_with_resume(built.agent, built.resume_session_id).await?;
    }
    Ok(ExitCode::SUCCESS)
}

/// 跑单轮 prompt（`--message`）。由 `oneshot` feature gate——裁掉时 hard fail
/// 提示重新带 feature 编译，不静默退化成 ACP。
#[cfg(feature = "oneshot")]
#[allow(clippy::too_many_arguments)]
async fn run_oneshot(
    agent: Arc<dyn AgentCore>,
    cwd: std::path::PathBuf,
    message: String,
    format: OutputFormat,
    resume: Option<SessionId>,
    track_denied: bool,
    goal: Option<Arc<defect_agent::session::GoalState>>,
) -> anyhow::Result<ExitCode> {
    defect_cli::oneshot::run(agent, cwd, message, format, resume, track_denied, goal).await
}

#[cfg(not(feature = "oneshot"))]
#[allow(clippy::too_many_arguments)]
async fn run_oneshot(
    _agent: Arc<dyn AgentCore>,
    _cwd: std::path::PathBuf,
    _message: String,
    _format: OutputFormat,
    _resume: Option<SessionId>,
    _track_denied: bool,
    _goal: Option<Arc<defect_agent::session::GoalState>>,
) -> anyhow::Result<ExitCode> {
    anyhow::bail!(
        "this binary was built without the `oneshot` feature; \
         rebuild with `--features oneshot` (on by default) to use --message / --goal"
    )
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
