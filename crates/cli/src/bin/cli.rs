//! `defect` binary entrypoint — minimal CLI lifecycle orchestration only.
//!
//! Default `AgentCore` assembly is encapsulated in
//! [`defect_cli::assembly::CliAgentBuilder`]:
//! downstream developers can trim the default feature set from that builder and layer on
//! their own
//! provider / tool / hook / observer / MCP factory.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use clap::Parser;
use defect_agent::session::AgentCore;
use defect_cli::args::{CliArgs, Command, OutputFormat};
use defect_cli::assembly::{CliAgentBuilder, ReplMode};
use defect_config::{LoadConfigOptions, load_dotenv_compat};
use defect_obs::init_tracing;

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cwd = env::current_dir()?;
    load_dotenv_compat(&cwd).map_err(|e| anyhow::anyhow!("dotenv load failed: {e}"))?;

    let cli = CliArgs::parse();

    // Management subcommands run instead of the agent and exit. `init` writes config, so
    // it must run before any config load / agent assembly (which would require a config
    // that may not exist yet).
    if let Some(command) = cli.command {
        match command {
            Command::Init(args) => {
                defect_cli::init::run(args).await?;
                return Ok(ExitCode::SUCCESS);
            }
        }
    }

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

    // Both `--message` (single-turn) and `--goal` (goal-driven multi-turn loop) are
    // headless, non-interactive modes.
    let headless = cli.message.is_some() || cli.goal.is_some();
    let mut builder = CliAgentBuilder::new(cwd.clone(), load_opts, config).repl(if cli.repl {
        ReplMode::Enabled
    } else {
        ReplMode::Disabled
    });
    // Headless mode: wrap `NonInteractivePolicy` to avoid hanging on permission prompts
    // when there is no TTY.
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

    // Exit priority: --goal > --message > --repl > ACP server.
    // Both --goal and --message reuse the oneshot runner: a single run_turn + event
    // consumption + exit code.
    // The only difference is in agent assembly — goal mode attaches a goal-gate hook, so
    // turns are internally extended across multiple rounds,
    // transparent to the CLI layer (still a single run_turn call).
    if let Some(prompt) = cli.goal.or(cli.message) {
        // Under `ask-writes`, only an `Ask` that is downgraded to `Deny` counts as an
        // "unattended gap"; `Deny` in `open`/`deny-all`/`read-only` modes is expected by
        // the user and does not contribute to the denied exit code.
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

/// Run a single-turn prompt (`--message`). Gated by the `oneshot` feature; when the
/// feature is disabled, hard-fail with a message to recompile with the feature enabled,
/// rather than silently falling back to ACP.
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

/// Starts the REPL. When the `repl` feature is enabled, runs the actual REPL; when it is
/// disabled, the flag is still parsed but fails at runtime with a hard error instructing
/// the user to rebuild with the feature enabled — it does not silently degrade to ACP.
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
