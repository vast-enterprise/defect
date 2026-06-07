//! CLI argument parsing.
//!
//! Aligned with `defect-config`'s `LoadConfigOptions::cli` — CLI flags take precedence.
//! CLI arguments — see config and `CliOverrides`.

use clap::{Parser, ValueEnum};

use defect_config::{
    CliOverrides, ProviderKind as ConfigProviderKind, SandboxMode, parse_cli_override,
};

/// Values for `--sandbox`. Mirrors [`SandboxMode`] locally so that clap can render the
/// possible values directly;
/// the config crate does not depend on clap, so it does not derive `ValueEnum` there
/// (following the same
/// "CLI-side parsing" pattern used by providers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SandboxModeArg {
    ReadOnly,
    AskWrites,
    Open,
    DenyAll,
}

impl From<SandboxModeArg> for SandboxMode {
    fn from(arg: SandboxModeArg) -> Self {
        match arg {
            SandboxModeArg::ReadOnly => SandboxMode::ReadOnly,
            SandboxModeArg::AskWrites => SandboxMode::AskWrites,
            SandboxModeArg::Open => SandboxMode::Open,
            SandboxModeArg::DenyAll => SandboxMode::DenyAll,
        }
    }
}

/// Output format for stdout in `--message` single-turn mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum OutputFormat {
    /// Plain text, no ANSI: assistant body to stdout, thoughts/tools to stderr.
    #[default]
    Text,
    /// One JSON line (NDJSON) per `AgentEvent` to stdout.
    Json,
    /// Silent mode; only prints the final result or error at the end.
    Quiet,
}

/// Headless agent over ACP/stdio.
#[derive(Debug, Parser)]
#[command(
    name = "defect",
    about = "Headless agent over ACP/stdio",
    long_about = "defect — headless agent over ACP/stdio.\n\n\
                  Auth env: ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY.\n\
                  Logging: RUST_LOG controls tracing-subscriber EnvFilter (default: info)."
)]
pub struct CliArgs {
    /// LLM provider to use. CLI flag takes precedence over the `DEFECT_PROVIDER`
    /// environment variable and config file.
    #[arg(long, env = "DEFECT_PROVIDER")]
    pub provider: Option<String>,

    /// Override the default model ID. CLI flag takes precedence over the `DEFECT_MODEL`
    /// environment variable.
    #[arg(long, env = "DEFECT_MODEL")]
    pub model: Option<String>,

    /// Override the sandbox permission mode. CLI flag takes precedence over config
    /// `[sandbox].mode`. Useful for CI: `--sandbox open` runs every tool without
    /// prompting. Note that `--repl` always forces `open` regardless.
    #[arg(long, value_enum)]
    pub sandbox: Option<SandboxModeArg>,

    /// Shortcut for `--sandbox open`: grants maximum permissions and runs every tool
    /// without prompting. Mutually exclusive with `--sandbox`.
    #[arg(long, conflicts_with = "sandbox")]
    pub yolo: bool,

    /// Run the entire session under a named subagent profile, located in
    /// `.defect/agents/<name>/` or `~/.config/defect/agents/<name>/`.
    /// The profile's model, system prompt, and tool allowlist become the session root.
    /// The CLI flag takes precedence over the `DEFECT_PROFILE` environment variable.
    #[arg(long, env = "DEFECT_PROFILE")]
    pub profile: Option<String>,

    /// Additional dotted-path config overrides; may be repeated.
    #[arg(long = "config", value_name = "KEY=VALUE")]
    pub config_override: Vec<String>,

    /// Resume a previous session. With a `SESSION_ID`, resume that specific session; bare
    /// `--resume` resumes the most recently active session for the current working
    /// directory. In ACP mode, the resumed session is returned on the first
    /// `session/new`; in `--repl` mode, it is loaded directly.
    #[arg(long, value_name = "SESSION_ID")]
    pub resume: Option<Option<String>>,

    /// Sandbox mode: ignore global/user config and store all state (config, sessions)
    /// under `<repo-root>/.defect/`. The user-level `~/.config/defect` config, agents,
    /// and skills are skipped entirely.
    #[arg(long)]
    pub local: bool,

    /// Run a minimal in-process REPL on stdin/stdout instead of the ACP server. Requires
    /// the `repl` build feature (enabled by default); a binary built with
    /// `--no-default-features` rejects this flag at runtime.
    #[arg(long)]
    pub repl: bool,

    /// Run a single prompt turn headlessly and exit (CI / scripting). The assistant
    /// output goes to stdout; the process exit code reflects the turn outcome. A value of
    /// `-`, or no value while stdin is piped, reads the prompt from stdin. Combine with
    /// `--resume` to continue a previous session. Mutually exclusive with `--repl`.
    /// Requires the `oneshot` build feature (on by default).
    #[arg(long, value_name = "PROMPT", conflicts_with = "repl")]
    pub message: Option<String>,

    /// Output format for `--message` / `--goal` mode.
    #[arg(long, value_enum, default_value_t = OutputFormat::default())]
    pub format: OutputFormat,

    /// Run a goal-driven autonomous loop and exit (for CI / scripting). The agent works
    /// across multiple turns until it calls the `goal_done` tool (goal achieved) or
    /// reaches `--max-turns`. Like `--message`, but continues until the goal is reached
    /// instead of stopping after one turn. Reads from stdin if the value is `-` or
    /// omitted while piped. Mutually exclusive with `--message` and `--repl`. Requires
    /// the `oneshot` build feature.
    #[arg(long, value_name = "OBJECTIVE", conflicts_with_all = ["message", "repl"])]
    pub goal: Option<String>,

    /// Maximum number of times the goal-gate may keep the agent going before giving up
    /// (maps to `[turn].max_hook_continues`). Only meaningful with `--goal`. When
    /// exceeded, the run exits with a non-zero (exhausted) code.
    #[arg(long, value_name = "N")]
    pub max_turns: Option<u32>,
}

impl CliArgs {
    /// Translates CLI flags into [`CliOverrides`] and feeds them to
    /// `defect_config::load_config`.
    pub fn to_overrides(&self) -> anyhow::Result<CliOverrides> {
        let config_overrides = self
            .config_override
            .iter()
            .map(|spec| parse_cli_override(spec).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        // `--yolo` is syntactic sugar for `--sandbox open` (clap ensures they are
        // mutually exclusive).
        let sandbox = if self.yolo {
            Some(SandboxMode::Open)
        } else {
            self.sandbox.map(SandboxMode::from)
        };
        Ok(CliOverrides {
            provider: self.provider.as_deref().map(ConfigProviderKind::from),
            model: self.model.clone(),
            sandbox,
            config_overrides,
        })
    }
}
