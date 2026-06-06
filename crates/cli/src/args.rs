//! CLI 参数解析。
//!
//! 与 `defect-config` 的 `LoadConfigOptions::cli` 对齐——CLI flag 优先级
//! 见 `docs/internal/config.md` §2 / `defect_config::CliOverrides`。

use clap::{Parser, ValueEnum};

use defect_config::{
    CliOverrides, ProviderKind as ConfigProviderKind, SandboxMode, parse_cli_override,
};

/// `--sandbox` 取值。本地镜像 [`SandboxMode`]，让 clap 直接渲染可选值；
/// config crate 不依赖 clap，故不在那侧 derive `ValueEnum`（沿用 provider
/// 的"CLI 侧解析"模式）。
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

/// `--message` 单轮模式的 stdout 输出形态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum OutputFormat {
    /// 纯文本，无 ANSI：助手正文到 stdout，思考/工具到 stderr。
    #[default]
    Text,
    /// 每个 `AgentEvent` 一行 JSON（NDJSON）到 stdout。
    Json,
    /// 过程静默，仅在结束时输出最终结果 / 错误。
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
    /// LLM provider to use. CLI flag wins over DEFECT_PROVIDER env and config.
    #[arg(long, env = "DEFECT_PROVIDER")]
    pub provider: Option<String>,

    /// Override the default model id. CLI flag wins over DEFECT_MODEL env.
    #[arg(long, env = "DEFECT_MODEL")]
    pub model: Option<String>,

    /// Override the sandbox permission mode. CLI flag wins over config
    /// `[sandbox].mode`. Useful for CI: `--sandbox open` runs every tool
    /// without prompting. Note `--repl` always forces `open` regardless.
    #[arg(long, value_enum)]
    pub sandbox: Option<SandboxModeArg>,

    /// Shortcut for `--sandbox open`: grant maximum permissions, run every
    /// tool without prompting. Mutually exclusive with `--sandbox`.
    #[arg(long, conflicts_with = "sandbox")]
    pub yolo: bool,

    /// Run the whole session as a named subagent profile (from
    /// `.defect/agents/<name>/` or `~/.config/defect/agents/<name>/`).
    /// Applies the profile's model, system prompt, and tool allowlist as the
    /// session root. CLI flag wins over DEFECT_PROFILE env.
    #[arg(long, env = "DEFECT_PROFILE")]
    pub profile: Option<String>,

    /// Additional dotted-path config overrides. May be repeated.
    #[arg(long = "config", value_name = "KEY=VALUE")]
    pub config_override: Vec<String>,

    /// Resume a previous session. With a SESSION_ID, resume that session;
    /// bare `--resume` resumes the most recently active session for the
    /// current working directory. In ACP mode the resumed session is
    /// returned on the first `session/new`; in `--repl` it is loaded
    /// directly.
    #[arg(long, value_name = "SESSION_ID")]
    pub resume: Option<Option<String>>,

    /// Sandbox mode: ignore global/user config and store all state
    /// (config, sessions) under `<repo-root>/.defect/`. The user-level
    /// `~/.config/defect` config, agents, and skills are skipped entirely.
    #[arg(long)]
    pub local: bool,

    /// Run a minimal in-process REPL on stdin/stdout instead of the ACP
    /// server. Requires the `repl` build feature (on by default); a binary
    /// built with `--no-default-features` rejects this flag at runtime.
    #[arg(long)]
    pub repl: bool,

    /// Run a single prompt turn headlessly and exit (CI / scripting). The
    /// assistant output goes to stdout; the process exit code reflects the
    /// turn outcome. A value of `-`, or no value while stdin is piped, reads
    /// the prompt from stdin. Combine with `--resume` to continue a previous
    /// session. Mutually exclusive with `--repl`. Requires the `oneshot`
    /// build feature (on by default).
    #[arg(long, value_name = "PROMPT", conflicts_with = "repl")]
    pub message: Option<String>,

    /// Output format for `--message` / `--goal` mode.
    #[arg(long, value_enum, default_value_t = OutputFormat::default())]
    pub format: OutputFormat,

    /// Run a goal-driven autonomous loop and exit (CI / scripting). The agent
    /// works across multiple turns until it calls the `goal_done` tool (goal
    /// achieved) or hits `--max-turns`. Like `--message` but keeps going until
    /// the goal is reached instead of stopping after one turn. Reads from
    /// stdin if the value is `-` or omitted while piped. Mutually exclusive
    /// with `--message` and `--repl`. Requires the `oneshot` build feature.
    #[arg(long, value_name = "OBJECTIVE", conflicts_with_all = ["message", "repl"])]
    pub goal: Option<String>,

    /// Maximum number of times the goal-gate may keep the agent going before
    /// giving up (maps to `[turn].max_hook_continues`). Only meaningful with
    /// `--goal`. When exceeded, the run exits with a non-zero (exhausted) code.
    #[arg(long, value_name = "N")]
    pub max_turns: Option<u32>,
}

impl CliArgs {
    /// 把 CLI flag 翻成 [`CliOverrides`]，喂给 `defect_config::load_config`。
    pub fn to_overrides(&self) -> anyhow::Result<CliOverrides> {
        let config_overrides = self
            .config_override
            .iter()
            .map(|spec| parse_cli_override(spec).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        // `--yolo` 是 `--sandbox open` 的糖（clap 已保证二者互斥）。
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
