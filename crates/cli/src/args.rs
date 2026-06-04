//! CLI 参数解析。
//!
//! 与 `defect-config` 的 `LoadConfigOptions::cli` 对齐——CLI flag 优先级
//! 见 `docs/internal/config.md` §2 / `defect_config::CliOverrides`。

use clap::Parser;

use defect_config::{CliOverrides, ProviderKind as ConfigProviderKind, parse_cli_override};

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

    /// Run a single prompt turn headlessly and exit. The assistant output is
    /// streamed to stdout; the process exits when the turn ends. Combine with
    /// `--resume` to continue a previous session. Mutually exclusive with
    /// `--repl`.
    #[arg(long, value_name = "PROMPT", conflicts_with = "repl")]
    pub message: Option<String>,
}

impl CliArgs {
    /// 把 CLI flag 翻成 [`CliOverrides`]，喂给 `defect_config::load_config`。
    pub fn to_overrides(&self) -> anyhow::Result<CliOverrides> {
        let config_overrides = self
            .config_override
            .iter()
            .map(|spec| parse_cli_override(spec).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(CliOverrides {
            provider: self.provider.as_deref().map(ConfigProviderKind::from),
            model: self.model.clone(),
            config_overrides,
        })
    }
}
