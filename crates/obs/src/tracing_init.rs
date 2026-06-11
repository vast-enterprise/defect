//! tracing-subscriber initialization.
//!
//! Process-level — must be called only once. `RUST_LOG` takes precedence over the config
//! file's `[tracing].filter`.

use tracing_subscriber::EnvFilter;

const DEFAULT_FILTER: &str = "info,toac=warn";

/// Initializes the global tracing subscriber.
///
/// Resolution order: `RUST_LOG` env > argument `filter` > `DEFAULT_FILTER`.
/// Output goes to stderr. When `jsonl` is true, each log record is emitted as one JSON
/// line (JSONL/NDJSON); otherwise human-readable text with ANSI colors auto-enabled when
/// stderr is a terminal.
pub fn init_tracing(filter: Option<&str>, jsonl: bool) -> anyhow::Result<()> {
    let default_filter = filter.unwrap_or(DEFAULT_FILTER);
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_target(true);
    if jsonl {
        builder
            .json()
            .try_init()
            .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))
    } else {
        builder
            .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
            .try_init()
            .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))
    }
}
