//! tracing-subscriber initialization.
//!
//! Process-level — must be called only once. `RUST_LOG` takes precedence over the config
//! file's `[tracing].filter`.

use tracing_subscriber::EnvFilter;

const DEFAULT_FILTER: &str = "info,toac=warn";

/// Initializes the global tracing subscriber.
///
/// Resolution order: `RUST_LOG` env > argument `filter` > `DEFAULT_FILTER`.
/// Output goes to stderr; ANSI colors are enabled automatically when stderr is a
/// terminal.
pub fn init_tracing(filter: Option<&str>) -> anyhow::Result<()> {
    let default_filter = filter.unwrap_or(DEFAULT_FILTER);
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))
}
