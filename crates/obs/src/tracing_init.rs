//! tracing-subscriber 初始化。
//!
//! 进程级——只能调用一次。`RUST_LOG` 优先于配置文件 `[tracing].filter`。

use tracing_subscriber::EnvFilter;

const DEFAULT_FILTER: &str = "info,toac=warn";

/// 初始化全局 tracing subscriber。
///
/// 解析顺序：`RUST_LOG` env > 入参 `filter` > [`DEFAULT_FILTER`]。
/// 输出走 stderr，自动检测终端启用 ANSI 颜色。
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
