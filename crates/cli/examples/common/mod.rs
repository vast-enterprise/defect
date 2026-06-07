//! `defect-cli` examples 的共用 helper。**仅供 examples/ 用**，不进 crate
//! 公共 API（`defect-cli` 是 binary crate，没有 lib target）。
//!
//! 与 [`crates/llm/examples/common/mod.rs`] 区别：本模块面向 ACP 形态——
//! - subscriber 一律 `with_writer(std::io::stderr)`（stdout 给 wire 用）
//! - 默认 EnvFilter 静默 toac 的 `INFO request` 事件（含 authorization
//!   header in plain text (see tracing design)
//!
//! See tracing design.

use std::io::IsTerminal;
use std::path::Path;

/// 装好 tracing：默认 `info,toac=warn`，环境变量 `RUST_LOG` 整体覆盖。
///
/// `toac=warn` 是默认 directive 的一部分——toac wire crate 的 request
/// 事件级别是 `info` 且包含 `headers={"authorization": "Bearer ..."}`，
/// 默认必须 silence 掉，避免 stderr 泄露凭证。需要看 wire 请求时显式
/// 用 `RUST_LOG=...,toac=debug`（debug 不打 headers）。
///
/// **stderr 强制**：stdio ACP 占用 stdout，subscriber 写 stdout 会污染
/// 协议线，客户端解码必炸。
///
/// # Panics
///
/// 重复初始化会 panic——examples 只在 main 调一次。
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,toac=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
}

/// 极简 `.env` 加载器：`KEY=VALUE` 一行一条，`#` 开头注释、空行跳过；
/// 支持外层 `"..."` / `'...'` 包裹去除。**已在进程 env 里的变量保留原值**，
/// 避免 .env 覆盖 shell 显式 export。读不到文件 / 解析失败仅 warn。
///
/// 与 `crates/cli/src/main.rs::load_env_file` 同款实现——examples 不能
/// 引用 binary crate 的私有 fn，只好复制。
pub fn load_env_file(path: &Path) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            eprintln!("warning: failed to read {}: {err}", path.display());
            return;
        }
    };
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if k.is_empty() || std::env::var_os(k).is_some() {
            continue;
        }
        // SAFETY: examples 入口未起其他线程。
        unsafe { std::env::set_var(k, v) };
    }
}
