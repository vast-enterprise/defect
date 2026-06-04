//! 路径解析 helper——session storage root 等。

use std::env;
use std::path::{Path, PathBuf};

/// 默认 session 持久化根目录。优先级：
/// 1. `XDG_STATE_HOME/defect/sessions`
/// 2. `$HOME/.local/state/defect/sessions`
///
/// # Errors
///
/// 当 `XDG_STATE_HOME` 与 `HOME` 均未设置时返回错误。
pub fn default_sessions_root() -> anyhow::Result<PathBuf> {
    if let Ok(xdg_state_home) = env::var("XDG_STATE_HOME") {
        return Ok(PathBuf::from(xdg_state_home).join("defect/sessions"));
    }
    if let Ok(home) = env::var("HOME") {
        return Ok(PathBuf::from(home).join(".local/state/defect/sessions"));
    }
    Err(anyhow::anyhow!(
        "cannot resolve session storage root: neither XDG_STATE_HOME nor HOME is set"
    ))
}

/// `--local` 沙盒模式的 session 持久化根目录：`<repo-root>/.defect/sessions`。
///
/// 项目根用 `.git` 探测（与配置层的项目层同源）；找不到 `.git` 时退回
/// `cwd/.defect/sessions`，让无仓库的沙盒目录也能用。
#[must_use]
pub fn local_sessions_root(cwd: &Path) -> PathBuf {
    let root = defect_config::find_repo_root(cwd).unwrap_or_else(|| cwd.to_path_buf());
    root.join(".defect/sessions")
}
