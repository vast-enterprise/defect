//! Path resolution helpers — session storage root, etc.

use std::env;
use std::path::{Path, PathBuf};

/// Default session persistence root directory. Priority:
/// 1. `XDG_STATE_HOME/defect/sessions`
/// 2. `$HOME/.local/state/defect/sessions`
///
/// # Errors
///
/// Returns an error when neither `XDG_STATE_HOME` nor `HOME` is set.
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

/// Session persistence root directory for `--local` sandbox mode:
/// `<repo-root>/.defect/sessions`.
///
/// The project root is detected via `.git` (same source as the project layer in the
/// config system); if no `.git` is found, falls back to `cwd/.defect/sessions` so that
/// sandbox directories without a repository can still be used.
#[must_use]
pub fn local_sessions_root(cwd: &Path) -> PathBuf {
    let root = defect_config::find_repo_root(cwd).unwrap_or_else(|| cwd.to_path_buf());
    root.join(".defect/sessions")
}
