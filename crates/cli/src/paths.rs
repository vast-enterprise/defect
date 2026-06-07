//! Path resolution helpers — session storage root, etc.

use std::path::{Path, PathBuf};

use directories::ProjectDirs;

/// Default session persistence root directory, resolved via the platform's standard
/// per-app state/data location (the `directories` crate):
/// - Linux: `$XDG_STATE_HOME/defect/sessions` (or `~/.local/state/defect/sessions`)
/// - macOS: `~/Library/Application Support/defect/sessions`
/// - Windows: `%LOCALAPPDATA%\defect\sessions`
///
/// On Linux `state_dir()` follows XDG; on macOS/Windows there is no separate state
/// directory, so we fall back to the per-app data directory.
///
/// # Errors
///
/// Returns an error when no home directory can be determined at all (e.g. a stripped
/// environment with no `HOME` on Unix or no profile dir on Windows) — in that case the
/// OS itself cannot tell us where per-user data belongs.
pub fn default_sessions_root() -> anyhow::Result<PathBuf> {
    let dirs = ProjectDirs::from("", "", "defect").ok_or_else(|| {
        anyhow::anyhow!(
            "cannot resolve session storage root: no home directory found for the current user"
        )
    })?;
    // state_dir() is Some only on Linux (XDG_STATE_HOME); elsewhere use data_dir().
    let base = dirs.state_dir().unwrap_or_else(|| dirs.data_dir());
    Ok(base.join("sessions"))
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
