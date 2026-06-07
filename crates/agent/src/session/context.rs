//! Environment context: factual information injected into the `# Environment` section of
//! the system prompt.
//!
//! The [`RunningContext`] aggregate holds session-varying parts (access method, cwd),
//! while platform/version/shell detection results that are invariant for the entire
//! process are cached with [`OnceLock`] — `os_info::get()` reads files and runs probes,
//! so recomputing it every turn is not worthwhile.

use std::path::Path;
use std::sync::OnceLock;

/// How the agent is connected — determines its understanding of the file and command
/// execution environment.
///
/// Note: The `defect` CLI binary itself runs as an ACP server over stdio; all real paths
/// currently go through [`Frontend::Acp`]. [`Frontend::Cli`] and [`Frontend::Headless`]
/// are variants reserved for future forms (bare CLI user, server backend service).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frontend {
    /// Direct CLI interaction (reserved for a future local CLI user mode).
    Cli,
    /// Accessed via the ACP protocol (editor / IDE client).
    ///
    /// `fs_delegated` / `shell_delegated` come from the ACP `initialize` handshake
    /// negotiation:
    /// `true` means file I/O / command execution is delegated to the client proxy,
    /// `false` means
    /// executed locally. The agent uses this to know whether it is facing a local
    /// environment or a remote proxy.
    Acp {
        fs_delegated: bool,
        shell_delegated: bool,
    },
    /// Headless (reserved for a background service on the server).
    Headless,
}

impl Frontend {
    /// Whether the filesystem is delegated to the client proxy. Only true for
    /// [`Frontend::Acp`] when `fs_delegated = true` is negotiated; all other variants
    /// read and write directly on the local side.
    fn fs_delegated(self) -> bool {
        matches!(
            self,
            Self::Acp {
                fs_delegated: true,
                ..
            }
        )
    }

    /// A single-line description rendered into the `# Environment` section.
    fn describe(self) -> String {
        match self {
            Self::Cli => "CLI".to_owned(),
            Self::Acp {
                fs_delegated,
                shell_delegated,
            } => format!(
                "ACP (fs: {}, shell: {})",
                delegation(fs_delegated),
                delegation(shell_delegated),
            ),
            Self::Headless => "headless".to_owned(),
        }
    }
}

fn delegation(delegated: bool) -> &'static str {
    if delegated { "delegated" } else { "local" }
}

/// Injects runtime environment context into the system prompt.
///
/// Static parts (platform, version, shell) are detected and cached internally by this
/// type; callers only need to provide the session‑varying [`Frontend`] and `cwd`.
pub struct RunningContext<'a> {
    pub frontend: Frontend,
    pub cwd: &'a Path,
}

impl<'a> RunningContext<'a> {
    pub fn new(frontend: Frontend, cwd: &'a Path) -> Self {
        Self { frontend, cwd }
    }

    /// Renders the body of the `# Environment` section (the title and separator are
    /// handled by
    /// [`crate::session::resolve_system_prompt`]).
    pub fn render(&self) -> String {
        let mut lines = Vec::with_capacity(6);
        lines.push(format!("- platform: {}", platform_line()));
        lines.push(format!("- defect version: {}", env!("CARGO_PKG_VERSION")));
        lines.push(format!("- frontend: {}", self.frontend.describe()));
        lines.push(format!("- cwd: {}", self.cwd.display()));
        lines.push(format!("- shell: {}", shell_line()));
        // The delegated filesystem (ACP) backchannel only supports text; `read_file` will
        // fail on images.
        // Explicitly instruct the model not to use `read_file` on images to avoid
        // unnecessary error round-trips.
        if self.frontend.fs_delegated() {
            lines.push(
                "- note: the filesystem is delegated and only supports text reads; \
                 do not use read_file on image or other binary files (it will fail)"
                    .to_owned(),
            );
        }
        lines.join("\n")
    }
}

/// Format: `linux / x86_64 (Ubuntu 22.04)`. OS/arch from compile-time `std` constants,
/// distro and version from runtime `os_info` detection. Cached for the entire process.
fn platform_line() -> &'static str {
    static PLATFORM: OnceLock<String> = OnceLock::new();
    PLATFORM.get_or_init(|| {
        let info = os_info::get();
        format!(
            "{} / {} ({} {})",
            std::env::consts::OS,
            std::env::consts::ARCH,
            info.os_type(),
            info.version(),
        )
    })
}

/// Default shell: reads `$SHELL`, falls back to `unknown`. Cached for the process
/// lifetime.
fn shell_line() -> &'static str {
    static SHELL: OnceLock<String> = OnceLock::new();
    SHELL.get_or_init(|| std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_owned()))
}

#[cfg(test)]
mod tests;
