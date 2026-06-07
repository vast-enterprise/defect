//! Filesystem backend abstraction.
//!
//! [`FsBackend`] is the trait boundary between the fs tool family (`read_file` /
//! `write_file` / `edit_file`) and the underlying I/O. Two v0 implementations:
//! - `defect_tools::fs::LocalFsBackend`: writes directly to disk
//! - `defect_acp::fs::AcpFsBackend`: delegates to the client via ACP `fs/read_text_file`
//!   / `fs/write_text_file` reverse requests
//!
//! Assembly is handled in the `defect-acp` `session/new` handler — the backend is
//! selected based on the client's [`FileSystemCapabilities`] negotiation result and
//! injected into [`crate::session::AgentCore::create_session`].

//! [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// A fingerprint of file content. Used with [`FsBackend::fingerprint`] and
/// [`Fingerprint::of`]:
/// `edit_file` records the fingerprint after reading, and takes it again before writing;
/// a mismatch indicates a concurrent write conflict.
///
/// Uses `(bytes, hash)` instead of a plain hash: comparing both length and hash reduces
/// the collision probability of a single `u64` hash to negligible. `DefaultHasher` is
/// only used for in-process one-shot comparisons, never persisted or shared across
/// processes, so the standard library's "unspecified but stable" semantics are
/// acceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    pub bytes: u64,
    pub hash: u64,
}

impl Fingerprint {
    /// Compute a fingerprint directly from a text string. `edit_file` uses this after
    /// reading `old_content` to avoid re-reading before writing.
    pub fn of(content: &str) -> Self {
        let mut h = DefaultHasher::new();
        content.hash(&mut h);
        Self {
            bytes: content.len() as u64,
            hash: h.finish(),
        }
    }
}

/// A no-op fs backend for testing only. All methods return [`FsError::NotPermitted`],
/// allowing test scenarios that require `Arc<dyn FsBackend>` (without actually running fs
/// tools) to skip setup.
///
/// In production, use `defect_tools::fs::LocalFsBackend` or
/// `defect_acp::fs::AcpFsBackend`.
pub struct NoopFsBackend;

impl FsBackend for NoopFsBackend {
    fn read_text(
        &self,
        _path: PathBuf,
        _line: Option<u32>,
        _limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>> {
        Box::pin(async {
            Err(FsError::NotPermitted(
                "NoopFsBackend cannot read".to_string(),
            ))
        })
    }

    fn write_text(&self, _path: PathBuf, _content: String) -> BoxFuture<'_, Result<(), FsError>> {
        Box::pin(async {
            Err(FsError::NotPermitted(
                "NoopFsBackend cannot write".to_string(),
            ))
        })
    }
}

/// Fs backend trait.
///
/// Two verbs cover all low-level operations of the v0 fs tool family:
/// - `edit_file` is composed at the tool layer (first [`read_text`] then
///   [`write_text`](FsBackend::write_text));
///   the backend is unaware of patch semantics
/// - Delete / move / mkdir are not part of the v0 fs tool family (ACP has no
///   corresponding inverse methods);
///   the LLM uses `bash`
///
/// Parameters use owned `PathBuf` / `String` to confine the future's lifetime to `&'_
/// self`,
/// avoiding explicit lifetime parameters; same trade-off as `LlmProvider::complete`.
///
/// [`read_text`]: FsBackend::read_text
pub trait FsBackend: Send + Sync {
    /// Reads the entire file as UTF-8 text.
    ///
    /// `line` / `limit` have the same semantics as ACP `ReadTextFileRequest`:
    /// - `line = Some(n)` starts reading from line n (1-based)
    /// - `limit = Some(k)` reads at most k lines
    /// - Both `None` reads the full file
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>>;

    /// Reads the raw bytes of an entire file. The `read_file` tool takes this path when
    /// it detects a binary type such as an image, passing the bytes to the caller for
    /// base64 encoding into a multimodal `tool_result`.
    ///
    /// The default implementation returns [`FsError::NotPermitted`] — the delegated
    /// backend (`AcpFsBackend`) uses the ACP `fs/read_text_file` reverse channel, which
    /// is text-only and cannot obtain binary data. In ACP environments, reading images is
    /// discouraged by the system prompt (the `# Environment` section notes that the
    /// frontend is delegated). The local backend (`LocalFsBackend`) overrides this to
    /// read directly from disk.
    fn read_bytes(&self, path: PathBuf) -> BoxFuture<'_, Result<Vec<u8>, FsError>> {
        Box::pin(async move {
            let _ = path;
            Err(FsError::NotPermitted(
                "this backend cannot read raw bytes (e.g. images); delegated environments only support text reads".to_string(),
            ))
        })
    }

    /// Write a UTF-8 text file, overwriting any existing content.
    ///
    /// The backend is responsible for ensuring the parent directory exists (`mkdir -p`
    /// semantics).
    ///
    /// Line-ending / atomicity responsibilities are split as:
    /// - Local backend performs line-ending normalization and atomic write via `tmp +
    ///   rename`
    /// - Delegated backend leaves the decision to the client
    fn write_text(&self, path: PathBuf, content: String) -> BoxFuture<'_, Result<(), FsError>>;

    /// Returns a "content fingerprint" used by `edit_file` to detect concurrent write
    /// conflicts in the read–modify–write window.
    ///
    /// The default implementation reads the full content via [`FsBackend::read_text`] and
    /// computes [`Fingerprint::of`] — this allows delegating backends (e.g.
    /// `AcpFsBackend`) to work without additional protocol methods. Local backends may
    /// override this method to use cheaper checks like mtime + size.
    fn fingerprint(&self, path: PathBuf) -> BoxFuture<'_, Result<Fingerprint, FsError>> {
        Box::pin(async move {
            let text = self.read_text(path, None, None).await?;
            Ok(Fingerprint::of(&text))
        })
    }
}

/// Fs backend error.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FsError {
    /// File not found.
    #[error("file not found: {0}")]
    NotFound(PathBuf),

    /// Operation not permitted: path out of bounds, binary file, client deny,
    /// insufficient permissions, etc.
    /// v0 uses a string placeholder; upgrade to an enum in a later iteration.
    #[error("operation not permitted: {0}")]
    NotPermitted(String),

    /// File exceeds the size threshold.
    #[error("file too large: {bytes} bytes > {limit}")]
    TooLarge { bytes: u64, limit: u64 },

    /// File was externally modified during a read-modify-write cycle.
    /// `edit_file` compares fingerprints via [`FsBackend::fingerprint`] before writing:
    /// a mismatch raises `Conflict`, prompting the LLM to re-read and re-edit instead of
    /// overwriting.
    #[error("file changed since last read: {0}")]
    Conflict(PathBuf),

    /// Underlying I/O or RPC failure.
    #[error("backend failure: {0}")]
    Backend(#[source] BoxError),
}

/// Resolves a request path to an absolute path within the workspace, verifying it does
/// not escape.
///
/// Behavior:
/// 1. Relative paths are joined with `workspace_root`; absolute paths are used as-is.
/// 2. Walks up from the target to find the nearest **existing** ancestor and
///    canonicalizes it
///    (on writes, the target itself and even multiple parent directories may not yet
///    exist).
/// 3. Checks that the real path of the existing ancestor starts with the real path of
///    `workspace_root` —
///    prevents symlink escape (e.g. `workspace/dir/link → /etc`).
/// 4. Appends the remaining non-existent path segments as-is, then appends the file name.
///
/// Both `LocalFsBackend` and `AcpFsBackend` implementations of [`crate::fs::FsBackend`]
/// call
/// this same function — in delegated mode the agent still enforces its own boundary, not
/// relying on the client.
///
/// # Errors
/// - [`FsError::NotPermitted`]: path escapes / no parent directory / no file name
/// - [`FsError::Backend`]: canonicalization of ancestor failed (IO error)
pub fn resolve_workspace_path(workspace_root: &Path, requested: &Path) -> Result<PathBuf, FsError> {
    let target = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        workspace_root.join(requested)
    };

    let parent = target.parent().ok_or_else(|| {
        FsError::NotPermitted(format!("path has no parent: {}", target.display()))
    })?;

    // Walk up from `parent` to find the nearest existing ancestor directory.
    // `canonicalize` requires the path to exist — in a write scenario the target
    // and even multiple parent directories may not yet exist, so we walk up to
    // the first real directory before calling `canonicalize`.
    let (existing_ancestor, missing_suffix) = find_existing_ancestor(parent).ok_or_else(|| {
        FsError::NotPermitted(format!(
            "no existing ancestor found for: {}",
            target.display()
        ))
    })?;

    let existing_canon =
        std::fs::canonicalize(existing_ancestor).map_err(|e| FsError::Backend(BoxError::new(e)))?;

    let root_canon =
        std::fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());

    if !existing_canon.starts_with(&root_canon) {
        return Err(FsError::NotPermitted(format!(
            "path {} escapes workspace root {}",
            target.display(),
            root_canon.display()
        )));
    }

    let file_name = target.file_name().ok_or_else(|| {
        FsError::NotPermitted(format!("path has no file component: {}", target.display()))
    })?;

    // Append the missing path segments back to the existing ancestor, then join the file
    // name.
    Ok(existing_canon.join(missing_suffix).join(file_name))
}

/// Walk upward from `path`, returning `(nearest existing ancestor, remaining path
/// segments)`.
///
/// The remaining path segments preserve their original relative structure (not
/// canonicalized),
/// so that reassembly retains the original semantics.
fn find_existing_ancestor(path: &Path) -> Option<(&Path, PathBuf)> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        if current.exists() {
            // The path segments collected from bottom to top need to be reversed before
            // joining.
            missing.reverse();
            return Some((current, missing.into_iter().collect()));
        }
        missing.push(current.file_name()?.to_os_string());
        current = current.parent()?;
    }
}

#[cfg(test)]
mod tests;
