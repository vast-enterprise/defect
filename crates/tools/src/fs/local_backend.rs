//! [`LocalFsBackend`]: a direct-to-disk [`FsBackend`] implementation.
//!
//! Local filesystem backend — implements two key invariants:
//! - **Line-ending normalization**: when writing to an existing file, normalizes
//!   new content to match the file's dominant line ending (CRLF / LF), avoiding mixed
//!   line endings.
//! - **Atomic writes**: performs full overwrites via a temporary file + `rename`,
//!   preventing partial files.
//!
//! Path validation is delegated to [`defect_agent::fs::resolve_workspace_path`] —
//! `LocalFsBackend` and `AcpFsBackend` share the same function.

use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs;

use defect_agent::error::BoxError;
use defect_agent::fs::{Fingerprint, FsBackend, FsError, resolve_workspace_path};
use futures::future::BoxFuture;

/// Hard upper bound for single-file size (shared by read and write).
pub const MAX_FS_BYTES: u64 = 10 * 1024 * 1024;

/// Monotonic in-process counter used during `tmp + rename` to prevent concurrent writes
/// to the same path within the same process.
static TMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// A disk-backed [`FsBackend`] implementation.
///
/// Holds the session's workspace root; all reads and writes are first validated by
/// [`resolve_workspace_path`].
pub struct LocalFsBackend {
    workspace_root: PathBuf,
}

impl LocalFsBackend {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }
}

impl FsBackend for LocalFsBackend {
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;

            let metadata = fs::metadata(&abs).await.map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => FsError::NotFound(abs.clone()),
                _ => FsError::Backend(BoxError::new(e)),
            })?;

            // Full reads are blocked by a hard size limit. Windowed reads (when `line` or
            // `limit` is `Some`) use a chunked-read path that streams line by line,
            // buffering only the requested window. This implements large-file
            // pagination, allowing the LLM to navigate log/data files larger than
            // 10 MiB via offset/limit without exceeding the overall memory budget.
            let windowed = line.is_some() || limit.is_some();
            if !windowed && metadata.len() > MAX_FS_BYTES {
                return Err(FsError::TooLarge {
                    bytes: metadata.len(),
                    limit: MAX_FS_BYTES,
                });
            }

            if windowed {
                return read_window_streaming(&abs, line, limit).await;
            }

            let bytes = fs::read(&abs).await.map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => FsError::NotFound(abs.clone()),
                _ => FsError::Backend(BoxError::new(e)),
            })?;

            if looks_binary(&bytes) {
                return Err(FsError::NotPermitted(format!(
                    "binary file: {}",
                    abs.display()
                )));
            }

            let text = String::from_utf8(bytes)
                .map_err(|e| FsError::NotPermitted(format!("file is not valid UTF-8: {e}")))?;

            Ok(slice_lines(&text, line, limit))
        })
    }

    fn read_bytes(&self, path: PathBuf) -> BoxFuture<'_, Result<Vec<u8>, FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;

            let metadata = fs::metadata(&abs).await.map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => FsError::NotFound(abs.clone()),
                _ => FsError::Backend(BoxError::new(e)),
            })?;
            if metadata.len() > MAX_FS_BYTES {
                return Err(FsError::TooLarge {
                    bytes: metadata.len(),
                    limit: MAX_FS_BYTES,
                });
            }

            fs::read(&abs).await.map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => FsError::NotFound(abs.clone()),
                _ => FsError::Backend(BoxError::new(e)),
            })
        })
    }

    /// Use mtime + size as the fingerprint — much cheaper than the default "read entire
    /// file + hash" approach, and sufficient for conflict detection semantics: a
    /// change in mtime or size is treated as a conflict.
    fn fingerprint(&self, path: PathBuf) -> BoxFuture<'_, Result<Fingerprint, FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;
            let metadata = fs::metadata(&abs).await.map_err(|e| match e.kind() {
                io::ErrorKind::NotFound => FsError::NotFound(abs.clone()),
                _ => FsError::Backend(BoxError::new(e)),
            })?;

            let size = metadata.len();
            let mtime_nanos = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);

            // Store `mtime_nanos` in the `hash` field and `size` in the `bytes` field —
            // [`Fingerprint`] equality compares the two fields directly as `(size,
            // mtime)`.
            Ok(Fingerprint {
                bytes: size,
                hash: mtime_nanos,
            })
        })
    }

    fn write_text(&self, path: PathBuf, content: String) -> BoxFuture<'_, Result<(), FsError>> {
        Box::pin(async move {
            let abs = resolve_workspace_path(&self.workspace_root, &path)?;

            if content.len() as u64 > MAX_FS_BYTES {
                return Err(FsError::TooLarge {
                    bytes: content.len() as u64,
                    limit: MAX_FS_BYTES,
                });
            }

            // Normalize line endings: only normalize when the file already exists, using
            // its existing line-ending convention; for new files, preserve the original
            // line endings from the LLM output.
            let final_content: Cow<'_, str> = match tokio::fs::read(&abs).await {
                Ok(prev_bytes) => {
                    let prev = String::from_utf8_lossy(&prev_bytes);
                    let target = detect_line_ending(&prev);
                    normalize(&content, target)
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => Cow::Borrowed(content.as_str()),
                Err(e) => return Err(FsError::Backend(BoxError::new(e))),
            };

            atomic_write(&abs, final_content.as_bytes())
                .await
                .map_err(|e| FsError::Backend(BoxError::new(e)))?;

            Ok(())
        })
    }
}

/// Streaming read window: scans the file line by line, accumulating content only within
/// the range [start, start+take).
///
/// Unlike [`slice_lines`], which requires the entire file to be in memory as a
/// [`String`], this approach uses `BufReader::read_line` and discards skipped lines
/// without counting them toward the byte budget. This means even files far exceeding
/// [`MAX_FS_BYTES`] won't cause memory issues as long as `limit` is tight enough.
///
/// Binary heuristic: rejects the file if a NUL byte is encountered during scanning,
/// matching the semantics of the full-path [`looks_binary`].
async fn read_window_streaming(
    path: &Path,
    line: Option<u32>,
    limit: Option<u32>,
) -> Result<String, FsError> {
    use tokio::io::AsyncBufReadExt;

    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => FsError::NotFound(path.to_path_buf()),
            _ => FsError::Backend(BoxError::new(e)),
        })?;
    let mut reader = tokio::io::BufReader::new(file);

    let start = line.unwrap_or(1).max(1) as usize - 1;
    let take = limit.unwrap_or(u32::MAX) as usize;

    let mut buf = Vec::new();
    let mut out = String::new();
    let mut idx: usize = 0;
    let mut accepted: usize = 0;
    let mut total_window_bytes: u64 = 0;

    while accepted < take {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .await
            .map_err(|e| FsError::Backend(BoxError::new(e)))?;
        if n == 0 {
            break; // EOF
        }
        if buf.contains(&0u8) {
            return Err(FsError::NotPermitted(format!(
                "binary file: {}",
                path.display()
            )));
        }

        if idx >= start {
            // Only accumulate lines within the window; reject if they exceed
            // `MAX_FS_BYTES` to prevent a single window from exhausting memory. The
            // window size is determined by the LLM-chosen `limit`; when the threshold is
            // hit, return `TooLarge` so the caller can retry with a smaller `limit`.
            total_window_bytes = total_window_bytes.saturating_add(n as u64);
            if total_window_bytes > MAX_FS_BYTES {
                return Err(FsError::TooLarge {
                    bytes: total_window_bytes,
                    limit: MAX_FS_BYTES,
                });
            }
            let chunk = std::str::from_utf8(&buf)
                .map_err(|e| FsError::NotPermitted(format!("file is not valid UTF-8: {e}")))?;
            out.push_str(chunk);
            accepted += 1;
        }
        idx += 1;
    }

    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnding {
    Lf,
    Crlf,
}

fn detect_line_ending(text: &str) -> LineEnding {
    let crlf = text.matches("\r\n").count();
    let total_lf = text.matches('\n').count();
    let lone_lf = total_lf.saturating_sub(crlf);
    if crlf > lone_lf {
        LineEnding::Crlf
    } else {
        LineEnding::Lf
    }
}

fn normalize(content: &str, target: LineEnding) -> Cow<'_, str> {
    match target {
        LineEnding::Lf => {
            if content.contains("\r\n") {
                Cow::Owned(content.replace("\r\n", "\n"))
            } else {
                Cow::Borrowed(content)
            }
        }
        LineEnding::Crlf => {
            // Normalize to LF first, then replace all LF with CRLF — this avoids
            // double-converting sequences like "\r\n\n" into "\r\r\n".
            let lf = content.replace("\r\n", "\n");
            Cow::Owned(lf.replace('\n', "\r\n"))
        }
    }
}

/// Binary heuristic: presence of `\0` or a high ratio of non-printable bytes. Only scans
/// the first 8 KiB.
fn looks_binary(bytes: &[u8]) -> bool {
    let head = bytes.get(..8 * 1024).unwrap_or(bytes);
    if head.is_empty() {
        return false;
    }
    if head.contains(&0u8) {
        return true;
    }
    let non_printable = head
        .iter()
        .filter(|&&b| b < 0x09 || (b > 0x0d && b < 0x20))
        .count();
    non_printable * 100 / head.len() > 30
}

/// Slices the text by `line` (1-based) and `limit`. Returns the full text when both are
/// `None`.
fn slice_lines(text: &str, line: Option<u32>, limit: Option<u32>) -> String {
    if line.is_none() && limit.is_none() {
        return text.to_string();
    }
    let start = line.unwrap_or(1).max(1) as usize - 1;
    let take = limit.unwrap_or(u32::MAX) as usize;
    let mut out = String::new();
    for (idx, l) in text.split_inclusive('\n').enumerate() {
        if idx < start {
            continue;
        }
        if idx >= start + take {
            break;
        }
        out.push_str(l);
    }
    out
}

/// Atomic write via `tmp + rename`. The temporary file is placed in the same parent
/// directory to avoid cross-device renames. The parent directory is created automatically
/// if it does not exist (`mkdir -p`).
async fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("path has no parent"))?;
    tokio::fs::create_dir_all(parent).await?;
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::other("path has no file component"))?;
    let nonce = TMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_path = parent.join(format!(
        ".{}.defect-{pid}-{nonce}.tmp",
        file_name.to_string_lossy()
    ));

    // RAII: automatically removes tmp on error paths to avoid leftover files.
    let cleanup = TmpCleanup {
        path: Some(tmp_path.clone()),
    };
    tokio::fs::write(&tmp_path, bytes).await?;
    tokio::fs::rename(&tmp_path, path).await?;
    cleanup.disarm();
    Ok(())
}

struct TmpCleanup {
    path: Option<PathBuf>,
}

impl TmpCleanup {
    fn disarm(mut self) {
        self.path = None;
    }
}

impl Drop for TmpCleanup {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            // Best-effort: leaving a .tmp file is better than leaving a partial target
            // file.
            let _ = std::fs::remove_file(&p);
        }
    }
}

#[cfg(test)]
mod tests;
