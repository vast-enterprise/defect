//! [`LocalFsBackend`]：直接打盘的 [`FsBackend`] 实现。
//!
//! Local filesystem backend — implements two key invariants:
//! - **行末符规范化**（§6.1）：写文件时如果文件已存在，按文件原有的主流
//!   行末符（CRLF / LF）规范化新内容，避免混合行末符
//! - **原子写**（§6.2）：通过临时文件 + `rename` 完成全量覆盖，避免半截
//!   文件
//!
//! 路径校验由 [`defect_agent::fs::resolve_workspace_path`] 兜底——
//! `LocalFsBackend` 与 `AcpFsBackend` 共用同一份函数。

use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs;

use defect_agent::error::BoxError;
use defect_agent::fs::{Fingerprint, FsBackend, FsError, resolve_workspace_path};
use futures::future::BoxFuture;

/// 单文件大小硬上限（read 与 write 共用）。详见 `tools-fs.md` §3.1 / §4.1。
pub const MAX_FS_BYTES: u64 = 10 * 1024 * 1024;

/// `tmp + rename` 时用到的进程内单调计数器，避免同进程同路径并发写打架。
static TMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// 直接打盘的 [`FsBackend`] 实现。
///
/// 持有 session 的 workspace root；所有 read / write 都先经过
/// [`resolve_workspace_path`] 校验。
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

            // 全文读：硬上限阻挡。窗口读（line / limit 任一为 Some）：
            // 走 chunked-read 路径——逐行流式扫描，只缓冲请求窗口。
            // v1 §3.1 的"大文件分页"：让 LLM 在不超过整体内存预算的前提下
            // 通过 offset/limit 巡读超过 10 MiB 的日志 / 数据文件。
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

    /// 用 mtime + size 做指纹——比走默认的"读全文 + hash"路径便宜得多，
    /// 且对 v1 conflict detection 的语义足够：mtime 变了 / size 变了
    /// 即视为冲突。
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

            // 把 mtime_nanos 折进 hash 字段，size 折进 bytes 字段——
            // [`Fingerprint`] 的等值比较直接对位 (size, mtime)。
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

            // 行末符规范化：仅在文件已存在时按原行末符规范；新文件保持 LLM 给的原貌。
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

/// 流式读窗口：逐行扫描文件，只在 [start, start+take) 范围内累积内容。
///
/// 与 [`slice_lines`] 的差异：后者要求整个文件已经在内存里（[`String`]）；
/// 这里走 `BufReader::read_line`，跳过的行直接丢弃，不进字节预算。这样
/// 即便文件远超 [`MAX_FS_BYTES`]，只要 `limit` 收得够紧就不会爆。
///
/// 二进制启发式：扫到的字节里出现 NUL 即拒，与全量路径的 [`looks_binary`]
/// 语义对齐。
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
            // 仅累积窗口内的行——超过 MAX_FS_BYTES 即拒，避免单次窗口
            // 自身把内存吃爆。窗口大小由 LLM 选 limit 决定，命中阈值
            // 时报 TooLarge 让上层据此把 limit 收小再试。
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
            // 先归一到 LF，再统一替换为 CRLF——避免 "\r\n\n" 类输入二次拼接成 "\r\r\n"。
            let lf = content.replace("\r\n", "\n");
            Cow::Owned(lf.replace('\n', "\r\n"))
        }
    }
}

/// 以 `\0` 出现 / 高比例非可打印字节作为二进制启发式。仅扫前 8 KiB。
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

/// 按 `line` (1-based 起始) / `limit` 切片。两者皆 None 时返回全文。
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

/// `tmp + rename` 原子写。tmp 文件落在同一父目录以避免跨设备 rename。
/// 父目录不存在时自动创建（`mkdir -p`）。
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

    // RAII：err 路径上自动 remove tmp，避免残留。
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
            // best-effort：失败留个 .tmp 比留半截目标文件好
            let _ = std::fs::remove_file(&p);
        }
    }
}

#[cfg(test)]
mod tests;
