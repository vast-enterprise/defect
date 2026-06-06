//! 文件系统后端抽象。
//!
//! [`FsBackend`] 是 fs 工具家族（`read_file` / `write_file` / `edit_file`）
//! 与底层 IO 之间的 trait 边界。两个 v0 实现：
//! - [`defect_tools::fs::LocalFsBackend`]：直接打盘
//! - [`defect_acp::fs::AcpFsBackend`]：走 ACP `fs/read_text_file` /
//!   `fs/write_text_file` 反向请求委托给客户端
//!
//! 装配权在 `defect-acp` 的 `session/new` handler——按客户端的
//! [`FileSystemCapabilities`] 协商结果选择后端，注入给
//! [`crate::session::AgentCore::create_session`]。
//!
//! 设计详见 `docs/internal/tools-fs.md` §2 与 `docs/inbound/acp-fs.md`。
//!
//! [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use thiserror::Error;

use crate::error::BoxError;

/// 文件内容的指纹。用于 [`FsBackend::fingerprint`] 与 [`Fingerprint::of`]：
/// `edit_file` 读取后记录指纹，写入前再次取指纹；不一致即视为并发写冲突。
///
/// 用 `(bytes, hash)` 而非单纯哈希：长度 + 哈希双重比较，把单 `u64` 哈希
/// 的碰撞概率压到可忽略。`DefaultHasher` 只用于进程内一次性比较，不持久化
/// 也不跨进程，所以可以容忍 std 默认实现的"未指定但稳定"语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint {
    pub bytes: u64,
    pub hash: u64,
}

impl Fingerprint {
    /// 直接对一段文本取指纹。`edit_file` 读到 old_content 后用这个先打个点，
    /// 避免在写前再读一次。
    pub fn of(content: &str) -> Self {
        let mut h = DefaultHasher::new();
        content.hash(&mut h);
        Self {
            bytes: content.len() as u64,
            hash: h.finish(),
        }
    }
}

/// 仅用于测试的 no-op fs 后端。所有方法都返回 [`FsError::NotPermitted`]，
/// 让需要 `Arc<dyn FsBackend>` 的测试场景（不实际跑 fs 工具）能跳过装配。
///
/// 真实运行时用 [`defect_tools::fs::LocalFsBackend`] 或
/// [`defect_acp::fs::AcpFsBackend`]。
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

/// fs 后端 trait。
///
/// 两个动词足够表达 v0 fs 工具家族的全部底层操作：
/// - `edit_file` 由工具层组合（先 [`read_text`] 再 [`write_text`]），
///   后端不感知 patch 语义
/// - 删除 / 移动 / mkdir 不进入 v0 fs 工具家族（ACP 没有对位反向方法），
///   LLM 用 `bash`
///
/// 入参用 owned `PathBuf` / `String`：把 future 的生命周期收敛到 `&'_ self`，
/// 避免显式生命周期参数；与 `LlmProvider::complete` 同款取舍。
///
/// [`read_text`]: FsBackend::read_text
pub trait FsBackend: Send + Sync {
    /// 读取整个文件的 UTF-8 文本。
    ///
    /// `line` / `limit` 与 ACP `ReadTextFileRequest` 同语义：
    /// - `line = Some(n)` 表示从第 n 行（1-based）开始读
    /// - `limit = Some(k)` 表示最多读 k 行
    /// - 两者皆 None 表示读全文
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<String, FsError>>;

    /// 读取整个文件的原始字节。`read_file` 工具在识别到图片等二进制类型时
    /// 走这条，把字节交给上层 base64 编码成多模态 tool_result。
    ///
    /// 默认实现返回 [`FsError::NotPermitted`]——委托后端（[`AcpFsBackend`]）
    /// 的 ACP `fs/read_text_file` 反向通道是纯文本的，拿不到二进制；ACP 环境
    /// 下读图片这件事由 system prompt 引导模型回避（`# Environment` 段会注明
    /// frontend 是 delegated）。本地后端（[`LocalFsBackend`]）重写为直接读盘。
    ///
    /// [`AcpFsBackend`]: defect_acp::fs::AcpFsBackend
    /// [`LocalFsBackend`]: defect_tools::fs::LocalFsBackend
    fn read_bytes(&self, path: PathBuf) -> BoxFuture<'_, Result<Vec<u8>, FsError>> {
        Box::pin(async move {
            let _ = path;
            Err(FsError::NotPermitted(
                "this backend cannot read raw bytes (e.g. images); delegated environments only support text reads".to_string(),
            ))
        })
    }

    /// 全量覆盖写一个 UTF-8 文本文件。
    ///
    /// 后端负责确保父目录存在（`mkdir -p` 语义）。
    ///
    /// 行末符 / 原子性的责任划分见 `docs/internal/tools-fs.md` §6：
    /// - 本地后端做行末符规范化与 `tmp + rename` 原子写
    /// - 委托后端把决定权交给客户端
    fn write_text(&self, path: PathBuf, content: String) -> BoxFuture<'_, Result<(), FsError>>;

    /// 取一份"内容指纹"。用于 `edit_file` 在 read → modify → write 的窗口
    /// 中检测并发写冲突。
    ///
    /// 默认实现走 [`FsBackend::read_text`] 全文读 + [`Fingerprint::of`]——这
    /// 让委托后端（如 [`AcpFsBackend`]）无需额外协议方法即可工作。本地后端
    /// 可重写此方法，用 mtime + size 做更便宜的判定。
    ///
    /// [`AcpFsBackend`]: defect_acp::fs::AcpFsBackend
    fn fingerprint(&self, path: PathBuf) -> BoxFuture<'_, Result<Fingerprint, FsError>> {
        Box::pin(async move {
            let text = self.read_text(path, None, None).await?;
            Ok(Fingerprint::of(&text))
        })
    }
}

/// fs 后端错误。
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FsError {
    /// 文件不存在。
    #[error("file not found: {0}")]
    NotFound(PathBuf),

    /// 操作被拒：路径越界 / 二进制 / 客户端 deny / 权限不足等。
    /// v0 用字符串占位；演进时再升枚举。
    #[error("operation not permitted: {0}")]
    NotPermitted(String),

    /// 文件超过大小阈值。
    #[error("file too large: {bytes} bytes > {limit}")]
    TooLarge { bytes: u64, limit: u64 },

    /// 文件在 read-modify-write 期间被外部修改。
    /// `edit_file` 在写入前用 [`FsBackend::fingerprint`] 比对：
    /// 不一致即抛 `Conflict`，提示 LLM 重读再编辑而不是覆盖。
    #[error("file changed since last read: {0}")]
    Conflict(PathBuf),

    /// 底层 IO / RPC 失败。
    #[error("backend failure: {0}")]
    Backend(#[source] BoxError),
}

/// 把请求路径解析到工作区内的绝对路径，并校验未越界。
///
/// 行为：
/// 1. 相对路径基于 `workspace_root` 拼接；绝对路径直接用
/// 2. 从目标路径向上查找最近的**已存在**祖先目录，对它做 canonicalize
///    （write 场景下目标本身乃至多级父目录都可能尚未存在）
/// 3. 校验已存在祖先的真实路径以 `workspace_root` 的真实路径开头——
///    防 symlink 越狱（`workspace/dir/link → /etc` 这类）
/// 4. 把剩余不存在的路径段原样拼回，再拼上文件名返回
///
/// [`crate::fs::FsBackend`] 的 `LocalFsBackend` / `AcpFsBackend` 实现都调用
/// 同一份函数——委托模式下 agent 仍自己守边界，不依赖客户端 enforce。
///
/// # Errors
/// - [`FsError::NotPermitted`]：路径越界 / 无父目录 / 无文件名
/// - [`FsError::Backend`]：祖先 canonicalize 失败（IO 错误）
pub fn resolve_workspace_path(workspace_root: &Path, requested: &Path) -> Result<PathBuf, FsError> {
    let target = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        workspace_root.join(requested)
    };

    let parent = target.parent().ok_or_else(|| {
        FsError::NotPermitted(format!("path has no parent: {}", target.display()))
    })?;

    // 从 parent 向上查找最近已存在的祖先目录。
    // canonicalize 要求路径存在——write 场景下目标乃至多级父目录都可能
    // 尚未创建，因此向上走到第一个真实存在的目录再 canonicalize。
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

    // 把不存在的那段路径原样拼回已存在祖先后面，再拼文件名。
    Ok(existing_canon.join(missing_suffix).join(file_name))
}

/// 从 `path` 开始向上走，返回 `(最近已存在的祖先, 剩余路径段)`。
///
/// 剩余路径段保持原有相对关系（不使用 canonicalize 后的形式），
/// 以便拼回时保留原始语义。
fn find_existing_ancestor(path: &Path) -> Option<(&Path, PathBuf)> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        if current.exists() {
            // 从下往上收集的路径段需要逆序拼回
            missing.reverse();
            return Some((current, missing.into_iter().collect()));
        }
        missing.push(current.file_name()?.to_os_string());
        current = current.parent()?;
    }
}

#[cfg(test)]
mod tests;
