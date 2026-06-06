//! 运行环境上下文：被注入 system prompt `# Environment` 段的事实信息。
//!
//! 聚合体 [`RunningContext`] 持有「随会话变化」的部分（接入方式、cwd），
//! 「整进程不变」的平台 / 版本 / shell 探测结果用 [`OnceLock`] 缓存——
//! `os_info::get()` 会读文件/跑探测，每个 turn 重算并不划算。

use std::path::Path;
use std::sync::OnceLock;

/// agent 被如何接入——决定它对文件与命令执行环境的认知。
///
/// 注意：defect 的命令行二进制本身就是跑在 stdio 上的 ACP server，当前真实
/// 路径都走 [`Frontend::Acp`]。[`Frontend::Cli`] / [`Frontend::Headless`] 是
/// 为将来的形态预留的变体（裸命令行 user、服务器后台服务）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frontend {
    /// 命令行直接交互（预留：未来的本地 CLI user 模式）。
    Cli,
    /// 经 ACP 协议接入（编辑器 / IDE 客户端）。
    ///
    /// `fs_delegated` / `shell_delegated` 来自 ACP `initialize` 握手协商：
    /// `true` 表示文件读写 / 命令执行委托给客户端代理，`false` 表示在
    /// 本地直接执行。agent 据此知道自己面对的是本地环境还是远端代理。
    Acp {
        fs_delegated: bool,
        shell_delegated: bool,
    },
    /// 无人值守（预留：服务器上的后台服务）。
    Headless,
}

impl Frontend {
    /// 文件系统是否委托给客户端代理。仅 [`Frontend::Acp`] 且协商出
    /// `fs_delegated = true` 时为真；其余形态都在本地直接读写。
    fn fs_delegated(self) -> bool {
        matches!(
            self,
            Self::Acp {
                fs_delegated: true,
                ..
            }
        )
    }

    /// 渲染进 `# Environment` 段的单行描述。
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

/// 注入 system prompt 的运行环境上下文。
///
/// 静态部分（平台、版本、shell）由本类型内部探测并缓存；调用方只需提供
/// 「随会话变化」的 [`Frontend`] 与 cwd。
pub struct RunningContext<'a> {
    pub frontend: Frontend,
    pub cwd: &'a Path,
}

impl<'a> RunningContext<'a> {
    pub fn new(frontend: Frontend, cwd: &'a Path) -> Self {
        Self { frontend, cwd }
    }

    /// 渲染 `# Environment` 段的正文（不含标题与分隔线，由
    /// [`crate::session::resolve_system_prompt`] 负责包裹）。
    pub fn render(&self) -> String {
        let mut lines = Vec::with_capacity(6);
        lines.push(format!("- platform: {}", platform_line()));
        lines.push(format!("- defect version: {}", env!("CARGO_PKG_VERSION")));
        lines.push(format!("- frontend: {}", self.frontend.describe()));
        lines.push(format!("- cwd: {}", self.cwd.display()));
        lines.push(format!("- shell: {}", shell_line()));
        // 委托文件系统（ACP）的反向通道只能传文本，read_file 读图片会失败。
        // 明确告诉模型别对图片用 read_file，避免无谓的报错往返。
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

/// `linux / x86_64 (Ubuntu 22.04)` 形式。OS / 架构来自编译期 std 常量，
/// 发行版与版本来自运行期 `os_info` 探测。整进程缓存。
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

/// 默认 shell：读 `$SHELL`，缺失时 `unknown`。整进程缓存。
fn shell_line() -> &'static str {
    static SHELL: OnceLock<String> = OnceLock::new();
    SHELL.get_or_init(|| std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_owned()))
}

#[cfg(test)]
mod tests;
