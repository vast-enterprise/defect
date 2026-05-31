//! 进程内最小 REPL —— `defect --repl`。
//!
//! 不走 ACP、不做 TUI：读 stdin 一行作为一个 prompt，跑一个 turn，把会话
//! 事件流以朴素彩色文本打到 stdout。定位是"开发期手搓 prompt 验一下 agent
//! 行为"的便捷入口，不是面向终端用户的正式前端。
//!
//! 整个模块由 `repl` feature gate（见 `Cargo.toml`）——关掉 feature 后既
//! 不编译这里，也不拖入 `owo-colors` / `libc`。
//!
//! ## 行编辑为何自己做
//!
//! 一开始偷懒让终端 canonical（cooked）模式做行编辑，有两个 bug：退格能把
//! 提示符也擦掉；删中文时按字节删而非按 unicode char 删（取决于 tty 的
//! `IUTF8` 标志）。所以读行阶段进 raw 模式自己接管（[`read_line`]）：维护一个
//! `String` 缓冲（`pop()` 天然按 `char` 删），每次按键用「回行首 + 清行 + 重绘
//! prompt+buffer」重画——提示符是重绘出来的删不掉，CJK 宽字符靠终端按显示
//! 宽度推进光标也对。raw 模式只在读行时开，turn 期间的事件渲染仍在 cooked
//! 模式，`\n` 正常。
//!
//! ## 与 ACP 路径的关系
//!
//! 复用同一个 [`AgentCore`]：用 [`Frontend::Cli`] 建 session、本地
//! `LocalFsBackend` / `LocalShellBackend`（REPL 跑在本机，文件与命令都直接
//! 执行，无委托）。事件流的消费逻辑是 `defect-acp` event pump 的极简版——
//! 那边翻译成 wire notification，这里翻译成终端文本。

use std::io::{Read, Write as _};
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent};
use defect_agent::event::AgentEvent;
use defect_agent::session::{AgentCore, Frontend, new_session_id};
use defect_tools::{LocalFsBackend, LocalShellBackend};
use futures::{FutureExt, StreamExt};
use owo_colors::OwoColorize;
use tokio::io::{AsyncWriteExt, Stdout};

/// 跑一个交互 REPL，直到 stdin EOF（Ctrl-D）或读到 `:q` / `:quit` / `:exit`。
///
/// `cwd` 是会话工作目录（本地 fs / shell 后端的根）。
pub async fn run(agent: Arc<dyn AgentCore>, cwd: PathBuf) -> anyhow::Result<()> {
    let mut out = tokio::io::stdout();

    // 本机直跑：fs/shell 都用 local 后端，frontend 标 Cli。
    let session_id = SessionId::new(new_session_id());
    let fs = Arc::new(LocalFsBackend::new(cwd.clone()));
    let shell = Arc::new(LocalShellBackend::new());
    let session = agent
        .create_session(session_id, cwd.clone(), Vec::new(), fs, shell, Frontend::Cli)
        .await
        .map_err(|e| anyhow::anyhow!("create_session failed: {e}"))?;

    let banner = format!(
        "defect repl — {} @ {}\n\
         type a prompt and hit enter; :q or Ctrl-D to quit.\n",
        session.current_model(),
        cwd.display(),
    );
    write(&mut out, &banner.dimmed().to_string()).await?;

    // 提示符（带 ANSI 着色，零显示宽度的颜色码不影响重绘对齐）。
    let prompt = "› ".cyan().bold().to_string();
    loop {
        out.flush().await?;
        // 读一行：进 raw 模式自己做行编辑（见模块文档）。放 spawn_blocking——
        // 读行在 turn 之间、不与事件流并发，阻塞读最简单。
        let prompt_for_read = prompt.clone();
        let outcome = tokio::task::spawn_blocking(move || read_line(&prompt_for_read)).await??;

        let line = match outcome {
            ReadOutcome::Line(line) => line,
            ReadOutcome::Interrupted => continue, // Ctrl-C：丢弃当前行，重新提示
            ReadOutcome::Eof => break,            // Ctrl-D（空行）
        };
        let prompt_text = line.trim();
        if prompt_text.is_empty() {
            continue;
        }
        if matches!(prompt_text, ":q" | ":quit" | ":exit") {
            break;
        }

        // 先订阅再 run_turn：事件在 turn 期间通过事件流推送，turn future
        // 只返回最终 StopReason。两者并发跑。
        let mut events = session.subscribe();
        let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt_text.to_owned()))];
        let turn = session.run_turn(prompt_blocks);
        tokio::pin!(turn);

        // 把事件流泵到 stdout，直到 turn future resolve（它返回时本轮事件
        // 已发完——TurnEnded 在 run_turn 返回前发出）。
        let stop = loop {
            tokio::select! {
                result = &mut turn => break result,
                ev = events.next() => {
                    // `None` 表示流提前关闭（不应发生）——忽略，继续等 turn future。
                    if let Some(ev) = ev {
                        render_event(&mut out, ev).await?;
                    }
                }
            }
        };

        // 排掉 turn 返回瞬间可能还在 buffer 里的尾部事件（如 TurnEnded）。
        while let Some(Some(ev)) = events.next().now_or_never() {
            render_event(&mut out, ev).await?;
        }

        match stop {
            Ok(reason) => write(&mut out, &format!("\n{}\n", stop_line(reason).dimmed())).await?,
            Err(e) => {
                write(&mut out, &format!("\n{} {e}\n", "turn error:".red().bold())).await?;
            }
        }
    }

    write(&mut out, &"bye.\n".dimmed().to_string()).await?;
    Ok(())
}

/// [`read_line`] 的结果。
enum ReadOutcome {
    /// 用户按回车提交的一行（不含行尾换行）。
    Line(String),
    /// Ctrl-C：放弃当前行。
    Interrupted,
    /// Ctrl-D（空缓冲）或 stdin EOF。
    Eof,
}

/// raw 模式下自己做行编辑，读一行。阻塞，须在 `spawn_blocking` 里调。
///
/// 支持：可打印字符（含多字节 UTF-8）、退格（Backspace / DEL，按 unicode
/// char 删）、回车提交、Ctrl-C 放弃、Ctrl-D（空缓冲）EOF。不支持光标移动 /
/// 历史——超出最小 REPL 范畴。
fn read_line(prompt: &str) -> std::io::Result<ReadOutcome> {
    // stdin 不是 TTY（管道 / 重定向）时没法进 raw 模式，也不需要——直接按
    // 行读，由上游缓冲做行边界。raw 模式的行编辑只在真终端才有意义。
    if !stdin_is_tty() {
        return read_line_cooked(prompt);
    }
    let _raw = RawMode::enable()?; // Drop 时恢复终端
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut buf = String::new();

    // 重绘当前行：回行首、清到行尾、写 prompt + buffer。
    macro_rules! redraw {
        () => {{
            write!(stdout, "\r\x1b[K{prompt}{buf}")?;
            stdout.flush()?;
        }};
    }
    redraw!();

    let mut byte = [0u8; 1];
    loop {
        if stdin.read(&mut byte)? == 0 {
            // stdin EOF。
            write!(stdout, "\r\n")?;
            stdout.flush()?;
            return Ok(ReadOutcome::Eof);
        }
        match byte[0] {
            b'\r' | b'\n' => {
                write!(stdout, "\r\n")?;
                stdout.flush()?;
                return Ok(ReadOutcome::Line(buf));
            }
            0x03 => {
                // Ctrl-C：放弃当前行。
                write!(stdout, "^C\r\n")?;
                stdout.flush()?;
                return Ok(ReadOutcome::Interrupted);
            }
            0x04 => {
                // Ctrl-D：仅空缓冲时作 EOF；否则忽略（不支持删字符语义）。
                if buf.is_empty() {
                    write!(stdout, "\r\n")?;
                    stdout.flush()?;
                    return Ok(ReadOutcome::Eof);
                }
            }
            0x7f | 0x08 => {
                // Backspace / DEL：删最后一个 unicode char，重绘。
                if buf.pop().is_some() {
                    redraw!();
                }
            }
            b if b < 0x20 => {
                // 其余控制字符（含未支持的转义序列引导符 ESC=0x1b）忽略。
                // ESC 后续字节会被当作普通输入读到——最小 REPL 不解析方向键等，
                // 把 ESC 整段吞掉以免污染缓冲。
                if b == 0x1b {
                    drain_escape(&mut stdin)?;
                }
            }
            b => {
                // 可打印 / UTF-8 起始字节：按 UTF-8 多字节读全，push 进缓冲。
                let mut bytes = vec![b];
                bytes.extend(read_utf8_continuation(&mut stdin, b)?);
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    buf.push_str(s);
                    redraw!();
                }
                // 非法 UTF-8 直接丢弃。
            }
        }
    }
}

/// stdin 是否连到终端。非 TTY（管道 / 文件重定向）时 raw 模式无意义。
fn stdin_is_tty() -> bool {
    // SAFETY: isatty 只读 fd 状态，无副作用。
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// 非 TTY 的降级读行：按行读，不做 raw 行编辑。空行（EOF）返回 [`ReadOutcome::Eof`]。
fn read_line_cooked(prompt: &str) -> std::io::Result<ReadOutcome> {
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{prompt}")?;
    stdout.flush()?;
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        return Ok(ReadOutcome::Eof);
    }
    // 去掉行尾换行（保留行内空白交给上层 trim）。
    let trimmed = line.trim_end_matches(['\r', '\n']);
    Ok(ReadOutcome::Line(trimmed.to_owned()))
}

/// 按 UTF-8 起始字节判断还需读几个续延字节，读回它们。
fn read_utf8_continuation(
    stdin: &mut impl Read,
    lead: u8,
) -> std::io::Result<Vec<u8>> {
    let extra = match lead {
        0xC0..=0xDF => 1,
        0xE0..=0xEF => 2,
        0xF0..=0xF7 => 3,
        _ => 0,
    };
    let mut rest = vec![0u8; extra];
    if extra > 0 {
        stdin.read_exact(&mut rest)?;
    }
    Ok(rest)
}

/// 吞掉一个 ANSI 转义序列（ESC 已被读走）。只为防止方向键等把后续字节当
/// 普通字符塞进缓冲——不解释序列含义。读到终结字节（字母 / `~`）即停。
fn drain_escape(stdin: &mut impl Read) -> std::io::Result<()> {
    let mut byte = [0u8; 1];
    // CSI 序列形如 `ESC [ ... <final>`；只在 `[` / `O` 引导时才继续吞。
    if stdin.read(&mut byte)? == 0 || !matches!(byte[0], b'[' | b'O') {
        return Ok(());
    }
    loop {
        if stdin.read(&mut byte)? == 0 {
            return Ok(());
        }
        // 参数 / 中间字节是 0x20..=0x3f；终结字节是 0x40..=0x7e。
        if byte[0] >= 0x40 {
            return Ok(());
        }
    }
}

/// 终端 raw 模式 RAII guard：构造时切 raw、Drop 时恢复原 termios。
///
/// 只关 canonical（行缓冲）与 echo、信号生成与软流控，让我们逐字节读、
/// 自己回显、自己处理 Ctrl-C/D；输出处理（OPOST）保持不变。
struct RawMode {
    fd: i32,
    original: libc::termios,
}

impl RawMode {
    fn enable() -> std::io::Result<Self> {
        let fd = libc::STDIN_FILENO;
        // SAFETY: 标准 termios FFI；termios 结构由 tcgetattr 填充后才使用。
        unsafe {
            let mut original: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut original) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut raw = original;
            // 关行规程：ICANON(行缓冲)、ECHO(回显)、ISIG(信号)、IEXTEN(扩展)。
            raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG | libc::IEXTEN);
            // 关输入软流控与 CR→NL 转换，让 Ctrl-S/Q、回车按原字节到达。
            raw.c_iflag &= !(libc::IXON | libc::ICRNL);
            // 每次至少读 1 字节、无超时。
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(fd, libc::TCSANOW, &raw) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { fd, original })
        }
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // SAFETY: 用构造时保存的原 termios 恢复；失败也无从处理，尽力而为。
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
        }
    }
}

/// 把单个 [`AgentEvent`] 渲染成终端文本。只处理对人有意义的几类，
/// 其余（LLM 调用边界、策略审计等）静默忽略——REPL 不是 observability。
async fn render_event(out: &mut Stdout, event: AgentEvent) -> anyhow::Result<()> {
    match event {
        AgentEvent::AssistantText { content } => {
            if let Some(text) = block_text(&content) {
                // 助手文本是增量 chunk，不换行，逐段拼出完整回复。
                write(out, &text).await?;
                out.flush().await?;
            }
        }
        AgentEvent::AssistantThought { content } => {
            if let Some(text) = block_text(&content) {
                write(out, &text.dimmed().italic().to_string()).await?;
                out.flush().await?;
            }
        }
        AgentEvent::ToolCallStarted { name, fields, .. } => {
            let title = fields.title.unwrap_or_else(|| name.clone());
            write(out, &format!("\n{} {}\n", "⚙".yellow(), title.yellow())).await?;
        }
        AgentEvent::ToolCallFinished { fields, .. } => {
            if let Some(status) = fields.status {
                write(out, &format!("  {} {status:?}\n", "↳".dimmed())).await?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// 从 [`ContentBlock`] 取文本；非文本块返回 `None`。
fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// turn 结束行的人类可读描述。
fn stop_line(reason: StopReason) -> String {
    let why = match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::MaxTurnRequests => "max_turn_requests",
        StopReason::Refusal => "refusal",
        StopReason::Cancelled => "cancelled",
        _ => "done",
    };
    format!("[{why}]")
}

/// 往 stdout 写一段字符串。
async fn write(out: &mut Stdout, s: &str) -> anyhow::Result<()> {
    out.write_all(s.as_bytes()).await?;
    Ok(())
}
