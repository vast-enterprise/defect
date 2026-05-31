//! 进程内最小 REPL —— `defect --repl`。
//!
//! 不走 ACP、不做 TUI：读 stdin 一行作为一个 prompt，跑一个 turn，把会话
//! 事件流以朴素彩色文本打到 stdout。定位是"开发期手搓 prompt 验一下 agent
//! 行为"的便捷入口，不是面向终端用户的正式前端。
//!
//! 整个模块由 `repl` feature gate（见 `Cargo.toml`）——关掉 feature 后既
//! 不编译这里，也不拖入 `owo-colors` / `crossterm`。
//!
//! ## 行编辑为何自己做
//!
//! 一开始偷懒让终端 canonical（cooked）模式做行编辑，有两个 bug：退格能把
//! 提示符也擦掉；删中文时按字节删而非按 unicode char 删。所以读行阶段进 raw
//! 模式自己接管（[`read_line`]）：维护一个 `String` 缓冲（`pop()` 天然按 `char`
//! 删），每次按键用「回行首 + 清行 + 重绘 prompt+buffer」重画——提示符是重绘
//! 出来的删不掉，CJK 宽字符靠终端按显示宽度推进光标也对。raw 模式只在读行时
//! 开，turn 期间的事件渲染仍在 cooked 模式，`\n` 正常。
//!
//! 用 [`crossterm`] 做 raw 模式与按键解析（Linux / macOS / Windows 一致）——
//! 它的 `event::read()` 返回已解析好的 [`KeyEvent`]（多字节 char 直接给到，
//! 不必手拼 UTF-8），raw 模式切换也跨平台。
//!
//! ## 与 ACP 路径的关系
//!
//! 复用同一个 [`AgentCore`]：用 [`Frontend::Cli`] 建 session、本地
//! `LocalFsBackend` / `LocalShellBackend`（REPL 跑在本机，文件与命令都直接
//! 执行，无委托）。事件流的消费逻辑是 `defect-acp` event pump 的极简版——
//! 那边翻译成 wire notification，这里翻译成终端文本。

use std::io::{IsTerminal, Write as _};
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
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

    // 持久订阅：循环外订阅**一次**，跨所有 turn 排空——含 session driver 自发的
    // 自主续转 turn（后台 subagent 完成后消化结果那一轮）。这是与 ACP
    // `spawn_session_pump` 同构的关键：事件消费的生命周期 = session 生命周期，
    // 而非单个 turn。早先按 turn 订阅时，turn 之间（卡在 read_line）无人排空，
    // 自主续转 turn 的事件全丢——表现为"subagent 后台返回后父 agent 没续上"。
    let mut events = session.subscribe();

    // 提示符（带 ANSI 着色，零显示宽度的颜色码不影响重绘对齐）。
    let prompt = "› ".cyan().bold().to_string();
    loop {
        out.flush().await?;

        // 读一行：进 raw 模式自己做行编辑（见模块文档）。放 spawn_blocking 阻塞读，
        // **同时**继续排空事件流——这样用户在敲下一条 prompt 时，后台任务完成触发的
        // 自主续转 turn 仍能实时打到屏幕上（不会等到下次 run_turn 才回放）。
        let prompt_for_read = prompt.clone();
        let mut read = tokio::task::spawn_blocking(move || read_line(&prompt_for_read));
        let outcome = loop {
            tokio::select! {
                joined = &mut read => break joined??,
                ev = events.next() => {
                    if let Some(ev) = ev {
                        render_event(&mut out, ev).await?;
                        out.flush().await?;
                    }
                }
            }
        };

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

        // run_turn future 只返回最终 StopReason；本轮事件经上面那条持久订阅推送。
        let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt_text.to_owned()))];
        let turn = session.run_turn(prompt_blocks);
        tokio::pin!(turn);

        // 把事件流泵到 stdout，直到 turn future resolve。
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

        // 排掉 turn 返回瞬间可能还在 buffer 里的尾部事件（如 TurnEnded）。下一轮
        // 输入阶段的 select 也会继续排空，这里只是尽早把本轮尾巴打全。
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
/// 用 crossterm 读结构化按键事件（跨平台、多字节 char 已解析好）。支持：可打印
/// 字符、退格（按 unicode char 删）、回车提交、Ctrl-C 放弃、Ctrl-D（空缓冲）EOF。
/// 不支持光标移动 / 历史——超出最小 REPL 范畴。
fn read_line(prompt: &str) -> std::io::Result<ReadOutcome> {
    // stdin 不是 TTY（管道 / 重定向）时没法进 raw 模式，也不需要——直接按行读，
    // 由上游缓冲做行边界。raw 模式的行编辑只在真终端才有意义。
    if !std::io::stdin().is_terminal() {
        return read_line_cooked(prompt);
    }
    let _raw = RawMode::enable()?; // Drop 时恢复终端
    let mut stdout = std::io::stdout().lock();
    let mut buf = String::new();

    // 重绘当前行：回行首、清到行尾、写 prompt + buffer。raw 模式下换行要 `\r\n`。
    macro_rules! redraw {
        () => {{
            write!(stdout, "\r\x1b[K{prompt}{buf}")?;
            stdout.flush()?;
        }};
    }
    redraw!();

    loop {
        // crossterm 的 read() 阻塞到下一个终端事件。只关心按键事件。
        let event = crossterm::event::read()?;
        let Event::Key(KeyEvent {
            code,
            modifiers,
            kind,
            ..
        }) = event
        else {
            continue; // resize / focus / paste / 鼠标等——忽略
        };
        // Windows 会同时上报 Press 与 Release；只认 Press（与 Release==默认 Press 的
        // unix 行为对齐），否则每个键会被处理两次。
        if kind == KeyEventKind::Release {
            continue;
        }
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        match code {
            KeyCode::Enter => {
                write!(stdout, "\r\n")?;
                stdout.flush()?;
                return Ok(ReadOutcome::Line(buf));
            }
            KeyCode::Char('c') if ctrl => {
                write!(stdout, "^C\r\n")?;
                stdout.flush()?;
                return Ok(ReadOutcome::Interrupted);
            }
            // Ctrl-D：仅空缓冲时作 EOF；非空时落到 `_ => {}` 被忽略。
            KeyCode::Char('d') if ctrl && buf.is_empty() => {
                write!(stdout, "\r\n")?;
                stdout.flush()?;
                return Ok(ReadOutcome::Eof);
            }
            // 删最后一个 unicode char（pop 天然按 char）；空缓冲时 pop 返回 None、不重绘。
            KeyCode::Backspace if buf.pop().is_some() => {
                redraw!();
            }
            KeyCode::Char(c) if !ctrl => {
                // 可打印字符（crossterm 已把多字节解析成一个 char）。
                buf.push(c);
                redraw!();
            }
            // 方向键 / Tab / 其它控制键：最小 REPL 不处理，忽略。
            _ => {}
        }
    }
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

/// 终端 raw 模式 RAII guard：构造时进 raw、Drop 时恢复。跨平台由 crossterm 负责
/// （unix 是 termios、Windows 是 console mode），我们不直接碰平台 API。
struct RawMode;

impl RawMode {
    fn enable() -> std::io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // 失败也无从处理，尽力而为（与终端状态恢复同语义）。
        let _ = disable_raw_mode();
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
