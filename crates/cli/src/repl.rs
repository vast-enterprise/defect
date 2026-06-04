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

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use defect_agent::event::AgentEvent;
use defect_agent::session::{AgentCore, Frontend, TurnError, new_session_id};
use defect_tools::{LocalFsBackend, LocalShellBackend};
use futures::{FutureExt, StreamExt};
use owo_colors::OwoColorize;
use tokio::io::{AsyncWriteExt, Stdout};
use tokio::sync::mpsc;

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
        .create_session(
            session_id,
            cwd.clone(),
            Vec::new(),
            fs,
            shell,
            Frontend::Cli,
        )
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
    // `spawn_session_pump` 同构的关键：事件消费的生命周期 = session 生命周期。
    let mut events = session.subscribe();

    // 输入读取：单独的 blocking 线程跑 raw 模式 + crossterm 读键，把按键经 channel
    // 发给主 task。**关键**：所有 stdout 写都只在主 task 做（行重绘 + 事件渲染），
    // blocking 线程一个字节都不写——这样既无 stdout 锁竞争（早先 read_line 长期
    // 持锁导致后台 turn 事件被阻塞、空闲完全静默），也能在事件来时干净地「擦输入行
    // → 打事件 → 重绘输入行」，输入与输出不再交错。
    let (key_tx, mut key_rx) = mpsc::channel::<KeyMsg>(64);
    let _input = InputReader::spawn(key_tx);

    let mut editor = LineEditor::new("› ".cyan().bold().to_string());
    editor.redraw(&mut out).await?;

    // turn 进行中用户敲的下一条 prompt 排在这里——同一 session 不能并发 turn，
    // 故 turn 结束后再依次跑。FIFO。
    let mut pending: std::collections::VecDeque<String> = std::collections::VecDeque::new();

    'session: loop {
        // 取下一条 prompt：优先消费上一个 turn 期间排队的行；否则进输入阶段读新行。
        let line = if let Some(queued) = pending.pop_front() {
            editor.echo_submitted(&mut out, &queued).await?; // 回显，让用户看到即将跑哪条
            queued
        } else {
            // 输入阶段：收按键拼行 / 同时实时渲染事件。直到拼出一整行或退出。
            let mut submitted: Option<String> = None;
            while submitted.is_none() {
                tokio::select! {
                    key = key_rx.recv() => match key {
                        Some(KeyMsg::Line(text)) => submitted = Some(text),          // 非 TTY 逐行
                        Some(KeyMsg::Edit(key)) => submitted = editor.on_key(key, &mut out).await?,
                        Some(KeyMsg::Interrupt) => editor.clear_line(&mut out).await?, // Ctrl-C 丢弃当前行
                        Some(KeyMsg::Eof) | None => break 'session,                   // Ctrl-D / 输入关闭
                    },
                    ev = events.next() => {
                        if let Some(ev) = ev {
                            editor.render_event(&mut out, ev).await?;
                        }
                    }
                }
            }
            submitted.expect("loop exits only when submitted is Some")
        };

        let prompt_text = line.trim();
        if prompt_text.is_empty() {
            editor.redraw(&mut out).await?;
            continue;
        }
        if matches!(prompt_text, ":q" | ":quit" | ":exit") {
            break;
        }

        // 跑 turn：future 只返回最终 StopReason，本轮事件经持久订阅推送，期间仍要
        // 排空事件流渲染、**并继续消费按键**（让用户能编辑/排队下一条）。turn slot
        // 可能被后台自主续转 turn 占着 → TurnInProgress 退避重试（仿 ACP），不直接报错。
        let (stop, queued) = run_user_turn(
            session.as_ref(),
            prompt_text.to_owned(),
            &mut events,
            &mut key_rx,
            &mut editor,
            &mut out,
        )
        .await?;
        // 本 turn 期间用户回车提交的行进队列，turn 结束后依次跑。
        pending.extend(queued);

        // 成功路径：turn 的 TurnEnded 事件已驱动 end_streaming（收尾 + 重绘 prompt），
        // 这里无需再打状态行。仅 fatal error（不发 TurnEnded）需显式落屏 + 回到 prompt。
        match stop {
            Ok(_) => {
                // 兜底：极少数情况下没收到 TurnEnded（如空 prompt 早退）也要回到 prompt。
                editor.ensure_idle(&mut out).await?;
            }
            Err(e) => {
                editor
                    .print_error(&mut out, &format!("{} {e}", "turn error:".red().bold()))
                    .await?;
            }
        }
    }

    write(&mut out, &"\r\nbye.\r\n".dimmed().to_string()).await?;
    Ok(())
}

/// 跑一个用户 turn，期间：
/// - 持续排空事件流渲染；
/// - **持续消费按键**——turn 进行中用户可编辑下一条 prompt（进 buffer，流式态下
///   静默、turn 结束重绘时显示），回车则把该行**排队**（同一 session 不能并发
///   turn，故等本 turn 结束后再跑）。
///
/// 撞上 [`TurnError::TurnInProgress`]（后台自主续转 turn 正占着 slot）时退避重试。
///
/// 返回 `(turn 最终结果, 本 turn 期间排队的行)`。`Eof`（Ctrl-D）也按"输入关闭"
/// 处理：不强行打断在跑的 turn，但记下，由调用方在 turn 后决定（这里简单忽略，
/// turn 结束回到输入循环时再次读到 EOF 自然退出）。
async fn run_user_turn(
    session: &dyn defect_agent::session::Session,
    prompt_text: String,
    events: &mut defect_agent::session::EventStream,
    key_rx: &mut mpsc::Receiver<KeyMsg>,
    editor: &mut LineEditor,
    out: &mut Stdout,
) -> anyhow::Result<(Result<StopReason, TurnError>, Vec<String>)> {
    // 退避参数与 ACP run_prompt_turn 一致：自主续转 turn 通常很短，退避几次即可拿到 slot。
    const MAX_RETRIES: u32 = 100;
    const BACKOFF: Duration = Duration::from_millis(20);

    // turn 进行中用户回车提交的行，turn 结束后交回调用方排队跑。
    let mut queued: Vec<String> = Vec::new();
    // 按键 channel 是否仍开。关闭后（cooked 路径 EOF / 用户关闭输入）必须**停止**
    // select 它——否则 `recv()` 持续立即返回 `None`，把 select 拖成 busy-spin，
    // 反而饿死 turn future（这是一个隐蔽的死循环：进程像挂死，CPU 打满）。
    let mut keys_open = true;
    let mut attempt = 0u32;
    let result = loop {
        let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt_text.clone()))];
        let turn = session.run_turn(prompt_blocks);
        tokio::pin!(turn);

        let result = loop {
            tokio::select! {
                // 优先排空事件再看 turn 是否结束——turn future 与事件流可能同时就绪
                // （turn 飞快结束、AssistantText/TurnEnded 已在 buffer 里）。若让 select
                // 随机选中 turn 分支先 break，buffer 里的尾部事件就漏渲染了。biased +
                // 先 poll events 保证事件不丢。
                biased;
                ev = events.next() => {
                    if let Some(ev) = ev {
                        editor.render_event(out, ev).await?;
                    }
                }
                // turn 进行中也消费按键：编辑下一条 prompt / 回车排队。channel 关闭后
                // 禁用本臂（见 keys_open 注释）。
                key = key_rx.recv(), if keys_open => {
                    match key {
                        None => keys_open = false,
                        Some(msg) => {
                            if let Some(line) = handle_key_during_turn(session, msg, editor, out).await? {
                                queued.push(line);
                            }
                        }
                    }
                }
                r = &mut turn => break r,
            }
        };

        // turn 已结束，但 buffer 里可能还有刚 send、未被 poll 的尾部事件（TurnEnded 等）。
        // 立即就绪的全排掉，不丢。
        while let Some(Some(ev)) = events.next().now_or_never() {
            editor.render_event(out, ev).await?;
        }

        match result {
            Err(TurnError::TurnInProgress) if attempt < MAX_RETRIES => {
                attempt += 1;
                // 退避期间仍要排空事件 + 消费按键——占着 slot 的自主续转 turn 正在产出。
                let sleep = tokio::time::sleep(BACKOFF);
                tokio::pin!(sleep);
                loop {
                    tokio::select! {
                        () = &mut sleep => break,
                        ev = events.next() => {
                            if let Some(ev) = ev {
                                editor.render_event(out, ev).await?;
                            }
                        }
                        key = key_rx.recv(), if keys_open => {
                            match key {
                                None => keys_open = false,
                                Some(msg) => {
                                    if let Some(line) = handle_key_during_turn(session, msg, editor, out).await? {
                                        queued.push(line);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            other => break other,
        }
    };
    Ok((result, queued))
}

/// turn 进行中处理一个按键消息。编辑动作更新 buffer（流式态下静默，turn 结束重绘
/// 时显示）；回车返回 `Some(line)` 让调用方排队。Ctrl-C **打断正在跑的 turn**——
/// 调 [`Session::cancel_turn`]（幂等），turn loop 在下一个检查点退出并发出
/// `TurnEnded{Cancelled}`，由事件渲染收尾；同时清掉当前编辑行。Ctrl-D 在 turn
/// 进行中不打断，忽略（turn 结束回输入循环会再次读到而退出）。channel 关闭
/// （`None`）由调用方的 select 守卫处理，不进本函数。
async fn handle_key_during_turn(
    session: &dyn defect_agent::session::Session,
    msg: KeyMsg,
    editor: &mut LineEditor,
    out: &mut Stdout,
) -> anyhow::Result<Option<String>> {
    match msg {
        KeyMsg::Line(text) => Ok(Some(text)),
        KeyMsg::Edit(key) => editor.on_key(key, out).await,
        KeyMsg::Interrupt => {
            // 打断在跑的 turn：底层 CancellationToken 被 cancel，turn 在下一个
            // 检查点（LLM 流 drain / 主循环 / 权限等待）退出。turn future 随后
            // 返回 Cancelled，事件流的 TurnEnded 负责收尾重绘——这里不直接动屏。
            session.cancel_turn();
            editor.clear_line(out).await?;
            Ok(None)
        }
        KeyMsg::Eof => Ok(None),
    }
}

/// 输入读取线程发给主 task 的消息。
enum KeyMsg {
    /// 一次行编辑动作（可打印字符 / 退格），主 task 据此更新 buffer 并重绘。
    Edit(KeyEvent),
    /// 用户提交了一整行（回车，TTY）或一行文本（非 TTY 的逐行读）。
    Line(String),
    /// Ctrl-C：放弃当前输入行。
    Interrupt,
    /// Ctrl-D（空缓冲）/ stdin EOF / 输入关闭。
    Eof,
}

/// 输入读取线程：raw 模式下跑 crossterm 读键循环，把按键经 channel 发给主 task。
/// **不写 stdout**——所有显示都由主 task 统一负责（见模块文档"行编辑为何自己做"）。
///
/// 非 TTY（管道 / 重定向）时退化为逐行读，每行发一条 [`KeyMsg::Line`]。
///
/// **raw 模式由本结构（主 task 侧）持有 [`RawMode`] guard**，而非读键线程——
/// 这是退出时终端不被搞乱的关键：Ctrl-D / `:q` 退出时读键线程通常仍**阻塞在
/// `crossterm::event::read()`**（读完一个键就回去等下一个，从不主动结束），若把
/// guard 放在那条线程的栈上，进程退出时它从没被 drop，`disable_raw_mode()` 不跑，
/// 终端就停在 raw 模式（无回显、光标错位）。把 guard 挂在主 task 返回时会 drop 的
/// `InputReader` 上，无论正常退出还是 unwind，`disable_raw_mode()` 都会执行。
/// 这些调用作用于全局 tty，跨线程安全——读键线程仍阻塞着也能把终端还原。
struct InputReader {
    handle: Option<std::thread::JoinHandle<()>>,
    /// raw 模式 guard（仅 TTY）。drop 时还原终端，见结构体文档。
    _raw: Option<RawMode>,
}

impl InputReader {
    fn spawn(tx: mpsc::Sender<KeyMsg>) -> Self {
        let tty = std::io::stdin().is_terminal();
        // 在主 task 侧进 raw（仅 TTY），guard 由 InputReader 持有。进不去就退化：
        // 不持 guard，读键线程照跑（crossterm 在非 raw 下仍能读，只是行为降级）。
        let raw = if tty { RawMode::enable().ok() } else { None };
        let handle = std::thread::spawn(move || {
            if tty {
                read_keys_raw(&tx);
            } else {
                read_lines_cooked(&tx);
            }
        });
        Self {
            handle: Some(handle),
            _raw: raw,
        }
    }
}

impl Drop for InputReader {
    fn drop(&mut self) {
        // raw 模式在此还原：`_raw` guard 随本结构 drop 调用 disable_raw_mode()。
        // 读键线程可能仍阻塞在 read()——不 join、不强杀（无可移植手段），随进程退出
        // 回收；终端状态已由上面的 guard 在本（主）线程还原，与那条线程是否结束无关。
        if let Some(h) = self.handle.take() {
            drop(h);
        }
    }
}

/// raw 模式读键循环（TTY）。每个有意义的按键发一条 [`KeyMsg`]；行内 buffer 在
/// 主 task 维护，这里只把按键**原样**转发（除回车/Ctrl-C/Ctrl-D 解释成控制消息）。
/// Ctrl-D 的"仅空缓冲才 EOF"语义需要 buffer 状态，所以这里跟踪一个**长度镜像**。
fn read_keys_raw(tx: &mpsc::Sender<KeyMsg>) {
    // raw 模式已由调用方（`InputReader::spawn`）在主 task 侧开启并持有 guard——
    // 不在本线程持有，否则线程阻塞在 read() 时进程退出，guard 永不 drop，终端
    // 停在 raw 模式（见 `InputReader` 文档）。这里只管读键。
    // buffer 长度镜像：仅用于判定 Ctrl-D（空缓冲 = EOF）与退格是否有内容可删。
    // 真正的 buffer 内容在主 task 的 LineEditor 里。
    let mut len = 0usize;
    loop {
        let Ok(event) = crossterm::event::read() else {
            let _ = tx.blocking_send(KeyMsg::Eof);
            return;
        };
        let Event::Key(key) = event else {
            continue; // resize / focus / paste / 鼠标——忽略
        };
        // Windows 同时上报 Press 与 Release；只认 Press。
        if key.kind == KeyEventKind::Release {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let msg = match key.code {
            KeyCode::Enter => {
                len = 0;
                // 行内容在主 task；这里发空 Line 触发"提交"，实际文本由主 task 给出。
                // 但主 task 需要文本——故改为：Enter 也走 Edit，由主 task 判定提交。见下。
                KeyMsg::Edit(key)
            }
            KeyCode::Char('c') if ctrl => {
                len = 0;
                KeyMsg::Interrupt
            }
            KeyCode::Char('d') if ctrl && len == 0 => KeyMsg::Eof,
            KeyCode::Char('d') if ctrl => continue, // 非空缓冲的 Ctrl-D：忽略
            KeyCode::Backspace => {
                len = len.saturating_sub(1);
                KeyMsg::Edit(key)
            }
            KeyCode::Char(_) if !ctrl => {
                len += 1;
                KeyMsg::Edit(key)
            }
            _ => continue, // 方向键 / Tab / 其它控制键：忽略
        };
        if tx.blocking_send(msg).is_err() {
            return; // 主 task 退出
        }
    }
}

/// 非 TTY 逐行读：每行发一条 [`KeyMsg::Line`]，EOF 发 [`KeyMsg::Eof`]。
fn read_lines_cooked(tx: &mpsc::Sender<KeyMsg>) {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                let _ = tx.blocking_send(KeyMsg::Eof);
                return;
            }
            Ok(_) => {
                let trimmed = line.trim_end_matches(['\r', '\n']).to_owned();
                if tx.blocking_send(KeyMsg::Line(trimmed)).is_err() {
                    return;
                }
            }
        }
    }
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

/// 主 task 侧的单行编辑器 + 输出协调器。**所有 stdout 写都经它**。
///
/// 用一个显示状态机解决"流式输出 vs 用户输入行"的冲突：
/// - **空闲态**（`streaming = false`）：屏幕底部是 prompt + 用户正在敲的 buffer。
///   按键更新 buffer 并就地重绘。
/// - **流式态**（`streaming = true`）：一个 turn 正在产出（助手文本是逐 chunk 的
///   增量事件）。此时**直接把文本追加到屏幕**，绝不重绘 prompt——否则每个 chunk
///   之间的"擦行+重画 prompt"会把刚打的助手文本抹碎（这正是之前输出乱的根因）。
///
/// 进入流式态：第一个内容事件惰性触发（先擦掉用户半行输入，转 streaming）。
/// 退出流式态：`TurnEnded` 时换到干净行、重绘 prompt + 被打断的 buffer。
/// 用户在流式态敲的字静默进 buffer，turn 结束重绘时显示出来。
///
/// 是否 raw（TTY）决定换行用 `\r\n` 还是 `\n`、是否做光标控制。
struct LineEditor {
    prompt: String,
    buf: String,
    /// 是否在 raw 终端（TTY）。非 TTY（管道）时不做任何光标控制、换行用 `\n`。
    tty: bool,
    /// 是否处于流式输出态（一个 turn 正在产出）。
    streaming: bool,
    /// 流式输出时，光标是否在行首（用于 turn 结束时决定要不要补换行）。
    at_line_start: bool,
}

impl LineEditor {
    fn new(prompt: String) -> Self {
        Self {
            prompt,
            buf: String::new(),
            tty: std::io::stdin().is_terminal(),
            streaming: false,
            at_line_start: true,
        }
    }

    /// 重绘当前输入行（空闲态）：回行首、清到行尾、写 prompt + buffer。
    async fn redraw(&self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.tty {
            write(out, &format!("\r\x1b[K{}{}", self.prompt, self.buf)).await?;
        } else {
            write(out, &self.prompt).await?;
        }
        out.flush().await?;
        Ok(())
    }

    /// 回显一条"即将运行"的 prompt（来自 turn 期间排队的行）：清当前行、打
    /// `prompt + line` 并换行，让用户看到下一条跑的是什么。
    async fn echo_submitted(&mut self, out: &mut Stdout, line: &str) -> anyhow::Result<()> {
        self.buf.clear();
        if self.tty {
            write(out, &format!("\r\x1b[K{}{}\r\n", self.prompt, line)).await?;
        } else {
            write(out, &format!("{}{}\n", self.prompt, line)).await?;
        }
        out.flush().await?;
        Ok(())
    }

    /// 处理一个行编辑按键。回车返回 `Some(line)`（buffer 取空）表示提交；其余 `None`。
    /// 流式态下只更新 buffer、**不重绘**（重绘会插进正在流的输出里）——等 turn 结束
    /// 重绘时再显示用户敲的内容。
    async fn on_key(&mut self, key: KeyEvent, out: &mut Stdout) -> anyhow::Result<Option<String>> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => {
                if !self.streaming {
                    write(out, "\r\n").await?;
                    out.flush().await?;
                }
                return Ok(Some(std::mem::take(&mut self.buf)));
            }
            KeyCode::Backspace => {
                let changed = self.buf.pop().is_some();
                // 重绘只在空闲态、且确有删除时做；流式态让 buffer 静默跟手。
                if changed && !self.streaming {
                    self.redraw(out).await?;
                }
            }
            KeyCode::Char(c) if !ctrl => {
                self.buf.push(c);
                if !self.streaming {
                    self.redraw(out).await?;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Ctrl-C：丢弃当前输入行内容并重绘空 prompt（仅空闲态有意义）。
    async fn clear_line(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        self.buf.clear();
        if !self.streaming {
            self.redraw(out).await?;
        }
        Ok(())
    }

    /// 进入流式态（若尚未）：擦掉用户正在敲的输入行，后续内容直接追加。
    async fn enter_streaming(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if !self.streaming {
            if self.tty {
                write(out, "\r\x1b[K").await?; // 擦掉 prompt + 半行输入
            }
            self.streaming = true;
            self.at_line_start = true;
        }
        Ok(())
    }

    /// 退出流式态：补一个换行（若光标不在行首）、重绘 prompt + 被打断的 buffer。
    async fn end_streaming(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.streaming {
            if !self.at_line_start {
                write(out, if self.tty { "\r\n" } else { "\n" }).await?;
            }
            self.streaming = false;
            self.redraw(out).await?;
        }
        Ok(())
    }

    /// 渲染一个 [`AgentEvent`]。内容事件惰性进入流式态后直接追加文本；`TurnEnded`
    /// 退出流式态并重绘 prompt。只处理对人有意义的几类，其余忽略。
    async fn render_event(&mut self, out: &mut Stdout, event: AgentEvent) -> anyhow::Result<()> {
        match event {
            AgentEvent::AssistantText { content } => {
                if let Some(text) = block_text(&content) {
                    self.stream_text(out, &text).await?;
                }
            }
            AgentEvent::AssistantThought { content } => {
                if let Some(text) = block_text(&content) {
                    self.stream_text(out, &text.dimmed().italic().to_string())
                        .await?;
                }
            }
            AgentEvent::ToolCallStarted { name, fields, .. } => {
                let title = fields.title.unwrap_or(name);
                self.stream_text(out, &format!("\n{} {}\n", "⚙".yellow(), title.yellow()))
                    .await?;
            }
            AgentEvent::ToolCallFinished { fields, .. } => {
                if let Some(status) = fields.status {
                    self.stream_text(out, &format!("{} {status:?}\n", "  ↳".dimmed()))
                        .await?;
                }
            }
            AgentEvent::TurnEnded { .. } => {
                self.end_streaming(out).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// 流式追加一段文本：确保已进入流式态、写文本（raw 下 `\n`→`\r\n`）、更新行首状态。
    async fn stream_text(&mut self, out: &mut Stdout, text: &str) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.enter_streaming(out).await?;
        write(out, &nl(text, self.tty)).await?;
        out.flush().await?;
        self.at_line_start = text.ends_with('\n');
        Ok(())
    }

    /// 确保回到空闲态：若仍在流式态（没收到 TurnEnded）就收尾并重绘 prompt；
    /// 已空闲则 no-op（避免与 TurnEnded 的重绘重复打 prompt）。
    async fn ensure_idle(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.streaming {
            self.end_streaming(out).await?;
        }
        Ok(())
    }

    /// 打印一行错误状态（turn fatal error；这类不发 TurnEnded 事件），再回到空闲 prompt。
    async fn print_error(&mut self, out: &mut Stdout, text: &str) -> anyhow::Result<()> {
        // 若正流式中，先收尾换行。
        if self.streaming && !self.at_line_start {
            write(out, if self.tty { "\r\n" } else { "\n" }).await?;
        }
        self.streaming = false;
        if self.tty {
            write(out, &format!("\r\x1b[K{text}\r\n")).await?;
        } else {
            write(out, &format!("{text}\n")).await?;
        }
        self.redraw(out).await?;
        Ok(())
    }
}

/// 把字符串里的 `\n` 在 raw 模式下换成 `\r\n`（raw 终端不做 ONLCR 转换，光标只
/// 下移不回首列）。非 raw（非 TTY）原样返回。
fn nl(s: &str, tty: bool) -> String {
    if tty {
        s.replace('\n', "\r\n")
    } else {
        s.to_owned()
    }
}

/// 从 [`ContentBlock`] 取文本；非文本块返回 `None`。
fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// 往 stdout 写一段字符串。
async fn write(out: &mut Stdout, s: &str) -> anyhow::Result<()> {
    out.write_all(s.as_bytes()).await?;
    Ok(())
}
