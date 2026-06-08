//! In-process minimal REPL — `defect --repl`.
//!
//! Does not use ACP or TUI: reads one line from stdin as a prompt, runs one turn, and
//! prints the session event stream to stdout as plain colored text. Its purpose is a
//! convenient entry point for "hand-crafting prompts during development to quickly test
//! agent behavior", not a polished frontend for end users.
//!
//! The entire module is gated by the `repl` feature (see `Cargo.toml`) — when the feature
//! is disabled, this code is not compiled and `owo-colors` / `crossterm` are not pulled
//! in.
//!
//! ## Why line editing is done manually
//!
//! Initially we relied on the terminal's canonical (cooked) mode for line editing, which
//! had two bugs: backspace could erase the prompt, and deleting Chinese characters
//! removed bytes instead of whole Unicode chars. So we switch to raw mode during line
//! reading and handle it ourselves: maintain a `String` buffer (where `pop()` naturally
//! deletes by `char`), and on each key press redraw by "carriage return + clear line +
//! redraw prompt+buffer" — the prompt is redrawn and thus cannot be erased, and CJK wide
//! characters work correctly because the terminal advances the cursor by display width.
//! Raw mode is only active during line reading; event rendering during a turn still runs
//! in cooked mode, so `\n` works normally.
//!
//! We use [`crossterm`] for raw mode and key event parsing (consistent across Linux /
//! macOS / Windows) — its `event::read()` returns already-parsed [`KeyEvent`] values
//! (multi-byte chars are delivered directly, no need to manually assemble UTF-8), and raw
//! mode switching is cross-platform.
//!
//! ## Relationship with the ACP path
//!
//! Reuses the same [`AgentCore`]: creates a session with
//! [`Frontend::Cli`](defect_agent::session::Frontend::Cli), and uses local
//! `LocalFsBackend` / `LocalShellBackend` (the REPL runs on the local machine, files and
//! commands are executed directly, no delegation). The event stream consumption logic is
//! a minimal version of the `defect-acp` event pump — that one translates events into
//! wire notifications, while this one translates them into terminal text.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use defect_agent::event::AgentEvent;
use defect_agent::llm::{Message, MessageContent, Role};
use defect_agent::session::{AgentCore, TurnError};
use futures::{FutureExt, StreamExt};
use owo_colors::OwoColorize;
use tokio::io::{AsyncWriteExt, Stdout};
use tokio::sync::mpsc;

use crate::session_open::open_session;

/// The user input prompt. Shared by the live input line and history replay so that a
/// replayed user message looks identical to one freshly typed (the two used to diverge:
/// live input showed `› …` while replay showed `user> …`).
const USER_PROMPT: &str = "› ";

/// Run an interactive REPL until stdin EOF (Ctrl-D) or until `:q` / `:quit` / `:exit` is
/// read.
///
/// `cwd` is the session working directory (the root of the local filesystem / shell
/// backend). When `resume = Some(id)`, the session is resumed (replaying history to the
/// terminal) instead of creating a new one.
pub async fn run(
    agent: Arc<dyn AgentCore>,
    cwd: PathBuf,
    resume: Option<SessionId>,
) -> anyhow::Result<()> {
    let mut out = tokio::io::stdout();

    let session = open_session(&agent, &cwd, resume).await?;

    let banner = format!(
        "defect repl — {} @ {}\n\
         type a prompt and hit enter; :q or Ctrl-D to quit.\n",
        session.current_model(),
        cwd.display(),
    );
    write(&mut out, &banner.dimmed().to_string()).await?;

    // Resume: replay the restored history transcript to the terminal so the user can see
    // the context.
    let history = session.history_snapshot();
    if !history.is_empty() {
        write(
            &mut out,
            &format!("— resumed {} message(s) —\n", history.len())
                .dimmed()
                .to_string(),
        )
        .await?;
        for message in &history {
            render_history_message(&mut out, message).await?;
        }
    }

    // Persistent subscription: subscribe **once** outside the loop, draining across all
    // turns — including the session driver's autonomous continuation turn (the round
    // where the background subagent digests results after completion). This is the key
    // isomorphism with ACP `spawn_session_pump`: the event consumption lifetime equals
    // the session lifetime.
    let mut events = session.subscribe();

    // Input reading: a dedicated blocking thread runs raw mode + crossterm key reading,
    // forwarding keypresses to the main task via a channel. **Key point**: all stdout
    // writes happen only in the main task (line redraw + event rendering); the blocking
    // thread never writes a single byte. This avoids stdout lock contention (previously,
    // `read_line` holding the lock for long periods blocked background turn events,
    // causing idle silence) and allows cleanly "erasing the input line → showing the
    // event → redrawing the input line" when events arrive, so input and output no longer
    // interleave.
    let (key_tx, mut key_rx) = mpsc::channel::<KeyMsg>(64);
    let _input = InputReader::spawn(key_tx);

    let mut editor = LineEditor::new(USER_PROMPT.cyan().bold().to_string());
    editor.redraw(&mut out).await?;

    // Prompts entered while a turn is in progress are queued here — the same session
    // cannot run concurrent turns, so they are processed sequentially after the current
    // turn finishes. FIFO.
    let mut pending: std::collections::VecDeque<String> = std::collections::VecDeque::new();

    'session: loop {
        // Fetch the next prompt: first consume lines queued during the previous turn;
        // otherwise enter the input phase to read a new line.
        let line = if let Some(queued) = pending.pop_front() {
            editor.echo_submitted(&mut out, &queued).await?; // Echo so the user sees which line is about to be run.
            queued
        } else {
            // Input phase: collect keystrokes to build a line while rendering events in
            // real time, until a complete line is submitted or the session exits.
            let mut submitted: Option<String> = None;
            while submitted.is_none() {
                tokio::select! {
                    key = key_rx.recv() => match key {
                        Some(KeyMsg::Line(text)) => submitted = Some(text),          // Non-TTY line-by-line input
                        Some(KeyMsg::Edit(key)) => submitted = editor.on_key(key, &mut out).await?,
                        Some(KeyMsg::Interrupt) => editor.clear_line(&mut out).await?, // Ctrl-C discards the current line
                        Some(KeyMsg::Eof) | None => break 'session,                   // Ctrl-D / input closed
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

        // Run the turn: the future returns only the final `StopReason`; events for this
        // turn are pushed via a persistent subscription. During the turn, we must still
        // drain the event stream for rendering **and continue consuming key presses** (so
        // the user can edit or queue the next prompt). The turn slot may be occupied by
        // an auto-advancing background turn → `TurnInProgress` backoff retry (similar to
        // ACP), rather than failing immediately.
        let (stop, queued) = run_user_turn(
            session.as_ref(),
            prompt_text.to_owned(),
            &mut events,
            &mut key_rx,
            &mut editor,
            &mut out,
        )
        .await?;
        // Lines submitted by the user during this turn are queued and processed after the
        // turn ends.
        pending.extend(queued);

        // On the success path, the turn's `TurnEnded` event already drove `end_streaming`
        // (cleanup + prompt redraw), so no status line is needed here. Only a fatal error
        // (which does not emit `TurnEnded`) requires explicit screen flush + return to
        // prompt.
        match stop {
            Ok(_) => {
                // Fallback: ensure we return to the prompt even when `TurnEnded` was not
                // received (e.g. early exit on an empty prompt).
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

/// Runs a user turn, during which:
/// - Continuously drains the event stream for rendering.
/// - **Continuously consumes key presses** — while the turn is running, the user can edit
///   the next prompt (into the buffer; silently in streaming mode, shown when the turn
///   ends and redraws). Pressing Enter **queues** that line (the same session cannot run
///   concurrent turns, so it waits until this turn finishes).
///
/// On encountering [`TurnError::TurnInProgress`] (a background auto-renew turn is
/// occupying the slot), backs off and retries.
///
/// Returns `(turn final result, lines queued during this turn)`. `Eof` (Ctrl-D) is also
/// treated as "input closed": it does not forcibly interrupt the running turn, but
/// records it, leaving the caller to decide after the turn (here it is simply ignored;
/// when the turn ends and the input loop resumes, reading EOF again will exit naturally).
async fn run_user_turn(
    session: &dyn defect_agent::session::Session,
    prompt_text: String,
    events: &mut defect_agent::session::EventStream,
    key_rx: &mut mpsc::Receiver<KeyMsg>,
    editor: &mut LineEditor,
    out: &mut Stdout,
) -> anyhow::Result<(Result<StopReason, TurnError>, Vec<String>)> {
    // Backoff parameters match those in ACP `run_prompt_turn`: self-renewing turns are
    // usually short, so a few retries suffice to acquire a slot.
    const MAX_RETRIES: u32 = 100;
    const BACKOFF: Duration = Duration::from_millis(20);

    // Lines submitted by the user while the turn was in progress; returned to the caller
    // for queued execution after the turn finishes.
    let mut queued: Vec<String> = Vec::new();
    // Whether the key channel is still open. Once closed (EOF on the cooked path / user
    // closes input), we **must stop** selecting on it — otherwise `recv()` keeps
    // returning `None` immediately, turning the select into a busy-spin that starves the
    // turn future (a subtle infinite loop: the process appears hung, CPU at 100%).
    let mut keys_open = true;
    let mut attempt = 0u32;
    let result = loop {
        let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt_text.clone()))];
        let turn = session.run_turn(prompt_blocks);
        tokio::pin!(turn);

        let result = loop {
            tokio::select! {
                // Drain events before checking whether the turn has finished — the turn
                // future and the event stream may become ready simultaneously (the turn
                // finishes quickly, with `AssistantText`/`TurnEnded` already buffered).
                // If `select` randomly picks the turn branch and breaks first, trailing
                // events in the buffer would be missed. Using `biased;` and polling
                // events first ensures no events are lost.
                biased;
                ev = events.next() => {
                    if let Some(ev) = ev {
                        editor.render_event(out, ev).await?;
                    }
                }
                // During a turn, key events are still consumed: editing the next prompt
                // or queuing Enter. Once the channel is closed, this arm is disabled (see
                // the `keys_open` comment).
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

        // The turn has ended, but the buffer may still contain tail events (TurnEnded,
        // etc.) that were just sent and not yet polled. Drain all immediately ready
        // events without dropping any.
        while let Some(Some(ev)) = events.next().now_or_never() {
            editor.render_event(out, ev).await?;
        }

        match result {
            Err(TurnError::TurnInProgress) if attempt < MAX_RETRIES => {
                attempt += 1;
                // During backoff, continue draining events and consuming key presses —
                // the auto-renewing turn that holds the slot is still producing output.
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

/// Handles a key event while a turn is in progress.
/// Edit actions update the buffer (silently in streaming mode; the display is updated
/// when the turn ends).
/// Enter returns `Some(line)` for the caller to enqueue.
/// Ctrl-C **interrupts the running turn** — calls
/// [`Session::cancel_turn`](defect_agent::session::Session::cancel_turn) (idempotent);
/// the turn loop exits at the next checkpoint and emits `TurnEnded{Cancelled}`, which the
/// event renderer handles; the current edit line is also cleared.
/// Ctrl-D does not interrupt during a turn; it is ignored (when the turn ends and the
/// input loop resumes, the same Ctrl-D will be read again and cause exit).
/// Channel closure (`None`) is handled by the caller's select guard and does not reach
/// this function.
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
            // Interrupts the running turn: the underlying `CancellationToken` is
            // cancelled, and the turn exits at the next checkpoint (LLM stream drain,
            // main loop, or permission wait). The turn future then returns `Cancelled`,
            // and the event stream's `TurnEnded` handles cleanup and redrawing — this
            // does not directly manipulate the screen.
            session.cancel_turn();
            editor.clear_line(out).await?;
            Ok(None)
        }
        KeyMsg::Eof => Ok(None),
    }
}

/// Messages sent from the input-reading thread to the main task.
enum KeyMsg {
    /// A single editing action (printable character or backspace); the main task updates
    /// the buffer and redraws accordingly.
    Edit(KeyEvent),
    /// The user submitted a full line (Enter in TTY mode, or one line of text in non-TTY
    /// line-by-line reading).
    Line(String),
    /// Ctrl-C: abort the current input line.
    Interrupt,
    /// Ctrl-D (empty buffer), stdin EOF, or input closed.
    Eof,
}

/// Input reading thread: runs a crossterm key-reading loop in raw mode and sends keys to
/// the main task via a channel.
/// **Does not write to stdout** — all display is handled by the main task (see module
/// docs "Why line editing is done ourselves").
///
/// When stdin is not a TTY (pipe / redirect), falls back to line-by-line reading, sending
/// one [`KeyMsg::Line`] per line.
///
/// **Raw mode is held by this struct (on the main task side) via a [`RawMode`] guard**,
/// not by the reading thread — this is critical for clean terminal restoration on exit:
/// when Ctrl-D or `:q` exits, the reading thread is typically still **blocked in
/// `crossterm::event::read()`** (it reads one key then waits for the next, never
/// terminating on its own). If the guard were on that thread's stack, it would never be
/// dropped when the process exits, so `disable_raw_mode()` would not run and the terminal
/// would remain in raw mode (no echo, misaligned cursor). By attaching the guard to
/// `InputReader`, which is dropped when the main task returns, `disable_raw_mode()`
/// executes on both normal exit and unwind. These calls operate on the global TTY and are
/// cross-thread safe — the terminal is restored even while the reading thread is still
/// blocked.
struct InputReader {
    handle: Option<std::thread::JoinHandle<()>>,
    /// Raw mode guard (TTY only). Restores the terminal on drop; see struct docs.
    _raw: Option<RawMode>,
}

impl InputReader {
    fn spawn(tx: mpsc::Sender<KeyMsg>) -> Self {
        let tty = std::io::stdin().is_terminal();
        // Enable raw mode on the main task side (TTY only); the guard is held by
        // `InputReader`. If enabling fails, degrade gracefully: the guard is not held,
        // but the key-reading thread still runs (crossterm can still read in non-raw
        // mode, albeit with degraded behavior).
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
        // Raw mode is restored here: the `_raw` guard calls `disable_raw_mode()` when
        // this struct is dropped. The key-reading thread may still be blocked in `read()`
        // — we do not join or forcibly kill it (no portable way exists), and it will be
        // reclaimed on process exit. Terminal state has already been restored by the
        // guard on this (main) thread, regardless of whether that thread has finished.
        if let Some(h) = self.handle.take() {
            drop(h);
        }
    }
}

/// Raw-mode key-reading loop (TTY). Each meaningful key press sends a [`KeyMsg`]; the
/// line buffer is maintained by the main task, so this loop only forwards keys **as-is**
/// (except Enter, Ctrl-C, and Ctrl-D, which are interpreted as control messages).
/// Ctrl-D's "EOF only on empty buffer" semantics require buffer state, so a **length
/// mirror** is tracked here.
fn read_keys_raw(tx: &mpsc::Sender<KeyMsg>) {
    // Raw mode is already enabled by the caller (`InputReader::spawn`) on the main task
    // side, which holds the guard — we do not hold it in this thread, because if the
    // thread blocks on `read()` and the process exits, the guard would never drop,
    // leaving the terminal stuck in raw mode (see `InputReader` docs). Here we just read
    // keys.
    // Buffer length mirror: used only to decide whether Ctrl-D (empty buffer = EOF) or
    // backspace has content to delete.
    // The actual buffer contents live in the main task's `LineEditor`.
    let mut len = 0usize;
    loop {
        let Ok(event) = crossterm::event::read() else {
            let _ = tx.blocking_send(KeyMsg::Eof);
            return;
        };
        let Event::Key(key) = event else {
            continue; // resize, focus, paste, mouse events — ignore
        };
        // Windows reports both Press and Release; only handle Press.
        if key.kind == KeyEventKind::Release {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let msg = match key.code {
            KeyCode::Enter => {
                len = 0;
                // The line content is handled by the main task; sending an empty `Line`
                // here triggers a "submit", but the actual text is provided by the main
                // task.
                // However, the main task needs the text — so this was changed: `Enter`
                // also goes through `Edit`, and the main task decides whether to submit.
                // See below.
                KeyMsg::Edit(key)
            }
            KeyCode::Char('c') if ctrl => {
                len = 0;
                KeyMsg::Interrupt
            }
            KeyCode::Char('d') if ctrl && len == 0 => KeyMsg::Eof,
            KeyCode::Char('d') if ctrl => continue, // Ctrl-D with non-empty buffer: ignore
            KeyCode::Backspace => {
                len = len.saturating_sub(1);
                KeyMsg::Edit(key)
            }
            KeyCode::Char(_) if !ctrl => {
                len += 1;
                KeyMsg::Edit(key)
            }
            _ => continue, // Arrow keys / Tab / other control keys: ignore
        };
        if tx.blocking_send(msg).is_err() {
            return; // the main task exits
        }
    }
}

/// Reads lines in non-TTY (cooked) mode: sends a [`KeyMsg::Line`] per line, and
/// [`KeyMsg::Eof`] on EOF.
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

/// RAII guard for terminal raw mode: enters raw mode on construction and restores it on
/// `Drop`. Cross-platform support is handled by crossterm (termios on Unix, console mode
/// on Windows); we do not interact with platform APIs directly.
struct RawMode;

impl RawMode {
    fn enable() -> std::io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // Failure is unrecoverable here; best-effort only (same semantics as terminal
        // state restoration).
        let _ = disable_raw_mode();
    }
}

/// Single-line editor + output coordinator on the main task side. **All stdout writes go
/// through it.**
///
/// Uses a display state machine to resolve the conflict between streaming output and user
/// input lines:
/// - **Idle state** (`streaming = false`): the bottom of the screen shows the prompt plus
///   the user's current buffer. Key presses update the buffer and redraw in place.
/// - **Streaming state** (`streaming = true`): a turn is producing output (assistant text
///   arrives as incremental chunk events). Text is **appended directly to the screen**
///   without ever redrawing the prompt — otherwise the "erase line + redraw prompt"
///   between chunks would fragment the just-printed assistant text (the root cause of
///   earlier garbled output).
///
/// Entering streaming state: lazily triggered by the first content event (first erases
/// the user's partial input line, then switches to streaming).
/// Exiting streaming state: on `TurnEnded`, move to a clean line, redraw the prompt + the
/// interrupted buffer.
/// Characters typed by the user while streaming are silently appended to the buffer and
/// shown when the turn ends and the display is redrawn.
///
/// Whether raw (TTY) determines if line breaks use `\r\n` or `\n`, and whether cursor
/// control is performed.
struct LineEditor {
    prompt: String,
    buf: String,
    /// Whether the terminal is in raw mode (TTY). When not a TTY (pipe), no cursor
    /// control is performed and newlines use `\n`.
    tty: bool,
    /// Whether the editor is in streaming output mode (a turn is in progress).
    streaming: bool,
    /// Whether the cursor is at the start of a line during streaming output (used to
    /// decide whether to add a newline when a turn ends).
    at_line_start: bool,
    /// The kind of the most recently streamed segment within the current turn. Used to
    /// insert a separating newline when the kind changes (e.g. thought → assistant text),
    /// so consecutive segments of different kinds do not run together on one line. `None`
    /// before the first segment of a turn.
    last_kind: Option<StreamKind>,
}

/// The kind of a streamed output segment. Distinct kinds get a separating newline at their
/// boundary; consecutive chunks of the same kind (e.g. multiple thought chunks) do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    /// Assistant reasoning / thinking (dimmed italic).
    Thought,
    /// Assistant message text.
    Text,
    /// Tool call lifecycle lines (`⚙ …`, `  ↳ …`).
    Tool,
}

impl LineEditor {
    fn new(prompt: String) -> Self {
        Self {
            prompt,
            buf: String::new(),
            tty: std::io::stdin().is_terminal(),
            streaming: false,
            at_line_start: true,
            last_kind: None,
        }
    }

    /// Redraw the current input line (idle state): carriage return, clear to end of line,
    /// write prompt + buffer.
    async fn redraw(&self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.tty {
            write(out, &format!("\r\x1b[K{}{}", self.prompt, self.buf)).await?;
        } else {
            write(out, &self.prompt).await?;
        }
        out.flush().await?;
        Ok(())
    }

    /// Echo a "will run" prompt for a line queued during a turn: clear the current line,
    /// print `prompt + line`, and add a newline so the user can see what will run next.
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

    /// Handle a line-editing key event. Enter returns `Some(line)` (with the buffer
    /// emptied) to signal submission; all other keys return `None`. In streaming mode,
    /// only update the buffer — do **not** redraw (redrawing would interfere with the
    /// ongoing stream output). The user's input will be shown when the turn ends and a
    /// redraw occurs.
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
                // Redraw only when idle and a deletion actually occurred; in streaming
                // mode the buffer silently follows input.
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

    /// Ctrl-C: discard the current input line and redraw an empty prompt (only meaningful
    /// in the idle state).
    async fn clear_line(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        self.buf.clear();
        if !self.streaming {
            self.redraw(out).await?;
        }
        Ok(())
    }

    /// Enter streaming mode (if not already): erase the input line the user is typing;
    /// subsequent output is appended directly.
    async fn enter_streaming(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if !self.streaming {
            if self.tty {
                write(out, "\r\x1b[K").await?; // Erase the prompt and partial input line
            }
            self.streaming = true;
            self.at_line_start = true;
        }
        Ok(())
    }

    /// Exit streaming mode: add a newline if the cursor is not at the start of a line,
    /// then redraw the prompt and the interrupted buffer.
    async fn end_streaming(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.streaming {
            if !self.at_line_start {
                write(out, if self.tty { "\r\n" } else { "\n" }).await?;
            }
            self.streaming = false;
            self.last_kind = None;
            self.redraw(out).await?;
        }
        Ok(())
    }

    /// Render an [`AgentEvent`]. Content events lazily enter streaming mode and append
    /// text directly; `TurnEnded` exits streaming mode and redraws the prompt. Only
    /// handle event types that are meaningful to the user; ignore the rest.
    async fn render_event(&mut self, out: &mut Stdout, event: AgentEvent) -> anyhow::Result<()> {
        match event {
            AgentEvent::AssistantText { content } => {
                if let Some(text) = block_text(&content) {
                    self.stream_text(out, &text, StreamKind::Text).await?;
                }
            }
            AgentEvent::AssistantThought { content } => {
                if let Some(text) = block_text(&content) {
                    self.stream_text(
                        out,
                        &text.dimmed().italic().to_string(),
                        StreamKind::Thought,
                    )
                    .await?;
                }
            }
            AgentEvent::ToolCallStarted { name, fields, .. } => {
                let title = fields.title.unwrap_or(name);
                self.stream_text(
                    out,
                    &format!("{} {}\n", "⚙".yellow(), title.yellow()),
                    StreamKind::Tool,
                )
                .await?;
            }
            AgentEvent::ToolCallFinished { fields, .. } => {
                if let Some(status) = fields.status {
                    self.stream_text(
                        out,
                        &format!("{} {status:?}\n", "  ↳".dimmed()),
                        StreamKind::Tool,
                    )
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

    /// Stream a chunk of text of a given [`StreamKind`]: ensure streaming mode is active,
    /// insert a separating newline if this segment's kind differs from the previous one
    /// and the cursor is mid-line (so e.g. thinking and the assistant reply do not run
    /// together), write the text (converting `\n` to `\r\n` in raw mode), and update the
    /// line-start state.
    async fn stream_text(
        &mut self,
        out: &mut Stdout,
        text: &str,
        kind: StreamKind,
    ) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.enter_streaming(out).await?;
        // Boundary separation: when the kind changes (thought → text, text → tool, …) and
        // we are not already at the start of a fresh line, break the line first. Same-kind
        // chunks (the common streaming case) are concatenated without inserting breaks.
        if self.last_kind.is_some_and(|prev| prev != kind) && !self.at_line_start {
            write(out, if self.tty { "\r\n" } else { "\n" }).await?;
            self.at_line_start = true;
        }
        write(out, &nl(text, self.tty)).await?;
        out.flush().await?;
        self.at_line_start = text.ends_with('\n');
        self.last_kind = Some(kind);
        Ok(())
    }

    /// Ensure the session returns to idle: if still streaming (no `TurnEnded` received),
    /// finalize and redraw the prompt; if already idle, no-op (to avoid double-printing
    /// the prompt with the `TurnEnded` redraw).
    async fn ensure_idle(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.streaming {
            self.end_streaming(out).await?;
        }
        Ok(())
    }

    /// Print an error line (turn fatal error; these do not emit a TurnEnded event), then
    /// return to the idle prompt.
    async fn print_error(&mut self, out: &mut Stdout, text: &str) -> anyhow::Result<()> {
        // If currently streaming, finish the line first.
        if self.streaming && !self.at_line_start {
            write(out, if self.tty { "\r\n" } else { "\n" }).await?;
        }
        self.streaming = false;
        self.last_kind = None;
        if self.tty {
            write(out, &format!("\r\x1b[K{text}\r\n")).await?;
        } else {
            write(out, &format!("{text}\n")).await?;
        }
        self.redraw(out).await?;
        Ok(())
    }
}

/// Replace `\n` with `\r\n` in raw mode (raw terminals do not perform ONLCR translation,
/// so the cursor moves down but not to the first column). In non-raw (non-TTY) mode,
/// return the string unchanged.
fn nl(s: &str, tty: bool) -> String {
    if tty {
        s.replace('\n', "\r\n")
    } else {
        s.to_owned()
    }
}

/// Extracts text from a [`ContentBlock`]; returns `None` for non-text blocks.
fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// Writes a string to stdout.
async fn write(out: &mut Stdout, s: &str) -> anyhow::Result<()> {
    out.write_all(s.as_bytes()).await?;
    Ok(())
}

/// On resume, replay a historical message using the **same visual language as live
/// rendering** (see [`LineEditor::render_event`] / [`LineEditor::echo_submitted`]): a user
/// message gets the `› ` prompt, assistant text is printed bare, thoughts are dimmed
/// italic, tool calls show `⚙ name`, and tool results show `  ↳ status`. Keeping the two
/// paths identical avoids the jarring style break a resumed session used to show. Display
/// only; does not affect session state. Newlines rely on the terminal's cooked mode (raw
/// mode is not yet active).
async fn render_history_message(out: &mut Stdout, message: &Message) -> anyhow::Result<()> {
    for content in message.content.iter() {
        match (message.role, content) {
            // User text mirrors the live input line: `› <text>`.
            (Role::User, MessageContent::Text { text }) => {
                write(out, &format!("{}{text}\n", USER_PROMPT.cyan().bold())).await?;
            }
            // Assistant text is streamed bare during a live turn — replay it the same way.
            (Role::Assistant, MessageContent::Text { text }) => {
                write(out, &format!("{text}\n")).await?;
            }
            (_, MessageContent::Thinking { text, .. }) => {
                write(out, &format!("{}\n", text.dimmed().italic())).await?;
            }
            (_, MessageContent::ToolUse { name, .. }) => {
                write(out, &format!("{} {}\n", "⚙".yellow(), name.yellow())).await?;
            }
            (_, MessageContent::ToolResult { is_error, .. }) => {
                let label = if *is_error { "Failed" } else { "Completed" };
                write(out, &format!("{} {label}\n", "  ↳".dimmed())).await?;
            }
            _ => {}
        }
    }
    Ok(())
}
