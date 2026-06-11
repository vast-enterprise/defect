//! Single-turn unattended mode —— `defect --message <prompt>`.
//!
//! Purpose: in CI / scripts, "run one prompt, produce a result, exit by
//! success/failure". On **equal footing** with the interactive REPL and the
//! ACP server; the three only share the [`AgentCore`] kernel:
//!
//! - Does **not** reuse the REPL's [`crate::repl`] rendering (that stack carries
//!   ANSI / line editing, bound to the `repl` feature's crossterm/owo-colors).
//!   This module ships its own minimal, ANSI-free event projection, so under
//!   `--no-default-features --features oneshot` it builds a slim CI binary with
//!   no TUI dependencies.
//! - Connects in-process directly to `AgentCore` (like the REPL), bypassing the
//!   wire —— CI runs its own agent and does not need ACP's cross-process
//!   generality.
//!
//! ## Output contract: stdout = agent content, stderr = framework logs
//!
//! **All agent content** (assistant body / thinking / tool calls) goes to
//! **stdout** as a single stream in event order; framework-level diagnostics
//! (denial warnings, turn errors, unreached goals) go through `tracing` —— and
//! `tracing` is uniformly written to **stderr** by `defect_obs::init_tracing`.
//! So `2>/dev/null` cleanly filters out framework noise while preserving the
//! agent's complete work record; the two streams no longer share a cursor or
//! run together.
//!
//! ## Exit codes (CI's lifeline for judging success/failure)
//!
//! Priority high to low: `TurnError`(1) > `Refusal`(3) > `MaxTokens`(2) >
//! `MaxTurnRequests`(7) > `Cancelled`(5) > unattended denial (`denied`, 4) >
//! goal unreached (6) > `EndTurn`(0). `MaxTokens` (a single response truncated
//! by the output limit) and `MaxTurnRequests` (the per-turn call budget
//! exhausted) are distinct conditions and carry distinct codes. See
//! `ExitOutcome`.
//!
//! ## Non-interactive permissions
//!
//! The caller (`bin/cli.rs`) is responsible for wrapping the session's policy in
//! [`defect_agent::policy::NonInteractivePolicy`], so that `Ask` degrades to
//! `Deny` and it does not hang waiting for input in a TTY-less environment. This
//! module listens for `PolicyDecision::Deny` in the event stream: once one
//! occurs, it logs a warning via `tracing` (→ stderr) and sets the `denied`
//! flag, so even if the turn ends normally with `EndTurn` it exits with a
//! non-zero code —— fail loud, letting CI know "an operation was denied, this
//! result is not trustworthy".

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent, ToolCallId};
use defect_agent::event::AgentEvent;
use defect_agent::policy::PolicyDecision;
use defect_agent::session::{AgentCore, TurnError};
use futures::{FutureExt, StreamExt};
use tokio::io::{AsyncWriteExt, Stdout};

use crate::args::OutputFormat;
use crate::session_open::{LocalSessionOpts, open_local_session};

/// Runs a single-turn prompt and returns the process exit code.
///
/// Only when `track_denied = true` (the caller wrapped `NonInteractivePolicy`)
/// is `PolicyDecision::Deny` treated as an "unattended gap" that affects the
/// exit code —— modes like `deny-all`, where the user knowingly expects
/// denials, should pass `false` to avoid spuriously reporting non-zero.
///
/// # Errors
///
/// Session open failure, stdin read failure, stdout/stderr write failure.
/// When `goal` is `Some` (`--goal` mode): if the goal is unreached after the
/// turn ends (turns exhausted without ever calling `goal_done`), exits with the
/// Exhausted code —— prevents CI from mistaking "ran out of turns without
/// reaching the goal" for success.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    agent: Arc<dyn AgentCore>,
    cwd: PathBuf,
    message: String,
    format: OutputFormat,
    resume: Option<SessionId>,
    track_denied: bool,
    goal: Option<Arc<defect_agent::session::GoalState>>,
    shell_output_max_bytes: usize,
) -> anyhow::Result<ExitCode> {
    let prompt = resolve_prompt(message).await?;

    let mut out = tokio::io::stdout();

    let session = open_local_session(
        &agent,
        &cwd,
        LocalSessionOpts {
            resume,
            shell_output_max_bytes,
        },
    )
    .await?;

    // Subscribe once outside the loop, draining across this turn (including the
    // driver's autonomous turn continuations) —— isomorphic to the interactive
    // REPL / ACP event pump. The turn future only yields the final StopReason;
    // this turn's content is pushed through the event stream.
    let mut events = session.subscribe();
    let mut sink = EventSink::new(format, track_denied);

    let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt))];
    let turn = session.run_turn(prompt_blocks);
    tokio::pin!(turn);

    let result = loop {
        tokio::select! {
            // biased + drain events first: the turn future and the event stream
            // may become ready at the same time (turn finishes fast, trailing
            // events already in the buffer). Poll events first to avoid dropping
            // any rendering.
            biased;
            ev = events.next() => {
                if let Some(ev) = ev {
                    sink.emit(&mut out, ev).await?;
                }
            }
            r = &mut turn => break r,
        }
    };

    // Turn has ended, but the buffer may still hold just-sent, not-yet-polled
    // trailing events —— drain everything that is immediately ready.
    while let Some(Some(ev)) = events.next().now_or_never() {
        sink.emit(&mut out, ev).await?;
    }

    // goal mode: turn ended normally but the goal is unreached (turns exhausted
    // without ever calling goal_done) → Exhausted.
    let goal_unreached = goal.as_ref().is_some_and(|g| !g.is_reached());
    if goal_unreached {
        tracing::warn!(
            "goal not reached: the agent stopped (or ran out of turns) without calling `goal_done`"
        );
    }
    let outcome = ExitOutcome::from(&result, sink.denied, goal_unreached);
    sink.finish(&mut out, &result, &outcome).await?;
    out.flush().await?;
    Ok(outcome.code())
}

/// Resolves the prompt source: `-`, or read from stdin when stdin is piped;
/// otherwise use the literal value.
async fn resolve_prompt(message: String) -> anyhow::Result<String> {
    use std::io::IsTerminal;

    let from_stdin = message == "-" || (message.is_empty() && !std::io::stdin().is_terminal());
    if from_stdin {
        use tokio::io::AsyncReadExt;
        let mut buf = String::new();
        tokio::io::stdin().read_to_string(&mut buf).await?;
        Ok(buf)
    } else {
        Ok(message)
    }
}

/// Reduction from the turn result to the process exit code.
enum ExitOutcome {
    Success,
    Denied,
    Cancelled,
    /// A single LLM response was truncated by the output `max_tokens` limit.
    MaxTokens,
    /// The per-turn LLM-call budget (`request_limit`) was exhausted.
    MaxRequests,
    Refusal,
    Error,
    /// goal mode: turn ended normally but the goal is unreached (turns exhausted
    /// / model gave up).
    GoalUnreached,
}

impl ExitOutcome {
    fn from(result: &Result<StopReason, TurnError>, denied: bool, goal_unreached: bool) -> Self {
        match result {
            Err(_) => Self::Error,
            Ok(StopReason::Refusal) => Self::Refusal,
            Ok(StopReason::MaxTokens) => Self::MaxTokens,
            Ok(StopReason::MaxTurnRequests) => Self::MaxRequests,
            Ok(StopReason::Cancelled) => Self::Cancelled,
            // EndTurn (and any future success-class terminal states): denied >
            // goal unreached > success.
            Ok(_) if denied => Self::Denied,
            Ok(_) if goal_unreached => Self::GoalUnreached,
            Ok(_) => Self::Success,
        }
    }

    /// Numeric exit code (0 = success).
    fn raw(&self) -> u8 {
        match self {
            Self::Success => 0,
            Self::Error => 1,
            Self::MaxTokens => 2,
            Self::Refusal => 3,
            Self::Denied => 4,
            Self::Cancelled => 5,
            Self::GoalUnreached => 6,
            Self::MaxRequests => 7,
        }
    }

    fn code(&self) -> ExitCode {
        ExitCode::from(self.raw())
    }
}

/// Event projector: writes the [`AgentEvent`] stream to stdout/stderr according
/// to the [`OutputFormat`].
struct EventSink {
    format: OutputFormat,
    track_denied: bool,
    /// Whether an unattended denial has occurred.
    denied: bool,
    /// `ToolCallId → tool name`, used to report which tool was involved on a
    /// `PolicyDecision::Deny`.
    tool_names: HashMap<ToolCallId, String>,
    /// In text format: whether stdout is still mid-line (the last thing written
    /// was not a `\n`). In goal mode it inserts newlines between multi-turn
    /// assistant outputs, avoiding "previous tail + next head" running together
    /// on one line.
    mid_line: bool,
    /// In text format: whether we are currently inside a thinking block. A
    /// thinking block's multiple deltas share one `[thinking] ` prefix (printed
    /// only on the first delta), so consecutive deltas merge into one block.
    in_thought: bool,
}

impl EventSink {
    fn new(format: OutputFormat, track_denied: bool) -> Self {
        Self {
            format,
            track_denied,
            denied: false,
            tool_names: HashMap::new(),
            mid_line: false,
            in_thought: false,
        }
    }

    async fn emit(&mut self, out: &mut Stdout, event: AgentEvent) -> anyhow::Result<()> {
        // Record tool names (needed for every format, used for denial reports).
        if let AgentEvent::ToolCallStarted { id, name, fields } = &event {
            let label = fields.title.clone().unwrap_or_else(|| name.clone());
            self.tool_names.insert(id.clone(), label);
        }

        // Unattended denial: framework-level diagnostic via tracing (→ stderr) +
        // set the flag (fail loud).
        if self.track_denied
            && let AgentEvent::PolicyDecision {
                id,
                decision: PolicyDecision::Deny,
            } = &event
        {
            self.denied = true;
            let tool = self
                .tool_names
                .get(id)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            tracing::warn!(
                tool = %tool,
                "tool denied: no operator present to approve (non-interactive)"
            );
        }

        match self.format {
            OutputFormat::Json => self.emit_json(out, &event).await,
            OutputFormat::Text => self.emit_text(out, &event).await,
            OutputFormat::Quiet => Ok(()),
        }
    }

    /// NDJSON: one line per event. `AgentEvent` already derives Serialize
    /// (`LlmCallStarted.request` is `#[serde(skip)]`, so it never enters JSON).
    async fn emit_json(&self, out: &mut Stdout, event: &AgentEvent) -> anyhow::Result<()> {
        let line = serde_json::to_string(event)?;
        write_raw(out, &line).await?;
        write_raw(out, "\n").await
    }

    /// Plain text: **all agent content** (body / thinking / tools) goes to
    /// stdout as a single stream in event order —— framework logs (tracing) go
    /// to stderr, the two never interfere.
    ///
    /// Boundary newlines: assistant body often has no trailing newline, while
    /// what immediately follows may be a thinking / tool line or the next
    /// generation. Before switching to a "non-body line" (thinking / tool) or
    /// starting a new generation segment, if stdout is still mid-line insert a
    /// `\n`, so each segment occupies whole lines and does not run together.
    async fn emit_text(&mut self, out: &mut Stdout, event: &AgentEvent) -> anyhow::Result<()> {
        match event {
            // A new LLM generation starts: if the previous assistant body had no
            // newline, insert one before starting the new segment.
            AgentEvent::LlmCallStarted { .. } | AgentEvent::TurnEnded { .. } => {
                self.end_thought(out).await?;
                self.break_line(out).await?;
            }
            AgentEvent::AssistantText { content } => {
                if let Some(text) = block_text(content)
                    && !text.is_empty()
                {
                    self.end_thought(out).await?;
                    write_raw(out, &text).await?;
                    out.flush().await?;
                    self.mid_line = !text.ends_with('\n');
                }
            }
            // A thinking block's multiple deltas share one `[thinking] ` prefix
            // —— consecutive deltas merge, with each delta written raw.
            AgentEvent::AssistantThought { content } => {
                if let Some(text) = block_text(content)
                    && !text.is_empty()
                {
                    if !self.in_thought {
                        self.break_line(out).await?;
                        write(out, "[thinking] ").await?;
                        self.in_thought = true;
                    }
                    write_raw(out, &text).await?;
                    out.flush().await?;
                    self.mid_line = !text.ends_with('\n');
                }
            }
            AgentEvent::ToolCallStarted { name, fields, .. } => {
                self.end_thought(out).await?;
                self.break_line(out).await?;
                let title = fields.title.clone().unwrap_or_else(|| name.clone());
                write(out, &format!("[tool] {title}\n")).await?;
                out.flush().await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Ends the current thinking block: if inside one, clears the flag and
    /// inserts a `\n`. Called before switching to body / tool / a new generation
    /// / turn end, so the thinking block occupies whole lines.
    async fn end_thought(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.in_thought {
            self.in_thought = false;
            self.break_line(out).await?;
        }
        Ok(())
    }

    /// If stdout is still mid-line, insert a `\n` to close it and flush. Used
    /// before a "non-body line" (thinking / tool) or a new generation segment,
    /// so the previous assistant body occupies its own whole line —— even within
    /// a single stream, content is separated by line.
    async fn break_line(&mut self, out: &mut Stdout) -> anyhow::Result<()> {
        if self.mid_line {
            write_raw(out, "\n").await?;
            out.flush().await?;
            self.mid_line = false;
        }
        Ok(())
    }

    /// Wrap-up output after the turn ends. Framework-level diagnostics (turn
    /// errors) go through tracing (→ stderr).
    async fn finish(
        &self,
        out: &mut Stdout,
        result: &Result<StopReason, TurnError>,
        outcome: &ExitOutcome,
    ) -> anyhow::Result<()> {
        if let Err(e) = result {
            tracing::error!(error = %e, "turn error");
        }
        match self.format {
            OutputFormat::Text => {
                // When the streamed assistant body has no trailing newline,
                // insert one to avoid running into the following shell prompt;
                // if already at line start (mid_line=false) skip it, to not emit
                // an extra blank line.
                if self.mid_line {
                    write_raw(out, "\n").await?;
                }
            }
            OutputFormat::Json => {
                // Final summary line: terminal state + exit-code semantics.
                let summary = serde_json::json!({
                    "type": "oneshot_result",
                    "stop_reason": result.as_ref().ok().map(|r| format!("{r:?}")),
                    "error": result.as_ref().err().map(|e| e.to_string()),
                    "denied": self.denied,
                    "exit_code": outcome.raw(),
                });
                write_raw(out, &summary.to_string()).await?;
                write_raw(out, "\n").await?;
            }
            OutputFormat::Quiet => {}
        }
        Ok(())
    }
}

/// Extracts text from a [`ContentBlock`]; non-text blocks return `None`.
fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// Writes a string to any async writer.
async fn write<W>(out: &mut W, s: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    out.write_all(s.as_bytes()).await?;
    Ok(())
}

/// Alias for `write`, semantically emphasizing "write as-is, with no
/// decoration".
async fn write_raw<W>(out: &mut W, s: &str) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write(out, s).await
}
