//! Session-level background task table.
//!
//! ## Problem
//!
//! Tools (primarily `spawn_agent { run_in_background: true }`) want to fire-and-forget
//! a task without blocking the initiating turn. However, the turn main loop's
//! `run_tools_concurrently` holds tool tasks in a function-local `JoinSet` — when the
//! function returns, the `JoinSet` is dropped and tasks are aborted, so no task can
//! outlive the turn that created it.
//!
//! [`BackgroundTasks`] moves task `JoinHandle`s to the **session level** (same lifetime
//! as `events` / `history`), allowing tasks to outlive their initiating turn. It also
//! uses a **session-level [`CancellationToken`]** (not a turn child token) to mint
//! per-task child tokens, making cancellation lifecycle independent of the initiating
//! turn.
//!
//! ## Reflow (phase 1: passive)
//!
//! When a task completes, it pushes a [`BackgroundOutcome`] into the `completed` queue.
//! `DefaultSession::run_turn` calls [`drain_completed`](BackgroundTasks::drain_completed)
//! before each turn, bringing pending results into history as **prefix blocks** of the
//! current user prompt — the LLM sees the results alongside the next user input.
//! Phase 2 (active continuation) is handled by the session input loop competing for a
//! new turn when a background task completes.
//!
//! ## Introspection and single-point cancellation (control plane)
//!
//! Tasks **do not disappear immediately after completion**: each task retains a
//! [`TaskEntry`] in the `tasks` table, recording status (running / completed / failed /
//! cancelled) and a **shared handle to the task's history**.
//!
//! The progress "block" granularity is deliberately set to **message blocks submitted to
//! the LLM** ([`crate::llm::Message`]) — not streaming deltas. Streaming
//! `AssistantText` / `AssistantThought` chunks produce several words per chunk (mapping
//! to ACP `AgentMessageChunk`), which are unhelpful for understanding "what is this
//! subagent doing now". The meaningful granularity is at the turn / tool-call boundary.
//! The main loop drains the entire batch, coalesces them into a single assistant
//! `Message`, and appends it to history — that is the moment a "block" is sent to the
//! AI. Therefore, `spawn_agent` shares the sub-turn history `Arc` into this table
//! (the sub-turn appends to it), and `peek` snapshots that history directly, taking
//! the **most recent N message blocks** — a single source of truth (identical to what
//! is fed to the LLM), no replay/coalesce of streaming deltas needed elsewhere.
//!
//! This allows the main agent to inspect a background sub-agent's progress with
//! `inspect_background_task`, or cancel a single task early via
//! [`cancel_task`](BackgroundTasks::cancel_task) without affecting other tasks
//! (each task has its own child token). Completed task entries are evicted by FIFO
//! upper bound to prevent unbounded growth in long sessions.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::llm::{Message, MessageContent, Role};
use crate::session::History;

/// Default number of recent message blocks returned by `inspect_background_task` when
/// `recent_blocks` is not specified.
const DEFAULT_RECENT_BLOCKS: usize = 10;

/// How many **finished** task entries to keep in the `tasks` table. Running entries don't
/// count toward the cap—they must remain to be cancelable/peekable. When the cap is
/// exceeded, the oldest finished entry is evicted.
const FINISHED_TASKS_CAP: usize = 64;

/// Configuration for the background task **progress view**.
///
/// The goal is to give the main agent a **bird's-eye** view of what a subagent is
/// currently doing, **not** to flood the main agent's context with the full text of
/// sub-turns. Therefore the defaults are conservative — assistant/thinking text is
/// **omitted** by default (`block_text_limit = 0`, reporting only metadata like "there is
/// an assistant text / thinking"); tool calls, which are naturally short, are kept as-is.
/// Users can increase `block_text_limit` when more detail is needed.
///
/// The source of truth for configuration lives on the agent side (here).
/// `defect-config`'s `ToolsConfig.background` reuses this struct directly (same
/// cross-crate reuse pattern as `TurnConfig` / `SessionCapabilitiesConfig` — config
/// depends on agent, not the other way around).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundProgressConfig {
    /// How many recent message blocks `inspect_background_task` returns by default when
    /// called without the `recent_blocks` argument.
    /// `0` is treated as `1` (at least one block is always returned, otherwise peek would
    /// always be empty).
    pub default_recent_blocks: usize,
    /// Maximum number of Unicode scalar values for the **body** of a single block,
    /// applied to free-form text such as assistant messages, thinking blocks, and tool
    /// results. Text exceeding this limit is truncated at the boundary with an ellipsis
    /// marker. `0` means no body text is kept (only the block's type and metadata, e.g.
    /// tool name) — this is the default, and minimizes pollution of the main agent's
    /// context.
    pub block_text_limit: usize,
}

impl Default for BackgroundProgressConfig {
    fn default() -> Self {
        Self {
            default_recent_blocks: DEFAULT_RECENT_BLOCKS,
            // By default, only summary/metadata is provided, not the full body — the goal
            // is an overview, not context transfer.
            block_text_limit: 0,
        }
    }
}

impl BackgroundProgressConfig {
    /// Normalize `recent_blocks`: if the caller passes `Some(n)`, use `n` (at least 1);
    /// if `None`, use the config default (at least 1).
    fn resolve_recent(&self, requested: Option<usize>) -> usize {
        requested.unwrap_or(self.default_recent_blocks).max(1)
    }
}

/// The outcome produced after a background task completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundOutcome {
    /// The task ID (same as returned by `spawn`), used for backflow message annotation
    /// and external diagnostics.
    pub task_id: String,
    /// Task label (primarily from the `spawn_agent` profile name), included in the return
    /// message so the model or user can identify the source.
    pub label: String,
    /// The result of the background task.
    pub result: BackgroundResult,
}

/// The final result of a background task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundResult {
    /// Completed successfully, containing the task's final text output.
    Completed(String),
    /// Failure (including cancellation), with an error description.
    Failed(String),
}

impl BackgroundResult {
    fn is_error(&self) -> bool {
        matches!(self, BackgroundResult::Failed(_))
    }

    fn text(&self) -> &str {
        match self {
            BackgroundResult::Completed(t) | BackgroundResult::Failed(t) => t,
        }
    }
}

/// Lifecycle status of a background task, exposed via the `inspect_background_task`
/// control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Still running.
    Running,
    /// The task completed successfully.
    Completed,
    /// The task failed.
    Failed,
    /// Canceled by [`cancel_task`](BackgroundTasks::cancel_task) /
    /// [`cancel_all`](BackgroundTasks::cancel_all).
    Canceled,
}

impl TaskStatus {
    /// Stable lowercase string name for control-plane tool output.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Canceled => "canceled",
        }
    }

    fn is_terminal(&self) -> bool {
        !matches!(self, TaskStatus::Running)
    }
}

/// The role/category of a progress block. Directly corresponds to the content of a
/// [`crate::llm::Message`] submitted to the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// User/task input message (including backflow of background results, tool result
    /// re-injection, etc.).
    User,
    /// Text produced by the assistant.
    AssistantText,
    /// The assistant's chain of thought.
    Thought,
    /// A tool call initiated by the assistant.
    ToolUse,
    /// Tool result (fed back to the model).
    ToolResult,
    /// Other (multimodal / provider activity, etc.), normalized for display.
    Other,
}

impl BlockKind {
    /// Stable lowercase string name for control-plane tool output.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockKind::User => "user",
            BlockKind::AssistantText => "assistant",
            BlockKind::Thought => "thought",
            BlockKind::ToolUse => "tool_use",
            BlockKind::ToolResult => "tool_result",
            BlockKind::Other => "other",
        }
    }

    /// Whether this kind of block's text is "free-form body" — subject to the limit in
    /// [`BackgroundProgressConfig::block_text_limit`]. Tool call names are inherently
    /// one-line summaries, not body text, and are not subject to the limit.
    fn is_free_form_body(&self) -> bool {
        matches!(
            self,
            BlockKind::User | BlockKind::AssistantText | BlockKind::Thought | BlockKind::ToolResult
        )
    }
}

/// A single progress block returned by `peek`: kind + text summary (truncated per
/// configuration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressBlock {
    pub kind: BlockKind,
    pub text: String,
}

/// A snapshot of a task in the control plane (returned by `list` / `peek`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub task_id: String,
    pub label: String,
    pub status: TaskStatus,
    /// Total number of progress blocks currently in this task's history (only populated
    /// by `peek`; `list` returns `0` because it does not read history).
    pub block_count: usize,
    /// Recent blocks (empty for `list`; contains the latest N blocks for `peek`).
    pub recent: Vec<ProgressBlock>,
}

/// Truncate free-form text to a character limit (splits on Unicode scalar boundaries,
/// never breaking a character). `limit == 0` returns an empty string (metadata only).
/// Appends ` …(+N more chars)` to indicate truncation.
fn truncate_body(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    let total = text.chars().count();
    if total <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{kept} …(+{} more chars)", total - limit)
}

/// Extract a human-readable text snippet from a
/// [`ToolResultBody`](crate::llm::ToolResultBody) (for a bird's-eye summary only).
fn tool_result_text(body: &crate::llm::ToolResultBody) -> String {
    use crate::llm::{ToolResultBody, ToolResultContent};
    match body {
        ToolResultBody::Text { text } => text.clone(),
        ToolResultBody::Json { value } => value.to_string(),
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text.clone(),
                ToolResultContent::Image { .. } => "<image>".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Maps a [`MessageContent`] to a progress block, truncating the body to `limit` (free
/// text only).
fn block_of_content(content: &MessageContent, role: Role, limit: usize) -> ProgressBlock {
    let (kind, raw): (BlockKind, String) = match content {
        MessageContent::Text { text } => {
            let kind = if role == Role::Assistant {
                BlockKind::AssistantText
            } else {
                BlockKind::User
            };
            (kind, text.clone())
        }
        MessageContent::Thinking { text, .. } => (BlockKind::Thought, text.clone()),
        // The tool name is a one-line summary and should not be truncated as body text;
        // parameters are excluded from the bird's-eye view (see the langfuse trace for
        // details).
        MessageContent::ToolUse { name, .. } => (BlockKind::ToolUse, name.clone()),
        MessageContent::ToolResult { output, .. } => {
            (BlockKind::ToolResult, tool_result_text(output))
        }
        MessageContent::Image { .. } => (BlockKind::Other, "<image>".to_string()),
        MessageContent::ProviderActivity { kind, .. } => {
            (BlockKind::Other, format!("provider activity: {kind:?}"))
        }
    };
    let text = if kind.is_free_form_body() {
        truncate_body(&raw, limit)
    } else {
        raw
    };
    ProgressBlock { kind, text }
}

/// Extracts the **most recent `n`** message blocks from a history snapshot (flattens each
/// [`Message`]'s content fragments into individual blocks while preserving chronological
/// order), truncating each block's body to `limit`. Returns `(total_blocks,
/// last_n_blocks)`.
fn recent_blocks_of(messages: &[Message], n: usize, limit: usize) -> (usize, Vec<ProgressBlock>) {
    let mut all: Vec<ProgressBlock> = Vec::new();
    for m in messages {
        for c in m.content.iter() {
            all.push(block_of_content(c, m.role, limit));
        }
    }
    let total = all.len();
    let skip = total.saturating_sub(n);
    (total, all.into_iter().skip(skip).collect())
}

/// An entry in the `tasks` table.
struct TaskEntry {
    label: String,
    status: TaskStatus,
    /// Cancellation token specific to this task (child of the session-level token).
    /// `cancel_task` calls `cancel` on it individually.
    cancel: CancellationToken,
    /// Shared handle to this task's history. `peek` uses it to snapshot the message
    /// blocks submitted to the LLM.
    /// `Some`: the tool that spawned the task (`spawn_agent`) shared the child turn's
    /// history via `Arc`;
    /// `None`: the task does not expose history (no progress to query; `peek` only
    /// returns status).
    history: Option<Arc<dyn History>>,
    /// The `JoinHandle` that keeps the task alive past the turn that spawned it. Set to
    /// `None` after completion.
    handle: Option<JoinHandle<()>>,
    /// Sequence number for termination order (only present on finished entries), used for
    /// FIFO eviction.
    finished_seq: Option<u64>,
}

struct BackgroundInner {
    /// Monotonically increasing task ID counter.
    next_id: u64,
    /// Monotonically increasing "finish order" counter for FIFO eviction of finished
    /// entries.
    next_finished_seq: u64,
    /// All tasks (running + recently finished). When finished entries exceed
    /// [`FINISHED_TASKS_CAP`], the oldest are evicted.
    tasks: BTreeMap<String, TaskEntry>,
    /// Completed results pending drain (FIFO). Emptied by `drain_completed`. Orthogonal
    /// to the `tasks` table: this drives passive draining, while `tasks` supports
    /// control-plane queries and interrupts.
    completed: Vec<BackgroundOutcome>,
}

impl BackgroundInner {
    /// Marks a task as finished, records its finish sequence number, and evicts the
    /// oldest finished entries up to the capacity limit.
    fn finish(&mut self, id: &str, status: TaskStatus) {
        let seq = self.next_finished_seq;
        self.next_finished_seq += 1;
        if let Some(entry) = self.tasks.get_mut(id) {
            entry.status = status;
            entry.handle = None;
            entry.finished_seq = Some(seq);
        }
        self.prune_finished();
    }

    /// When finished entries exceed the cap, evict the oldest ones by finish sequence.
    /// Running entries are never evicted.
    fn prune_finished(&mut self) {
        let mut finished: Vec<(u64, String)> = self
            .tasks
            .iter()
            .filter_map(|(id, e)| e.finished_seq.map(|seq| (seq, id.clone())))
            .collect();
        if finished.len() <= FINISHED_TASKS_CAP {
            return;
        }
        finished.sort_by_key(|(seq, _)| *seq);
        let drop_count = finished.len() - FINISHED_TASKS_CAP;
        for (_, id) in finished.into_iter().take(drop_count) {
            self.tasks.remove(&id);
        }
    }
}

/// Session-level background task table. `Clone` is cheap (inner `Arc`) — `DefaultSession`
/// holds one copy, cloned to tools via `ToolContext`.
#[derive(Clone)]
pub struct BackgroundTasks {
    /// Session-level cancellation token. Each task derives its token via `child_token()`,
    /// so `cancel_all` cancels all tasks at once, while cancelling any single task does
    /// not affect the others.
    cancel: CancellationToken,
    /// Notifies when a task completes. Each time a task result is enqueued, `notify_one`
    /// is called — the session driver waits on this and, when woken, starts an autonomous
    /// turn to continue processing (phase two). Passive backpressure does not rely on it.
    completed_notify: Arc<Notify>,
    /// Progress view configuration (default block count / body limit). `peek` renders
    /// based on this.
    progress_config: BackgroundProgressConfig,
    inner: Arc<Mutex<BackgroundInner>>,
}

impl BackgroundTasks {
    /// Constructs a new instance with a session-level cancellation token and a
    /// progress-view configuration. `session_cancel` is owned by the session and is
    /// cancelled when the session terminates.
    #[must_use]
    pub fn new(
        session_cancel: CancellationToken,
        progress_config: BackgroundProgressConfig,
    ) -> Self {
        Self {
            cancel: session_cancel,
            completed_notify: Arc::new(Notify::new()),
            progress_config,
            inner: Arc::new(Mutex::new(BackgroundInner {
                next_id: 0,
                next_finished_seq: 0,
                tasks: BTreeMap::new(),
                completed: Vec::new(),
            })),
        }
    }

    /// Wait for a "task completion enqueued" event. The session driver uses this to drive
    /// proactive continuation.
    ///
    /// Uses `Notify`: the driver first calls `notified()` to obtain a future, then checks
    /// the queue, then awaits — avoiding missed notifications that arrive between checks
    /// (`Notify`'s permit semantics guarantee that already-fired notifies are not lost).
    pub async fn wait_for_completion(&self) {
        self.completed_notify.notified().await;
    }

    /// Whether there are completed results waiting to be collected. The driver checks
    /// this after waking up to decide whether to start a turn.
    #[must_use]
    pub fn has_completed(&self) -> bool {
        !self
            .inner
            .lock()
            .expect("BackgroundTasks mutex poisoned")
            .completed
            .is_empty()
    }

    /// Spawns a background task and returns its ID **immediately**.
    ///
    /// `make_fut` receives two handles: a [`CancellationToken`] specific to this task (a
    /// child of the session-level token, which the task body should use to observe
    /// cancellation) and a [`TaskHandle`] (the task body shares its history `Arc` into
    /// the table via [`TaskHandle::attach_history`], allowing the control plane to peek
    /// at the **message chunks submitted to the LLM**). On completion, the result is
    /// placed in the `completed` queue and the corresponding entry in the `tasks` table
    /// is marked as terminal (the entry is retained for later inspection).
    ///
    /// The closure form that "receives token/handle and then creates the future" is used
    /// because both must be minted inside `spawn`, and the future needs to capture them —
    /// accepting a future directly would not allow obtaining a token whose lifetime is
    /// independent of the turn.
    pub fn spawn<F, Fut>(&self, label: String, make_fut: F) -> String
    where
        F: FnOnce(CancellationToken, TaskHandle) -> Fut,
        Fut: Future<Output = BackgroundResult> + Send + 'static,
    {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let id = format!("bg-{}", inner.next_id);
        inner.next_id += 1;

        let task_cancel = self.cancel.child_token();
        let handle = TaskHandle {
            inner: self.inner.clone(),
            task_id: id.clone(),
        };
        // The task body can detect whether it was cancelled, so that completion
        // distinguishes between `Failed` and `Canceled` states.
        let cancel_for_task = task_cancel.clone();
        let fut = make_fut(task_cancel.clone(), handle);

        let inner_arc = self.inner.clone();
        let notify = self.completed_notify.clone();
        let id_for_task = id.clone();
        let label_for_task = label.clone();
        let join = tokio::spawn(async move {
            let result = fut.await;
            // Distinguish between a task error and an explicit cancellation: the latter
            // records the status as `Canceled`, the former as `Failed`.
            let status = if cancel_for_task.is_cancelled() {
                TaskStatus::Canceled
            } else if result.is_error() {
                TaskStatus::Failed
            } else {
                TaskStatus::Completed
            };
            if let Ok(mut inner) = inner_arc.lock() {
                inner.finish(&id_for_task, status);
                inner.completed.push(BackgroundOutcome {
                    task_id: id_for_task,
                    label: label_for_task,
                    result,
                });
            }
            // Wakes the session driver waiting on `wait_for_completion` (active
            // continuation).
            // Uses `notify_one` instead of `notify_waiters`: the former **retains a
            // permit** when no waiters exist,
            // so the next `notified().await` returns immediately — avoiding lost wakeups
            // when a task completes
            // before the driver parks. Single consumer (exactly one driver), so
            // `notify_one` semantics are correct.
            // Notify outside the lock.
            notify.notify_one();
        });

        inner.tasks.insert(
            id.clone(),
            TaskEntry {
                label,
                status: TaskStatus::Running,
                cancel: task_cancel,
                history: None,
                handle: Some(join),
                finished_seq: None,
            },
        );
        id
    }

    /// Drain all completed results (clears the queue). Called by `run_turn` before
    /// starting a turn to passively collect results.
    pub fn drain_completed(&self) -> Vec<BackgroundOutcome> {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        std::mem::take(&mut inner.completed)
    }

    /// Number of currently running tasks. Used for diagnostics / control plane.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.inner
            .lock()
            .expect("BackgroundTasks mutex poisoned")
            .tasks
            .values()
            .filter(|e| e.status == TaskStatus::Running)
            .count()
    }

    /// Returns a snapshot of all tasks (running + recently finished), **without reading
    /// history** (`recent` is empty, `block_count` is 0). Sorted by task ID in ascending
    /// order. Used by `inspect_background_task` when called without arguments.
    #[must_use]
    pub fn list(&self) -> Vec<TaskSnapshot> {
        let inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        inner
            .tasks
            .iter()
            .map(|(id, e)| TaskSnapshot {
                task_id: id.clone(),
                label: e.label.clone(),
                status: e.status,
                block_count: 0,
                recent: Vec::new(),
            })
            .collect()
    }

    /// Take a snapshot of a single task, including the most recent `recent_blocks`
    /// message blocks submitted to the LLM (`None` uses the config default). Returns
    /// `None` if the task does not exist (never spawned or already evicted); blocks are
    /// empty if the task does not expose history.
    ///
    /// Implementation: clone the task's history `Arc` while holding the table lock, then
    /// release the table lock before snapshotting (snapshotting uses the history's own
    /// lock). This avoids performing a potentially expensive deep copy of history while
    /// holding the table lock, which would block spawn/finish.
    #[must_use]
    pub fn peek(&self, id: &str, recent_blocks: Option<usize>) -> Option<TaskSnapshot> {
        let n = self.progress_config.resolve_recent(recent_blocks);
        let limit = self.progress_config.block_text_limit;
        let (label, status, history) = {
            let inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
            let entry = inner.tasks.get(id)?;
            (entry.label.clone(), entry.status, entry.history.clone())
        };
        let (block_count, recent) = match history {
            Some(h) => recent_blocks_of(&h.snapshot(), n, limit),
            None => (0, Vec::new()),
        };
        Some(TaskSnapshot {
            task_id: id.to_string(),
            label,
            status,
            block_count,
            recent,
        })
    }

    /// Cancel a single task early: cancels only its dedicated child token, without
    /// affecting other tasks.
    ///
    /// Returns `Some(true)` if a running task was found and cancellation was requested;
    /// `Some(false)` if the task exists but is already in a terminal state (no-op);
    /// `None` if no such id exists. Cancellation is **cooperative** — the task body must
    /// observe its cancel token and exit; the status transitions to `Canceled` only when
    /// the task actually finishes.
    pub fn cancel_task(&self, id: &str) -> Option<bool> {
        let inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let entry = inner.tasks.get(id)?;
        if entry.status.is_terminal() {
            return Some(false);
        }
        entry.cancel.cancel();
        Some(true)
    }

    /// Cancels all background tasks (called when the session ends). Idempotent.
    pub fn cancel_all(&self) {
        self.cancel.cancel();
    }
}

/// A handle given to a background task, allowing it to share its history `Arc` into the
/// task table so the control plane can peek at the message chunks it submits to the LLM.
/// `Clone` is cheap (inner `Arc` + small string).
#[derive(Clone)]
pub struct TaskHandle {
    inner: Arc<Mutex<BackgroundInner>>,
    task_id: String,
}

impl TaskHandle {
    /// Shares this task's history handle into the task table. Called by the `spawn_agent`
    /// background path before constructing a child turn, passing the child turn's history
    /// `Arc` — afterwards `peek` can snapshot the message chunks the child agent has
    /// committed. The task entry may have already been evicted (in extreme cases the task
    /// finishes instantly and is dropped by FIFO), in which case the operation is
    /// silently ignored.
    pub fn attach_history(&self, history: Arc<dyn History>) {
        if let Ok(mut inner) = self.inner.lock()
            && let Some(entry) = inner.tasks.get_mut(&self.task_id)
        {
            entry.history = Some(history);
        }
    }
}

/// Formats a background task outcome into a text block that is fed back into the
/// conversation.
///
/// The wording is structured as a "deferred tool result return", clearly marking the
/// source (task id + label) and success/failure, to prevent the model from
/// misinterpreting it as user speech.
/// Phase 2 will replace this with the proper ingest path using `IngestSource::Background`,
/// at which point this function will be superseded by the corresponding payload.
#[must_use]
pub fn format_background_outcome(outcome: &BackgroundOutcome) -> String {
    let status = if outcome.result.is_error() {
        "failed"
    } else {
        "completed"
    };
    format!(
        "⟨background task {} ({}) {}⟩\n{}",
        outcome.task_id,
        outcome.label,
        status,
        outcome.result.text()
    )
}

#[cfg(test)]
mod tests;
