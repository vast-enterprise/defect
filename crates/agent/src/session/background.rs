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
//! per-task child tokens, making cancellation lifecycle independent of the initiating turn.
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

/// `inspect_background_task` 不带 `recent_blocks` 时默认返回多少条最近消息块。
const DEFAULT_RECENT_BLOCKS: usize = 10;

/// `tasks` 表里保留多少个**已结束**的任务条目。运行中的条目不计入上限——它们必须留着
/// 才能被 cancel / peek。超过上限时按结束顺序淘汰最旧的那条。
const FINISHED_TASKS_CAP: usize = 64;

/// 后台任务**进度视图**的配置。
///
/// 目的：给主 agent 一个"这个 subagent 此刻大致在干嘛"的**鸟瞰**，而**不是**把子 turn
/// 的完整正文灌回主 agent 上下文。所以默认偏保守——assistant/思考的正文默认**不留**
/// （`block_text_limit = 0`，只报"有一条 assistant 文本 / 思考"这类元信息）；工具调用
/// 这类本就简短的块原样保留。需要更细时用户再放大 `block_text_limit`。
///
/// 配置真相源在 agent 侧（这里），`defect-config` 的 `ToolsConfig.background` 直接复用本
/// 结构（与 `TurnConfig` / `SessionCapabilitiesConfig` 同款跨 crate 复用——config 依赖
/// agent，不能反向）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundProgressConfig {
    /// `inspect_background_task` 不带 `recent_blocks` 参数时，默认返回多少条最近消息块。
    /// `0` 视作 `1`（至少给一条，否则 peek 永远空）。
    pub default_recent_blocks: usize,
    /// 单个 block 的**正文**字符上限（按 Unicode 标量计），作用于 assistant 文本 / 思考 /
    /// 工具结果这类自由正文。超出即在边界截断并加省略标记。`0` = 不留正文（只报块的
    /// 类型与元信息，如工具名）——默认值，最不污染主 agent 上下文。
    pub block_text_limit: usize,
}

impl Default for BackgroundProgressConfig {
    fn default() -> Self {
        Self {
            default_recent_blocks: DEFAULT_RECENT_BLOCKS,
            // 默认只给摘要 / 元信息，不灌正文——目的是鸟瞰，不是搬运上下文。
            block_text_limit: 0,
        }
    }
}

impl BackgroundProgressConfig {
    /// 规整 `recent_blocks`：调用方传 `Some(n)` 用 `n`（至少 1），`None` 用配置默认（至少 1）。
    fn resolve_recent(&self, requested: Option<usize>) -> usize {
        requested.unwrap_or(self.default_recent_blocks).max(1)
    }
}

/// 一个后台任务完成后的产物。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundOutcome {
    /// 任务 id（`spawn` 返回的同一个），用于回流消息标注与外部诊断。
    pub task_id: String,
    /// 任务标签（首要来源：`spawn_agent` 的 profile 名），进回流消息让模型/用户辨识来源。
    pub label: String,
    /// 任务结果。
    pub result: BackgroundResult,
}

/// 后台任务的最终结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundResult {
    /// 正常完成，携带任务的最终文本输出。
    Completed(String),
    /// 失败（含被取消），携带错误描述。
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

/// 后台任务的生命周期状态。供 `inspect_background_task` 控制面展示。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// 仍在运行。
    Running,
    /// 正常跑完。
    Completed,
    /// 失败结束。
    Failed,
    /// 被 [`cancel_task`](BackgroundTasks::cancel_task) / [`cancel_all`](BackgroundTasks::cancel_all)
    /// 取消。
    Canceled,
}

impl TaskStatus {
    /// 稳定的小写字符串名，进控制面工具输出。
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

/// 一条进度 block 的角色/类别。直接对位提交给 LLM 的 [`crate::llm::Message`] 内容。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// 用户/任务输入消息（含回流进来的后台结果、工具结果回灌等）。
    User,
    /// 助手产出的文本。
    AssistantText,
    /// 助手的思考链。
    Thought,
    /// 助手发起的一次工具调用。
    ToolUse,
    /// 工具结果（喂回给模型的）。
    ToolResult,
    /// 其它（多模态 / provider 活动等），归一展示。
    Other,
}

impl BlockKind {
    /// 稳定的小写字符串名，进控制面工具输出。
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

    /// 该类 block 的文本是否为**自由正文**——受 [`BackgroundProgressConfig::block_text_limit`]
    /// 约束。工具调用名这类本就是一行摘要，不算正文、不受上限约束。
    fn is_free_form_body(&self) -> bool {
        matches!(
            self,
            BlockKind::User | BlockKind::AssistantText | BlockKind::Thought | BlockKind::ToolResult
        )
    }
}

/// peek 返回的单条进度 block：类别 + 文本摘要（已按配置截断）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressBlock {
    pub kind: BlockKind,
    pub text: String,
}

/// 一个任务在控制面里的快照（`list` / `peek` 返回）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub task_id: String,
    pub label: String,
    pub status: TaskStatus,
    /// 该任务 history 里现有的消息块总数（`peek` 才填；`list` 为 `0`，因为列举不读 history）。
    pub block_count: usize,
    /// 最近的若干 block（`list` 不带、为空；`peek` 带最近 N 个）。
    pub recent: Vec<ProgressBlock>,
}

/// 把自由正文按字符上限截断（按 Unicode 标量切，不会切坏字符）。`limit == 0` ⇒ 空串
/// （只留元信息）。截断时附 ` …(+N more chars)` 标记，让查看者知道有省略。
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

/// 把 [`ToolResultBody`](crate::llm::ToolResultBody) 提一段可读文本出来（仅用于鸟瞰摘要）。
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

/// 把一条 [`MessageContent`] 映射成一条进度 block，正文按 `limit` 截断（仅自由正文）。
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
        // 工具名是一行摘要，不当正文截断；参数不进鸟瞰（要看细节去 langfuse trace）。
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

/// 从一份 history snapshot 里取**最近 `n` 条**消息块（把每条 [`Message`] 的各 content 片段
/// 摊平成独立 block，保持时间顺序），正文按 `limit` 截断。返回 `(总块数, 最近 n 块)`。
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

/// `tasks` 表里的一条任务条目。
struct TaskEntry {
    label: String,
    status: TaskStatus,
    /// 本任务专属取消令牌（session 级 token 的子 token）。`cancel_task` 取它单独 `cancel`。
    cancel: CancellationToken,
    /// 指向该任务 history 的共享句柄。`peek` 经它 snapshot 出**提交给 LLM 的消息块**。
    /// `Some`：发起任务的工具（`spawn_agent`）把子 turn 的 history `Arc` 共享了进来；
    /// `None`：任务没暴露 history（无进度可查，peek 只回状态）。
    history: Option<Arc<dyn History>>,
    /// 运行中的 `JoinHandle`，使任务活过发起它的 turn。结束后置 `None`。
    handle: Option<JoinHandle<()>>,
    /// 结束顺序序号（仅已结束条目有），供 FIFO 淘汰。
    finished_seq: Option<u64>,
}

struct BackgroundInner {
    /// 单调递增的任务 id 计数器。
    next_id: u64,
    /// 单调递增的"结束顺序"计数器，供已结束条目 FIFO 淘汰。
    next_finished_seq: u64,
    /// 全部任务（运行中 + 近期已结束）。已结束条目超过 [`FINISHED_TASKS_CAP`] 时淘汰最旧。
    tasks: BTreeMap<String, TaskEntry>,
    /// 已完成、待回流的结果（FIFO）。`drain_completed` 取空。与 `tasks` 表正交：
    /// 前者驱动被动回流，后者支撑控制面查询/中断。
    completed: Vec<BackgroundOutcome>,
}

impl BackgroundInner {
    /// 把一条任务标记为已结束、记录结束序号，并按上限淘汰最旧的已结束条目。
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

    /// 已结束条目超过上限时，按结束序号淘汰最旧的几条。运行中条目永不淘汰。
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

/// Session 级后台任务表。`Clone` 廉价（内部 `Arc`）——`DefaultSession` 持有一份，
/// 经 `ToolContext` clone 给工具。
#[derive(Clone)]
pub struct BackgroundTasks {
    /// session 级取消令牌。每个任务从它 `child_token()`，故 `cancel_all` 一次性掐掉全部，
    /// 且任意单个任务的取消不影响其他任务。
    cancel: CancellationToken,
    /// 任务完成通知。每当一个任务结果入队就 `notify_one`——session driver 等在它上面，
    /// 被唤醒后起一个自主 turn 主动续转（阶段二）。被动回流不依赖它。
    completed_notify: Arc<Notify>,
    /// 进度视图配置（默认返回块数 / 正文上限）。`peek` 据此渲染。
    progress_config: BackgroundProgressConfig,
    inner: Arc<Mutex<BackgroundInner>>,
}

impl BackgroundTasks {
    /// 用一个 session 级取消令牌 + 进度视图配置构造。`session_cancel` 由 session 持有、
    /// 随 session 终结而取消。
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

    /// 等到"有任务完成入队"事件。session driver 用它驱动主动续转。
    ///
    /// 用 `Notify`：driver 先 `notified()` 拿到 future、再检查队列、然后 await——避免漏掉
    /// 在两次检查之间到达的通知（`Notify` 的 permit 语义保证已发生的 notify 不丢）。
    pub async fn wait_for_completion(&self) {
        self.completed_notify.notified().await;
    }

    /// 当前是否有已完成、待回流的结果。driver 唤醒后先查它再决定起不起 turn。
    #[must_use]
    pub fn has_completed(&self) -> bool {
        !self
            .inner
            .lock()
            .expect("BackgroundTasks mutex poisoned")
            .completed
            .is_empty()
    }

    /// Spawn 一个后台任务，**立即**返回它的 id。
    ///
    /// `make_fut` 收到两个句柄：本任务专属的 [`CancellationToken`]（session 级 token 的
    /// 子 token，任务体应在其上感知取消）与一个 [`TaskHandle`]（任务体把自己的 history
    /// `Arc` 经 [`TaskHandle::attach_history`] 共享进表，供控制面 peek **提交给 LLM 的
    /// 消息块**）。任务完成时结果进 `completed` 队列、并把 `tasks` 表里本条标记为终态
    /// （保留条目以便事后查询）。
    ///
    /// 取这个"收 token / handle 再造 future"的闭包形态，是因为两者都要在 spawn 内部 mint，
    /// 而 future 需要捕获它们——直接收 future 就拿不到这个生命周期独立于 turn 的 token。
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
        // 任务体感知"是否被取消"，用于完成时区分 Failed / Canceled 状态。
        let cancel_for_task = task_cancel.clone();
        let fut = make_fut(task_cancel.clone(), handle);

        let inner_arc = self.inner.clone();
        let notify = self.completed_notify.clone();
        let id_for_task = id.clone();
        let label_for_task = label.clone();
        let join = tokio::spawn(async move {
            let result = fut.await;
            // 区分"任务报错"与"被显式取消"：后者把状态记成 Canceled，前者记 Failed。
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
            // 唤醒等在 wait_for_completion 上的 session driver（主动续转）。
            // 用 notify_one 而非 notify_waiters：前者在无等待者时**保留一个 permit**，
            // 下次 notified().await 立即返回——避免"任务在 driver park 之前就完成"导致的
            // 丢唤醒。单消费者（恰好一个 driver），notify_one 语义正合适。在锁外 notify。
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

    /// 取出全部已完成结果（清空队列）。`run_turn` 在起 turn 前调用做被动回流。
    pub fn drain_completed(&self) -> Vec<BackgroundOutcome> {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        std::mem::take(&mut inner.completed)
    }

    /// 当前运行中的任务数。供诊断 / 控制面用。
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

    /// 列出所有任务（运行中 + 近期已结束）的快照，**不读 history**（`recent` 空、
    /// `block_count` 为 0）。按 task id 升序。供 `inspect_background_task` 无参列举。
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

    /// 取单个任务的快照，带最近 `recent_blocks` 条**提交给 LLM 的消息块**（`None` ⇒ 用配置
    /// 默认）。任务不存在（从未 spawn / 已被淘汰）返回 `None`；任务未暴露 history 则块为空。
    ///
    /// 实现：clone 出该任务的 history `Arc`（在锁内）后**释放表锁**，再 snapshot（snapshot
    /// 走 history 自己的锁）——避免在持表锁期间做可能较重的历史深拷贝、阻塞 spawn/finish。
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

    /// 提前中断单个任务：取它专属的子 token 单独 `cancel`，不波及其它任务。
    ///
    /// 返回 `Some(true)` 表示找到了一个运行中的任务并已请求取消；`Some(false)` 表示任务
    /// 存在但已是终态（无操作）；`None` 表示无此 id。取消是**协作式**的——任务体须在它的
    /// cancel token 上感知并退出，状态在任务实际结束时才翻成 `Canceled`。
    pub fn cancel_task(&self, id: &str) -> Option<bool> {
        let inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let entry = inner.tasks.get(id)?;
        if entry.status.is_terminal() {
            return Some(false);
        }
        entry.cancel.cancel();
        Some(true)
    }

    /// 取消所有后台任务（session 终结时调用）。幂等。
    pub fn cancel_all(&self) {
        self.cancel.cancel();
    }
}

/// 交给后台任务体的句柄：让任务把自己的 history `Arc` 共享进任务表，从而控制面能 peek
/// 它**提交给 LLM 的消息块**。`Clone` 廉价（内部 `Arc` + 小字符串）。
#[derive(Clone)]
pub struct TaskHandle {
    inner: Arc<Mutex<BackgroundInner>>,
    task_id: String,
}

impl TaskHandle {
    /// 把本任务的 history 句柄共享进任务表。`spawn_agent` 后台路径在构造子 turn 前调用，
    /// 传子 turn 的 history `Arc`——之后 `peek` 就能 snapshot 出子 agent 已提交的消息块。
    /// 任务条目可能已被淘汰（极端情况下任务瞬时结束并被 FIFO 挤掉），那时静默忽略。
    pub fn attach_history(&self, history: Arc<dyn History>) {
        if let Ok(mut inner) = self.inner.lock()
            && let Some(entry) = inner.tasks.get_mut(&self.task_id)
        {
            entry.history = Some(history);
        }
    }
}

/// 把一个后台任务结果格式化成回流到对话里的文本块内容。
///
/// 措辞按"延迟工具结果回流"组织，明确标注来源（task id + label）与成败，避免模型误判为用户发言。
/// 阶段二会换成 `IngestSource::Background` 的正派 ingest 路径（§5.1），届时此函数被相应载荷取代。
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
