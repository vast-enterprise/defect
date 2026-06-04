//! Session 级后台任务表。
//!
//! 设计详见 `docs/proposals/task-arrange.md` §3.1。
//!
//! ## 解决什么
//!
//! 工具（首要场景 `spawn_agent { run_in_background: true }`）想 fire-and-forget 地跑一个
//! 任务，让发起它的 turn **不阻塞**。但 turn 主循环的 `run_tools_concurrently` 用一个**函数
//! 局部** `JoinSet` 持有工具 task——函数返回时 `JoinSet` drop、task 被 abort。所以没有任何
//! 任务能活过发起它的 turn。
//!
//! [`BackgroundTasks`] 把任务的 `JoinHandle` 挪到 **session 级**持有（与 `events` / `history`
//! 同档生命周期），任务因此活过发起它的 turn；并用一个 **session 级 [`CancellationToken`]**
//! （不是 turn 的子 token）给每个任务 mint 子 token，使后台任务的取消生命周期独立于发起它
//! 的 turn。
//!
//! ## 回流（阶段一：被动）
//!
//! 任务完成后把 [`BackgroundOutcome`] push 进 `completed` 队列。`DefaultSession::run_turn`
//! 在每次起 turn 之前 [`drain_completed`](BackgroundTasks::drain_completed)，把待回流结果作为
//! **本轮 user prompt 的前缀块**带入 history——结果搭着用户下一次输入一起被 LLM 看到。
//! 阶段二（主动续转）改由 session input loop 在后台完成时立即竞争一个新 turn，见
//! `docs/proposals/task-arrange.md` §3.2。
//!
//! ## 内省与单点中断（控制面）
//!
//! 任务**完成后不立刻从表里消失**：每个任务在 `tasks` 表里保留一条 [`TaskEntry`]，记录
//! 状态（运行 / 完成 / 失败 / 取消）与一个**有界进度环** [`ProgressRing`]。发起任务的工具
//! （`spawn_agent`）把子 turn 的"最近几个 block"（assistant 文本 / 思考 / 工具调用起止）经
//! [`ProgressSink`] 喂进这个环。于是主 agent 能用 `inspect_background_task` 查某个后台
//! subagent 的进度，用 `cancel_background_task` 经 [`cancel_task`](BackgroundTasks::cancel_task)
//! 提前掐掉单个任务——不波及其它任务（每个任务一个独立子 token）。完成的任务条目按 FIFO
//! 上限淘汰，避免长会话无界增长。

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// 进度环默认容量（block 数）。够覆盖"最近几个 block"的查询，又不让单个长跑任务
/// 把内存吃光。可经 [`BackgroundProgressConfig::ring_cap`] 覆盖。
const DEFAULT_PROGRESS_RING_CAP: usize = 64;

/// `tasks` 表里保留多少个**已结束**的任务条目。运行中的条目不计入上限——它们必须留着
/// 才能被 cancel / peek。超过上限时按结束顺序淘汰最旧的那条。
const FINISHED_TASKS_CAP: usize = 64;

/// 后台任务**进度视图**的配置。
///
/// 目的：给主 agent 一个"这个 subagent 此刻大致在干嘛"的**鸟瞰**，而**不是**把子 turn
/// 的完整正文灌回主 agent 上下文。所以默认偏保守——只留元信息（工具调用标题这类本就是
/// 摘要的东西），assistant/thought 正文默认**不留**（`block_text_limit = 0`）。需要更细
/// 时用户再放大 `block_text_limit`。
///
/// 配置真相源在 agent 侧（这里），`defect-config` 的 `ToolsConfig.background` 直接复用本
/// 结构（与 `TurnConfig` / `SessionCapabilitiesConfig` 同款跨 crate 复用——config 依赖
/// agent，不能反向）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundProgressConfig {
    /// 每个任务的进度环最多保留多少个 block。溢出淘汰最旧。`0` 视作 `1`（至少留一格，
    /// 否则环恒空、peek 永远看不到东西）。
    pub ring_cap: usize,
    /// 单个 block 的**正文**字符上限（按 Unicode 标量计），仅作用于 assistant 文本 /
    /// 思考这类自由正文。超出即在边界截断并加省略标记。`0` = 不留正文（只记"发生了
    /// 一段 assistant 文本 / 思考"这一元信息）——默认值，最不污染。工具调用的标题
    /// **不受**本上限约束：它本身就是一行摘要。
    pub block_text_limit: usize,
}

impl Default for BackgroundProgressConfig {
    fn default() -> Self {
        Self {
            ring_cap: DEFAULT_PROGRESS_RING_CAP,
            // 默认只给摘要 / 元信息，不灌正文——目的是鸟瞰，不是搬运上下文。
            block_text_limit: 0,
        }
    }
}

impl BackgroundProgressConfig {
    /// 规整成可用值：`ring_cap` 至少 1。
    fn normalized_ring_cap(&self) -> usize {
        self.ring_cap.max(1)
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

/// 后台任务进度环里的一个 block 的类别。直接对位 `spawn_agent` 桥接的子 turn 事件。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressKind {
    /// 子 agent 的 assistant 文本增量。
    AssistantText,
    /// 子 agent 的思考链增量。
    Thought,
    /// 子 agent 发起了一次工具调用。
    ToolStart,
    /// 子 agent 的一次工具调用结束。
    ToolFinish,
}

impl ProgressKind {
    /// 稳定的小写字符串名，进控制面工具输出。
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            ProgressKind::AssistantText => "assistant",
            ProgressKind::Thought => "thought",
            ProgressKind::ToolStart => "tool_start",
            ProgressKind::ToolFinish => "tool_finish",
        }
    }

    /// 这类 block 的文本是否为**自由正文**（assistant 文本 / 思考）——受
    /// [`BackgroundProgressConfig::block_text_limit`] 约束。工具调用的标题
    /// 本身就是一行摘要，不算正文、不受上限约束。
    fn is_free_form_body(&self) -> bool {
        matches!(self, ProgressKind::AssistantText | ProgressKind::Thought)
    }
}

/// 进度环里的一个 block：类别 + 文本摘要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressBlock {
    pub kind: ProgressKind,
    pub text: String,
}

/// 有界进度环。每个任务一个，由 [`ProgressSink`] 写、由 `peek` 读。最多保留 `cap` 个
/// block，溢出时淘汰最旧的。
struct ProgressRing {
    blocks: VecDeque<ProgressBlock>,
    cap: usize,
}

impl ProgressRing {
    fn with_cap(cap: usize) -> Self {
        Self {
            blocks: VecDeque::new(),
            cap,
        }
    }

    fn push(&mut self, block: ProgressBlock) {
        while self.blocks.len() >= self.cap {
            self.blocks.pop_front();
        }
        self.blocks.push_back(block);
    }

    /// 取最近 `n` 个 block（保持时间顺序：旧→新）。
    fn recent(&self, n: usize) -> Vec<ProgressBlock> {
        let skip = self.blocks.len().saturating_sub(n);
        self.blocks.iter().skip(skip).cloned().collect()
    }
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

/// 写进度环的句柄。`Clone` 廉价（内部 `Arc`）。交给发起任务的工具（`spawn_agent`），
/// 让它把子 turn 的"最近几个 block"流式喂进对应任务的进度环。任务结束后写入静默无害
/// （环还在表里、可被最后 peek 到）。
///
/// 截断策略在**写入这一处**统一执行（不在读取处）——无论谁产事件，进环的正文就已经
/// 按 [`BackgroundProgressConfig::block_text_limit`] 收敛，省内存也省读取期再处理。
#[derive(Clone)]
pub struct ProgressSink {
    ring: Arc<Mutex<ProgressRing>>,
    /// 自由正文（assistant / thought）的字符上限快照。工具调用标题不受其约束。
    block_text_limit: usize,
}

impl ProgressSink {
    /// 追加一个进度 block。自由正文按 `block_text_limit` 截断；工具调用标题原样保留
    /// （本就是摘要）。锁中毒时静默丢弃（进度环纯诊断，不该让任务崩）。
    pub fn push(&self, kind: ProgressKind, text: String) {
        let text = if kind.is_free_form_body() {
            truncate_body(&text, self.block_text_limit)
        } else {
            text
        };
        if let Ok(mut ring) = self.ring.lock() {
            ring.push(ProgressBlock { kind, text });
        }
    }
}

/// 一个任务在控制面里的快照（`list` / `peek` 返回）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub task_id: String,
    pub label: String,
    pub status: TaskStatus,
    /// 进度环里现有的 block 总数。
    pub block_count: usize,
    /// 最近的若干 block（`list` 不带、为空；`peek` 带最近 N 个）。
    pub recent: Vec<ProgressBlock>,
}

/// `tasks` 表里的一条任务条目。
struct TaskEntry {
    label: String,
    status: TaskStatus,
    /// 本任务专属取消令牌（session 级 token 的子 token）。`cancel_task` 取它单独 `cancel`。
    cancel: CancellationToken,
    /// 进度环的共享句柄——与交给任务的 [`ProgressSink`] 同一个 `Arc`。
    ring: Arc<Mutex<ProgressRing>>,
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
    /// 进度视图配置（环容量 / 正文上限）。`spawn` 据此给每个任务建环与 sink。
    progress_config: BackgroundProgressConfig,
    inner: Arc<Mutex<BackgroundInner>>,
}

impl BackgroundTasks {
    /// 用一个 session 级取消令牌 + 进度视图配置构造。`session_cancel` 由 session 持有、
    /// 随 session 终结而取消。
    #[must_use]
    pub fn new(session_cancel: CancellationToken, progress_config: BackgroundProgressConfig) -> Self {
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
    /// 子 token，任务体应在其上感知取消）与一个 [`ProgressSink`]（往本任务进度环里喂"最近
    /// 几个 block"，供控制面 peek）。任务完成时结果进 `completed` 队列、并把 `tasks` 表里
    /// 本条标记为终态（保留条目以便事后查询）。
    ///
    /// 取这个"收 token / sink 再造 future"的闭包形态，是因为两者都要在 spawn 内部 mint，
    /// 而 future 需要捕获它们——直接收 future 就拿不到这个生命周期独立于 turn 的 token。
    pub fn spawn<F, Fut>(&self, label: String, make_fut: F) -> String
    where
        F: FnOnce(CancellationToken, ProgressSink) -> Fut,
        Fut: Future<Output = BackgroundResult> + Send + 'static,
    {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let id = format!("bg-{}", inner.next_id);
        inner.next_id += 1;

        let task_cancel = self.cancel.child_token();
        let ring = Arc::new(Mutex::new(ProgressRing::with_cap(
            self.progress_config.normalized_ring_cap(),
        )));
        let sink = ProgressSink {
            ring: ring.clone(),
            block_text_limit: self.progress_config.block_text_limit,
        };
        // 任务体感知"是否被取消"，用于完成时区分 Failed / Canceled 状态。
        let cancel_for_task = task_cancel.clone();
        let fut = make_fut(task_cancel.clone(), sink);

        let inner_arc = self.inner.clone();
        let notify = self.completed_notify.clone();
        let id_for_task = id.clone();
        let label_for_task = label.clone();
        let handle = tokio::spawn(async move {
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
                ring,
                handle: Some(handle),
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

    /// 列出所有任务（运行中 + 近期已结束）的快照，**不带**进度 block（`recent` 为空、
    /// 只给 `block_count`）。按 task id 升序。供 `inspect_background_task` 无参列举。
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
                block_count: e.ring.lock().map(|r| r.blocks.len()).unwrap_or(0),
                recent: Vec::new(),
            })
            .collect()
    }

    /// 取单个任务的快照，带最近 `n` 个进度 block。任务不存在（从未 spawn / 已被淘汰）
    /// 返回 `None`。供 `inspect_background_task` 带 task_id 查进度。
    #[must_use]
    pub fn peek(&self, id: &str, n: usize) -> Option<TaskSnapshot> {
        let inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let entry = inner.tasks.get(id)?;
        let (block_count, recent) = entry
            .ring
            .lock()
            .map(|r| (r.blocks.len(), r.recent(n)))
            .unwrap_or((0, Vec::new()));
        Some(TaskSnapshot {
            task_id: id.to_string(),
            label: entry.label.clone(),
            status: entry.status,
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
#[path = "background/test.rs"]
mod test;
