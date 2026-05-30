//! Hook 系统：主循环的扩展点。
//!
//! 设计详见 `docs/internal/hooks.md`。
//!
//! ## 抽象层次
//!
//! - [`HookEvent`]：主循环 emit 的载荷（5 件 Sync 拦截 + 8 件 Async 观察 enum 占位）
//! - [`HookHandler`]：单个执行器（Builtin / Command / Prompt 三种 v0 形态在子 crate 实现）
//! - [`HookMatcher`]：单条 hook 的匹配条件（按 tool / glob / safety 过滤）
//! - [`HookEngine`]：主循环面向的派发器；持有 handler 表、执行 pipeline、合并 outcome
//!
//! v0 主循环只 emit 5 件 Sync 拦截事件；8 件 Async 观察事件仅 enum 占位。
//!
//! ## Async 观察事件的现状（未落地）
//!
//! 核心 gap 是 [`HookEngine::observe`] 的 **AgentEvent → HookEvent 投影器缺失**：
//! 没有任何 task 订阅 [`crate::event::AgentEvent`] 流、把它投影成 [`HookEvent`]
//! 再调 `observe()`。`DefaultHookEngine::observe()` 当前是空函数。
//!
//! 数据源大多已就绪——8 件里 6 件能直接从现有 `AgentEvent` 投影
//! （`TurnStart`←`TurnStarted`、`TurnEnd`←`TurnEnded`、`PreLlmCall`←`LlmCallStarted`、
//! `PostLlmCall`←`LlmCallFinished`、`PostCompact`←`ContextCompressed`、
//! `PermissionAsk`←`PolicyDecision{Ask}`）。两个例外要先补数据源：
//! - `PreCompact`：`AgentEvent` 只有压缩**完成后**的 `ContextCompressed`，没有压缩前事件 → 需在 turn loop 新增 emit 点
//! - `SessionEnd`：`AgentEvent` 无任何 session 终结事件 → 需先在 session 生命周期造终结信号
//!
//! 完整落地步骤见 `docs/internal/hooks.md` §11。
//!
//! ## 默认实现
//!
//! [`NoopHookEngine`]：所有 fire 直接返回 `Pass`，observe 直接丢弃；session/turn 装配
//! 时若没有显式 hook 引擎走这个，保持"hook 未配置 = 主循环行为不变"。
//!
//! [`DefaultHookEngine`]：用 [`arc_swap::ArcSwap`] 持有 handler 表，按 §3.4 的
//! pipeline 语义串行调度；matcher / 超时 / panic 捕获按 hooks.md §3.5 的降级表
//! 处理。

use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{
    ContentBlock, RequestPermissionRequest, SessionId, StopReason as AcpStopReason, ToolCallId,
    ToolCallUpdateFields,
};
use arc_swap::ArcSwap;
use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;
use crate::llm::Usage;
use crate::tool::SafetyClass;

pub mod builtin;
pub mod command;
pub mod prompt;

/// `DefaultHookEngine` 的默认 per-handler 超时（hooks.md §8）。
const DEFAULT_HANDLER_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// HookEvent
// ---------------------------------------------------------------------------

/// 主循环 emit 给 hook 引擎的事件。
///
/// **借用形式**：构造时不付 clone 代价；engine 内部按需 clone 到 owned 形态
/// 喂 Command / Prompt handler。具体 owned 形态见 §4 / `docs/internal/hooks.md`。
///
/// 类别（详见 `docs/internal/hooks.md` §1.1）：
/// - **Sync 拦截**（v0 实际 emit）：`SessionStart` / `UserPromptSubmit` /
///   `PreToolUse` / `PostToolUse` / `PostToolUseFailure`
/// - **Async 观察**（v0 仅 enum 占位，未落地——投影器缺失，见模块级文档）：
///   `SessionEnd` / `TurnStart` / `TurnEnd` / `PreLlmCall` / `PostLlmCall` /
///   `PreCompact` / `PostCompact` / `PermissionAsk`
#[non_exhaustive]
#[derive(Debug)]
pub enum HookEvent<'a> {
    // ── Sync 拦截 ──
    SessionStart {
        source: SessionSource<'a>,
        cwd: &'a Path,
    },
    UserPromptSubmit {
        content: &'a [ContentBlock],
    },
    PreToolUse {
        id: &'a ToolCallId,
        name: &'a str,
        args: &'a Value,
        safety: SafetyClass,
    },
    PostToolUse {
        id: &'a ToolCallId,
        name: &'a str,
        fields: &'a ToolCallUpdateFields,
    },
    PostToolUseFailure {
        id: &'a ToolCallId,
        name: &'a str,
        error: &'a str,
    },

    // ── Async 观察（v0 仅占位、未落地：投影器缺失，见模块级文档） ──
    SessionEnd {
        reason: AcpStopReason,
    },
    TurnStart {
        prompt: &'a [ContentBlock],
    },
    TurnEnd {
        reason: AcpStopReason,
        usage: &'a Usage,
    },
    PreLlmCall {
        model: &'a str,
        attempt: u32,
    },
    PostLlmCall {
        model: &'a str,
        attempt: u32,
        usage: &'a Usage,
        error: Option<&'a str>,
    },
    PreCompact {
        tokens_before: u64,
    },
    PostCompact {
        tokens_before: u64,
        tokens_after: u64,
    },
    PermissionAsk {
        id: &'a ToolCallId,
        request: &'a RequestPermissionRequest,
    },
}

/// `SessionStart` 的"是新建还是 resume"提示。
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SessionSource<'a> {
    /// 全新创建的 session。
    New,
    /// resume 既有 session。
    Resume { session_id: &'a SessionId },
}

/// 事件类别枚举——用于 handler 表分桶 / matcher 校验。
///
/// 1:1 对应 [`HookEvent`] 的变体；引擎内部按 kind 索引 handler 列表。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEventKind {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    SessionEnd,
    TurnStart,
    TurnEnd,
    PreLlmCall,
    PostLlmCall,
    PreCompact,
    PostCompact,
    PermissionAsk,
}

impl HookEvent<'_> {
    /// 该事件所属的类别。配置加载与引擎派发都按 kind 分桶。
    pub fn kind(&self) -> HookEventKind {
        match self {
            Self::SessionStart { .. } => HookEventKind::SessionStart,
            Self::UserPromptSubmit { .. } => HookEventKind::UserPromptSubmit,
            Self::PreToolUse { .. } => HookEventKind::PreToolUse,
            Self::PostToolUse { .. } => HookEventKind::PostToolUse,
            Self::PostToolUseFailure { .. } => HookEventKind::PostToolUseFailure,
            Self::SessionEnd { .. } => HookEventKind::SessionEnd,
            Self::TurnStart { .. } => HookEventKind::TurnStart,
            Self::TurnEnd { .. } => HookEventKind::TurnEnd,
            Self::PreLlmCall { .. } => HookEventKind::PreLlmCall,
            Self::PostLlmCall { .. } => HookEventKind::PostLlmCall,
            Self::PreCompact { .. } => HookEventKind::PreCompact,
            Self::PostCompact { .. } => HookEventKind::PostCompact,
            Self::PermissionAsk { .. } => HookEventKind::PermissionAsk,
        }
    }

    /// 事件名 snake_case——env 注入 / stdin envelope / 模板渲染共用。
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::SessionStart { .. } => "session_start",
            Self::UserPromptSubmit { .. } => "user_prompt_submit",
            Self::PreToolUse { .. } => "pre_tool_use",
            Self::PostToolUse { .. } => "post_tool_use",
            Self::PostToolUseFailure { .. } => "post_tool_use_failure",
            Self::SessionEnd { .. } => "session_end",
            Self::TurnStart { .. } => "turn_start",
            Self::TurnEnd { .. } => "turn_end",
            Self::PreLlmCall { .. } => "pre_llm_call",
            Self::PostLlmCall { .. } => "post_llm_call",
            Self::PreCompact { .. } => "pre_compact",
            Self::PostCompact { .. } => "post_compact",
            Self::PermissionAsk { .. } => "permission_ask",
        }
    }

    /// 判断该事件是否属于 Sync 拦截类别。
    pub fn is_sync(&self) -> bool {
        matches!(
            self.kind(),
            HookEventKind::SessionStart
                | HookEventKind::UserPromptSubmit
                | HookEventKind::PreToolUse
                | HookEventKind::PostToolUse
                | HookEventKind::PostToolUseFailure
        )
    }
}

// ---------------------------------------------------------------------------
// HookHandler 与 outcome
// ---------------------------------------------------------------------------

/// Handler 自我宣告的语义类别。引擎按此校验"能否挂在该事件上"。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookCapability {
    /// 仅适用于 Async 观察事件——只能日志/审计；返回非 `Pass` outcome
    /// 由引擎丢弃 + warn。**当前无代码路径用到**：observe 投影 task 未落地
    /// （见模块级文档），暂无 handler 真正以此 capability 被调度。
    Observe,
    /// 完整能力——可 block / patch / append；仅适用于 Sync 拦截事件。
    Intercept,
}

/// 单条 hook 的匹配条件。
///
/// 形态与 `defect-config` 的 `HookMatcher` 一致；agent crate 不依赖 config，
/// 这里独立定义、CLI 装配时把 config 形态翻成 agent 形态（详见
/// `docs/internal/hooks.md` §5.3）。
///
/// 字段全空 = 匹配该事件下所有触发。
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct HookMatcher {
    /// 按工具名精确匹配（仅 `*ToolUse*` 事件）。
    pub tool: Option<String>,
    /// 按工具名 glob 匹配（仅 `*ToolUse*` 事件）。
    pub tool_glob: Option<String>,
    /// 按 [`SafetyClass`] 过滤（仅 `PreToolUse`）；任一匹配即命中。空 vec = 不过滤。
    pub safety: Vec<SafetyClass>,
}

impl HookMatcher {
    /// 该 matcher 是否命中给定事件。
    ///
    /// 匹配语义：
    /// - 非工具事件（`SessionStart` / `UserPromptSubmit` / Async 类）matcher
    ///   字段全部应当为空；填了字段也直接命中（hooks.md §3.3 的"不向上报错"原则）
    /// - `tool` 与事件 `name` 精确匹配（区分大小写）
    /// - `tool_glob` 用 [`glob_match`] 简单匹配（仅 `*` / `?` 通配符；hooks.md §5.3
    ///   明确不上 regex）
    /// - `safety` 仅对 `PreToolUse` 生效，命中条件为 `safety.contains(event.safety)`
    pub fn matches(&self, event: &HookEvent<'_>) -> bool {
        let (name, safety) = match event {
            HookEvent::PreToolUse {
                name, safety: s, ..
            } => (Some(*name), Some(*s)),
            HookEvent::PostToolUse { name, .. } | HookEvent::PostToolUseFailure { name, .. } => {
                (Some(*name), None)
            }
            _ => (None, None),
        };

        if let Some(expected) = &self.tool
            && name.is_none_or(|n| n != expected)
        {
            return false;
        }
        if let Some(pat) = &self.tool_glob
            && name.is_none_or(|n| !glob_match(pat, n))
        {
            return false;
        }
        // safety 过滤仅对 PreToolUse 生效；其他工具事件上 safety 空 = 不过滤，
        // 非空 = 该 matcher 写错了用法但不阻塞——按 §3.3 静默跳过该条件。
        if !self.safety.is_empty()
            && let Some(s) = safety
            && !self.safety.contains(&s)
        {
            return false;
        }
        true
    }
}

/// 极简 glob 匹配：仅支持 `*`（任意 0+ 字符）与 `?`（单字符）。
///
/// 不引入 `globset` / `glob` crate 是为了：
/// 1. 工具名 glob 的复杂度天花板就这两通配符，引重型依赖收益不大
/// 2. workspace 当前 0 依赖于 glob，新加要走 deps review；hook matcher 落地不阻塞
fn glob_match(pattern: &str, text: &str) -> bool {
    // 所有 `[1..]` / `[i..]` 索引在 `first()` / `0..=len()` 的前置条件下
    // 一定不越界；为简洁起见这里 allow 掉 crate 级 indexing_slicing 警告。
    #[allow(clippy::indexing_slicing)]
    fn helper(p: &[u8], t: &[u8]) -> bool {
        if p.is_empty() {
            return t.is_empty();
        }
        match p.first().copied() {
            Some(b'*') => {
                let rest = &p[1..];
                if rest.is_empty() {
                    return true;
                }
                for i in 0..=t.len() {
                    if helper(rest, &t[i..]) {
                        return true;
                    }
                }
                false
            }
            Some(b'?') => !t.is_empty() && helper(&p[1..], &t[1..]),
            Some(c) => {
                if t.first().copied() == Some(c) {
                    helper(&p[1..], &t[1..])
                } else {
                    false
                }
            }
            None => t.is_empty(),
        }
    }
    helper(pattern.as_bytes(), text.as_bytes())
}

/// 共享给 handler 的轻量上下文。
#[non_exhaustive]
pub struct HookCtx<'a> {
    pub session_id: &'a SessionId,
    pub cwd: &'a Path,
    pub cancel: CancellationToken,
}

impl<'a> HookCtx<'a> {
    pub fn new(session_id: &'a SessionId, cwd: &'a Path, cancel: CancellationToken) -> Self {
        Self {
            session_id,
            cwd,
            cancel,
        }
    }
}

/// Handler 返回给引擎的结果。
///
/// 三个字段独立可组合（详见 `docs/internal/hooks.md` §3.1）：一个 handler
/// 可一次返回"修改 args + 追加 system context"。`Pass` = 全字段默认值。
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct HookOutcome {
    /// 早退理由。`Some` 时引擎不再调用后续 handler；其他字段忽略。
    pub block: Option<String>,
    /// 修改 in-flight 事件数据。具体补丁形态由事件决定。
    pub patch: Option<HookPatch>,
    /// 追加内容到 system prompt / tool_result.content（具体落点见 §3.2）。
    pub append: Vec<ContentBlock>,
}

/// 事件可被 patch 的形态。具体允许哪种 patch 见
/// `docs/internal/hooks.md` §3.2。
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum HookPatch {
    /// `PreToolUse`：替换工具参数。
    ToolArgs(Value),
    /// `UserPromptSubmit`：在用户原文前后追加。**不允许完全替换**（见 §3.6）。
    UserPrompt {
        prepend: Vec<ContentBlock>,
        append: Vec<ContentBlock>,
    },
}

/// Handler 失败原因。
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("hook handler timed out")]
    Timeout,

    #[error("hook handler failed: {0}")]
    HandlerFailed(#[source] BoxError),

    /// handler 信任未通过 / 未注册等配置层错误。
    #[error("hook configuration error: {0}")]
    Configuration(String),
}

/// 处理 [`HookEvent`] 的执行器。
///
/// `BoxFuture` 是 workspace [No async_trait] 约定。
///
/// [No async_trait]: ../../CLAUDE.MD
pub trait HookHandler: Send + Sync {
    /// handler 的语义类别。配置加载时按事件类别 + capability 校验。
    fn capability(&self) -> HookCapability;

    /// 执行 hook。返回 outcome 给引擎；具体效果见
    /// `docs/internal/hooks.md` §3.2。
    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>>;
}

// ---------------------------------------------------------------------------
// HookEngine
// ---------------------------------------------------------------------------

/// 主循环面向的派发器。
///
/// - [`Self::fire`]：Sync 拦截事件入口；阻塞主循环直到所有匹配 handler 跑完，
///   返回合并 outcome
/// - [`Self::observe`]：Async 观察入口；本应由订阅 [`crate::event::AgentEvent`]
///   流的 fan-out task 投影后调用，不阻塞主循环。**该投影 task 尚未落地**
///   （见模块级文档与 `docs/internal/hooks.md` §11），故当前 `observe` 无调用方。
///
/// 默认实现 [`DefaultHookEngine`]；测试 / 默认 session 装配走 [`NoopHookEngine`]。
pub trait HookEngine: Send + Sync {
    fn fire<'a>(&'a self, event: HookEvent<'a>, ctx: HookCtx<'a>) -> BoxFuture<'a, HookOutcome>;

    fn observe<'a>(&'a self, event: HookEvent<'a>, ctx: HookCtx<'a>);
}

// ---------------------------------------------------------------------------
// NoopHookEngine
// ---------------------------------------------------------------------------

/// 默认 hook 引擎：所有 fire 返回 `Pass`，observe 直接丢弃。
///
/// session / turn 装配时若没有显式注入 hook 引擎走它——保证"未配置 hook
/// = 主循环行为完全不变"，与 [`crate::http::NoopHttpClient`] 同款。
#[derive(Debug, Default)]
pub struct NoopHookEngine;

impl HookEngine for NoopHookEngine {
    fn fire<'a>(&'a self, _event: HookEvent<'a>, _ctx: HookCtx<'a>) -> BoxFuture<'a, HookOutcome> {
        Box::pin(async { HookOutcome::default() })
    }

    fn observe<'a>(&'a self, _event: HookEvent<'a>, _ctx: HookCtx<'a>) {}
}

// ---------------------------------------------------------------------------
// DefaultHookEngine
// ---------------------------------------------------------------------------

/// 一条已装配 hook：matcher + handler + 单条超时。
///
/// 由 CLI 装配期 / 测试构造；engine 仅按事件类别索引并 `matches` 后串行调用。
pub struct HandlerEntry {
    pub matcher: HookMatcher,
    pub handler: Arc<dyn HookHandler>,
    /// `None` = 用引擎默认（5 秒）；CLI 把 TOML 上的 `timeout_sec` 翻进来。
    pub timeout: Option<Duration>,
}

impl HandlerEntry {
    pub fn new(matcher: HookMatcher, handler: Arc<dyn HookHandler>) -> Self {
        Self {
            matcher,
            handler,
            timeout: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// 一份按事件类别分桶的 handler 表。
///
/// 装配在 [`DefaultHookEngine`] 内，外部用 [`DefaultHookEngine::reload`]
/// 整体替换——`ArcSwap` 让运行期热加载几乎零开销。
#[derive(Default)]
pub struct HandlerTable {
    /// 按事件类别索引的 handler 列表。声明顺序即 pipeline 执行顺序
    /// （详见 `docs/internal/hooks.md` §3.4）。
    pub buckets: std::collections::HashMap<HookEventKind, Vec<HandlerEntry>>,
}

impl HandlerTable {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn handlers(&self, kind: HookEventKind) -> &[HandlerEntry] {
        self.buckets.get(&kind).map(Vec::as_slice).unwrap_or(&[])
    }

    /// 在指定事件类别下追加一条已装配 hook。
    pub fn push(&mut self, kind: HookEventKind, entry: HandlerEntry) {
        self.buckets.entry(kind).or_default().push(entry);
    }
}

/// 默认 hook 引擎：按 hooks.md §3.4 的 pipeline 语义串行调度。
///
/// - 用 [`ArcSwap`] 持有 [`HandlerTable`]，[`Self::reload`] 可整体热替换
/// - `fire` 内部按 matcher 过滤 → 串行 await，每个 handler 看到的是前序
///   patch 应用之后的事件
/// - 单条 handler 超时 / panic / 错误按 §3.5 表降级；具体规则见
///   [`Self::merge_outcome`]
pub struct DefaultHookEngine {
    table: ArcSwap<HandlerTable>,
}

impl DefaultHookEngine {
    pub fn new() -> Self {
        Self {
            table: ArcSwap::from_pointee(HandlerTable::empty()),
        }
    }

    /// 用一份新的 handler 表整体替换当前表；运行期热加载用。
    ///
    /// 旧表在所有正在跑的 fire/observe 调用结束后由 `Arc` 自动回收。
    pub fn reload(&self, table: HandlerTable) {
        self.table.store(Arc::new(table));
    }

    /// 当前 handler 表的快照引用。仅供测试 / 诊断观察用。
    #[doc(hidden)]
    pub fn snapshot(&self) -> Arc<HandlerTable> {
        self.table.load_full()
    }
}

impl Default for DefaultHookEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HookEngine for DefaultHookEngine {
    fn fire<'a>(&'a self, event: HookEvent<'a>, ctx: HookCtx<'a>) -> BoxFuture<'a, HookOutcome> {
        let table = self.table.load_full();
        Box::pin(async move {
            let kind = event.kind();
            let entries = table.handlers(kind);
            if entries.is_empty() {
                return HookOutcome::default();
            }

            // Owned 副本：pipeline 之间累积的 patch 落在这里，下一个 handler
            // 看到的是改写后的 event。
            let mut state = OwnedHookEvent::from_borrowed(&event);
            let mut accumulated = HookOutcome::default();

            for entry in entries {
                // 借用形式重新构造一次 event 给 handler 看（详见 hooks.md §2.2）
                let borrowed = state.borrow();
                if !entry.matcher.matches(&borrowed) {
                    continue;
                }

                let timeout = entry.timeout.unwrap_or(DEFAULT_HANDLER_TIMEOUT);
                let handler_ctx = HookCtx::new(ctx.session_id, ctx.cwd, ctx.cancel.clone());
                // 在 BoxFuture 上跑 catch_unwind：handler 内 panic 转 HandlerFailed
                // 后由下方 outcome 处理逻辑按 §3.5 降级。
                let fut =
                    AssertUnwindSafe(entry.handler.handle(&borrowed, handler_ctx)).catch_unwind();
                let result = match tokio::time::timeout(timeout, fut).await {
                    Ok(Ok(Ok(outcome))) => Ok(outcome),
                    Ok(Ok(Err(err))) => Err(err),
                    Ok(Err(panic)) => Err(HookError::HandlerFailed(BoxError::new(PanicError(
                        panic_message(&panic),
                    )))),
                    Err(_elapsed) => Err(HookError::Timeout),
                };

                if let Some(block) = merge_outcome(&mut state, &mut accumulated, kind, result) {
                    accumulated.block = Some(block);
                    return accumulated;
                }
            }

            accumulated
        })
    }

    fn observe<'a>(&'a self, _event: HookEvent<'a>, _ctx: HookCtx<'a>) {
        // 未落地：observe-only handler 还没接入 AgentEvent fan-out。
        // 核心缺口是 AgentEvent → HookEvent 投影 task（订阅 session 事件流、
        // 投影后调本方法）尚未实现；config 也还没有 async 事件桶。
        // 完整落地步骤见 `docs/internal/hooks.md` §11，且本方法目前无调用方。
    }
}

// ---------------------------------------------------------------------------
// Pipeline 内部：owned event + outcome 合并
// ---------------------------------------------------------------------------

/// `HookEvent` 的 owned 副本——pipeline 之间携带 patch 演化。
///
/// 字段命名跟 [`HookEvent`] 1:1 对应；只承载 v0 实际 emit 的 5 件套。
/// Async 类事件不进 pipeline（observe 不会走 fire），所以这里不收 owned
/// 表示也无影响；遇到 Async 事件 fire 直接 noop（见上方 `for entry`
/// 之前的 borrow，matcher 永远命中但 owned 形态保留原始借用语义）。
enum OwnedHookEvent {
    SessionStart {
        source: OwnedSessionSource,
        cwd: std::path::PathBuf,
    },
    UserPromptSubmit {
        content: Vec<ContentBlock>,
    },
    PreToolUse {
        id: ToolCallId,
        name: String,
        args: Value,
        safety: SafetyClass,
    },
    PostToolUse {
        id: ToolCallId,
        name: String,
        fields: ToolCallUpdateFields,
    },
    PostToolUseFailure {
        id: ToolCallId,
        name: String,
        error: String,
    },
    /// Async 事件——不进 pipeline 的 patch 累积逻辑，只为补完 enum 完整性。
    /// `fire` 里其实不会走到这条分支（async 事件不应当通过 fire 入口）。
    Other,
}

#[derive(Debug, Clone)]
enum OwnedSessionSource {
    New,
    Resume { session_id: SessionId },
}

impl OwnedHookEvent {
    fn from_borrowed(event: &HookEvent<'_>) -> Self {
        match event {
            HookEvent::SessionStart { source, cwd } => Self::SessionStart {
                source: match source {
                    SessionSource::New => OwnedSessionSource::New,
                    SessionSource::Resume { session_id } => OwnedSessionSource::Resume {
                        session_id: (*session_id).clone(),
                    },
                },
                cwd: cwd.to_path_buf(),
            },
            HookEvent::UserPromptSubmit { content } => Self::UserPromptSubmit {
                content: content.to_vec(),
            },
            HookEvent::PreToolUse {
                id,
                name,
                args,
                safety,
            } => Self::PreToolUse {
                id: (*id).clone(),
                name: (*name).to_string(),
                args: (*args).clone(),
                safety: *safety,
            },
            HookEvent::PostToolUse { id, name, fields } => Self::PostToolUse {
                id: (*id).clone(),
                name: (*name).to_string(),
                fields: (*fields).clone(),
            },
            HookEvent::PostToolUseFailure { id, name, error } => Self::PostToolUseFailure {
                id: (*id).clone(),
                name: (*name).to_string(),
                error: (*error).to_string(),
            },
            _ => Self::Other,
        }
    }

    fn borrow(&self) -> HookEvent<'_> {
        match self {
            Self::SessionStart { source, cwd } => HookEvent::SessionStart {
                source: match source {
                    OwnedSessionSource::New => SessionSource::New,
                    OwnedSessionSource::Resume { session_id } => {
                        SessionSource::Resume { session_id }
                    }
                },
                cwd,
            },
            Self::UserPromptSubmit { content } => HookEvent::UserPromptSubmit { content },
            Self::PreToolUse {
                id,
                name,
                args,
                safety,
            } => HookEvent::PreToolUse {
                id,
                name,
                args,
                safety: *safety,
            },
            Self::PostToolUse { id, name, fields } => HookEvent::PostToolUse { id, name, fields },
            Self::PostToolUseFailure { id, name, error } => {
                HookEvent::PostToolUseFailure { id, name, error }
            }
            Self::Other => unreachable!("OwnedHookEvent::Other should never be borrowed"),
        }
    }
}

/// 把 handler 返回的 [`HookOutcome`] 合并进 pipeline 状态。
///
/// 返回 `Some(reason)` 表示 pipeline 早退（block 命中 + 该事件允许 block）。
/// 返回 `None` 表示继续下一个 handler。`HookError` 按 hooks.md §3.5 的表降级：
/// - 允许 block 的事件（`UserPromptSubmit` / `PreToolUse`）：错误等价 block
/// - 不允许 block 的事件：错误降为 warning，继续 pipeline
fn merge_outcome(
    state: &mut OwnedHookEvent,
    accumulated: &mut HookOutcome,
    kind: HookEventKind,
    result: Result<HookOutcome, HookError>,
) -> Option<String> {
    let outcome = match result {
        Ok(o) => o,
        Err(err) => {
            return handle_handler_error(kind, err);
        }
    };

    // ── block 字段处理 ──
    if let Some(reason) = outcome.block {
        if event_allows_block(kind) {
            return Some(reason);
        }
        tracing::warn!(
            kind = ?kind,
            reason = %reason,
            "hook outcome.block ignored: event does not allow block"
        );
    }

    // ── patch 字段处理 ──
    if let Some(patch) = outcome.patch {
        match (kind, patch) {
            (HookEventKind::PreToolUse, HookPatch::ToolArgs(new_args)) => {
                if let OwnedHookEvent::PreToolUse { args, .. } = state {
                    *args = new_args.clone();
                }
                accumulated.patch = Some(HookPatch::ToolArgs(new_args));
            }
            (HookEventKind::UserPromptSubmit, HookPatch::UserPrompt { prepend, append }) => {
                if let OwnedHookEvent::UserPromptSubmit { content } = state {
                    let mut next = Vec::with_capacity(prepend.len() + content.len() + append.len());
                    next.extend(prepend.clone());
                    next.append(content);
                    next.extend(append.clone());
                    *content = next;
                }
                // accumulated.patch 反映"组合后的"前后追加块——下一个 handler
                // 看到的 state.content 已经把它合进去了，这里给 caller 保留
                // 一份合并视图（caller 通常在 turn 主循环里用不到，因为 state
                // 已经直接被改写）。
                accumulated.patch = match accumulated.patch.take() {
                    Some(HookPatch::UserPrompt {
                        prepend: old_prepend,
                        append: old_append,
                    }) => {
                        let mut combined_prepend = prepend.clone();
                        combined_prepend.extend(old_prepend);
                        let mut combined_append = old_append;
                        combined_append.extend(append.clone());
                        Some(HookPatch::UserPrompt {
                            prepend: combined_prepend,
                            append: combined_append,
                        })
                    }
                    _ => Some(HookPatch::UserPrompt { prepend, append }),
                };
            }
            (kind, patch) => {
                tracing::warn!(
                    kind = ?kind,
                    patch_kind = patch.kind_str(),
                    "hook outcome.patch ignored: not allowed for this event kind"
                );
            }
        }
    }

    // ── append 字段处理 ──
    if !outcome.append.is_empty() {
        if event_allows_append(kind) {
            accumulated.append.extend(outcome.append);
        } else {
            tracing::warn!(
                kind = ?kind,
                count = outcome.append.len(),
                "hook outcome.append ignored: event has no landing site"
            );
        }
    }

    None
}

fn handle_handler_error(kind: HookEventKind, err: HookError) -> Option<String> {
    if event_allows_block(kind) {
        // 保守：handler 出问题就别让事件继续——把错误信息当作 block reason。
        tracing::info!(kind = ?kind, error = %err, "hook handler error treated as block");
        Some(err.to_string())
    } else {
        tracing::warn!(kind = ?kind, error = %err, "hook handler error downgraded to warning");
        None
    }
}

fn event_allows_block(kind: HookEventKind) -> bool {
    matches!(
        kind,
        HookEventKind::UserPromptSubmit | HookEventKind::PreToolUse
    )
}

fn event_allows_append(kind: HookEventKind) -> bool {
    matches!(
        kind,
        HookEventKind::SessionStart
            | HookEventKind::UserPromptSubmit
            | HookEventKind::PostToolUse
            | HookEventKind::PostToolUseFailure
    )
}

impl HookPatch {
    fn kind_str(&self) -> &'static str {
        match self {
            Self::ToolArgs(_) => "tool_args",
            Self::UserPrompt { .. } => "user_prompt",
        }
    }
}

// catch_unwind payload → 文本，避免依赖具体 panic 类型
#[derive(Debug, thiserror::Error)]
#[error("hook handler panicked: {0}")]
struct PanicError(String);

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use agent_client_protocol_schema::ContentBlock;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ctx<'a>(session_id: &'a SessionId, cwd: &'a Path) -> HookCtx<'a> {
        HookCtx::new(session_id, cwd, CancellationToken::new())
    }

    /// 简单 handler：返回固定 outcome，记录 handle 调用次数与每次看到的 args（仅
    /// `PreToolUse` 用）。
    struct StubHandler {
        outcome: Mutex<HookOutcome>,
        observed_args: Mutex<Vec<Value>>,
        observed_user_prompt: Mutex<Vec<Vec<ContentBlock>>>,
        calls: AtomicUsize,
    }

    impl StubHandler {
        fn new(outcome: HookOutcome) -> Arc<Self> {
            Arc::new(Self {
                outcome: Mutex::new(outcome),
                observed_args: Mutex::new(Vec::new()),
                observed_user_prompt: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl HookHandler for StubHandler {
        fn capability(&self) -> HookCapability {
            HookCapability::Intercept
        }

        fn handle<'a>(
            &'a self,
            event: &'a HookEvent<'a>,
            _ctx: HookCtx<'a>,
        ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let HookEvent::PreToolUse { args, .. } = event {
                self.observed_args
                    .lock()
                    .expect("stub mutex")
                    .push((*args).clone());
            }
            if let HookEvent::UserPromptSubmit { content } = event {
                self.observed_user_prompt
                    .lock()
                    .expect("stub mutex")
                    .push(content.to_vec());
            }
            let outcome = self.outcome.lock().expect("stub mutex").clone();
            Box::pin(async move { Ok(outcome) })
        }
    }

    /// Handler 永久挂起——用于测试超时。
    struct HangHandler;

    impl HookHandler for HangHandler {
        fn capability(&self) -> HookCapability {
            HookCapability::Intercept
        }

        fn handle<'a>(
            &'a self,
            _event: &'a HookEvent<'a>,
            _ctx: HookCtx<'a>,
        ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
            Box::pin(std::future::pending())
        }
    }

    /// Handler panic——用于测试 catch_unwind 降级。
    struct PanicHandler;

    impl HookHandler for PanicHandler {
        fn capability(&self) -> HookCapability {
            HookCapability::Intercept
        }

        fn handle<'a>(
            &'a self,
            _event: &'a HookEvent<'a>,
            _ctx: HookCtx<'a>,
        ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
            Box::pin(async { panic!("boom") })
        }
    }

    fn pre_tool_use<'a>(
        id: &'a ToolCallId,
        name: &'a str,
        args: &'a Value,
        safety: SafetyClass,
    ) -> HookEvent<'a> {
        HookEvent::PreToolUse {
            id,
            name,
            args,
            safety,
        }
    }

    #[tokio::test]
    async fn noop_engine_returns_pass() {
        let engine = NoopHookEngine;
        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = engine.fire(ev, ctx(&session_id, cwd)).await;
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn default_engine_empty_table_returns_pass() {
        let engine = DefaultHookEngine::new();
        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::UserPromptSubmit { content: &[] };
        let outcome = engine.fire(ev, ctx(&session_id, cwd)).await;
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[test]
    fn event_kind_matches_variant() {
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        assert_eq!(ev.kind(), HookEventKind::SessionStart);
        assert!(ev.is_sync());

        let ev = HookEvent::SessionEnd {
            reason: AcpStopReason::EndTurn,
        };
        assert_eq!(ev.kind(), HookEventKind::SessionEnd);
        assert!(!ev.is_sync());
    }

    #[test]
    fn reload_swaps_table() {
        let engine = DefaultHookEngine::new();
        assert!(engine.snapshot().buckets.is_empty());
        let mut t = HandlerTable::empty();
        t.buckets.insert(HookEventKind::PreToolUse, Vec::new());
        engine.reload(t);
        assert!(
            engine
                .snapshot()
                .buckets
                .contains_key(&HookEventKind::PreToolUse)
        );
    }

    // ----- matcher -----

    #[test]
    fn matcher_filters_by_tool_name() {
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let bash = pre_tool_use(&id, "bash", &args, SafetyClass::Destructive);
        let edit = pre_tool_use(&id, "edit", &args, SafetyClass::Mutating);
        let m = HookMatcher {
            tool: Some("bash".to_string()),
            ..Default::default()
        };
        assert!(m.matches(&bash));
        assert!(!m.matches(&edit));
    }

    #[test]
    fn matcher_filters_by_glob() {
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let mcp_a = pre_tool_use(&id, "mcp.fs.read", &args, SafetyClass::ReadOnly);
        let mcp_b = pre_tool_use(&id, "mcp.git.status", &args, SafetyClass::ReadOnly);
        let plain = pre_tool_use(&id, "bash", &args, SafetyClass::Destructive);
        let m = HookMatcher {
            tool_glob: Some("mcp.*".to_string()),
            ..Default::default()
        };
        assert!(m.matches(&mcp_a));
        assert!(m.matches(&mcp_b));
        assert!(!m.matches(&plain));
    }

    #[test]
    fn matcher_filters_by_safety() {
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let destructive = pre_tool_use(&id, "bash", &args, SafetyClass::Destructive);
        let read_only = pre_tool_use(&id, "fs.read", &args, SafetyClass::ReadOnly);
        let m = HookMatcher {
            safety: vec![SafetyClass::Destructive, SafetyClass::Network],
            ..Default::default()
        };
        assert!(m.matches(&destructive));
        assert!(!m.matches(&read_only));
    }

    #[test]
    fn matcher_empty_matches_anything() {
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let m = HookMatcher::default();
        assert!(m.matches(&pre_tool_use(&id, "anything", &args, SafetyClass::ReadOnly)));
        let cwd = std::path::Path::new("/");
        assert!(m.matches(&HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        }));
    }

    // ----- pipeline -----

    #[tokio::test]
    async fn pipeline_pre_tool_use_passes_patched_args_downstream() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();

        let h1 = StubHandler::new(HookOutcome {
            patch: Some(HookPatch::ToolArgs(serde_json::json!({"redacted": true}))),
            ..Default::default()
        });
        let h2 = StubHandler::new(HookOutcome::default());

        table.push(
            HookEventKind::PreToolUse,
            HandlerEntry::new(HookMatcher::default(), h1.clone()),
        );
        table.push(
            HookEventKind::PreToolUse,
            HandlerEntry::new(HookMatcher::default(), h2.clone()),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"command": "echo hi"});
        let outcome = engine
            .fire(
                pre_tool_use(&id, "bash", &args, SafetyClass::Destructive),
                ctx(&session_id, cwd),
            )
            .await;

        // h2 看到的 args 是 h1 改写之后的
        let observed = h2.observed_args.lock().expect("mutex").clone();
        assert_eq!(observed, vec![serde_json::json!({"redacted": true})]);

        assert!(matches!(
            outcome.patch,
            Some(HookPatch::ToolArgs(ref v)) if v == &serde_json::json!({"redacted": true})
        ));
    }

    #[tokio::test]
    async fn pipeline_blocks_early() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();

        let h1 = StubHandler::new(HookOutcome {
            block: Some("nope".to_string()),
            ..Default::default()
        });
        let h2 = StubHandler::new(HookOutcome::default());

        table.push(
            HookEventKind::PreToolUse,
            HandlerEntry::new(HookMatcher::default(), h1.clone()),
        );
        table.push(
            HookEventKind::PreToolUse,
            HandlerEntry::new(HookMatcher::default(), h2.clone()),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let outcome = engine
            .fire(
                pre_tool_use(&id, "bash", &args, SafetyClass::ReadOnly),
                ctx(&session_id, cwd),
            )
            .await;

        assert_eq!(outcome.block.as_deref(), Some("nope"));
        assert_eq!(h2.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn post_tool_use_drops_stray_block_field() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        let h1 = StubHandler::new(HookOutcome {
            block: Some("ignored".to_string()),
            append: vec![ContentBlock::from("hello")],
            ..Default::default()
        });
        table.push(
            HookEventKind::PostToolUse,
            HandlerEntry::new(HookMatcher::default(), h1),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let fields = ToolCallUpdateFields::default();
        let ev = HookEvent::PostToolUse {
            id: &id,
            name: "bash",
            fields: &fields,
        };
        let outcome = engine.fire(ev, ctx(&session_id, cwd)).await;
        // Post* 不允许 block；append 落地
        assert!(outcome.block.is_none());
        assert_eq!(outcome.append.len(), 1);
    }

    #[tokio::test]
    async fn session_start_drops_stray_block() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        let h1 = StubHandler::new(HookOutcome {
            block: Some("ignored".to_string()),
            append: vec![ContentBlock::from("preload")],
            ..Default::default()
        });
        table.push(
            HookEventKind::SessionStart,
            HandlerEntry::new(HookMatcher::default(), h1),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = engine.fire(ev, ctx(&session_id, cwd)).await;
        // SessionStart 不允许 block；append 落地
        assert!(outcome.block.is_none());
        assert_eq!(outcome.append.len(), 1);
    }

    #[tokio::test]
    async fn user_prompt_submit_pipeline_combines_prepend_append() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        let h1 = StubHandler::new(HookOutcome {
            patch: Some(HookPatch::UserPrompt {
                prepend: vec![ContentBlock::from("[hint] ")],
                append: vec![],
            }),
            ..Default::default()
        });
        let h2 = StubHandler::new(HookOutcome::default());
        table.push(
            HookEventKind::UserPromptSubmit,
            HandlerEntry::new(HookMatcher::default(), h1),
        );
        table.push(
            HookEventKind::UserPromptSubmit,
            HandlerEntry::new(HookMatcher::default(), h2.clone()),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let original = vec![ContentBlock::from("hello")];
        let ev = HookEvent::UserPromptSubmit { content: &original };
        let _ = engine.fire(ev, ctx(&session_id, cwd)).await;

        // h2 看到的是 [hint, hello]
        let observed = h2.observed_user_prompt.lock().expect("mutex").clone();
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].len(), 2);
    }

    #[tokio::test]
    async fn pre_tool_use_handler_timeout_blocks() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        table.push(
            HookEventKind::PreToolUse,
            HandlerEntry::new(HookMatcher::default(), Arc::new(HangHandler))
                .with_timeout(Duration::from_millis(10)),
        );
        engine.reload(table);
        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let args = Value::Null;
        let outcome = engine
            .fire(
                pre_tool_use(&id, "bash", &args, SafetyClass::ReadOnly),
                ctx(&session_id, cwd),
            )
            .await;
        // PreToolUse 上 error → block
        assert!(outcome.block.is_some());
    }

    #[tokio::test]
    async fn post_tool_use_handler_panic_downgrades_to_warning() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        table.push(
            HookEventKind::PostToolUse,
            HandlerEntry::new(HookMatcher::default(), Arc::new(PanicHandler)),
        );
        // 跟一个正常 handler，确认 pipeline 没被中断
        let h2 = StubHandler::new(HookOutcome {
            append: vec![ContentBlock::from("after-panic")],
            ..Default::default()
        });
        table.push(
            HookEventKind::PostToolUse,
            HandlerEntry::new(HookMatcher::default(), h2.clone()),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let id = ToolCallId::new("c1");
        let fields = ToolCallUpdateFields::default();
        let ev = HookEvent::PostToolUse {
            id: &id,
            name: "bash",
            fields: &fields,
        };
        let outcome = engine.fire(ev, ctx(&session_id, cwd)).await;
        assert!(outcome.block.is_none());
        assert_eq!(outcome.append.len(), 1);
        assert_eq!(h2.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn glob_basic() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*", ""));
        assert!(glob_match("a*c", "abc"));
        assert!(glob_match("a*c", "ac"));
        assert!(!glob_match("a*c", "abd"));
        assert!(glob_match("???", "abc"));
        assert!(!glob_match("???", "abcd"));
        assert!(glob_match("mcp.*", "mcp.fs.read"));
    }
}
