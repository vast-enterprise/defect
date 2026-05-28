//! Hook 系统：主循环的扩展点。
//!
//! 设计详见 `docs/internal/hooks.md`。
//!
//! ## 抽象层次
//!
//! - [`HookEvent`]：主循环 emit 的载荷（5 件 Sync 拦截 + 7 件 Async 观察 enum 打桩）
//! - [`HookHandler`]：单个执行器（Builtin / Command / Prompt 三种 v0 形态在子 crate 实现）
//! - [`HookEngine`]：主循环面向的派发器；持有 handler 表、执行 pipeline、合并 outcome
//!
//! v0 主循环只 emit 5 件 Sync 拦截事件；其余变体编译期占位、运行时不触发。
//!
//! ## 默认实现
//!
//! [`NoopHookEngine`]：所有 fire 直接返回 `Pass`，observe 直接丢弃；session/turn 装配
//! 时若没有显式 hook 引擎走这个，保持"hook 未配置 = 主循环行为不变"。
//!
//! [`DefaultHookEngine`]：用 [`arc_swap::ArcSwap`] 持有 handler 表，预留热加载口子；
//! v0 仅承诺 trait + 骨架，handler 注册/调度的细节随 Phase D-E 落地。

use std::path::Path;
use std::sync::Arc;

use agent_client_protocol::schema::{
    ContentBlock, RequestPermissionRequest, SessionId, StopReason as AcpStopReason, ToolCallId,
    ToolCallUpdateFields,
};
use arc_swap::ArcSwap;
use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;
use crate::llm::Usage;
use crate::tool::SafetyClass;

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
/// - **Async 观察**（v0 enum 打桩，订阅 [`crate::event::AgentEvent`] 即可使用）：
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

    // ── Async 观察（v0 不 emit，仅占位） ──
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
    /// 由引擎丢弃 + warn。
    Observe,
    /// 完整能力——可 block / patch / append；仅适用于 Sync 拦截事件。
    Intercept,
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
/// - [`Self::observe`]：Async 观察入口；fan-out task 内部调用，不阻塞主循环
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
// DefaultHookEngine（骨架）
// ---------------------------------------------------------------------------

/// 一份按事件类别分桶的 handler 表。
///
/// 装配在 [`DefaultHookEngine`] 内，外部用 [`DefaultHookEngine::reload`]
/// 整体替换——`ArcSwap` 让运行期热加载几乎零开销。
#[derive(Default)]
pub struct HandlerTable {
    /// 按事件类别索引的 handler 列表。声明顺序即 pipeline 执行顺序
    /// （详见 `docs/internal/hooks.md` §3.4）。
    pub buckets: std::collections::HashMap<HookEventKind, Vec<Arc<dyn HookHandler>>>,
}

impl HandlerTable {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn handlers(&self, kind: HookEventKind) -> &[Arc<dyn HookHandler>] {
        self.buckets.get(&kind).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// 默认 hook 引擎。骨架阶段：
///
/// - 持有一份可热加载的 [`HandlerTable`]（[`ArcSwap`]）
/// - `fire` / `observe` 当前等价 `NoopHookEngine`——pipeline 与降级语义留待
///   Phase D 落地（详见 `docs/internal/hooks.md` §10）
///
/// 把骨架先放进 trait 定义里是为了让 `DefaultAgentCoreBuilder::hook_engine`
/// 在 Phase A 就有"非 noop"可注入；handler 注册的 setter 后续补。
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
    fn fire<'a>(&'a self, event: HookEvent<'a>, _ctx: HookCtx<'a>) -> BoxFuture<'a, HookOutcome> {
        // Phase A：不调度任何 handler；保留 table 快照让后续 phase 补完。
        // 当前 buckets 永远为空，等价 NoopHookEngine。
        let table = self.table.load_full();
        Box::pin(async move {
            let _ = table.handlers(event.kind());
            HookOutcome::default()
        })
    }

    fn observe<'a>(&'a self, _event: HookEvent<'a>, _ctx: HookCtx<'a>) {
        // Phase A：observe-only handler 还没接入 AgentEvent fan-out，留空。
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn noop_engine_returns_pass() {
        let engine = NoopHookEngine;
        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let cancel = CancellationToken::new();
        let ctx = HookCtx::new(&session_id, cwd, cancel);
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = engine.fire(ev, ctx).await;
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn default_engine_empty_table_returns_pass() {
        let engine = DefaultHookEngine::new();
        let session_id = SessionId::new("s1");
        let cwd = std::path::Path::new("/");
        let cancel = CancellationToken::new();
        let ctx = HookCtx::new(&session_id, cwd, cancel);
        let ev = HookEvent::UserPromptSubmit { content: &[] };
        let outcome = engine.fire(ev, ctx).await;
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
}
