//! Hook system: extension points for the agent main loop.
//!
//! ## Abstraction layers
//!
//! - [`HookStep`](step::HookStep)：主循环在各 step 边界调用的拦截点（按事件名分桶）
//! - [`StepHandler`]：单个执行器（Builtin / Command / Prompt 三种形态在子模块实现）
//! - [`HookMatcher`]：单条 hook 的匹配条件（按 tool / glob / safety 过滤）
//! - [`HookEngine`]：主循环面向的派发器；持有 [`HandlerTable`]、执行 pipeline、合并 verdict
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

use agent_client_protocol_schema::SessionId;
use arc_swap::ArcSwap;
use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;
use crate::tool::SafetyClass;

pub mod builtin;
pub mod command;
pub mod prompt;
pub mod step;

/// `DefaultHookEngine` 的默认 per-handler 超时（hooks.md §8）。
const DEFAULT_HANDLER_TIMEOUT: Duration = Duration::from_secs(5);

/// 单条 hook 的匹配条件。
///
/// 形态与 `defect-config` 的 `HookMatcher` 一致；agent crate 不依赖 config，
/// 这里独立定义、CLI 装配时把 config 形态翻成 agent 形态（详见
/// See hooks design for trust model.
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
    /// Step 模型的匹配：按工具名 + safety（都从 step 信封取，非工具 step 传 `None`）。
    ///
    /// 字段全空 = 命中所有。`tool` 精确、`tool_glob` 通配、`safety` 任一命中（空 vec = 不过滤）。
    pub fn matches_step(&self, tool: Option<&str>, safety: Option<SafetyClass>) -> bool {
        if let Some(expected) = &self.tool
            && tool.is_none_or(|n| n != expected)
        {
            return false;
        }
        if let Some(pat) = &self.tool_glob
            && tool.is_none_or(|n| !tool_name_matches(pat, n))
        {
            return false;
        }
        if !self.safety.is_empty() && safety.is_none_or(|s| !self.safety.contains(&s)) {
            return false;
        }
        true
    }
}

/// 工具名 glob 匹配，统一走 [`globset`]（与 skill triggers / search 同款）。
///
/// 工具名以 `.` 分隔（如 `mcp.fs.read`），不是文件路径——`globset` 默认把 `*`
/// 视作"不跨 `/`"，但工具名里没有 `/`，所以 `mcp.*` 能正常匹配整串。模式在每次
/// 匹配时编译（工具名匹配调用稀疏、模式短，编译开销可忽略）。非法模式不 panic：
/// 记一条 warn 并当作不匹配（matcher 失配 = 该 hook 不触发，安全侧）。
fn tool_name_matches(pattern: &str, name: &str) -> bool {
    match globset::Glob::new(pattern) {
        Ok(glob) => glob.compile_matcher().is_match(name),
        Err(err) => {
            tracing::warn!(%pattern, %err, "invalid tool_glob pattern; treating as no-match");
            false
        }
    }
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

/// **Step 模型的 handler**（迁移目标）。引擎给它一个挂载点的输入信封（[`step::HookStep::to_envelope`]
/// 的产物），它产出一个 verdict JSON——引擎再用 [`step::HookStep::apply_verdict`] 把 verdict 应用回
/// step。两种 hook 都实现它：内部 Rust hook 直接算 verdict；command/prompt hook 把信封喂子进程/LLM、
/// Parse output into a verdict.
///
/// 返回 `Ok(None)` = 不干预（等价空 verdict）；`Ok(Some(verdict))` = 应用该 verdict；`Err` = 失败，
/// 由引擎按降级表处理。
pub trait StepHandler: Send + Sync {
    /// 处理一个挂载点：输入信封 → verdict JSON。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>>;
}

// ---------------------------------------------------------------------------
// HookEngine
// ---------------------------------------------------------------------------

/// 主循环面向的派发器（step 模型）。
///
/// 唯一入口 [`Self::dispatch`]：给定一个挂载点的 [`step::HookStep`]，引擎按 `event_name` 找到匹配
/// handler，逐个把 step 的信封喂进去、把 verdict 应用回 step（数据轴累积），合并出最终
/// [`step::HookControl`]（控制轴早退）。step 上的字段改动（注入 / 改 args / 填产出…）就地生效；
/// Summary: what the caller should read + control indication.
///
/// 默认实现 [`DefaultHookEngine`]；测试 / 默认 session 装配走 [`NoopHookEngine`]。
pub trait HookEngine: Send + Sync {
    /// **默认实现返回 [`step::HookControl::Proceed`]**（= 不干预），[`NoopHookEngine`] 即用它。
    /// [`DefaultHookEngine`] 覆盖它走真实派发。
    fn dispatch<'a>(
        &'a self,
        _step: &'a mut dyn step::HookStep,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, step::HookControl> {
        Box::pin(async { step::HookControl::Proceed })
    }
}

// ---------------------------------------------------------------------------
// NoopHookEngine
// ---------------------------------------------------------------------------

/// 默认 hook 引擎：`dispatch` 走 trait 默认实现（`Proceed`，即不干预）。
///
/// session / turn 装配时若没有显式注入 hook 引擎走它——保证"未配置 hook
/// = 主循环行为完全不变"，与 [`crate::http::NoopHttpClient`] 同款。
#[derive(Debug, Default)]
pub struct NoopHookEngine;

impl HookEngine for NoopHookEngine {}

// ---------------------------------------------------------------------------
// DefaultHookEngine
// ---------------------------------------------------------------------------

/// 一份按 step `event_name` 分桶的 handler 表。
///
/// 装配在 [`DefaultHookEngine`] 内，外部用 [`DefaultHookEngine::reload`]
/// 整体替换——`ArcSwap` 让运行期热加载几乎零开销。
#[derive(Default)]
pub struct HandlerTable {
    /// 按 step `event_name`（snake_case）索引的 handler 列表。声明顺序即 pipeline 执行顺序。
    pub step_buckets: std::collections::HashMap<&'static str, Vec<StepHandlerEntry>>,
}

/// 一条已装配的 step handler：name + matcher + handler + 单条超时。
pub struct StepHandlerEntry {
    /// 展示名，仅用于 tracing / 可观测性里标识这条 hook。默认匿名标签
    /// （见 [`Self::new`]）；装配方可用 [`Self::with_name`] 覆盖。
    pub name: String,
    pub matcher: HookMatcher,
    pub handler: Arc<dyn StepHandler>,
    pub timeout: Option<Duration>,
}

/// 未命名 hook 在 tracing 里的占位名。
pub const ANONYMOUS_HOOK_NAME: &str = "anonymous";

impl StepHandlerEntry {
    pub fn new(matcher: HookMatcher, handler: Arc<dyn StepHandler>) -> Self {
        Self {
            name: ANONYMOUS_HOOK_NAME.to_string(),
            matcher,
            handler,
            timeout: None,
        }
    }

    /// 设置展示名。`None` 保持匿名占位（[`ANONYMOUS_HOOK_NAME`]）。
    pub fn with_name(mut self, name: Option<String>) -> Self {
        if let Some(name) = name {
            self.name = name;
        }
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

impl HandlerTable {
    pub fn empty() -> Self {
        Self::default()
    }

    /// 某 step `event_name` 下已装配的 step handler。
    pub fn step_handlers(&self, event_name: &str) -> &[StepHandlerEntry] {
        self.step_buckets
            .get(event_name)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// 在某 step `event_name` 下追加一条 step handler。
    pub fn push_step(&mut self, event_name: &'static str, entry: StepHandlerEntry) {
        self.step_buckets.entry(event_name).or_default().push(entry);
    }
}

/// 默认 hook 引擎：按 hooks.md §3.4 的 pipeline 语义串行调度。
///
/// - 用 [`ArcSwap`] 持有 [`HandlerTable`]，[`Self::reload`] 可整体热替换
/// - `fire` 内部按 matcher 过滤 → 串行 await，每个 handler 看到的是前序
///   patch 应用之后的事件
/// - 单条 handler 超时 / panic / 错误按 §3.5 表降级
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
    fn dispatch<'a>(
        &'a self,
        step: &'a mut dyn step::HookStep,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, step::HookControl> {
        let table = self.table.load_full();
        Box::pin(async move {
            let entries = table.step_handlers(step.event_name());
            if entries.is_empty() {
                return step::HookControl::Proceed;
            }

            // matcher 用工具名 / safety 过滤——从 step 信封里取（仅 *ToolApply* step 带这些字段）。
            let envelope_json = with_common_header(step.to_envelope(), step.event_name(), &ctx);
            let tool = envelope_json.get("tool").and_then(Value::as_str);
            let safety = envelope_json
                .get("safety")
                .and_then(Value::as_str)
                .and_then(parse_safety);

            for entry in entries {
                if !entry.matcher.matches_step(tool, safety) {
                    continue;
                }
                // 每个 handler 看到的是上一个 handler 改写后的信封 + 通用头。
                let envelope = with_common_header(step.to_envelope(), step.event_name(), &ctx);
                let timeout = entry.timeout.unwrap_or(DEFAULT_HANDLER_TIMEOUT);
                let handler_ctx = HookCtx::new(ctx.session_id, ctx.cwd, ctx.cancel.clone());
                let fut = AssertUnwindSafe(entry.handler.handle_step(&envelope, handler_ctx))
                    .catch_unwind();
                let verdict = match tokio::time::timeout(timeout, fut).await {
                    Ok(Ok(Ok(v))) => v,
                    Ok(Ok(Err(err))) => {
                        tracing::warn!(event = %step.event_name(), hook = %entry.name, error = %err, "step hook handler error; skipped");
                        continue;
                    }
                    Ok(Err(panic)) => {
                        tracing::warn!(event = %step.event_name(), hook = %entry.name, panic = %panic_message(&panic), "step hook handler panicked; skipped");
                        continue;
                    }
                    Err(_elapsed) => {
                        tracing::warn!(event = %step.event_name(), hook = %entry.name, "step hook handler timed out; skipped");
                        continue;
                    }
                };
                let Some(verdict) = verdict else { continue };
                match step.apply_verdict(&verdict) {
                    // 控制轴早退：非 Proceed 即停止 pipeline。
                    Ok(step::HookControl::Proceed) => {}
                    Ok(control) => return control,
                    Err(err) => {
                        tracing::warn!(event = %step.event_name(), hook = %entry.name, error = %err, "step verdict malformed; skipped");
                    }
                }
            }
            step::HookControl::Proceed
        })
    }
}

/// 把通用头并进 step 专属信封。通用头：`session_id` / `cwd` / `hook_event`。
///
/// step 自身不持有 `HookCtx`（零借用、`Send`），所以通用上下文由引擎在派发时统一补上——
/// 用户 hook 因此在每个信封里都能拿到 session / cwd / 事件名。step 专属字段优先（不被覆盖）。
fn with_common_header(envelope: Value, event_name: &str, ctx: &HookCtx<'_>) -> Value {
    let Value::Object(mut map) = envelope else {
        return envelope;
    };
    map.entry("session_id")
        .or_insert_with(|| Value::String(ctx.session_id.0.to_string()));
    map.entry("cwd")
        .or_insert_with(|| Value::String(ctx.cwd.to_string_lossy().into_owned()));
    map.entry("hook_event")
        .or_insert_with(|| Value::String(event_name.to_string()));
    Value::Object(map)
}

/// 信封里的 `safety` 字段（snake_case）→ [`SafetyClass`]。未知 / 缺省 → `None`。
fn parse_safety(s: &str) -> Option<SafetyClass> {
    match s {
        "read_only" => Some(SafetyClass::ReadOnly),
        "mutating" => Some(SafetyClass::Mutating),
        "destructive" => Some(SafetyClass::Destructive),
        "network" => Some(SafetyClass::Network),
        _ => None,
    }
}

// catch_unwind payload → 文本，避免依赖具体 panic 类型
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
mod tests {
    use super::*;
    use agent_client_protocol_schema::StopReason as AcpStopReason;

    fn ctx<'a>(session_id: &'a SessionId, cwd: &'a Path) -> HookCtx<'a> {
        HookCtx::new(session_id, cwd, CancellationToken::new())
    }

    #[test]
    fn glob_basic() {
        // 迁移到 globset 后的工具名匹配语义（`.` 非路径分隔符，`*`/`?` 照常）。
        assert!(tool_name_matches("*.rs", "main.rs"));
        assert!(tool_name_matches("*", ""));
        assert!(tool_name_matches("a*c", "abc"));
        assert!(tool_name_matches("a*c", "ac"));
        assert!(!tool_name_matches("a*c", "abd"));
        assert!(tool_name_matches("???", "abc"));
        assert!(!tool_name_matches("???", "abcd"));
        assert!(tool_name_matches("mcp.*", "mcp.fs.read"));
        // 非法模式不 panic，当作不匹配。
        assert!(!tool_name_matches("[bad", "anything"));
    }

    // ----- step 模型派发（迁移 slice 1）-----

    /// 返回固定 verdict 的 step handler。
    struct StubStepHandler {
        verdict: Value,
    }

    impl StepHandler for StubStepHandler {
        fn handle_step<'a>(
            &'a self,
            _envelope: &'a Value,
            _ctx: HookCtx<'a>,
        ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
            let v = self.verdict.clone();
            Box::pin(async move { Ok(Some(v)) })
        }
    }

    #[tokio::test]
    async fn dispatch_routes_to_step_handler_by_event_name() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        table.push_step(
            "before_turn_end",
            StepHandlerEntry::new(
                HookMatcher::default(),
                Arc::new(StubStepHandler {
                    verdict: serde_json::json!({
                        "control": "continue",
                        "additional_context": ["keep going"],
                    }),
                }),
            ),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let mut step = step::BeforeTurnEnd {
            stop_reason: AcpStopReason::EndTurn,
            continues_so_far: 0,
            voluntary: true,
            feedback: Vec::new(),
        };
        let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
        assert_eq!(control, step::HookControl::Continue);
        // verdict 的注入落到了 step 上。
        assert_eq!(step.feedback.len(), 1);
    }

    #[tokio::test]
    async fn dispatch_no_handler_returns_proceed() {
        let engine = DefaultHookEngine::new();
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let mut step = step::BeforeToolApply {
            tool_name: "bash".to_string(),
            safety: crate::tool::SafetyClass::ReadOnly,
            args: serde_json::json!({}),
            result: None,
        };
        let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
        assert_eq!(control, step::HookControl::Proceed);
    }

    #[tokio::test]
    async fn dispatch_matcher_filters_by_tool() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        // 只匹配 tool=="edit" 的 handler；step 的 tool 是 "bash" → 不命中。
        table.push_step(
            "before_tool_apply",
            StepHandlerEntry::new(
                HookMatcher {
                    tool: Some("edit".to_string()),
                    ..Default::default()
                },
                Arc::new(StubStepHandler {
                    verdict: serde_json::json!({"control": "break"}),
                }),
            ),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let mut step = step::BeforeToolApply {
            tool_name: "bash".to_string(),
            safety: crate::tool::SafetyClass::ReadOnly,
            args: serde_json::json!({}),
            result: None,
        };
        let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
        // 不命中 → Proceed。
        assert_eq!(control, step::HookControl::Proceed);
    }

    #[tokio::test]
    async fn dispatch_matcher_filters_by_safety() {
        let engine = DefaultHookEngine::new();
        let mut table = HandlerTable::empty();
        // 只匹配 Destructive 的 handler；step 的 safety 是 ReadOnly → 不命中。
        table.push_step(
            "before_tool_apply",
            StepHandlerEntry::new(
                HookMatcher {
                    safety: vec![SafetyClass::Destructive],
                    ..Default::default()
                },
                Arc::new(StubStepHandler {
                    verdict: serde_json::json!({"control": "break"}),
                }),
            ),
        );
        engine.reload(table);

        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let mut step = step::BeforeToolApply {
            tool_name: "bash".to_string(),
            safety: SafetyClass::ReadOnly,
            args: serde_json::json!({}),
            result: None,
        };
        let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
        assert_eq!(control, step::HookControl::Proceed);

        // safety 命中（Destructive）→ handler 跑，返回 break。
        let mut step2 = step::BeforeToolApply {
            tool_name: "bash".to_string(),
            safety: SafetyClass::Destructive,
            args: serde_json::json!({}),
            result: None,
        };
        let control2 = engine.dispatch(&mut step2, ctx(&session_id, cwd)).await;
        assert!(matches!(control2, step::HookControl::Break { .. }));
    }

    #[tokio::test]
    async fn dispatch_merges_common_header() {
        let engine = DefaultHookEngine::new();
        // 用一个回显信封的 handler 确认通用头被并入。
        struct EchoHandler {
            seen: std::sync::Arc<std::sync::Mutex<Option<Value>>>,
        }
        impl StepHandler for EchoHandler {
            fn handle_step<'a>(
                &'a self,
                envelope: &'a Value,
                _ctx: HookCtx<'a>,
            ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
                *self.seen.lock().unwrap() = Some(envelope.clone());
                Box::pin(async { Ok(None) })
            }
        }
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut table = HandlerTable::empty();
        table.push_step(
            "after_session_enter",
            StepHandlerEntry::new(
                HookMatcher::default(),
                Arc::new(EchoHandler { seen: seen.clone() }),
            ),
        );
        engine.reload(table);

        let session_id = SessionId::new("sess-9");
        let cwd = Path::new("/repo");
        let mut step = step::AfterSessionEnter {
            cwd: "/repo".to_string(),
            source: step::SessionSource::New,
            additional_context: Vec::new(),
        };
        let _ = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
        let env = seen.lock().unwrap().clone().expect("handler saw envelope");
        assert_eq!(env["session_id"], "sess-9");
        assert_eq!(env["cwd"], "/repo");
        assert_eq!(env["hook_event"], "after_session_enter");
    }
}
