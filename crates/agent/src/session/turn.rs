//! Turn 主循环。
//!
//! 设计详见 `docs/internal/turn-loop.md`。本文件按 §3 的状态机落地。
//!
//! 关键依赖：
//! - [`History`]：消息历史的读写
//! - [`ToolRegistry`]：工具查找
//! - [`LlmProvider`]：LLM 调用
//! - [`EventEmitter`]：事件发布（`Arc` 共享，使工具 task 也能 emit）
//! - [`PermissionGate`]：权限请求等待

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason as AcpStopReason};
use tokio_util::sync::CancellationToken;

use crate::event::AgentEvent;
use crate::fs::FsBackend;
use crate::hooks::{HookCtx, HookEngine};
use crate::http::HttpClient;
use crate::llm::{
    CompletionRequest, HostedCapabilities, LlmProvider, Message, MessageContent, Role,
    SamplingParams, StopReason as LlmStopReason, ToolChoice, Usage,
};
use crate::policy::SandboxPolicy;
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::{History, ToolRegistry, TurnError};
use crate::shell::ShellBackend;

const DEFAULT_PROMPT_FILE: &str = "AGENTS.md";

#[path = "turn/request_audit.rs"]
mod request_audit;

#[path = "turn/compact.rs"]
mod compact;

#[path = "turn/content.rs"]
mod content;

#[path = "turn/llm.rs"]
mod llm_drive;

#[path = "turn/tools.rs"]
mod tools;

#[path = "turn/hooks.rs"]
mod hooks;

use content::content_block_to_message_content;
use hooks::UserPromptHookFlow;
use llm_drive::{assistant_message, real_input_tokens};
use tools::{Approved, DecisionFlow, approved_tool_name, tool_results_message};

pub(crate) use request_audit::RequestAuditTracker;

/// LLM 调用次数上限策略。详见 `docs/internal/turn-loop.md` §6.1。
#[derive(Debug, Clone, Copy)]
pub enum TurnRequestLimit {
    /// 不设上限。
    Unbounded,
    /// 固定上限：到 N 即返回 [`AcpStopReason::MaxTurnRequests`]。
    Fixed(u32),
    /// 自适应：每当本轮"有 tool_use 被批准执行"视为推进，
    /// 上限自动 +1；否则按 [`Self::Fixed`] 终止。
    Adaptive {
        initial: u32,
        expand_on_progress: bool,
    },
}

impl TurnRequestLimit {
    fn initial_cap(&self) -> Option<u32> {
        match *self {
            Self::Unbounded => None,
            Self::Fixed(n) => Some(n),
            Self::Adaptive { initial, .. } => Some(initial),
        }
    }

    fn expand_on_progress(&self) -> bool {
        matches!(
            self,
            Self::Adaptive {
                expand_on_progress: true,
                ..
            }
        )
    }
}

/// turn 配置。详见 `docs/internal/turn-loop.md` §9。
#[derive(Debug, Clone)]
pub struct TurnConfig {
    pub model: String,
    pub allowed_models: Option<Vec<String>>,
    pub base_prompt: BasePromptConfig,
    pub system_prompt: Option<String>,
    pub prompt: PromptConfig,
    pub sampling: SamplingParams,
    pub request_limit: TurnRequestLimit,
    /// 压缩阈值的**绝对值显式覆盖**（token 数）。`Some` 时优先于
    /// [`Self::compact_ratio`] 推算。`None` 则按 ratio 自动推算。
    pub compact_threshold_tokens: Option<u64>,
    /// 压缩阈值占模型 `context_window` 的比例（如 `0.85` = 用量过 85% 触发）。
    /// `None` = 不按比例自动压缩（且无绝对值时本 turn 完全不压缩）。
    /// 仅在 `compact_threshold_tokens` 为 `None` 且模型公开 `context_window`
    /// 时生效。详见 `session/turn/compact.rs`。
    pub compact_ratio: Option<f64>,
    pub max_llm_retries: u32,
    /// `0` = 不限。v0 默认不限。
    pub max_concurrent_tools: usize,
    /// `before turn-end` hook 强制续命的硬上限——防止 hook 一直 `Continue` 把 turn
    /// 拖成死循环。见 `docs/internal/hook-step-context.md` §5.7。默认 3。
    pub max_hook_continues: u32,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            allowed_models: None,
            base_prompt: BasePromptConfig::default(),
            system_prompt: None,
            prompt: PromptConfig::default(),
            sampling: SamplingParams::default(),
            request_limit: TurnRequestLimit::Adaptive {
                initial: 32,
                expand_on_progress: true,
            },
            compact_threshold_tokens: None,
            // 默认按 context_window 的 85% 触发——留 ~15% 给摘要输出与浮动，
            // 落在 codex(90%)/Claude(~93%)/opencode(window-20k) 的合理区间内。
            compact_ratio: Some(0.85),
            max_llm_retries: 3,
            max_concurrent_tools: 0,
            max_hook_continues: DEFAULT_MAX_HOOK_CONTINUES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BasePromptConfig {
    pub file: Option<PathBuf>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptConfig {
    pub file: String,
    pub text: Option<String>,
    pub provider_overlays: std::collections::BTreeMap<String, String>,
    pub model_overlays: std::collections::BTreeMap<String, String>,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            file: DEFAULT_PROMPT_FILE.to_owned(),
            text: None,
            provider_overlays: std::collections::BTreeMap::new(),
            model_overlays: std::collections::BTreeMap::new(),
        }
    }
}
/// turn 一次执行的全部依赖与累计状态。
///
/// 本 struct 由 [`crate::session::DefaultSession`] 在每次 `run_turn` 时构造，
/// 借用 session 的子组件、跑完即销毁。
pub struct TurnRunner<'a> {
    pub history: &'a dyn History,
    pub tools: &'a dyn ToolRegistry,
    pub provider: &'a dyn LlmProvider,
    pub policy: &'a dyn SandboxPolicy,
    pub events: Arc<EventEmitter>,
    pub permissions: &'a PermissionGate,
    pub cancel: CancellationToken,
    pub config: &'a TurnConfig,
    /// 本 turn 解析定型的 system prompt。`Arc<str>`：每次 `build_request` 都
    /// `clone` 进 `CompletionRequest.system`，Arc 让其退化成引用计数。
    pub system_prompt: Option<Arc<str>>,
    pub cwd: &'a std::path::Path,
    pub fs: Arc<dyn FsBackend>,
    pub shell: Arc<dyn ShellBackend>,
    pub http: Arc<dyn HttpClient>,
    /// session 启动期裁决出的 hosted capability 集合。
    /// 每轮 turn 装配请求时直接复用，不再重新查询。
    pub hosted_capabilities: HostedCapabilities,
    /// Hook 引擎。turn 主循环在 4 个时刻 emit Sync 事件
    /// （`UserPromptSubmit` / `PreToolUse` / `PostToolUse` / `PostToolUseFailure`）
    /// 等 hook 跑完再继续。详见 `docs/internal/hooks.md` §7.1。
    pub hooks: &'a dyn HookEngine,
    /// 当前 session id。`HookCtx` 注入用——hook handler 按 session 维度路由 / 审计。
    pub session_id: &'a SessionId,
    /// session 级后台任务句柄。`Some` 时工具的 `run_in_background` 能力开启
    /// （经 [`crate::tool::ToolContext::background`] 注入给工具）；嵌套子 agent
    /// turn 传 `None`，结构性禁止后台任务自我繁殖。详见 `docs/proposals/task-arrange.md`。
    pub background: Option<crate::session::BackgroundTasks>,
    /// 本 turn 输入的摄入来源——决定 `before_ingest` step 信封的 `source` 字段。
    /// 用户 turn = `User`；session driver 起的后台续转 turn = `Background`（§5.1）。
    pub ingest_source: crate::hooks::step::IngestSource,
    /// 请求稳定性诊断：对比相邻两次实际发给 provider 的请求快照，
    /// 帮助定位 prompt cache 低命中率的高波动来源。
    pub(crate) request_audit: &'a RequestAuditTracker,
}

impl<'a> TurnRunner<'a> {
    /// 跑完一次 turn。
    pub async fn run(&self, prompt: Vec<ContentBlock>) -> Result<AcpStopReason, TurnError> {
        // ① UserPromptSubmit hook（Sync 拦截）
        // 在 prompt 落 history 之前给 hook 改写 / 拦截的机会。详见
        // `docs/internal/hooks.md` §7.1。
        let prompt = match self.fire_user_prompt_submit(prompt).await {
            UserPromptHookFlow::Continue(p) => p,
            UserPromptHookFlow::Refused => {
                // hook block：不 emit UserPromptCommitted，不进 history；
                // 直接返回 Refusal，让 ACP 桥接以此回 PromptResponse。
                return Ok(AcpStopReason::Refusal);
            }
        };

        self.events
            .emit(AgentEvent::UserPromptCommitted {
                content: prompt.clone(),
            })
            .await;
        self.history.append(Message {
            role: Role::User,
            content: prompt
                .into_iter()
                .map(content_block_to_message_content)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect(),
        });

        // after Ingest hook：输入已并入 history。仅可注入。
        {
            let mut step = crate::hooks::step::AfterIngest {
                committed_len: 1,
                additional_context: Vec::new(),
            };
            let _ = self.hooks.dispatch(&mut step, self.hook_ctx()).await;
            if !step.additional_context.is_empty() {
                self.append_user_feedback(step.additional_context);
            }
        }

        self.events.emit(AgentEvent::TurnStarted).await;

        // after turn enter hook：turn 作用域已进入。可注入 system context / Break 拒该 turn。
        // 注：现状埋点在 prompt 摄入之后（设计 §6 标注的"落地调整：埋点前移"待后续）。
        {
            let mut step = crate::hooks::step::AfterTurnEnter {
                is_subagent: false,
                agent_type: None,
                additional_context: Vec::new(),
            };
            let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;
            if !step.additional_context.is_empty() {
                self.append_user_feedback(step.additional_context);
            }
            if let crate::hooks::step::HookControl::Break { .. } = control {
                return Ok(AcpStopReason::EndTurn);
            }
        }

        let result = self.run_inner().await;

        if let Ok(outcome) = &result {
            self.events
                .emit(AgentEvent::TurnEnded {
                    reason: outcome.reason,
                    usage: outcome.usage,
                })
                .await;
        }
        // Err 路径不发 TurnEnded：桥接层据 future outcome 自行决定 wire 响应。

        result.map(|outcome| outcome.reason)
    }

    async fn run_inner(&self) -> Result<TurnOutcome, TurnError> {
        let mut state = TurnState::new(self.config.request_limit, self.config.max_hook_continues);
        loop {
            if self.cancel.is_cancelled() {
                return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
            }

            self.maybe_compact().await?;

            let mut req = self.build_request();

            // before Generate hook：可改 request（model）/ short-circuit（填合成 assistant，跳过 LLM）/ Break。
            let mut before_gen = crate::hooks::step::BeforeGenerate {
                model: req.model.clone(),
                message_count: req.messages.len(),
                attempt: state.request_count.saturating_add(1),
                assistant_text: None,
            };
            let bg_control = self.hooks.dispatch(&mut before_gen, self.hook_ctx()).await;
            req.model = before_gen.model;
            if let Some(text) = before_gen.assistant_text {
                // short-circuit：用合成 assistant 回复跳过真实 LLM 调用，然后走 before-turn-end 判定。
                self.history.append(Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text { text }].into(),
                });
                if self.decide_turn_end(&mut state).await {
                    continue;
                }
                return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
            }
            if let crate::hooks::step::HookControl::Break { reason } = bg_control {
                return Ok(turn_outcome(&state, reason));
            }

            let (mut stream, attempt) = self.call_llm_with_retry(&req, &mut state).await?;

            let outcome = self.drain_provider_stream(&mut stream, &mut state).await?;

            if outcome.cancelled {
                return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
            }

            // 流已 drain，本次调用的 usage 到齐——现在发 LlmCallFinished，带**单次**
            // 调用的真 usage（outcome.usage，非 turn 累计 state.usage）。
            self.events
                .emit(AgentEvent::LlmCallFinished {
                    model: req.model.clone(),
                    attempt,
                    usage: outcome.usage,
                    error: None,
                })
                .await;

            // after Generate hook：观察（usage / stop / error）。无可填产出；要干预下一轮走 before-turn-end。
            let stop_reason_for_hook = match outcome.stop {
                LlmStopReason::EndTurn | LlmStopReason::StopSequence => AcpStopReason::EndTurn,
                LlmStopReason::Refusal => AcpStopReason::Refusal,
                LlmStopReason::MaxTokens => AcpStopReason::MaxTokens,
                LlmStopReason::ToolUse => AcpStopReason::EndTurn,
            };
            let mut after_gen = crate::hooks::step::AfterGenerate {
                model: req.model.clone(),
                usage: outcome.usage,
                stop: stop_reason_for_hook,
                error: None,
            };
            let _ = self.hooks.dispatch(&mut after_gen, self.hook_ctx()).await;

            // 把本次调用回报的真实输入 token 喂给 history，作为压缩阈值判断的
            // 精确基线（详见 `session/turn/compact.rs`）。本次发出的 messages 即
            // req.messages，其真实输入量就是 outcome.usage 的输入侧三项之和。
            if let Some(real_input) = real_input_tokens(&outcome.usage) {
                self.history.record_input_tokens(real_input);
            }

            let assistant = assistant_message(&outcome);
            if !assistant.content.is_empty() {
                self.history.append(assistant);
            }

            // 被动停止（Refusal / MaxTokens）：不经 before-turn-end hook（hook 不能续命这些），
            // 直接退出。见 `docs/internal/hook-step-context.md` §5.7。
            match outcome.stop {
                LlmStopReason::EndTurn | LlmStopReason::StopSequence => {
                    // 自愿停止 → before-turn-end 判定点。
                    if self.decide_turn_end(&mut state).await {
                        continue;
                    }
                    return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
                }
                LlmStopReason::Refusal => {
                    return Ok(turn_outcome(&state, AcpStopReason::Refusal));
                }
                LlmStopReason::MaxTokens => {
                    return Ok(turn_outcome(&state, AcpStopReason::MaxTokens));
                }
                LlmStopReason::ToolUse => {}
            }

            if outcome.tool_uses.is_empty() {
                // 自愿停止（没要工具）→ 同一个 before-turn-end 判定点。
                if self.decide_turn_end(&mut state).await {
                    continue;
                }
                return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
            }

            // before Permission hook（v0 仅 observe 打桩；policy 仍是放行权威，见 hooks.md §7.3）。
            for tu in &outcome.tool_uses {
                let mut bp = crate::hooks::step::BeforePermission {
                    tool: tu.name.clone(),
                    decision: "ask".to_string(),
                    resolved: None,
                };
                let _ = self.hooks.dispatch(&mut bp, self.hook_ctx()).await;
            }

            let approved = match self.decide_permissions(&outcome.tool_uses).await? {
                DecisionFlow::Continue(list) => list,
                DecisionFlow::Cancelled => {
                    return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
                }
            };

            // after Permission hook（v0 仅 observe 打桩）。
            for a in &approved {
                let (tool, granted) = match a {
                    Approved::Run { .. } => (approved_tool_name(a), true),
                    Approved::Denied { .. } | Approved::FailedArgs { .. } => {
                        (approved_tool_name(a), false)
                    }
                };
                let mut ap = crate::hooks::step::AfterPermission { tool, granted };
                let _ = self.hooks.dispatch(&mut ap, self.hook_ctx()).await;
            }

            let progressed = approved.iter().any(|a| matches!(a, Approved::Run { .. }));
            if progressed {
                state.note_progress();
            }

            let results = self.run_tools_concurrently(approved).await;

            // after ToolBatch hook：整批并行工具结束后、下次 LLM 调用前。可注入 / Break（graceful）。
            let mut batch = crate::hooks::step::AfterToolBatch {
                results: results
                    .iter()
                    .map(|r| crate::hooks::step::ToolBatchEntry {
                        tool_name: r.name.clone(),
                        is_error: r.is_error,
                    })
                    .collect(),
                additional_context: Vec::new(),
            };
            let batch_control = self.hooks.dispatch(&mut batch, self.hook_ctx()).await;

            self.history.append(tool_results_message(results));
            if !batch.additional_context.is_empty() {
                self.append_user_feedback(batch.additional_context);
            }
            if let crate::hooks::step::HookControl::Break { reason } = batch_control {
                return Ok(turn_outcome(&state, reason));
            }

            if state.exceeded_request_cap() {
                return Ok(turn_outcome(&state, AcpStopReason::MaxTurnRequests));
            }
        }
    }

    fn build_request(&self) -> CompletionRequest {
        let req = CompletionRequest {
            model: self.config.model.clone(),
            system: self.system_prompt.clone(),
            messages: self.history.snapshot(),
            tools: self.tools.schemas(),
            tool_choice: ToolChoice::Auto,
            sampling: self.config.sampling.clone(),
            hosted_capabilities: self.hosted_capabilities,
        };
        self.request_audit.record(&req);
        req
    }

    /// 压缩触发判定 + 编排。详见 `session/turn/compact.rs`。
    ///
    /// 阈值解析顺序：
    /// 1. `compact_threshold_tokens`（绝对值）显式覆盖；
    /// 2. 否则按 `model_info(model).context_window * compact_ratio` 自动推算；
    /// 3. 两者都拿不到 → 不压缩（保持 v0 语义）。
    async fn maybe_compact(&self) -> Result<(), TurnError> {
        let Some(threshold) = self.compact_threshold() else {
            return Ok(());
        };
        let Some(estimate) = self.history.token_estimate() else {
            return Ok(());
        };
        if estimate < threshold {
            return Ok(());
        }

        // before Compact hook：hook 可 `Skip` 否决本次压缩（变更型 step，无"填产出"）。
        let mut before = crate::hooks::step::BeforeCompact {
            token_estimate: estimate,
            threshold,
        };
        if let crate::hooks::step::HookControl::Skip =
            self.hooks.dispatch(&mut before, self.hook_ctx()).await
        {
            tracing::info!("compaction vetoed by before-compact hook");
            return Ok(());
        }

        let Some(report) = compact::run(self, threshold).await? else {
            // 没有安全的压缩边界（如单个超长轮次）——本次跳过，不发事件。
            return Ok(());
        };
        self.events
            .emit(AgentEvent::ContextCompressed {
                tokens_before: report.tokens_before,
                tokens_after: report.tokens_after,
            })
            .await;

        // after Compact hook：观察 + 可注入（注入物落 history）。
        let mut after = crate::hooks::step::AfterCompact {
            tokens_before: report.tokens_before,
            tokens_after: report.tokens_after,
            additional_context: Vec::new(),
        };
        let _ = self.hooks.dispatch(&mut after, self.hook_ctx()).await;
        if !after.additional_context.is_empty() {
            self.append_user_feedback(after.additional_context);
        }
        Ok(())
    }

    /// 解析本 turn 的压缩阈值（token 数）。`None` = 本 turn 不主动压缩。
    fn compact_threshold(&self) -> Option<u64> {
        if let Some(explicit) = self.config.compact_threshold_tokens {
            return Some(explicit);
        }
        let ratio = self.config.compact_ratio?;
        let context_window = self
            .provider
            .model_info(&self.config.model)?
            .context_window?;
        // ratio 落在 (0, 1]；context_window * ratio 向下取整。
        let threshold = (context_window as f64 * ratio).floor() as u64;
        (threshold > 0).then_some(threshold)
    }

    pub(super) fn hook_ctx(&self) -> HookCtx<'_> {
        HookCtx::new(self.session_id, self.cwd, self.cancel.clone())
    }
}

// ----- internal types -----

#[derive(Clone, Copy)]
struct TurnOutcome {
    reason: AcpStopReason,
    usage: Usage,
}

/// `before turn-end` hook 强制续命次数的**默认**上限。可被
/// [`TurnConfig::max_hook_continues`] 覆盖（配置项 `[turn].max_hook_continues`）。
/// 见 `docs/internal/hook-step-context.md` §5.7。
pub(crate) const DEFAULT_MAX_HOOK_CONTINUES: u32 = 3;

struct TurnState {
    request_count: u32,
    usage: Usage,
    cap: Option<u32>,
    expand_on_progress: bool,
    /// 本 turn 已被 `before turn-end` hook 续命几次。上限 [`Self::max_stop_hook_continues`]。
    stop_hook_continues: u32,
    /// 续命硬上限（来自 [`TurnConfig::max_hook_continues`]）。防 hook 无限 `Continue`。
    max_stop_hook_continues: u32,
}

impl TurnState {
    fn new(limit: TurnRequestLimit, max_hook_continues: u32) -> Self {
        Self {
            request_count: 0,
            usage: Usage::default(),
            cap: limit.initial_cap(),
            expand_on_progress: limit.expand_on_progress(),
            stop_hook_continues: 0,
            max_stop_hook_continues: max_hook_continues,
        }
    }

    fn note_progress(&mut self) {
        if self.expand_on_progress
            && let Some(cap) = self.cap.as_mut()
        {
            *cap = cap.saturating_add(1);
        }
    }

    fn exceeded_request_cap(&self) -> bool {
        match self.cap {
            None => false,
            Some(cap) => self.request_count >= cap,
        }
    }

    /// 是否还允许 `before turn-end` hook 续命（未达硬上限）。
    fn may_stop_hook_continue(&self) -> bool {
        self.stop_hook_continues < self.max_stop_hook_continues
    }

    /// 记一次续命。
    fn note_stop_hook_continue(&mut self) {
        self.stop_hook_continues = self.stop_hook_continues.saturating_add(1);
    }
}

// ----- helpers -----

fn turn_outcome(state: &TurnState, reason: AcpStopReason) -> TurnOutcome {
    TurnOutcome {
        reason,
        usage: state.usage,
    }
}

#[cfg(test)]
mod test;
