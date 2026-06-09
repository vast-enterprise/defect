//! Turn main loop.
//!
//! Turn main loop — the "heart" of the agent. This file implements the state machine.
//!
//! Key dependencies:
//! - [`History`]: read/write message history
//! - [`ToolRegistry`]: tool lookup
//! - [`LlmProvider`]: LLM invocation
//! - [`EventEmitter`]: event emission (shared via `Arc` so tool tasks can also emit)
//! - [`PermissionGate`]: wait for permission requests

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

mod request_audit;

mod compact;

mod microcompact;

mod compaction_slot;

pub use compaction_slot::CompactionSlot;

mod sanitize;

mod content;

mod llm_drive;

mod tools;

mod hooks;

use content::content_block_to_message_content;
use hooks::UserPromptHookFlow;
use llm_drive::{assistant_message, real_input_tokens};
use tools::{
    Approved, DecisionFlow, approved_tool_name, reject_oversized_results, tool_results_message,
};

pub(crate) use request_audit::RequestAuditTracker;
// Out-of-band `/compact` slash command reuses the same synchronous compaction primitive
// as the turn loop's hard-watermark fallback, so the two share boundary selection and
// summary logic instead of forking a second compaction path.
pub(crate) use compact::{CompactionCtx, run_sync as run_sync_compaction};

/// Strategy for capping LLM calls.
#[derive(Debug, Clone, Copy)]
pub enum TurnRequestLimit {
    /// No upper limit.
    Unbounded,
    /// Fixed limit: returns [`AcpStopReason::MaxTurnRequests`] after N turns.
    Fixed(u32),
    /// Adaptive: each time a tool use is approved and executed in the current turn, that
    /// counts as progress and the limit is automatically incremented by 1; otherwise,
    /// termination follows [`Self::Fixed`].
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

/// Configuration for a turn.
#[derive(Debug, Clone)]
pub struct TurnConfig {
    /// The selected provider vendor (the provider half of the selection key). Together
    /// with [`Self::model`], this is used to resolve the actual provider entry in the
    /// registry by the `(vendor, model)` pair.
    pub provider: String,
    pub model: String,
    pub allowed_models: Option<Vec<String>>,
    pub base_prompt: BasePromptConfig,
    pub system_prompt: Option<String>,
    pub prompt: PromptConfig,
    pub sampling: SamplingParams,
    pub request_limit: TurnRequestLimit,
    /// Explicit absolute override for the compaction threshold (in tokens). When `Some`,
    /// takes precedence over the value inferred from [`Self::compact_ratio`]. When
    /// `None`, the threshold is automatically derived from the ratio.
    pub compact_threshold_tokens: Option<u64>,
    /// Compression threshold as a fraction of the model's `context_window` (e.g. `0.85` =
    /// trigger when usage exceeds 85%). This is the **hard watermark**: when reached, if
    /// no background compaction is in flight, the turn main loop performs **synchronous**
    /// compaction as a fallback (blocking the current turn but guaranteeing the context
    /// is not exceeded). `None` = no automatic ratio-based compression (and if no
    /// absolute threshold is set either, no compression occurs for this turn). Only
    /// effective when `compact_threshold_tokens` is `None` and the model exposes a
    /// `context_window`. See `session/turn/compact.rs` for details.
    pub compact_ratio: Option<f64>,
    /// **Background full compaction** toggle. When `true`, once the soft watermark
    /// derived from [`Self::compact_soft_ratio`] is exceeded, a summarization compaction
    /// is started **asynchronously** (without blocking the current turn); it quietly
    /// compresses history before the hard watermark is hit. When disabled, only
    /// synchronous compaction at the hard watermark remains. See
    /// `session/turn/compaction_slot.rs`.
    pub background_compact_enabled: bool,
    /// Soft watermark for background compaction, as a fraction of `context_window`
    /// (default `0.7`). Must be less than `compact_ratio` (the hard watermark) to leave a
    /// window between soft and hard for background summarization to complete. Only
    /// effective when `background_compact_enabled` is set and a threshold can be derived.
    pub compact_soft_ratio: Option<f64>,
    /// Enables **micro‑compaction**. When `true`, each turn first runs a micro‑compaction
    /// (cleans oversized `tool_result` in older turns, without calling the LLM or
    /// deleting messages) at the water level above [`Self::microcompact_ratio`],
    /// deferring expensive full compaction. See `session/turn/microcompact.rs`.
    pub microcompact_enabled: bool,
    /// Micro‑compaction watermark as a fraction of `context_window` (default `0.6`).
    /// Typically below the soft watermark — micro‑compaction is the cheapest first line
    /// of defense. Only effective when `microcompact_enabled` and a threshold can be
    /// derived.
    pub microcompact_ratio: Option<f64>,
    pub max_llm_retries: u32,
    /// `0` = unlimited. The default is unlimited.
    pub max_concurrent_tools: usize,
    /// Hard upper limit on forced continuations from the `before turn-end` hook —
    /// prevents infinite loops from repeated hook `Continue` calls. Default: 3.
    pub max_hook_continues: u32,
    /// Maximum subagent nesting depth (vertical recursion limit) for this turn.
    /// `spawn_agent` uses this to pass the "remaining depth" to tools via
    /// [`crate::tool::ToolContext::subagent_depth`]; a child agent's nested turn receives
    /// `subagent_max_depth` = parent's remaining depth minus one. `0` means this turn's
    /// tool set contains no `spawn_agent`, structurally forbidding dispatch — replacing
    /// the old hardcoded gate of "whitelist never contains `spawn_agent`". Default:
    /// `DEFAULT_SUBAGENT_MAX_DEPTH`.
    pub subagent_max_depth: u32,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            provider: String::new(),
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
            // Trigger compaction at 85% of `context_window` (hard watermark), reserving
            // ~15% for summary output and headroom — within the reasonable range of codex
            // (90%), Claude (~93%), and opencode (window-20k).
            compact_ratio: Some(0.85),
            // Background compaction enabled by default: starts async summarization at
            // soft=0.7, aiming to finish before hard=0.85 is reached.
            background_compact_enabled: true,
            compact_soft_ratio: Some(0.7),
            // Micro‑compaction enabled by default: at 0.6 it evicts old large
            // `tool_result`s — the cheapest first line of defense.
            microcompact_enabled: true,
            microcompact_ratio: Some(0.6),
            max_llm_retries: 3,
            max_concurrent_tools: 0,
            max_hook_continues: DEFAULT_MAX_HOOK_CONTINUES,
            subagent_max_depth: DEFAULT_SUBAGENT_MAX_DEPTH,
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
/// All dependencies and accumulated state for a single turn execution.
///
/// This struct is constructed by [`crate::session::DefaultSession`] on each `run_turn`
/// call,
/// borrowing sub-components of the session and being dropped after the turn completes.
pub struct TurnRunner<'a> {
    pub history: &'a dyn History,
    pub tools: &'a dyn ToolRegistry,
    pub provider: &'a dyn LlmProvider,
    /// The active policy for this turn's snapshot. Owned as an `Arc` rather than
    /// borrowed: it flows with [`crate::tool::ToolContext`] into `spawn_agent`, where
    /// child agents wrap it with
    /// [`NonInteractivePolicy`](crate::policy::NonInteractivePolicy) — must be the
    /// parent's actual policy at this moment.
    pub policy: Arc<dyn SandboxPolicy>,
    pub events: Arc<EventEmitter>,
    pub permissions: &'a PermissionGate,
    pub cancel: CancellationToken,
    pub config: &'a TurnConfig,
    /// The system prompt resolved for this turn. `Arc<str>`: each `build_request` call
    /// `clone`s it into `CompletionRequest.system`; the `Arc` reduces this to a reference
    /// count increment.
    pub system_prompt: Option<Arc<str>>,
    pub cwd: &'a std::path::Path,
    pub fs: Arc<dyn FsBackend>,
    pub shell: Arc<dyn ShellBackend>,
    pub http: Arc<dyn HttpClient>,
    /// Hosted capabilities determined at session startup.
    /// Reused directly on each turn when assembling requests, without re-querying.
    pub hosted_capabilities: HostedCapabilities,
    /// Hook engine. The turn main loop emits Sync events at four points
    /// (`UserPromptSubmit` / `PreToolUse` / `PostToolUse` / `PostToolUseFailure`).
    /// Waits for hooks to finish before proceeding.
    pub hooks: &'a dyn HookEngine,
    /// Current session ID. Injected into `HookCtx` so that hook handlers can route or
    /// audit by session.
    pub session_id: &'a SessionId,
    /// Session-level background task handle. When `Some`, enables the tool's
    /// `run_in_background` capability (injected into tools via
    /// [`crate::tool::ToolContext::background`]); nested sub-agent turns receive `None`,
    /// structurally preventing background task self-replication.
    pub background: Option<crate::session::BackgroundTasks>,
    /// Shared state for the `--goal` goal-driven loop. When `Some`, this session is
    /// running in goal mode: it is injected into the `goal_done` tool via
    /// [`crate::tool::ToolContext::goal`], and the `goal-gate` hook uses it in
    /// `before_turn_end` to allow or extend the session. `None` = non-goal mode
    /// (default).
    pub goal: Option<Arc<crate::session::GoalState>>,
    /// Session-level single-flight compaction slot. When `Some`, background full
    /// compaction is available — exceeding the soft watermark triggers an async summary
    /// compaction without blocking the current turn. Nested sub-agent turns pass `None`
    /// (sub-agent contexts are short-lived and should not spawn background tasks).
    /// Requires `Arc<dyn History>`/`Arc<dyn LlmProvider>` for the task to hold `'static`
    /// references across turns, hence the accompanying `history_arc`/`provider_arc`.
    pub compaction_slot: Option<crate::session::CompactionSlot>,
    /// `Arc<dyn History>` for `compaction_slot` (points to the same object as the
    /// `history` borrow). Held by the background compaction task across turns. When
    /// `None`, background compaction is unavailable and falls back to synchronous
    /// compaction only.
    pub history_arc: Option<Arc<dyn History>>,
    /// `Arc` for the provider used by `compaction_slot`. Same as above.
    pub provider_arc: Option<Arc<dyn LlmProvider>>,
    /// Session-level cancellation token; the background compaction task's cancellation
    /// token is derived from this one (independent of the turn's cancel, cancelled when
    /// the session ends). When `None`, background compaction uses the turn's `cancel`
    /// (sub-agent path).
    pub session_cancel: Option<CancellationToken>,
    /// The ingestion source for this turn's input — determines the `source` field of the
    /// `before_ingest` step envelope.
    /// User turns use `User`; background continuation turns started by the session driver
    /// use `Background`.
    pub ingest_source: crate::hooks::step::IngestSource,
    /// Request stability diagnostics: compares snapshots of two consecutive requests
    /// actually sent to the provider, helping locate sources of high volatility in low
    /// prompt cache hit rates.
    pub(crate) request_audit: &'a RequestAuditTracker,
}

impl<'a> TurnRunner<'a> {
    /// Runs a single turn.
    pub async fn run(&self, prompt: Vec<ContentBlock>) -> Result<AcpStopReason, TurnError> {
        // ① UserPromptSubmit hook (sync interception)
        // Gives hooks a chance to rewrite or intercept the prompt before it lands in
        // history.
        let prompt = match self.fire_user_prompt_submit(prompt).await {
            UserPromptHookFlow::Continue(p) => p,
            UserPromptHookFlow::Refused => {
                // Hook blocked: do not emit `UserPromptCommitted`, do not append to
                // history; return `Refusal` directly so the ACP bridge responds with
                // `PromptResponse`.
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

        // After Ingest hook: input has been merged into history. Injection only.
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

        // After-turn-enter hook: the turn scope has been entered. Allows injecting system
        // context or a Break to reject the turn.
        // Note: currently the hook point is placed after prompt ingestion (moving the hook
        // point earlier is a deferred adjustment).
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
        // The `Err` path does not emit `TurnEnded`; the bridge layer decides the wire
        // response based on the future outcome.

        result.map(|outcome| outcome.reason)
    }

    async fn run_inner(&self) -> Result<TurnOutcome, TurnError> {
        let mut state = TurnState::new(self.config.request_limit, self.config.max_hook_continues);
        loop {
            if self.cancel.is_cancelled() {
                return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
            }

            self.manage_context().await?;

            let mut req = self.build_request();

            // Before Generate hook: can modify request (model), short-circuit (fill in
            // synthetic assistant to skip LLM), or Break.
            let mut before_gen = crate::hooks::step::BeforeGenerate {
                model: req.model.clone(),
                message_count: req.messages.len(),
                attempt: state.request_count.saturating_add(1),
                assistant_text: None,
            };
            let bg_control = self.hooks.dispatch(&mut before_gen, self.hook_ctx()).await;
            req.model = before_gen.model;
            if let Some(text) = before_gen.assistant_text {
                // Short-circuit: use a synthetic assistant reply to skip the real LLM
                // call, then proceed to the before-turn-end check.
                self.history.append(Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text { text }].into(),
                });
                if self
                    .decide_turn_end(&mut state, AcpStopReason::EndTurn, true)
                    .await
                {
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

            // The stream has been drained and all usage for this call is available — emit
            // `LlmCallFinished` with the **per-call** actual usage (`outcome.usage`, not
            // the turn-accumulated `state.usage`).
            self.events
                .emit(AgentEvent::LlmCallFinished {
                    model: req.model.clone(),
                    attempt,
                    usage: outcome.usage,
                    error: None,
                })
                .await;

            // After the Generate hook: observe (usage / stop / error). No output to fill;
            // to intervene, route the next turn through before-turn-end.
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

            // Feed the actual input token count returned by this call into `history` as
            // the precise baseline for compaction threshold decisions (see
            // `session/turn/compact.rs`). The messages sent in this call are
            // `req.messages`, and their real input size is the sum of the three
            // input-side fields in `outcome.usage`.
            if let Some(real_input) = real_input_tokens(&outcome.usage) {
                self.history.record_input_tokens(real_input);
            }

            let assistant = assistant_message(&outcome);
            if !assistant.content.is_empty() {
                self.history.append(assistant);
            }

            // Passive stop (Refusal / MaxTokens): skip the before-turn-end hook (the hook
            // cannot extend these), exit directly.
            match outcome.stop {
                LlmStopReason::EndTurn | LlmStopReason::StopSequence => {
                    // Voluntary stop → before-turn-end decision point.
                    if self
                        .decide_turn_end(&mut state, AcpStopReason::EndTurn, true)
                        .await
                    {
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
                // Voluntary stop (no tool requested) → same before-turn-end decision
                // point.
                if self
                    .decide_turn_end(&mut state, AcpStopReason::EndTurn, true)
                    .await
                {
                    continue;
                }
                return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
            }

            // Before the Permission hook (currently only observes/stubs; policy still
            // delegates to the authority).
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

            // After permission hook (currently only observes/stubs).
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

            let mut results = self.run_tools_concurrently(approved).await;

            // Reject any single tool result that exceeds the model's context window: it can
            // never fit, so appending it as-is would only blow up the next request. Replace
            // it with an actionable error before it enters history. See
            // `reject_oversized_results`.
            let rejected = reject_oversized_results(&mut results, self.context_window());
            if rejected > 0 {
                tracing::warn!(
                    rejected,
                    "rejected oversized tool result(s) exceeding the context window"
                );
            }

            // After `ToolBatch` hook: after all parallel tools finish, before the next
            // LLM call. Allows injection or graceful break.
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
                // Hitting the per-turn request cap is an involuntary stop. Still consult
                // the before-turn-end hook: in goal mode the goal gate decides whether to
                // keep working (resetting the request budget for the next round, bounded
                // by `max_hook_continues`). Without a continuing hook, the turn stops with
                // `MaxTurnRequests` as before.
                if self
                    .decide_turn_end(&mut state, AcpStopReason::MaxTurnRequests, false)
                    .await
                {
                    continue;
                }
                return Ok(turn_outcome(&state, AcpStopReason::MaxTurnRequests));
            }
        }
    }

    fn build_request(&self) -> CompletionRequest {
        // Before sending the request, pair any orphaned `tool_use` (left over from an
        // interruption, with no matching `tool_result`) — otherwise the provider will
        // permanently reject the request. Only patch the copy sent to the provider; the
        // true history remains untouched. See `sanitize`.
        let messages = sanitize::sanitize_tool_pairing(self.history.snapshot());
        let req = CompletionRequest {
            model: self.config.model.clone(),
            system: self.system_prompt.clone(),
            messages,
            tools: self.tools.schemas(),
            tool_choice: ToolChoice::Auto,
            sampling: self.config.sampling.clone(),
            hosted_capabilities: self.hosted_capabilities,
        };
        self.request_audit.record(&req);
        req
    }

    /// Layered context management: micro → soft (background) → hard (synchronous
    /// fallback). Called at the start of each main loop iteration.
    ///
    /// Three water levels (see `compact_thresholds` for details):
    /// 1. **micro** (default 0.6·window): if micro-compaction is enabled, first evict old
    ///    large `tool_result` entries — no LLM calls, no message deletion, near-zero
    ///    latency, deferring expensive full compaction.
    /// 2. **soft** (default 0.7·window): if background compaction is enabled,
    ///    **asynchronously** start a summary compaction; the turn does not block, quietly
    ///    compacting before hitting hard (single-flight, won't re-start if one is already
    ///    in flight).
    /// 3. **hard** (default 0.85·window, equivalent to the old `compact_ratio`
    ///    semantics): at this level compaction is mandatory — if a background compaction
    ///    is already in flight, `await` its completion; otherwise compact
    ///    **synchronously** as a fallback.
    ///
    /// micro/soft require the model to expose `context_window`; hard also supports an
    /// absolute override via `compact_threshold_tokens`. If any level cannot obtain its
    /// threshold, that level is skipped (preserving the conservative "no information, no
    /// compaction" semantics).
    async fn manage_context(&self) -> Result<(), TurnError> {
        let thresholds = self.compact_thresholds();
        // All three thresholds absent → no proactive compaction this turn (preserves the
        // existing semantics).
        if thresholds.is_empty() {
            return Ok(());
        }
        let Some(estimate) = self.history.token_estimate() else {
            return Ok(());
        };

        // ① micro: synchronous, cheapest. May reduce `estimate` below soft/hard
        // thresholds, so re-fetch after compaction.
        let estimate = if self.config.microcompact_enabled
            && thresholds.micro.is_some_and(|t| estimate >= t)
        {
            self.run_microcompact().await;
            self.history.token_estimate().unwrap_or(estimate)
        } else {
            estimate
        };

        // ② soft: crossing the threshold triggers an async background compaction without
        // blocking the current round.
        if self.config.background_compact_enabled
            && let (Some(soft), Some(hard)) = (thresholds.soft, thresholds.hard)
            && estimate >= soft
            && estimate < hard
        {
            self.spawn_background_compaction(hard).await;
            // Non-blocking – continue assembling requests this round; summary persistence
            // happens in a later round (or later).
            return Ok(());
        }

        // ③ hard: must compact.
        if let Some(hard) = thresholds.hard
            && estimate >= hard
        {
            self.compact_hard(estimate, hard).await?;
        }
        Ok(())
    }

    /// Run a micro-compact and write back via `replace`. Best-effort: does nothing if
    /// there is nothing to clean up.
    async fn run_microcompact(&self) {
        let messages = self.history.snapshot();
        let Some((rebuilt, report)) = microcompact::run(&messages) else {
            return;
        };
        self.history.replace(rebuilt);
        tracing::info!(
            cleared = report.cleared,
            tokens_before = report.tokens_before,
            tokens_after = report.tokens_after,
            "context microcompacted"
        );
        self.events
            .emit(AgentEvent::ContextMicrocompacted {
                tokens_before: report.tokens_before,
                tokens_after: report.tokens_after,
                cleared: report.cleared,
            })
            .await;
    }

    /// Spawns a single-flight background full compaction when the soft threshold is
    /// exceeded. Requires a slot and `Arc` references to history and provider (only
    /// available in the top-level turn; child agent turns silently skip this, leaving it
    /// to the synchronous compaction at the hard threshold).
    async fn spawn_background_compaction(&self, hard_threshold: u64) {
        let (Some(slot), Some(history_arc), Some(provider_arc)) = (
            self.compaction_slot.as_ref(),
            self.history_arc.as_ref(),
            self.provider_arc.as_ref(),
        ) else {
            return;
        };
        if slot.is_in_flight() {
            return; // Single-flight: a compaction is already in flight, do not start another.
        }

        // The compaction cancel token is independent of the turn cancel — the summary
        // should be allowed to finish even if the originating turn has ended; however, it
        // is cancelled when the session ends. For sub-agent paths that have no
        // `session_cancel`, it falls back to the turn cancel.
        let cancel = self
            .session_cancel
            .clone()
            .unwrap_or_else(|| self.cancel.clone())
            .child_token();
        let ctx = compact::CompactionCtx {
            provider: provider_arc.clone(),
            model: self.config.model.clone(),
            sampling: self.config.sampling.clone(),
            tools: self.tools.schemas(),
            cancel,
        };
        let events = self.events.clone();
        let on_done: Arc<
            dyn Fn(crate::session::CompactionReport) -> futures::future::BoxFuture<'static, ()>
                + Send
                + Sync,
        > = Arc::new(move |report| {
            // Return a future so that `emit` is awaited inside the compaction task body —
            // no detached task is spawned, and event emission is governed by the
            // compaction task's cancel/track semantics.
            let events = events.clone();
            Box::pin(async move {
                events
                    .emit(AgentEvent::ContextCompressed {
                        tokens_before: report.tokens_before,
                        tokens_after: report.tokens_after,
                    })
                    .await;
            })
        });
        let started = slot.try_spawn(history_arc.clone(), ctx, hard_threshold, on_done);
        if started {
            tracing::info!(hard_threshold, "background compaction started");
        }
    }

    /// Hard threshold fallback: wait for an in-flight background compaction to finish, or
    /// run a synchronous compaction.
    async fn compact_hard(&self, estimate: u64, hard: u64) -> Result<(), TurnError> {
        // A background compaction is already in flight; wait for it to finish to avoid
        // redundant work.
        if let Some(slot) = self.compaction_slot.as_ref()
            && slot.is_in_flight()
        {
            slot.await_in_flight().await;
            return Ok(());
        }

        // Before the compact hook: the hook may `Skip` to veto this compaction (a
        // mutating step).
        let mut before = crate::hooks::step::BeforeCompact {
            token_estimate: estimate,
            threshold: hard,
        };
        if let crate::hooks::step::HookControl::Skip =
            self.hooks.dispatch(&mut before, self.hook_ctx()).await
        {
            tracing::info!("compaction vetoed by before-compact hook");
            return Ok(());
        }

        let ctx = self.sync_compaction_ctx();
        let Some(report) = compact::run_sync(self.history, &ctx, hard).await else {
            // No safe compaction boundary (e.g., a single very long turn) — skip this
            // round, no event emitted.
            return Ok(());
        };
        self.events
            .emit(AgentEvent::ContextCompressed {
                tokens_before: report.tokens_before,
                tokens_after: report.tokens_after,
            })
            .await;

        // After the compact hook: observe and allow injection (injected content goes into
        // history).
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

    /// The [`compact::CompactionCtx`] for synchronous compaction. Wrapping a borrowed
    /// provider in a temporary `Arc` is not feasible (a trait object borrow cannot be
    /// `Arc`), so the synchronous path requires `provider_arc`. Falling back when it is
    /// missing is impossible (the top-level always has one; child agents use a borrowed
    /// `provider`—see the `sync_compaction_ctx` implementation).
    fn sync_compaction_ctx(&self) -> compact::CompactionCtx {
        compact::CompactionCtx {
            provider: self
                .provider_arc
                .clone()
                .expect("sync compaction requires provider_arc"),
            model: self.config.model.clone(),
            sampling: self.config.sampling.clone(),
            tools: self.tools.schemas(),
            cancel: self.cancel.clone(),
        }
    }

    /// Parse the three-tier compaction thresholds (in tokens) for this turn. Any tier set
    /// to `None` means that tier is not triggered.
    /// The model's context window in tokens, if the provider exposes it. `None` ⇒ unknown
    /// (no ceiling can be enforced for compaction or oversized-result rejection).
    fn context_window(&self) -> Option<u64> {
        self.provider
            .model_info(&self.config.model)
            .and_then(|m| m.context_window)
    }

    fn compact_thresholds(&self) -> CompactThresholds {
        let window = self.context_window();

        // For `hard`, an absolute threshold takes precedence; otherwise, use `ratio *
        // window`.
        let hard = self.config.compact_threshold_tokens.or_else(|| {
            let ratio = self.config.compact_ratio?;
            ratio_threshold(window?, ratio)
        });
        // micro/soft can only be derived from window (absolute overrides apply only to
        // hard).
        let from_ratio =
            |ratio: Option<f64>| ratio.and_then(|r| window.and_then(|w| ratio_threshold(w, r)));
        CompactThresholds {
            micro: from_ratio(self.config.microcompact_ratio),
            soft: from_ratio(self.config.compact_soft_ratio),
            hard,
        }
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

/// Three-tier compaction watermarks (in tokens). Each `None` means that tier is not
/// triggered this turn.
#[derive(Clone, Copy)]
struct CompactThresholds {
    micro: Option<u64>,
    soft: Option<u64>,
    hard: Option<u64>,
}

impl CompactThresholds {
    /// All three thresholds absent — no proactive compaction this turn.
    fn is_empty(&self) -> bool {
        self.micro.is_none() && self.soft.is_none() && self.hard.is_none()
    }
}

/// `context_window * ratio` rounded down. `ratio` is in `(0, 1]`. `0` → `None` (no
/// trigger).
fn ratio_threshold(context_window: u64, ratio: f64) -> Option<u64> {
    let threshold = (context_window as f64 * ratio).floor() as u64;
    (threshold > 0).then_some(threshold)
}

/// Default upper limit for forced continuations in the `before turn-end` hook. Can be
/// overridden by [`TurnConfig::max_hook_continues`] (config key
/// `[turn].max_hook_continues`). See docs on hook step context exit semantics.
pub(crate) const DEFAULT_MAX_HOOK_CONTINUES: u32 = 3;

/// Default upper bound for subagent vertical recursion depth. Counted from the top-level
/// turn: N levels means the top turn can spawn subagents, their children can spawn
/// further, and so on, until the Nth level (where `subagent_max_depth` reaches 0) can no
/// longer call `spawn_agent`. The default of 4 leaves room for orchestrations like "main
/// agent → coordinator subagent → worker subagent" while preventing runaway vertical
/// growth. Horizontal runaway is separately guarded by `request_limit`.
pub(crate) const DEFAULT_SUBAGENT_MAX_DEPTH: u32 = 1;

struct TurnState {
    request_count: u32,
    usage: Usage,
    cap: Option<u32>,
    expand_on_progress: bool,
    /// How many times this turn has been extended by the `before turn-end` hook. Cap is
    /// [`Self::max_stop_hook_continues`].
    stop_hook_continues: u32,
    /// Hard upper limit for life-extending continues (from
    /// [`TurnConfig::max_hook_continues`]). Prevents hooks from `Continue`ing
    /// indefinitely.
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

    /// Reset the per-turn request budget back to its initial state. Called when a
    /// `before turn-end` hook keeps the turn alive (e.g. goal mode continuing), so the
    /// `request_limit` behaves as a *per-logical-turn* budget rather than a single budget
    /// shared across the whole multi-turn run. The cap returns to its initial value
    /// (re-reading the configured strategy), discarding any `expand_on_progress` growth.
    fn reset_request_budget(&mut self, limit: TurnRequestLimit) {
        self.request_count = 0;
        self.cap = limit.initial_cap();
    }

    fn exceeded_request_cap(&self) -> bool {
        match self.cap {
            None => false,
            Some(cap) => self.request_count >= cap,
        }
    }

    /// Whether the `before turn-end` hook is still allowed to continue (has not reached
    /// the hard limit).
    fn may_stop_hook_continue(&self) -> bool {
        self.stop_hook_continues < self.max_stop_hook_continues
    }

    /// Records one stop-hook continuation.
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
mod tests;
