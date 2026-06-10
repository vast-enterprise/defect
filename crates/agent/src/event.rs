//! The event stream published by the agent main loop to external consumers.
//!
//! ## Decoupling-by-shape
//!
//! The main loop emits only [`AgentEvent`] — an internal enum — and three independent
//! consumers each take what they need:
//!
//! ```text
//!                ┌──► defect-acp     (translated into SessionUpdate / PromptResponse)
//! AgentEvent ────┼──► defect-storage (jsonl persistence)
//!                └──► tracing        (structured logging, observability)
//! ```
//!
//! We define the enum **variants** ourselves (decoupling the persistence format from the
//! wire, and expressing semantics absent from the wire such as turn boundaries and LLM
//! calls), but we **reuse ACP's passive data structures** (`ToolCallUpdateFields`,
//! `ContentBlock`, `StopReason`, etc.) as field types wherever possible, to avoid
//! reinventing fields.

use std::sync::Arc;

use agent_client_protocol_schema::{
    ContentBlock, PermissionOptionId, StopReason as AcpStopReason, ToolCallId, ToolCallUpdateFields,
};
use serde::{Deserialize, Serialize};

use crate::llm::{Message, Usage};
use crate::policy::PolicyDecision;

/// Events published by the agent main loop.
///
/// Final-state semantics: the event stream for a turn starts with
/// [`AgentEvent::TurnStarted`] and ends with [`AgentEvent::TurnEnded`]. After
/// `TurnEnded`, no more events for that turn are produced — `defect-acp` stops pushing
/// `session/update` and responds with `PromptResponse` upon seeing it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    // ---------- turn boundary ----------
    /// A prompt turn has started.
    TurnStarted,

    /// The user prompt has been committed to history by the main loop.
    UserPromptCommitted { content: Vec<ContentBlock> },

    /// A prompt turn ended. `reason` directly borrows the semantic category from ACP.
    TurnEnded {
        reason: AcpStopReason,
        /// Cumulative token usage for this turn (field-wise sum from
        /// [`crate::llm::ProviderChunk::Usage`]).
        usage: Usage,
    },

    /// A prompt turn failed permanently (e.g. a provider error after retries) and was
    /// rolled back. Everything the turn appended to history — starting with the user
    /// prompt — has been discarded in memory; consumers that persist or mirror history
    /// (storage) must drop the same tail so a failed turn leaves no orphan to be replayed
    /// on reload or re-sent on the next request. Audit-only on the wire (the ACP bridge
    /// reports failure via the JSON-RPC error, not this event).
    TurnAborted,

    // ---------- Assistant output (pushed to wire) ----------
    /// Incremental assistant text. Maps to ACP `SessionUpdate::AgentMessageChunk`.
    AssistantText { content: ContentBlock },

    /// Assistant thought chain delta. Maps to ACP `SessionUpdate::AgentThoughtChunk`.
    AssistantThought { content: ContentBlock },

    // ---------- Tool calls (pushed to wire) ----------
    /// A tool call start declaration.
    /// Maps to ACP `SessionUpdate::ToolCall` (status = Pending).
    ToolCallStarted {
        id: ToolCallId,
        name: String,
        fields: ToolCallUpdateFields,
    },

    /// Tool call progress delta.
    /// Maps to ACP `SessionUpdate::ToolCallUpdate`.
    ToolCallProgress {
        id: ToolCallId,
        fields: ToolCallUpdateFields,
    },

    /// Tool call finished (success/failure is indicated by `fields.status`).
    /// Maps to ACP `SessionUpdate::ToolCallUpdate` (with a terminal status).
    ToolCallFinished {
        id: ToolCallId,
        fields: ToolCallUpdateFields,
    },

    // ---------- Permission decisions (partially pushed to wire) ----------
    /// The sandbox policy makes a decision about a tool call. `Ask` triggers the ACP
    /// `session/request_permission`; `Allow` / `Deny` are only audited and not sent over
    /// the wire.
    PolicyDecision {
        id: ToolCallId,
        decision: PolicyDecision,
    },

    /// User's response to a [`PolicyDecision::Ask`]. Audit-only, not sent on the wire.
    PermissionResolved {
        id: ToolCallId,
        outcome: PermissionResolution,
    },

    // Main loop orchestration (not sent over the wire; storage / tracing only)
    /// A single LLM provider call has started.
    LlmCallStarted {
        model: String,
        /// The attempt number (1-based). Retries are driven by the main loop.
        attempt: u32,
        /// A snapshot of the request sent to the provider (system message + full message
        /// history).
        ///
        /// Used by observability to reconstruct the generation's `input` as a standard
        /// chat message array (including the system message). Not sent over the wire;
        /// storage currently ignores this field.
        ///
        /// Wrapped in `Arc`: when events are fanned out to subscribers via
        /// [`crate::session::EventEmitter`], each subscriber clones the event. With long
        /// contexts, deep-copying the entire message history repeatedly is expensive. The
        /// snapshot is read-only once inside the event, so `Arc` reduces clone to a
        /// reference-count increment.
        /// `#[serde(skip)]`: the serde derive on `AgentEvent` is not currently used, and
        /// we prefer not to enable serde's `rc` feature for it—on deserialization this
        /// field takes the default empty snapshot.
        #[serde(skip)]
        request: Arc<LlmRequestSnapshot>,
    },

    /// A single LLM provider call has finished. `error` being `Some` indicates failure
    /// (the retry hint determines whether to proceed to the next attempt).
    LlmCallFinished {
        model: String,
        attempt: u32,
        usage: Usage,
        /// Error description on failure (the full error object is not stored here — it
        /// goes into tracing).
        error: Option<String>,
    },

    /// The main loop compressed / truncated the history.
    ContextCompressed {
        tokens_before: u64,
        tokens_after: u64,
    },

    /// The main loop performed a **micro-compaction**: it cleaned up oversized
    /// `tool_result` bodies from older turns (without calling the LLM or deleting
    /// messages). `cleared` is the number of `tool_result` entries actually cleaned. This
    /// is distinguished from [`Self::ContextCompressed`] so that observability and the UI
    /// can display them separately.
    ContextMicrocompacted {
        tokens_before: u64,
        tokens_after: u64,
        cleared: usize,
    },

    // ---------- subagent nesting (observability only) ----------
    /// A **leaf** event produced inside a `spawn_agent` sub-agent turn, bridged from the
    /// sub-turn's isolated event stream into the parent session's event stream.
    ///
    /// Design intent: the sub-agent runs in a fresh, isolated context (its own
    /// [`crate::session::EventEmitter`]), and the parent agent **cannot see** its
    /// intermediate steps — this is the isolation contract of `spawn_agent`. However,
    /// observability (langfuse) wants to display the sub-turn's LLM calls / tool calls
    /// nested under the parent's `spawn_agent` tool call span. So `spawn_agent` attaches
    /// a bridging subscriber to the sub-emitter, wrapping each sub-event as this variant
    /// and forwarding it to the parent emitter.
    ///
    /// ## Flattening (supports recursive subagents)
    ///
    /// `inner` **is always a leaf event** (never another `Subagent`). Nesting depth is
    /// expressed by the **ancestor chain** [`Self::Subagent::ancestor_path`], not by
    /// nested `Box` wrappers: the chain lists ids from the top-level `spawn_agent` tool
    /// call down to the current layer. Each bridging layer **prepends** its own
    /// `parent_tool_call_id` to the chain head, leaving the leaf `inner` unchanged (see
    /// the bridge closure in `spawn_agent.rs`). The projector uses the full chain to
    /// locate the parent mount point — the chain is globally unique, naturally avoids
    /// `ToolCallId` collisions across sub-sessions, and the projector does not need to
    /// recursively unwrap.
    ///
    /// **Consumption contract**: only the langfuse projector processes this (emitting
    /// nested generations/spans under the parent tool span). All other consumers
    /// (`defect-storage` persistence, `defect-acp` wire projection, REPL rendering)
    /// **ignore** it — the isolation contract remains unchanged for them.
    Subagent {
        /// Ancestor chain of `ToolCallId`s from the top-level `spawn_agent` tool call
        /// down to the current subagent layer.
        /// The head is the top-level `spawn_agent` (directly attached to the parent turn
        /// trace), and the tail is the `spawn_agent` that initiated this leaf event.
        /// Depth equals `ancestor_path.len()`.
        ancestor_path: Vec<ToolCallId>,
        /// The profile name of the subagent that initiated this leaf event (e.g.
        /// `weebs-in`), used for naming / metadata of nested spans.
        agent_type: String,
        /// The bridged child turn **leaf** event (never another `Subagent`). `Box`
        /// prevents the enum from growing unbounded due to self-reference.
        inner: Box<AgentEvent>,
    },
}

/// A snapshot of an LLM call request, containing only the fields needed for observability
/// to reconstruct the generation `input` (system prompt + full message history). Does not
/// include tools, sampling parameters, etc.
///
/// Defined separately rather than embedded in `CompletionRequest` to avoid making
/// `AgentEvent` depend on the full request type, and to keep the snapshot minimal and
/// serialization-stable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmRequestSnapshot {
    /// The system prompt, if any. Observability reconstructs it as a single
    /// `{role:"system"}` entry.
    pub system: Option<Arc<str>>,
    /// The full message history sent to the provider.
    pub messages: Vec<Message>,
}

/// The user's response to an [`Ask`](crate::policy::Ask).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionResolution {
    /// The user selected an option; `option_id` is provided by the ACP
    /// `PermissionOption`.
    Selected { option_id: PermissionOptionId },
    /// The user cancelled the turn before making a selection.
    Cancelled,
}
