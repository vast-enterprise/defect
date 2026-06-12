//! Translation of `AgentEvent` into Langfuse ingestion events.
//!
//! [`TraceProjector`] is a **stateful, per-session** projector (isomorphic to
//! `defect-storage`'s `RecordProjector`). The main loop calls [`TraceProjector::project`]
//! for each incoming [`AgentEvent`] and returns 0..N [`IngestionEvent`]s to the reporter.
//!
//! ## Hierarchy (one trace per turn)
//!
//! ```text
//! trace (turn)
//! └── step (span)                 One per turn: one llm_call + the tools it triggers
//!     ├── llm_call (generation)
//!     └── tool (span)             Sibling of llm_call under the same step
//!         └── (spawn_agent) → subagent (span)
//!             └── step (span)     Recursive isomorphic structure inside subagent
//!                 ├── llm_call
//!                 └── tool / spawn_agent → subagent → ...
//! ```
//!
//! Key point: the **step span** is the container for each turn. Both `llm_call`
//! (generation) and the tools triggered in that turn hang under it as siblings. This
//! makes the generation duration reflect **pure LLM call** time (it ends at
//! `LlmCallFinished`, no longer delayed to the next turn or wrapping tool execution
//! time); tool duration lives inside the step.
//!
//! ## Recursive subagent (flattened ancestor_path)
//!
//! [`AgentEvent::Subagent`] carries an `ancestor_path` (a chain of `ToolCallId`s from the
//! top-level `spawn_agent` tool call to the current layer), and `inner` is always a leaf
//! event. The projector **deterministically** derives all span ids from this chain, so no
//! per-layer anchoring is needed; arbitrary depth is handled uniformly. Each subagent
//! layer is an independent **scope** (sharing the same step/gen/tool projection logic as
//! the top-level turn).
//!
//! ## ID strategy
//!
//! - **traceId**: Generated once with `Uuid::new_v4()` at `TurnStarted`, reused within
//!   the turn. **Must not** use an auto-incrementing `{session}-turn-{seq}` — resuming
//!   would cause id collisions.
//! - **scope prefix**: Top-level = `{trace}`; subagent path `[A,B]` =
//!   `{trace}-sub-A-sub-B`. The subagent span's id **is** its scope prefix.
//! - **step / generation / tool span id**: Derived from the scope prefix + sequence
//!   number / `ToolCallId`, globally unique and deterministic — the parent of a subagent
//!   span (the tool span that spawned it) is computed directly from the path.
//! - **anchor**: The only state that needs to be stored is the **top-level `spawn_agent`
//!   tool call id → trace_id** (trace_id is random and not derivable). All subagents in
//!   the same turn share that trace_id, so only the top-level anchor is needed.
//! - **envelope id**: One `Uuid::new_v4()` per ingestion event, for Langfuse
//!   deduplication.
//!
//! ## Timestamps
//!
//! `AgentEvent` carries no timestamps; the caller passes `now` (an RFC3339 string). The
//! projector does not read the clock itself, making it easier to test and deterministic.

use std::collections::HashMap;

use agent_client_protocol_schema::{
    ContentBlock, StopReason, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::event::{AgentEvent, LlmRequestSnapshot};
use defect_agent::llm::{Message, MessageContent, Role, Usage};

use super::model::{EventKind, IngestionEvent, ObservationBody, ObservationLevel, TraceBody};

/// Deployment environment label (written into the `environment` field of traces and
/// observations).
const DEFAULT_ENVIRONMENT: &str = "production";
/// The Langfuse trace name for each agent turn.
const TRACE_NAME: &str = "turn";
/// The container span name for each turn (one `llm_call` plus the tools it triggers).
const STEP_NAME: &str = "step";
/// The name of the Langfuse generation that corresponds to an LLM call.
const GENERATION_NAME: &str = "llm_call";
/// The wire-level name of the `spawn_agent` tool. The canonical source is
/// `defect_agent::tool::spawn_agent::SPAWN_AGENT_TOOL_NAME` (which is `pub(crate)` and
/// thus inaccessible across crates),
/// so this is a copy of the wire name — the projector uses it to anchor **top-level**
/// `spawn_agent` tool calls to a `trace_id`.
const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";
/// The name prefix for a subagent's own span (the layer separate from the tool span that
/// spawned it).
const SUBAGENT_SPAN_NAME: &str = "subagent";

/// Per-session projection state.
pub struct TraceProjector {
    session_id: String,
    /// Metadata for the current **top-level** turn; `None` when not inside a turn (before
    /// `TurnStarted` / after `TurnEnded`). Note that subagent events may still arrive (in
    /// the background) after the top-level turn ends, at which point `turn` is `None`,
    /// but the subagent scope remains alive and the trace_id is retrieved via
    /// [`Self::anchors`].
    turn: Option<TurnMeta>,
    /// Temporarily stores the user prompt text. The main loop emits `UserPromptCommitted`
    /// **before** `TurnStarted`, so when the prompt arrives the turn has not been created
    /// yet — it is stashed here and consumed when `TurnStarted` builds the turn.
    pending_input: Option<String>,
    /// Top-level `spawn_agent` tool call id → its owning trace_id. Subagent events look
    /// up the trace_id via `ancestor_path[0]` (trace_id is random and not derivable; all
    /// nested subagents in the same turn share this trace_id, so only the top-level hop
    /// is anchored). Cleared when the subagent (path length 1) ends.
    anchors: HashMap<String, String>,
    /// All active scopes: `scope prefix` → state. **Session-level** — top-level turn
    /// scopes (prefix = `trace_id`) coexist with subagent scopes (prefix =
    /// `{trace}-sub-...`). Subagent scopes may survive across turn boundaries
    /// (background), so they are not cleared with the turn; each is removed when its
    /// corresponding `TurnEnded` fires.
    scopes: HashMap<String, ScopeState>,
}

/// Metadata for the current top-level turn (trace-level; does not include step/gen/tool
/// projection state — those live in `scopes[trace_id]`, which is isomorphic to a subagent
/// scope).
struct TurnMeta {
    trace_id: String,
    /// The user prompt text, written into the trace input.
    input: Option<String>,
    /// The final assistant text for the entire turn (written into the trace output).
    final_output: String,
}

/// Projection state for a scope (top-level turn or a subagent layer) of steps,
/// generations, and tools.
///
/// Top-level and subagent scopes share this structure — this is the observability-side
/// manifestation of "a subagent is just an agent with a parent": the same step container,
/// generation, and tool span logic, differing only in the mount point (`step_parent`) and
/// id prefix (`prefix`).
struct ScopeState {
    /// ID prefix: top-level = `{trace}`; subagent = `{trace}-sub-A-sub-B`.
    /// For a subagent scope, `prefix` is also the subagent span's id.
    prefix: String,
    /// The parent observation for step spans in this scope: top-level = `None` (attached
    /// directly to the trace); subagent = `Some(subagent span id)` = `Some(prefix)`.
    step_parent: Option<String>,
    /// The currently active step span id (`None` = no `llm_call` yet).
    current_step_id: Option<String>,
    /// The sequence number of this step, used to derive the step id.
    step_seq: u32,
    /// The current in-progress generation within this step.
    current_gen: Option<PendingGeneration>,
    /// Tool call ID → assigned span ID (pairing Started/Finished events).
    tool_spans: HashMap<String, String>,
}

/// Accumulated state for an in-progress generation. Flushed into a single
/// generation-update when finalized by `LlmCallFinished`.
struct PendingGeneration {
    id: String,
    parent_step_id: String,
    model: String,
    /// Accumulated assistant reply text.
    output: String,
    /// Accumulated thinking text (stored in generation's `metadata.reasoning`, not in
    /// `output`).
    thinking: String,
    /// Token usage for this call (from `LlmCallFinished.usage`).
    usage: Usage,
    /// Error message (from `LlmCallFinished.error`).
    error: Option<String>,
}

impl ScopeState {
    fn new(prefix: String, step_parent: Option<String>) -> Self {
        Self {
            prefix,
            step_parent,
            current_step_id: None,
            step_seq: 0,
            current_gen: None,
            tool_spans: HashMap::new(),
        }
    }
}

impl TraceProjector {
    /// Creates a new per-session projector.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn: None,
            pending_input: None,
            anchors: HashMap::new(),
            scopes: HashMap::new(),
        }
    }

    /// Translates an event into 0..N ingestion events. `now` is an RFC3339 timestamp.
    /// `new_id` supplies a unique id (envelope id / trace id) — injected for
    /// deterministic testing.
    pub fn project(
        &mut self,
        event: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        match event {
            AgentEvent::TurnStarted => self.on_turn_started(now, new_id),
            AgentEvent::UserPromptCommitted { content } => {
                self.on_user_prompt(&content);
                Vec::new()
            }
            AgentEvent::LlmCallStarted {
                model,
                attempt,
                request,
            } => self.on_top_llm_started(model, attempt, request.as_ref(), now, new_id),
            AgentEvent::AssistantText { content } => {
                self.accumulate_top_text(&content);
                Vec::new()
            }
            AgentEvent::AssistantThought { content } => {
                self.accumulate_top_thinking(&content);
                Vec::new()
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                self.on_top_llm_finished(usage, error, now, new_id)
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                self.on_top_tool_started(id.to_string(), name, fields.raw_input, now, new_id)
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                self.on_top_tool_finished(&id.to_string(), &fields, now, new_id)
            }
            AgentEvent::ContextCompressed {
                tokens_before,
                tokens_after,
            } => self.on_context_compressed(tokens_before, tokens_after, None, now, new_id),
            AgentEvent::ContextMicrocompacted {
                tokens_before,
                tokens_after,
                cleared,
            } => {
                self.on_context_compressed(tokens_before, tokens_after, Some(cleared), now, new_id)
            }
            AgentEvent::TurnEnded { reason, usage } => {
                self.on_turn_ended(reason, usage, now, new_id)
            }
            AgentEvent::Subagent {
                ancestor_path,
                agent_type,
                inner,
            } => {
                let path: Vec<String> = ancestor_path.iter().map(ToString::to_string).collect();
                self.on_subagent(&path, agent_type, *inner, now, new_id)
            }
            // Do not report: progress increments and permission audits (not included in
            // langfuse for this release).
            AgentEvent::ToolCallProgress { .. }
            | AgentEvent::PolicyDecision { .. }
            | AgentEvent::PermissionResolved { .. } => Vec::new(),
            _ => Vec::new(),
        }
    }

    // ---- Top-level turn events ----

    fn on_turn_started(
        &mut self,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let trace_id = new_id();
        let input = self.pending_input.take();
        let body = TraceBody {
            id: trace_id.clone(),
            name: Some(TRACE_NAME.into()),
            session_id: Some(self.session_id.clone()),
            // Include input at trace creation so the UI can immediately display user
            // input without waiting for TurnEnded.
            input: input.clone().map(serde_json::Value::String),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        // Top-level scope: prefix is `trace_id`, steps are attached directly to the trace
        // (`step_parent` is `None`).
        self.scopes
            .insert(trace_id.clone(), ScopeState::new(trace_id.clone(), None));
        self.turn = Some(TurnMeta {
            trace_id: trace_id.clone(),
            input,
            final_output: String::new(),
        });
        vec![IngestionEvent::trace(
            new_id(),
            now.to_string(),
            EventKind::TraceCreate,
            &body,
        )]
    }

    fn on_user_prompt(&mut self, content: &[ContentBlock]) {
        let text = content_text(content);
        if !text.is_empty() {
            self.pending_input = Some(text);
        }
    }

    fn on_top_llm_started(
        &mut self,
        model: String,
        attempt: u32,
        request: &LlmRequestSnapshot,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_llm_started(scope, &trace_id, model, attempt, request, now, new_id)
    }

    fn accumulate_top_text(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(turn) = self.turn.as_mut()
        {
            turn.final_output.push_str(&text.text);
            let trace_id = turn.trace_id.clone();
            if let Some(scope) = self.scopes.get_mut(&trace_id)
                && let Some(pg) = scope.current_gen.as_mut()
            {
                pg.output.push_str(&text.text);
            }
        }
    }

    fn accumulate_top_thinking(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone())
            && let Some(scope) = self.scopes.get_mut(&trace_id)
            && let Some(pg) = scope.current_gen.as_mut()
        {
            pg.thinking.push_str(&text.text);
        }
    }

    fn on_top_llm_finished(
        &mut self,
        usage: Usage,
        error: Option<String>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        note_llm_finished(scope, usage, error);
        flush_generation(scope, &trace_id, now, new_id)
    }

    fn on_top_tool_started(
        &mut self,
        tool_call_id: String,
        name: String,
        raw_input: Option<serde_json::Value>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        // Top-level `spawn_agent` tool call: anchor the `trace_id` so that later subagent
        // events (including background and cross-turn) can retrieve it via
        // `ancestor_path[0]`.
        if name == SPAWN_AGENT_TOOL_NAME {
            self.anchors.insert(tool_call_id.clone(), trace_id.clone());
        }
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_tool_started(
            scope,
            &trace_id,
            &tool_call_id,
            name,
            raw_input,
            now,
            new_id,
        )
    }

    fn on_top_tool_finished(
        &mut self,
        tool_call_id: &str,
        fields: &ToolCallUpdateFields,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_tool_finished(scope, &trace_id, tool_call_id, fields, now, new_id)
    }

    /// When `cleared` is `Some`, a micro-compression (clearing `tool_result` without LLM
    /// involvement) is performed; when `None`, a full summary compression is done. Both
    /// produce structurally identical observations, differing only in `name`/`metadata`.
    /// Compression is a turn-level, cross-step operation (not part of any single
    /// `llm_call`), so it is attached directly to the trace (no parent).
    fn on_context_compressed(
        &mut self,
        tokens_before: u64,
        tokens_after: u64,
        cleared: Option<usize>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let mut meta = serde_json::Map::new();
        meta.insert("tokens_before".into(), tokens_before.into());
        meta.insert("tokens_after".into(), tokens_after.into());
        if let Some(cleared) = cleared {
            meta.insert("cleared_tool_results".into(), cleared.into());
        }
        let name = if cleared.is_some() {
            "context_microcompaction"
        } else {
            "context_compaction"
        };
        let body = ObservationBody {
            id: new_id(),
            trace_id,
            name: Some(name.into()),
            start_time: Some(now.to_string()),
            metadata: Some(serde_json::Value::Object(meta)),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            ..Default::default()
        };
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::EventCreate,
            &body,
        )]
    }

    fn on_turn_ended(
        &mut self,
        reason: StopReason,
        usage: Usage,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.take() else {
            return Vec::new();
        };
        let trace_id = turn.trace_id.clone();
        let mut events = Vec::new();
        // Finalize the top-level scope: flush any in-flight generation, close the current
        // step, then remove the scope.
        if let Some(mut scope) = self.scopes.remove(&trace_id) {
            events.extend(flush_generation(&mut scope, &trace_id, now, new_id));
            events.extend(close_current_step(&mut scope, &trace_id, now, new_id));
        }

        let mut meta = serde_json::Map::new();
        meta.insert(
            "stop_reason".into(),
            serde_json::to_value(reason).unwrap_or(serde_json::Value::Null),
        );
        if let Some(details) = usage_to_details(&usage) {
            meta.insert("usage".into(), serde_json::Value::Object(details));
        }
        let body = TraceBody {
            id: trace_id,
            name: Some(TRACE_NAME.into()),
            session_id: Some(self.session_id.clone()),
            input: turn.input.map(serde_json::Value::String),
            output: (!turn.final_output.is_empty())
                .then_some(serde_json::Value::String(turn.final_output)),
            metadata: Some(serde_json::Value::Object(meta)),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        events.push(IngestionEvent::trace(
            new_id(),
            now.to_string(),
            // A second send with the same trace_id acts as an update (merging
            // endTime/output/metadata).
            EventKind::TraceCreate,
            &body,
        ));
        events
    }

    // ---- subagent events (arbitrary depth, homogeneous) ----

    /// Processes a **leaf** event of a subagent child turn (the `inner` after unwrapping
    /// `AgentEvent::Subagent`).
    ///
    /// `path` is the chain of `ToolCallId`s from the top-level `spawn_agent` tool call
    /// down to the current layer. The `trace_id` is retrieved via `anchors[path[0]]`; the
    /// scope prefix and the parent of the subagent span (the tool span that spawned it)
    /// are deterministically derived from `path`. On first encounter with a given `path`,
    /// lazily create its subagent span and scope; subsequent `inner` events are
    /// dispatched into that scope (reusing the same step/gen/tool logic as the top
    /// level).
    ///
    /// If no top-level anchor is found (the `spawn_agent` tool call was never seen), the
    /// event is dropped — no orphans are created.
    fn on_subagent(
        &mut self,
        path: &[String],
        agent_type: String,
        inner: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(first) = path.first() else {
            return Vec::new();
        };
        let Some(trace_id) = self.anchors.get(first).cloned() else {
            // No top-level anchor: this `spawn_agent` tool call has never been seen
            // before — drop it, don't create an orphan.
            return Vec::new();
        };
        let prefix = scope_prefix(&trace_id, path);

        let mut events = Vec::new();
        // First time seeing this path: lazily create a dedicated subagent span (parent =
        // the tool span that initiated it, derived from path).
        if !self.scopes.contains_key(&prefix) {
            let parent_tool = parent_tool_span_id(&trace_id, path);
            let mut meta = serde_json::Map::new();
            meta.insert("agent_type".into(), agent_type.clone().into());
            let body = ObservationBody {
                id: prefix.clone(),
                trace_id: trace_id.clone(),
                parent_observation_id: Some(parent_tool),
                name: Some(format!("{SUBAGENT_SPAN_NAME}:{agent_type}")),
                start_time: Some(now.to_string()),
                metadata: Some(serde_json::Value::Object(meta)),
                environment: Some(DEFAULT_ENVIRONMENT.into()),
                ..Default::default()
            };
            events.push(IngestionEvent::observation(
                new_id(),
                now.to_string(),
                EventKind::SpanCreate,
                &body,
            ));
            // Subagent scope: the step is attached under this subagent span (= prefix).
            self.scopes.insert(
                prefix.clone(),
                ScopeState::new(prefix.clone(), Some(prefix.clone())),
            );
        }
        let scope = self
            .scopes
            .get_mut(&prefix)
            .expect("subagent scope just ensured");

        match inner {
            AgentEvent::LlmCallStarted {
                model,
                attempt,
                request,
            } => {
                events.extend(scope_llm_started(
                    scope,
                    &trace_id,
                    model,
                    attempt,
                    request.as_ref(),
                    now,
                    new_id,
                ));
            }
            AgentEvent::AssistantText { content } => {
                if let (ContentBlock::Text(text), Some(pg)) = (&content, scope.current_gen.as_mut())
                {
                    pg.output.push_str(&text.text);
                }
            }
            AgentEvent::AssistantThought { content } => {
                if let (ContentBlock::Text(text), Some(pg)) = (&content, scope.current_gen.as_mut())
                {
                    pg.thinking.push_str(&text.text);
                }
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                note_llm_finished(scope, usage, error);
                events.extend(flush_generation(scope, &trace_id, now, new_id));
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                events.extend(scope_tool_started(
                    scope,
                    &trace_id,
                    &id.to_string(),
                    name,
                    fields.raw_input,
                    now,
                    new_id,
                ));
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                events.extend(scope_tool_finished(
                    scope,
                    &trace_id,
                    &id.to_string(),
                    &fields,
                    now,
                    new_id,
                ));
            }
            // Sub-turn ended: finalize the in-progress generation, close the current
            // step, close the subagent span, and clear the session-level scope; also
            // clear the anchor for the top-level hop (path length 1).
            AgentEvent::TurnEnded { .. } => {
                events.extend(flush_generation(scope, &trace_id, now, new_id));
                events.extend(close_current_step(scope, &trace_id, now, new_id));
                let subagent_span_id = scope.prefix.clone();
                let body = ObservationBody {
                    id: subagent_span_id,
                    trace_id: trace_id.clone(),
                    end_time: Some(now.to_string()),
                    ..Default::default()
                };
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::SpanUpdate,
                    &body,
                ));
                self.scopes.remove(&prefix);
                if path.len() == 1 {
                    self.anchors.remove(first);
                }
            }
            // Remaining sub-turn events (TurnStarted, UserPromptCommitted, progress,
            // audit) are not reported individually.
            _ => {}
        }
        events
    }
}

// ---- scope generic projection (shared by top-level turn and subagent) ----

/// LLM call started: finalize the previous step (if any) → open a new step → create a
/// generation under the new step.
fn scope_llm_started(
    scope: &mut ScopeState,
    trace_id: &str,
    model: String,
    attempt: u32,
    request: &LlmRequestSnapshot,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    // Defensive: the previous generation should already have been finalized by its
    // `LlmCallFinished` (generation duration = pure LLM time); if it is still active,
    // flush it first to ensure `create` precedes `update`.
    let mut events = flush_generation(scope, trace_id, now, new_id);
    // Close the previous step (which includes the last llm_call and the tools triggered
    // in that round).
    events.extend(close_current_step(scope, trace_id, now, new_id));

    // Start a new step.
    scope.step_seq += 1;
    let step_id = format!("{}-step-{}", scope.prefix, scope.step_seq);
    scope.current_step_id = Some(step_id.clone());
    let step_body = ObservationBody {
        id: step_id.clone(),
        trace_id: trace_id.to_string(),
        parent_observation_id: scope.step_parent.clone(),
        name: Some(STEP_NAME.into()),
        start_time: Some(now.to_string()),
        environment: Some(DEFAULT_ENVIRONMENT.into()),
        ..Default::default()
    };
    events.push(IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanCreate,
        &step_body,
    ));

    // Attach the generation under the new step.
    let gen_id = format!("{step_id}-gen");
    scope.current_gen = Some(PendingGeneration {
        id: gen_id.clone(),
        parent_step_id: step_id.clone(),
        model: model.clone(),
        output: String::new(),
        thinking: String::new(),
        usage: Usage::default(),
        error: None,
    });
    let mut meta = serde_json::Map::new();
    meta.insert("attempt".into(), attempt.into());
    let gen_body = ObservationBody {
        id: gen_id,
        trace_id: trace_id.to_string(),
        parent_observation_id: Some(step_id),
        name: Some(GENERATION_NAME.into()),
        model: Some(model),
        start_time: Some(now.to_string()),
        // input is the standard chat messages array, with system as the first entry
        // {role:"system"}.
        input: Some(request_to_input(request)),
        metadata: Some(serde_json::Value::Object(meta)),
        environment: Some(DEFAULT_ENVIRONMENT.into()),
        ..Default::default()
    };
    events.push(IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::GenerationCreate,
        &gen_body,
    ));
    events
}

/// Record usage/error from `LlmCallFinished` into the current generation (written out at
/// finalization).
fn note_llm_finished(scope: &mut ScopeState, usage: Usage, error: Option<String>) {
    if let Some(pg) = scope.current_gen.as_mut() {
        pg.usage = usage;
        if error.is_some() {
            pg.error = error;
        }
    }
}

/// Finalize the current generation: output, thinking, usage, and endTime are written into
/// a generation-update event. No-op if there is no ongoing generation. Called from
/// `LlmCallFinished` (on the success path, after the stream has been drained and
/// output/thinking are fully collected) — the generation duration covers only the pure
/// LLM call, excluding tool execution.
fn flush_generation(
    scope: &mut ScopeState,
    trace_id: &str,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let Some(pg) = scope.current_gen.take() else {
        return Vec::new();
    };
    let mut meta = serde_json::Map::new();
    if !pg.thinking.is_empty() {
        // There is no dedicated ingestion field for thinking/reasoning; store it in
        // metadata to avoid polluting output.
        meta.insert("reasoning".into(), serde_json::Value::String(pg.thinking));
    }
    let body = ObservationBody {
        id: pg.id,
        trace_id: trace_id.to_string(),
        parent_observation_id: Some(pg.parent_step_id),
        name: Some(GENERATION_NAME.into()),
        model: Some(pg.model),
        end_time: Some(now.to_string()),
        output: (!pg.output.is_empty()).then_some(serde_json::Value::String(pg.output)),
        usage_details: usage_to_details(&pg.usage),
        metadata: (!meta.is_empty()).then_some(serde_json::Value::Object(meta)),
        level: pg.error.as_ref().map(|_| ObservationLevel::Error),
        status_message: pg.error,
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::GenerationUpdate,
        &body,
    )]
}

/// Close the current step span (write `end_time`). No-op if no step is in progress.
fn close_current_step(
    scope: &mut ScopeState,
    trace_id: &str,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let Some(step_id) = scope.current_step_id.take() else {
        return Vec::new();
    };
    let body = ObservationBody {
        id: step_id,
        trace_id: trace_id.to_string(),
        end_time: Some(now.to_string()),
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanUpdate,
        &body,
    )]
}

/// Tool call start → span-create, attached under the current step (sibling to llm_call).
fn scope_tool_started(
    scope: &mut ScopeState,
    trace_id: &str,
    tool_call_id: &str,
    name: String,
    raw_input: Option<serde_json::Value>,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let span_id = format!("{}-tool-{}", scope.prefix, tool_call_id);
    scope
        .tool_spans
        .insert(tool_call_id.to_string(), span_id.clone());
    let body = ObservationBody {
        id: span_id,
        trace_id: trace_id.to_string(),
        // Tool calls are always attached to the current step; in theory, a tool call
        // always follows an `llm_call`, so the step must exist. Defensively allow `None`
        // (out-of-order / no step) — fall back to attaching directly to the trace.
        parent_observation_id: scope.current_step_id.clone(),
        name: Some(name),
        start_time: Some(now.to_string()),
        input: raw_input,
        environment: Some(DEFAULT_ENVIRONMENT.into()),
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanCreate,
        &body,
    )]
}

/// Tool call finished → span-update (endTime + output + level).
fn scope_tool_finished(
    scope: &mut ScopeState,
    trace_id: &str,
    tool_call_id: &str,
    fields: &ToolCallUpdateFields,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    // Retrieve the span id assigned at Started; if missing (out of order), derive a new
    // one.
    let span_id = scope
        .tool_spans
        .remove(tool_call_id)
        .unwrap_or_else(|| format!("{}-tool-{}", scope.prefix, tool_call_id));
    let failed = matches!(fields.status, Some(ToolCallStatus::Failed));
    let body = ObservationBody {
        id: span_id,
        trace_id: trace_id.to_string(),
        end_time: Some(now.to_string()),
        output: fields.raw_output.clone(),
        level: failed.then_some(ObservationLevel::Error),
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanUpdate,
        &body,
    )]
}

// ---- id derivation ----

/// The id prefix for a scope: top-level (empty path) = `{trace}`; subagent path `[A,B]` =
/// `{trace}-sub-A-sub-B`. The subagent scope's prefix is also the id of its subagent
/// span.
fn scope_prefix(trace_id: &str, path: &[String]) -> String {
    let mut s = trace_id.to_string();
    for id in path {
        s.push_str("-sub-");
        s.push_str(id);
    }
    s
}

/// The parent observation id for a subagent span (the `spawn_agent` tool span that
/// initiated it).
/// Format: `{parent scope prefix}-tool-{subagent's initiating tool_call_id}`. `path` is
/// non-empty.
fn parent_tool_span_id(trace_id: &str, path: &[String]) -> String {
    let (last, parent_path) = path.split_last().expect("path is non-empty");
    format!("{}-tool-{}", scope_prefix(trace_id, parent_path), last)
}

// ---- data conversion helpers ----

/// Converts a [`Usage`] into a langfuse `usageDetails` map. Returns `None` (no report)
/// when all fields are `None`.
fn usage_to_details(usage: &Usage) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut map = serde_json::Map::new();
    if let Some(v) = usage.input_tokens {
        map.insert("input".into(), v.into());
    }
    if let Some(v) = usage.output_tokens {
        map.insert("output".into(), v.into());
    }
    if let Some(v) = usage.cache_read_input_tokens {
        map.insert("cache_read_input_tokens".into(), v.into());
    }
    if let Some(v) = usage.cache_creation_input_tokens {
        map.insert("cache_creation_input_tokens".into(), v.into());
    }
    (!map.is_empty()).then_some(map)
}

/// Concatenates text from `ContentBlock` items in a list, ignoring non-text blocks.
fn content_text(content: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in content {
        if let ContentBlock::Text(text) = block {
            out.push_str(&text.text);
        }
    }
    out
}

/// Reconstruct the request snapshot into the standard `input` for a Langfuse generation:
/// an array of chat messages.
///
/// The system prompt becomes the first entry `{role:"system"}`, followed by the full
/// message history.
/// This matches the Langfuse SDK's standard format (see observation-types docs) — the UI
/// renders it as conversation bubbles and supports playground replay.
fn request_to_input(request: &LlmRequestSnapshot) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = &request.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    for msg in &request.messages {
        messages.push(message_to_value(msg));
    }
    serde_json::Value::Array(messages)
}

/// Converts a single [`Message`] to a langfuse `{role, content}` object. The content
/// field collapses multimodal blocks into text or structured fragments (langfuse input
/// accepts arbitrary JSON, and the UI renders it as best it can).
fn message_to_value(msg: &Message) -> serde_json::Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let parts: Vec<serde_json::Value> = msg.content.iter().map(content_to_value).collect();
    // Use a plain string for single text content (most common and readable); otherwise
    // use an array.
    let content = match parts.as_slice() {
        [serde_json::Value::String(s)] => serde_json::Value::String(s.clone()),
        _ => serde_json::Value::Array(parts),
    };
    serde_json::json!({ "role": role, "content": content })
}

/// Converts [`MessageContent`] to a Langfuse content fragment.
fn content_to_value(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text { text } => serde_json::Value::String(text.clone()),
        MessageContent::Thinking { text, .. } => {
            serde_json::json!({ "type": "thinking", "text": text })
        }
        MessageContent::ToolUse { id, name, args } => {
            serde_json::json!({ "type": "tool_use", "id": id, "name": name, "input": args })
        }
        MessageContent::ToolResult {
            tool_use_id,
            is_error,
            ..
        } => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }),
        MessageContent::Image { mime, .. } => {
            serde_json::json!({ "type": "image", "mime": mime })
        }
        MessageContent::ProviderActivity {
            provider_id, kind, ..
        } => serde_json::json!({
            "type": "provider_activity",
            "provider_id": provider_id,
            "kind": format!("{kind:?}"),
        }),
    }
}
