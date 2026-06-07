//! Hook step-context: typestate + envelope.

//! ## One-liner
//!
//! Each mount point gets its own dedicated **step type** (typestate). The same step state
//! is consumed by two kinds of hooks — internal Rust hooks work directly with the strong
//! type and mutate fields; user-configured hooks observe the world through the JSON
//! envelope produced by [`HookStep::to_envelope`] and apply output back via
//! [`HookStep::apply_verdict`]. **Equal capability, identical surface** — only the medium
//! of expression differs.
//!
//! ## Two axioms
//!
//! 1. **Typestate**: Instead of one large enum with a variant field, each mount point is
//!    a concrete struct. The surface is locked at compile time; `Option` presence/absence
//!    encodes "already produced / will produce" — filling a "will produce" `Option` means
//!    short-circuit.
//! 2. **Call vs mutation asymmetry**: Call-type steps (Generate / ToolApply / Permission)
//!    fill an `Option` to skip; mutation-type steps (Compact / Ingest) degrade to
//!    veto/rewrite, without the "fill Option" path.
//!
//! ## Scope (Step 1)
//!
//! This module delivers **types + envelope + unit tests**, with **no mount points wired
//! in** (call-site integration is a follow-up PR). Currently implements the base
//! infrastructure ([`HookControl`] / [`HookStep`] / envelope conventions) plus 3
//! representative steps: [`BeforeTurnEnd`] (fork control), [`BeforeToolApply`] (call-type
//! short-circuit), [`AfterGenerate`] (observation). The remaining 10 steps are mechanical
//! fill-ins of the same shape.

use agent_client_protocol_schema::{ContentBlock, StopReason as AcpStopReason};
use serde_json::{Value, json};

use crate::llm::{ToolResultBody, Usage};
use crate::tool::SafetyClass;

/// The single source of truth for all mount-point `event_name`s (snake_case) — used by
/// the config layer for event-name validation and by the CLI for bucket assembly.
///
/// Order is irrelevant; to add a new step, append a line here and the config layer will
/// automatically recognize the new event name (no changes to the config crate are
/// needed).
pub const ALL_EVENT_NAMES: &[&str] = &[
    "after_session_enter",
    "after_turn_enter",
    "before_ingest",
    "after_ingest",
    "before_compact",
    "after_compact",
    "before_generate",
    "after_generate",
    "before_permission",
    "after_permission",
    "before_tool_apply",
    "after_tool_apply",
    "after_tool_batch",
    "before_turn_end",
];

/// Whether an event name is a known mount point. The config layer uses this to fail-fast
/// on misspelled event keys.
#[must_use]
pub fn is_known_event(name: &str) -> bool {
    ALL_EVENT_NAMES.contains(&name)
}

// Control Flow

/// A hook's instruction for **control flow** (axis two). Data injection (axis one) goes
/// through the step's `&mut` fields, not here.
///
/// Which variants are meaningful depends on the hook point: `Break` is available at any
/// step; `Continue` only at [`BeforeTurnEnd`]; `Skip` only at `before Compact`. The
/// engine downgrades out-of-place variants with a warning (see the validation in
/// `apply_verdict`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HookControl {
    /// No intervention in control flow — the step proceeds normally with any data changes
    /// already made on `ctx`. Corresponds to envelope `control: null`.
    #[default]
    Proceed,
    /// End the current turn with a final stop reason. Usable from any step.
    Break { reason: AcpStopReason },
    /// Does not end the turn; instead, loops back to the top of the cycle for another
    /// round. Only meaningful in [`BeforeTurnEnd`] (and must be injected beforehand; see
    /// design §4).
    Continue,
    /// Skip the actual call for this step. Only meaningful for `before Compact` (veto
    /// compaction).
    Skip,
}

/// Errors when parsing an envelope verdict.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum VerdictError {
    #[error("hook verdict `control` is not a known directive: {0:?}")]
    UnknownControl(String),

    #[error("hook verdict field `{field}` is malformed: {reason}")]
    Malformed { field: &'static str, reason: String },
}

// ---------------------------------------------------------------------------
// HookStep trait
// ---------------------------------------------------------------------------

/// The step state for a mount point. Consumed by two kinds of hooks:
/// - Internal Rust hook: takes `&mut Self` directly, mutates fields to inject data, and
///   returns a [`HookControl`] on its own.
/// - User-configured hook: [`Self::to_envelope`] produces JSON fed to stdin/templates;
///   the handler outputs JSON, which [`Self::apply_verdict`] applies back to the step
///   (mutating data) and parses into a [`HookControl`].
pub trait HookStep: Send {
    /// Event name (snake_case). Used in envelope headers and matchers.
    fn event_name(&self) -> &'static str;

    /// Projects the step into an **input envelope** — fed to command stdin / prompt
    /// templates. Contains a common header plus step-specific fields.
    fn to_envelope(&self) -> Value;

    /// Apply the handler's output verdict (JSON) back to this step: parse the common
    /// `control` / `additional_context` fields, then handle the step-specific "fill
    /// output" fields. Returns a control directive.
    ///
    /// # Errors
    ///
    /// Returns [`VerdictError`] if the verdict's `control` is an unknown value or the
    /// step-specific fields are malformed.
    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError>;
}

// Common envelope conventions

/// Parse the generic `control` field from a verdict. `null` / absent →
/// [`HookControl::Proceed`].
///
/// `break` may carry a `stop_reason` (default `end_turn`). Validation is the caller's
/// responsibility (which step allows which control).
fn parse_control(verdict: &Value) -> Result<HookControl, VerdictError> {
    // By default, `veto` is interpreted as `Break` (the veto semantics for most steps).
    // Turn-end and compact steps override this by calling `parse_control_veto` with their
    // own semantics.
    parse_control_veto(
        verdict,
        HookControl::Break {
            reason: AcpStopReason::EndTurn,
        },
    )
}

/// Like [`parse_control`], but interprets the abstract `"veto"` control (produced by
/// command hook exit 2) as `veto_as` — letting each step translate the veto according to
/// its own semantics (turn-end → Continue, compact → Skip, everything else → Break).
fn parse_control_veto(verdict: &Value, veto_as: HookControl) -> Result<HookControl, VerdictError> {
    let Some(ctrl) = verdict.get("control") else {
        return Ok(HookControl::Proceed);
    };
    match ctrl {
        Value::Null => Ok(HookControl::Proceed),
        Value::String(s) => match s.as_str() {
            "proceed" => Ok(HookControl::Proceed),
            "continue" => Ok(HookControl::Continue),
            "skip" => Ok(HookControl::Skip),
            "veto" => Ok(veto_as),
            "break" => {
                let reason = verdict
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .map_or(AcpStopReason::EndTurn, parse_stop_reason);
                Ok(HookControl::Break { reason })
            }
            other => Err(VerdictError::UnknownControl(other.to_string())),
        },
        other => Err(VerdictError::UnknownControl(other.to_string())),
    }
}

/// Parse the `additional_context` field of a verdict: accepts an array of strings (the
/// most natural form for a user hook), each converted into a text [`ContentBlock`].
/// Defaults to empty.
fn parse_additional_context(verdict: &Value) -> Result<Vec<ContentBlock>, VerdictError> {
    let Some(v) = verdict.get("additional_context") else {
        return Ok(Vec::new());
    };
    match v {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ContentBlock::from)
                    .ok_or_else(|| VerdictError::Malformed {
                        field: "additional_context",
                        reason: "each entry must be a string".to_string(),
                    })
            })
            .collect(),
        _ => Err(VerdictError::Malformed {
            field: "additional_context",
            reason: "must be an array of strings".to_string(),
        }),
    }
}

/// Returns the snake_case string representation of [`AcpStopReason`] for the envelope.
fn stop_reason_str(reason: AcpStopReason) -> &'static str {
    match reason {
        AcpStopReason::EndTurn => "end_turn",
        AcpStopReason::MaxTokens => "max_tokens",
        AcpStopReason::MaxTurnRequests => "max_turn_requests",
        AcpStopReason::Refusal => "refusal",
        AcpStopReason::Cancelled => "cancelled",
        _ => "end_turn",
    }
}

/// [`ToolResultBody`] → envelope JSON. Text/Json are passed through directly; multimodal
/// Content degrades to a text summary (image blocks are marked as placeholders) to keep
/// the hook envelope compact and readable.
fn tool_result_body_to_json(body: &ToolResultBody) -> Value {
    match body {
        ToolResultBody::Text { text } => Value::String(text.clone()),
        ToolResultBody::Json { value } => value.clone(),
        ToolResultBody::Content { blocks } => {
            use crate::llm::ToolResultContent;
            let text: String = blocks
                .iter()
                .map(|b| match b {
                    ToolResultContent::Text { text } => text.clone(),
                    ToolResultContent::Image { mime, .. } => format!("[image: {mime}]"),
                })
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
    }
}

/// Converts [`SafetyClass`] to a snake_case string for envelopes, symmetric with the
/// engine-side `parse_safety`.
fn safety_str(s: SafetyClass) -> &'static str {
    match s {
        SafetyClass::ReadOnly => "read_only",
        SafetyClass::Mutating => "mutating",
        SafetyClass::Destructive => "destructive",
        SafetyClass::Network => "network",
    }
}

/// Parses a snake_case string into an [`AcpStopReason`]; unknown values fall back to
/// `EndTurn`.
fn parse_stop_reason(s: &str) -> AcpStopReason {
    match s {
        "max_tokens" => AcpStopReason::MaxTokens,
        "max_turn_requests" => AcpStopReason::MaxTurnRequests,
        "refusal" => AcpStopReason::Refusal,
        "cancelled" => AcpStopReason::Cancelled,
        _ => AcpStopReason::EndTurn,
    }
}

// Step 1: before turn-end (control branch point, default Break)

/// `before turn-end`: the turn's only voluntary exit point. **Defaults to `Break`** — "do
/// nothing" = let it stop.
///
/// A hook returning [`HookControl::Continue`] extends the turn: it injects
/// [`Self::feedback`] into history (appended as a user message when committed), does not
/// end the turn, and loops back to the top for another round. `Continue` only takes
/// effect when [`Self::voluntary`] is true — involuntary stops (Refusal / MaxTokens /
/// Cancelled / MaxTurnRequests) ignore it; otherwise the hook could bypass the request
/// cap and extend indefinitely.
#[derive(Debug, Clone)]
pub struct BeforeTurnEnd {
    /// The reason this turn stopped.
    pub stop_reason: AcpStopReason,
    /// How many times this turn has been extended by a hook (the hook decides when to
    /// stop; a hard cap in the loop provides a safety net).
    pub continues_so_far: u32,
    /// Whether the stop is voluntary (LLM said EndTurn or returned empty tool_use).
    /// `Continue` only takes effect when voluntary.
    pub voluntary: bool,
    /// Feedback to inject into history when continuing the turn. `apply_verdict` fills
    /// this from the verdict's `additional_context`; internal Rust hooks push directly.
    /// On finalization, the loop appends it as a user message.
    pub feedback: Vec<ContentBlock>,
}

impl HookStep for BeforeTurnEnd {
    fn event_name(&self) -> &'static str {
        "before_turn_end"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "stop_reason": stop_reason_str(self.stop_reason),
            "continues_so_far": self.continues_so_far,
            "voluntary": self.voluntary,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // At turn-end, "veto" means Continue: command hook exit 2 here means "don't
        // stop".
        let control = parse_control_veto(verdict, HookControl::Continue)?;
        let ctx = parse_additional_context(verdict)?;
        // At turn-end, `additional_context` is the keep-alive feedback.
        self.feedback.extend(ctx);
        Ok(control)
    }
}

// ----------------------------------------------------------------------------
// Represents step 2: before ToolApply (call-type, short-circuit = fill result)
// ----------------------------------------------------------------------------

/// A synthetic tool result produced by a hook — setting [`BeforeToolApply::result`]
/// effectively "intercepts" the tool.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticToolResult {
    pub body: ToolResultBody,
    pub is_error: bool,
}

/// Entry point for call-type transformations, invoked before each `ToolApply`.
///
/// Two orthogonal intervention axes:
/// - **Modify args** (data axis): rewrite the parameters passed to the tool.
/// - **Fill result** (short-circuit): `Some` = skip the actual tool invocation and use
///   this synthetic output as the result; **the turn continues**.
///   This is fundamentally different from `Break` (which ends the entire turn) — do not
///   confuse "intercepting a single tool" with "ending the turn".
#[derive(Debug, Clone)]
pub struct BeforeToolApply {
    pub tool_name: String,
    /// The tool's safety level, placed in the envelope for the matcher's safety
    /// filtering.
    pub safety: SafetyClass,
    /// Modifiable tool arguments.
    pub args: Value,
    /// The result that will be produced. `None` = actually run the tool; `Some` =
    /// short-circuit.
    pub result: Option<SyntheticToolResult>,
}

impl HookStep for BeforeToolApply {
    fn event_name(&self) -> &'static str {
        "before_tool_apply"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "tool": self.tool_name,
            "safety": safety_str(self.safety),
            "args": self.args,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        let control = parse_control(verdict)?;

        // Data plane: update args.
        if let Some(new_args) = verdict.get("args") {
            self.args = new_args.clone();
        }

        // Short-circuit: fill `result`. The verdict's `result` field is a
        // `ToolResultBody` plus an optional `is_error`.
        if let Some(r) = verdict.get("result").filter(|r| !r.is_null()) {
            let body: ToolResultBody =
                serde_json::from_value(r.clone()).map_err(|e| VerdictError::Malformed {
                    field: "result",
                    reason: e.to_string(),
                })?;
            let is_error = verdict
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            self.result = Some(SyntheticToolResult { body, is_error });
        }

        Ok(control)
    }
}

// Step 3: after Generate (observational, outputs are non-Option)

/// `after Generate`: the LLM call has returned. **Observational** — usage / stop / error
/// are all present (non-`Option`), with no room to "fill in" outputs; to influence the
/// next round, use [`BeforeTurnEnd`]. Only `Break` and observation are meaningful.
#[derive(Debug, Clone)]
pub struct AfterGenerate {
    pub model: String,
    pub usage: Usage,
    pub stop: AcpStopReason,
    pub error: Option<String>,
}

impl HookStep for AfterGenerate {
    fn event_name(&self) -> &'static str {
        "after_generate"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "model": self.model,
            "usage": self.usage,
            "stop_reason": stop_reason_str(self.stop),
            "error": self.error,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // Observation-only: no output to fill, only control accepted (typically just
        // break); `additional_context` has no landing point here, so it is ignored.
        parse_control(verdict)
    }
}

// Scope step: after session enter / after turn enter (no output, injectable / breakable)

/// The source of the session: new or resumed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionSource {
    New,
    Resume,
}

/// `after session enter`: the session scope has been entered. Allows injecting a system
/// suffix or rejecting with `Break`.
#[derive(Debug, Clone)]
pub struct AfterSessionEnter {
    pub cwd: String,
    pub source: SessionSource,
    /// Suffix appended to the system prompt (`apply_verdict` fills this from
    /// `additional_context`).
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterSessionEnter {
    fn event_name(&self) -> &'static str {
        "after_session_enter"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "cwd": self.cwd,
            "source": match self.source { SessionSource::New => "new", SessionSource::Resume => "resume" },
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

/// `after turn enter`: the turn scope has been entered, but input for this round has not
/// yet been consumed. Injection or `Break` can reject this turn.
#[derive(Debug, Clone)]
pub struct AfterTurnEnter {
    pub is_subagent: bool,
    pub agent_type: Option<String>,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterTurnEnter {
    fn event_name(&self) -> &'static str {
        "after_turn_enter"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "is_subagent": self.is_subagent,
            "agent_type": self.agent_type,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Ingest step: before / after (mutation type: rewrite input / veto)
// ---------------------------------------------------------------------------

/// The source of the input to be ingested in the current turn.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestSource {
    /// First round: user prompt.
    User,
    /// Continuation turn: feedback injected before the turn ends.
    Continuation,
    /// Background task backflow: an autonomous continuation turn initiated by the session
    /// driver after a `run_in_background` subtask completes. Its input is a deferred tool
    /// result rather than a user utterance.
    Background,
}

/// `before Ingest`: called before ingesting the current turn's input. Can rewrite the
/// entire pending input or `Break` to reject the turn.
///
/// This is a mutation hook — the short-circuit is `Break` (reject), not "fill a result"
/// (there is no separable output). On an empty ingestion turn, `input` is empty.
///
/// The verdict supports two rewriting modes (not mutually exclusive; both can be given):
/// - `input` (`String` / array of `String`): **fully replace** the pending input.
/// - `prepend_input` (`String` / array of `String`): **prepend** text blocks before the
///   existing input, preserving original blocks (including non-text blocks like images).
///   Used to inject context (e.g., a skill's auto-activated L1 prompt) before the user's
///   prompt without losing the original multimodal content.
#[derive(Debug, Clone)]
pub struct BeforeIngest {
    pub source: IngestSource,
    /// The input to be ingested, which can be rewritten.
    pub input: Vec<ContentBlock>,
}

impl HookStep for BeforeIngest {
    fn event_name(&self) -> &'static str {
        "before_ingest"
    }

    fn to_envelope(&self) -> Value {
        // Expose the input text (concatenated `Text` blocks) so the hook can inspect or
        // rewrite it; non-text blocks are excluded from the envelope but remain in the
        // step.
        let text: String = self
            .input
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        json!({
            "source": match self.source {
                IngestSource::User => "user",
                IngestSource::Continuation => "continuation",
                IngestSource::Background => "background",
            },
            "input": text,
            "input_len": self.input.len(),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // The `input` field of the verdict can be a string (replacing the entire input
        // with a single text block) or an array of strings.
        if let Some(v) = verdict.get("input").filter(|v| !v.is_null()) {
            self.input = match v {
                Value::String(s) => vec![ContentBlock::from(s.as_str())],
                _ => parse_block_array(v, "input")?,
            };
        }
        // Prepend: insert text blocks before the existing `input`, preserving the
        // original blocks (including non-text ones). Applied after the full `input`
        // replacement — if both are given, the prepended content comes before the
        // replacement result.
        if let Some(v) = verdict.get("prepend_input").filter(|v| !v.is_null()) {
            let mut prefix = match v {
                Value::String(s) => vec![ContentBlock::from(s.as_str())],
                _ => parse_block_array(v, "prepend_input")?,
            };
            prefix.append(&mut self.input);
            self.input = prefix;
        }
        parse_control(verdict)
    }
}

/// `after Ingest`: input has been merged into history. Injection only.
#[derive(Debug, Clone)]
pub struct AfterIngest {
    pub committed_len: usize,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterIngest {
    fn event_name(&self) -> &'static str {
        "after_ingest"
    }

    fn to_envelope(&self) -> Value {
        json!({ "committed_len": self.committed_len })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Compact step: before (veto only) / after (observation)
// ---------------------------------------------------------------------------

/// `before Compact`: runs before compaction. Mutation type — short-circuit = `Skip`
/// (vetoes this compaction), no "fill result".
#[derive(Debug, Clone)]
pub struct BeforeCompact {
    pub token_estimate: u64,
    pub threshold: u64,
}

impl HookStep for BeforeCompact {
    fn event_name(&self) -> &'static str {
        "before_compact"
    }

    fn to_envelope(&self) -> Value {
        json!({ "token_estimate": self.token_estimate, "threshold": self.threshold })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // A "veto" in compact means skip this compression (Skip).
        parse_control_veto(verdict, HookControl::Skip)
    }
}

/// `after Compact`: compression is complete. Injection / observation only.
#[derive(Debug, Clone)]
pub struct AfterCompact {
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterCompact {
    fn event_name(&self) -> &'static str {
        "after_compact"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tokens_before": self.tokens_before, "tokens_after": self.tokens_after })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ----------------------------------------------------------------------------
// Generate step: before (modify request / short-circuit)
// ----------------------------------------------------------------------------

/// `before Generate`: runs before the LLM call. Call-site hook — can modify request
/// fields, or set `assistant_text` to short-circuit (skip the real LLM call with a
/// synthetic reply).
#[derive(Debug, Clone)]
pub struct BeforeGenerate {
    pub model: String,
    pub message_count: usize,
    pub attempt: u32,
    /// Short-circuit: `Some` = skip the LLM call and use this synthetic assistant text as
    /// the reply. On commit, it is built into a `Message`.
    pub assistant_text: Option<String>,
}

impl HookStep for BeforeGenerate {
    fn event_name(&self) -> &'static str {
        "before_generate"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "model": self.model,
            "message_count": self.message_count,
            "attempt": self.attempt,
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        if let Some(m) = verdict.get("model").and_then(Value::as_str) {
            self.model = m.to_string();
        }
        if let Some(a) = verdict.get("assistant").and_then(Value::as_str) {
            self.assistant_text = Some(a.to_string());
        }
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// Permission step: before (delegate; v0 only stubs) / after (observe)
// ---------------------------------------------------------------------------

/// `before Permission`: invoked before requesting user authorization. v0 only stubs
/// observe — the `resolved` fallback is not yet wired (policy remains the authority for
/// allow/deny; see hooks.md §7.3). Stub is in place for future use.
#[derive(Debug, Clone)]
pub struct BeforePermission {
    pub tool: String,
    /// The current policy decision (`"allow"`, `"deny"`, or `"ask"`).
    pub decision: String,
    /// Resolved result; not consumed in v0.
    pub resolved: Option<bool>,
}

impl HookStep for BeforePermission {
    fn event_name(&self) -> &'static str {
        "before_permission"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tool": self.tool, "decision": self.decision })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        // v0: only accepts `control` (typically `break`); `resolved` is kept as a
        // placeholder but not consumed here.
        if let Some(r) = verdict.get("resolved").and_then(Value::as_bool) {
            self.resolved = Some(r);
        }
        parse_control(verdict)
    }
}

/// `after Permission`: authorization result is determined. Observation only.
#[derive(Debug, Clone)]
pub struct AfterPermission {
    pub tool: String,
    pub granted: bool,
}

impl HookStep for AfterPermission {
    fn event_name(&self) -> &'static str {
        "after_permission"
    }

    fn to_envelope(&self) -> Value {
        json!({ "tool": self.tool, "granted": self.granted })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        parse_control(verdict)
    }
}

// ---------------------------------------------------------------------------
// ToolApply step: after (per-tool) / after ToolBatch (whole batch)
// ---------------------------------------------------------------------------

/// `after ToolApply` (per-tool): the tool has produced a result. Supports injection
/// (appending to `tool_result`) or `Break`.
#[derive(Debug, Clone)]
pub struct AfterToolApply {
    pub tool_name: String,
    pub is_error: bool,
    /// The result body produced by the tool (always present, not an `Option`) — placed
    /// into the envelope so the hook can see the tool's output.
    pub output: ToolResultBody,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterToolApply {
    fn event_name(&self) -> &'static str {
        "after_tool_apply"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "tool": self.tool_name,
            "is_error": self.is_error,
            "output": tool_result_body_to_json(&self.output),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

/// A summary entry for a batch of parallel tool results (for envelope use).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolBatchEntry {
    pub tool_name: String,
    pub is_error: bool,
}

/// `after ToolBatch`: a full batch of parallel tools has finished. Can inject / `Break`
/// (graceful, see proposal §7).
#[derive(Debug, Clone)]
pub struct AfterToolBatch {
    pub results: Vec<ToolBatchEntry>,
    pub additional_context: Vec<ContentBlock>,
}

impl HookStep for AfterToolBatch {
    fn event_name(&self) -> &'static str {
        "after_tool_batch"
    }

    fn to_envelope(&self) -> Value {
        json!({
            "results": self.results.iter().map(|e| json!({
                "tool": e.tool_name,
                "is_error": e.is_error,
            })).collect::<Vec<_>>(),
        })
    }

    fn apply_verdict(&mut self, verdict: &Value) -> Result<HookControl, VerdictError> {
        self.additional_context
            .extend(parse_additional_context(verdict)?);
        parse_control(verdict)
    }
}

// ----------------------------------------------------------------------------
// Pipeline: merging multiple verdicts on a single step
// ----------------------------------------------------------------------------

/// Applies a sequence of handler verdicts to the same step in declaration order,
/// producing the final [`HookControl`].
///
/// This is the step-level pipeline semantics (aligned with the existing `merge_outcome`):
/// - **Data accumulation**: each verdict's field mutations (changing args, injecting,
///   filling result, etc.) are applied sequentially to the same `&mut step`; later
///   handlers see the state modified by earlier ones.
/// - **Control short-circuit**: any verdict that returns something other than
///   [`HookControl::Proceed`] **stops the pipeline** and returns it — `Break` /
///   `Continue` / `Skip` all mean "the outcome is decided", and subsequent handlers
///   should not override it.
/// - **Error handling**: when a verdict fails to parse, `on_error` decides how to degrade
///   (returning `Some(control)` to short-circuit, or `None` to skip that verdict and
///   continue) — strategies like "treat a block event error as equivalent to block" are
///   left to the caller; this function does not hardcode them.
pub fn run_step_pipeline<S, I, F>(step: &mut S, verdicts: I, mut on_error: F) -> HookControl
where
    S: HookStep + ?Sized,
    I: IntoIterator<Item = Value>,
    F: FnMut(VerdictError) -> Option<HookControl>,
{
    for verdict in verdicts {
        match step.apply_verdict(&verdict) {
            Ok(HookControl::Proceed) => {}
            Ok(control) => return control,
            Err(err) => {
                if let Some(control) = on_error(err) {
                    return control;
                }
            }
        }
    }
    HookControl::Proceed
}

/// Parse the `ContentBlock` array from the verdict (string array to text blocks).
fn parse_block_array(v: &Value, field: &'static str) -> Result<Vec<ContentBlock>, VerdictError> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ContentBlock::from)
                    .ok_or(VerdictError::Malformed {
                        field,
                        reason: "each entry must be a string".to_string(),
                    })
            })
            .collect(),
        _ => Err(VerdictError::Malformed {
            field,
            reason: "must be an array of strings".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests;
