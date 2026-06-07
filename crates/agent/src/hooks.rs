//! Hook system: extension points for the agent main loop.
//!
//! ## Abstraction layers
//!
//! - [`HookStep`](step::HookStep): interception points called by the main loop at step
//!   boundaries (bucketed by event name)
//! - [`StepHandler`]: a single executor (implemented in submodules as Builtin / Command /
//!   Prompt)
//! - [`HookMatcher`]: matching conditions for a single hook (filtering by tool / glob /
//!   safety)
//! - [`HookEngine`]: the dispatcher the main loop interacts with; holds a
//!   [`HandlerTable`], executes the pipeline, and merges verdicts
//!
//! ## Default implementations
//!
//! [`NoopHookEngine`]: all `fire` calls return `Pass` directly, `observe` calls are
//! discarded; used when no explicit hook engine is provided during session/turn assembly,
//! preserving "no hook configured = main loop behavior unchanged".
//!
//! [`DefaultHookEngine`]: holds the handler table via [`arc_swap::ArcSwap`], dispatches
//! serially according to the pipeline semantics; matcher, timeout, and panic
//! capture are handled per the degradation table.

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

/// Default per-handler timeout for `DefaultHookEngine`.
const DEFAULT_HANDLER_TIMEOUT: Duration = Duration::from_secs(5);

/// Matching conditions for a single hook.
///
/// Shape is identical to `defect-config`'s `HookMatcher`; the agent crate does not depend
/// on config,
/// so this is defined independently and the CLI translates the config shape into the
/// agent shape at assembly time.
/// See hooks design for trust model.
///
/// All fields empty = match all triggers under that event.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct HookMatcher {
    /// Match by exact tool name (only for `*ToolUse*` events).
    pub tool: Option<String>,
    /// Glob match by tool name (only for `*ToolUse*` events).
    pub tool_glob: Option<String>,
    /// Filter by [`SafetyClass`] (only `PreToolUse`); any match triggers. Empty vec = no
    /// filter.
    pub safety: Vec<SafetyClass>,
}

impl HookMatcher {
    /// Matches a step model by tool name and safety (both taken from the step envelope;
    /// non-tool steps pass `None`).
    ///
    /// All fields empty = matches everything. `tool` is exact, `tool_glob` is a glob
    /// pattern, `safety` matches any (empty vec = no filter).
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

/// Tool name glob matching, using [`globset`] (same as skill triggers / search).
///
/// Tool names are dot-separated (e.g. `mcp.fs.read`), not file paths — `globset` treats
/// `*` as "does not cross `/`" by default, but tool names contain no `/`, so `mcp.*`
/// matches the whole string correctly. Patterns are compiled on each match (tool name
/// matches are infrequent and patterns are short, so compilation overhead is negligible).
/// Invalid patterns do not panic: a warn is logged and the match is treated as no-match
/// (matcher mismatch = the hook is not triggered, safe side).
fn tool_name_matches(pattern: &str, name: &str) -> bool {
    match globset::Glob::new(pattern) {
        Ok(glob) => glob.compile_matcher().is_match(name),
        Err(err) => {
            tracing::warn!(%pattern, %err, "invalid tool_glob pattern; treating as no-match");
            false
        }
    }
}

/// A lightweight context shared with the handler.
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

/// Reasons for handler failure.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("hook handler timed out")]
    Timeout,

    #[error("hook handler failed: {0}")]
    HandlerFailed(#[source] BoxError),

    /// Handler trust not established, unregistered, or other configuration-layer errors.
    #[error("hook configuration error: {0}")]
    Configuration(String),
}

/// **Step model handler** (migration target). The engine gives it an input envelope for a
/// mount point (produced by [`step::HookStep::to_envelope`]), and it produces a verdict
/// JSON — the engine then applies the verdict back to the step via
/// [`step::HookStep::apply_verdict`]. Both hook types implement this: internal Rust hooks
/// compute the verdict directly; command/prompt hooks feed the envelope to a
/// subprocess/LLM and parse the output into a verdict.
///
/// Returns `Ok(None)` = no intervention (equivalent to an empty verdict);
/// `Ok(Some(verdict))` = apply that verdict; `Err` = failure, handled by the engine
/// according to the degradation table.
pub trait StepHandler: Send + Sync {
    /// Process a mount point: input envelope → verdict JSON.
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>>;
}

// ---------------------------------------------------------------------------
// HookEngine
// ---------------------------------------------------------------------------

/// Dispatcher for the main loop (step model).
///
/// The sole entry point is [`Self::dispatch`]: given a [`step::HookStep`] for a mount
/// point, the engine finds matching handlers by `event_name`, feeds each handler the step
/// envelope, applies the verdict back to the step (accumulating on the data axis), and
/// merges the final [`step::HookControl`] (early exit on the control axis). Field
/// mutations on the step (injection, argument changes, output filling, etc.) take effect
/// in place. Summary: what the caller should read + control indication.
///
/// Default implementation is [`DefaultHookEngine`]; tests and default session setup use
/// [`NoopHookEngine`].
pub trait HookEngine: Send + Sync {
    /// **Default implementation returns [`step::HookControl::Proceed`]** (no
    /// intervention); [`NoopHookEngine`] uses this directly. [`DefaultHookEngine`]
    /// overrides it for real dispatch.
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

/// Default hook engine: `dispatch` uses the trait's default implementation (`Proceed`,
/// i.e., no-op).
///
/// When assembling a session/turn without an explicitly injected hook engine, this is
/// used — ensuring that "no hook configured = main loop behavior is completely
/// unchanged", analogous to [`crate::http::NoopHttpClient`].
#[derive(Debug, Default)]
pub struct NoopHookEngine;

impl HookEngine for NoopHookEngine {}

// ---------------------------------------------------------------------------
// DefaultHookEngine
// ---------------------------------------------------------------------------

/// A handler table bucketed by step `event_name`.
///
/// It is mounted inside [`DefaultHookEngine`] and replaced atomically via
/// [`DefaultHookEngine::reload`] — `ArcSwap` makes runtime hot-reloading nearly
/// zero-cost.
#[derive(Default)]
pub struct HandlerTable {
    /// Handler list indexed by step `event_name` (snake_case). Declaration order
    /// determines pipeline execution order.
    pub step_buckets: std::collections::HashMap<&'static str, Vec<StepHandlerEntry>>,
}

/// A fully assembled step handler: name, matcher, handler, and per-entry timeout.
pub struct StepHandlerEntry {
    /// Display name, used only in tracing / observability to identify this hook. Defaults
    /// to an anonymous label (see [`Self::new`]); assemblers can override it with
    /// [`Self::with_name`].
    pub name: String,
    pub matcher: HookMatcher,
    pub handler: Arc<dyn StepHandler>,
    pub timeout: Option<Duration>,
}

/// Placeholder name used in tracing for unnamed hooks.
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

    /// Sets the display name. `None` keeps the anonymous placeholder
    /// ([`ANONYMOUS_HOOK_NAME`]).
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

    /// Step handlers assembled under the step `event_name`.
    pub fn step_handlers(&self, event_name: &str) -> &[StepHandlerEntry] {
        self.step_buckets
            .get(event_name)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Appends a step handler under the given step `event_name`.
    pub fn push_step(&mut self, event_name: &'static str, entry: StepHandlerEntry) {
        self.step_buckets.entry(event_name).or_default().push(entry);
    }
}

/// Default hook engine: serial dispatch following the pipeline semantics.
///
/// - Uses [`ArcSwap`] to hold a [`HandlerTable`]; [`Self::reload`] enables full hot-swap
/// - `fire` internally filters by matcher → serial await, each handler sees the event
///   after
///   all prior patches have been applied
/// - Timeout, panic, or error in a single handler is downgraded per the degradation table
pub struct DefaultHookEngine {
    table: ArcSwap<HandlerTable>,
}

impl DefaultHookEngine {
    pub fn new() -> Self {
        Self {
            table: ArcSwap::from_pointee(HandlerTable::empty()),
        }
    }

    /// Atomically replace the entire handler table with a new one; used for runtime
    /// hot-reloading.
    ///
    /// The old table is automatically reclaimed by `Arc` once all in-flight
    /// `fire`/`observe` calls finish.
    pub fn reload(&self, table: HandlerTable) {
        self.table.store(Arc::new(table));
    }

    /// A snapshot reference to the current handler table. Intended for
    /// testing/diagnostics only.
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

            // The matcher filters by tool name and safety, which are extracted from the
            // step envelope (only *ToolApply* steps carry these fields).
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
                // Each handler sees the envelope as modified by the previous handler,
                // plus the common headers.
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
                    // Early exit on control: anything other than Proceed stops the
                    // pipeline.
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

/// Merge common headers into the step-specific envelope. Common headers: `session_id` /
/// `cwd` / `hook_event`.
///
/// The step itself does not hold a `HookCtx` (zero-borrow, `Send`), so the engine fills
/// in the common context at dispatch time — this ensures every user hook envelope
/// contains session, cwd, and event name. Step-specific fields take precedence (they are
/// not overwritten).
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

/// The `safety` field (snake_case) from the envelope maps to [`SafetyClass`]; unknown or
/// missing values yield `None`.
fn parse_safety(s: &str) -> Option<SafetyClass> {
    match s {
        "read_only" => Some(SafetyClass::ReadOnly),
        "mutating" => Some(SafetyClass::Mutating),
        "destructive" => Some(SafetyClass::Destructive),
        "network" => Some(SafetyClass::Network),
        _ => None,
    }
}

// Extract a text representation from a `catch_unwind` payload without depending on the
// concrete panic type.
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
mod tests;
