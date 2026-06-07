//! Builtin hook handlers.
//!
//! In-process Rust handlers with zero external dependencies. During CLI assembly, they
//! are looked up by name in [`BuiltinRegistry`], instantiated, and registered into
//! [`super::HandlerTable`] of `DefaultHookEngine`.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::{Map, Value};

use super::{HookCtx, HookError, StepHandler};
use crate::tool::SkillEntry;

/// Registry mapping builtin handler names to factory closures.
///
/// When the CLI assembles `DefaultHookEngine`, it feeds `HookHandlerSpec::Builtin { name
/// }` to
/// [`Self::lookup_step`]. Unknown names fail fast at config-load time, so users don't
/// discover
/// typos mid-turn.
///
/// The factory signature is `Fn() -> Arc<dyn HookHandler>`: handlers have no per-config
/// parameters, and multiple `[[hooks.*]]` entries referencing the same builtin share a
/// single
/// `Arc`. If a builtin later needs configuration parameters, upgrade `name` to a
/// structured
/// enum and switch the registry to `match` dispatch.
pub struct BuiltinRegistry {
    /// A map from name to `Arc<dyn StepHandler>` factory.
    step_factories: BTreeMap<String, Box<dyn Fn() -> Arc<dyn StepHandler> + Send + Sync>>,
}

impl BuiltinRegistry {
    /// Default registry: `tracing-audit` + `redact-secrets`.
    pub fn defaults() -> Self {
        let mut reg = Self {
            step_factories: BTreeMap::new(),
        };
        reg.register_step("tracing-audit", || Arc::new(TracingAuditHook));
        reg.register_step("redact-secrets", || Arc::new(RedactSecretsHook));
        reg
    }

    /// Register a builtin step handler factory. Duplicate names overwrite previous
    /// entries, allowing tests to stub and replace default behavior.
    pub fn register_step<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Arc<dyn StepHandler> + Send + Sync + 'static,
    {
        self.step_factories
            .insert(name.to_string(), Box::new(factory));
    }

    /// Look up a step handler by name. `None` means the configuration layer should
    /// fail-fast with an error.
    pub fn lookup_step(&self, name: &str) -> Option<Arc<dyn StepHandler>> {
        self.step_factories.get(name).map(|f| f())
    }

    /// Lists registered builtin names, used by the `defect hooks list` CLI.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.step_factories.keys().map(String::as_str)
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::defaults()
    }
}

// tracing-audit

/// Converts `Post*ToolUse` events into structured tracing records.
///
/// Intended to be attached to `[[hooks.post_tool_use]]` /
/// `[[hooks.post_tool_use_failure]]` for an audit trail; attaching it to other events
/// will cause [`StepHandler::handle_step`] to simply `Pass` through.
pub struct TracingAuditHook;

impl StepHandler for TracingAuditHook {
    /// Step model: consumes an `after_tool_apply` envelope `{tool, is_error}`, writes a
    /// structured audit log, and produces no verdict.
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        Box::pin(async move {
            let tool = envelope.get("tool").and_then(Value::as_str).unwrap_or("?");
            let is_error = envelope
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            tracing::info!(
                target: "defect_agent::hooks::audit",
                tool = %tool,
                outcome = if is_error { "error" } else { "ok" },
                "tool call completed",
            );
            Ok(None)
        })
    }
}

// ---------------------------------------------------------------------------
// redact-secrets
// ---------------------------------------------------------------------------

/// On `PreToolUse`, performs in-place replacement of likely sensitive fields in `args`.
///
/// Matches (case-insensitive substring): `password` / `secret` / `token` / `api_key`
/// / `apikey` / `authorization`. When matched, the field value is replaced with `"***"`
/// and patched into `args`.
///
/// Only operates when `args` is an `Object`; other shapes (arrays, strings) are left
/// untouched — the shape of `args` is defined by the tool itself, and deep recursive
/// rewriting could break tool semantics.
///
/// Does not handle `password=xxx` embedded inside a `bash` `command` string — that would
/// require shell lexing, which is beyond the stability guarantees of this builtin.
pub struct RedactSecretsHook;

const SECRET_KEY_NEEDLES: &[&str] = &[
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
];

impl StepHandler for RedactSecretsHook {
    /// Step model: consumes the `before_tool_apply` envelope `{tool, args}`, redacts
    /// potentially sensitive fields in `args` in place, and returns a `{args:
    /// <redacted>}` verdict if any were found (the engine applies it back to the step,
    /// modifying `args`).
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let verdict = envelope
            .get("args")
            .and_then(Value::as_object)
            .map(redact_object)
            .filter(|r| r.changed)
            .map(|r| serde_json::json!({ "args": Value::Object(r.value) }));
        Box::pin(async move { Ok(verdict) })
    }
}

struct Redacted {
    value: Map<String, Value>,
    changed: bool,
}

fn redact_object(obj: &Map<String, Value>) -> Redacted {
    let mut out = Map::with_capacity(obj.len());
    let mut changed = false;
    for (key, value) in obj {
        if key_is_secret(key) {
            out.insert(key.clone(), Value::String("***".to_string()));
            changed = true;
        } else {
            out.insert(key.clone(), value.clone());
        }
    }
    Redacted {
        value: out,
        changed,
    }
}

fn key_is_secret(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SECRET_KEY_NEEDLES
        .iter()
        .any(|needle| lower.contains(needle))
}

// ---------------------------------------------------------------------------
// skill-manifest
// ---------------------------------------------------------------------------

/// On `SessionStart`, appends the L1 manifest of available skills (`name + description`)
/// to the system prompt suffix, so the model is aware of which skills it can load on
/// demand via the `skill` tool.
///
/// This is the L1 injection point for progressive disclosure. Note that the
/// `skill` tool's own description already embeds the same catalog (see
/// [`crate::tool::SkillTool`]), so this hook is an **optional enhancement**: when
/// installed, it also places the manifest in the system prompt (more robust for clients
/// that do not count tool descriptions toward the attention budget). Both paths originate
/// from the same skill index, so they will not diverge.
///
/// Unlike other builtins, this handler holds a skill index and **cannot** be constructed
/// via the parameterless factory [`BuiltinRegistry::defaults`]. Instead, it is registered
/// during CLI assembly using a closure that captures the index (see `defect_cli::hooks`).
pub struct SkillManifestHook {
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillManifestHook {
    /// Constructs from a loaded skill index. The caller **must not** register this hook
    /// when `skills` is empty (the manifest would be an empty segment, wasting tokens).
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        Self { skills }
    }
}

/// Renders the session-start injection: a level-1 manifest (name + description for every
/// skill) plus the full body of each `always` skill (always-on, injected directly into
/// the system prompt). Returns `None` for an empty index (no empty segment injected).
fn render_skill_manifest(skills: &BTreeMap<String, SkillEntry>) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from(
        "## Available Skills\n\n\
         Load a skill's full instructions with the `skill` tool (by name) when the task matches:\n",
    );
    for (name, entry) in skills {
        out.push_str(&format!("- **{name}**: {}\n", entry.description));
    }
    // Always-on skills: inline the body of any skill marked `always: true` so the model
    // has those instructions from the start, without needing to call the `skill` tool.
    for (name, entry) in skills {
        if entry.always {
            out.push_str(&format!("\n## Skill: {name}\n\n{}\n", entry.body));
        }
    }
    Some(out)
}

impl StepHandler for SkillManifestHook {
    /// In the step model, inject the L1 skill manifest as `additional_context` during
    /// `after_session_enter`
    /// (the engine applies it back to the step, appending it to the system prompt
    /// suffix).
    fn handle_step<'a>(
        &'a self,
        _envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let verdict = render_skill_manifest(&self.skills)
            .map(|manifest| serde_json::json!({ "additional_context": [manifest] }));
        Box::pin(async move { Ok(verdict) })
    }
}

// ---------------------------------------------------------------------------
// skill-triggers
// ---------------------------------------------------------------------------

/// On `before_ingest`, automatically activate relevant skills based on the user prompt.
/// When a match is found, insert a **L1 hint** (e.g. "Detected skill X relevance; use the
/// `skill` tool if needed") before the prompt, rather than injecting the full skill body.
/// This follows progressive disclosure: the model decides whether to actually load the
/// skill.
///
/// Trigger conditions (any one triggers):
/// - **keyword**: any of the skill's `triggers.keywords` is a case-insensitive substring
///   of the prompt text.
/// - **glob**: any "path-like token" extracted from the prompt text matches one of the
///   skill's `triggers.globs`.
///
/// Skills with `always` trigger are already injected in full at session start, so they
/// are skipped here to avoid duplicate hints.
///
/// Like [`SkillManifestHook`], this hook holds a skill index and is registered via a
/// closure that captures the index (see `defect_cli::hooks`).
pub struct SkillTriggersHook {
    skills: Arc<BTreeMap<String, SkillEntry>>,
}

impl SkillTriggersHook {
    /// Constructs from the already-loaded skill index. The caller **must not** register
    /// this hook when `skills` is empty.
    pub fn new(skills: Arc<BTreeMap<String, SkillEntry>>) -> Self {
        Self { skills }
    }
}

/// Extract path-like tokens from a prompt string (best-effort, no NLP).
///
/// Split on whitespace, strip surrounding quotes/backticks/brackets and trailing
/// punctuation. A token is considered a path if it either:
/// (1) contains `/` (e.g. `crates/agent/src/foo.rs`); or (2) ends with an extension
/// `xxx.ext` (e.g. `Cargo.toml` / `main.rs`). Strip leading `./`. Bare words (no `/` and
/// no extension) are not paths — they are left for keyword matching.
fn extract_path_tokens(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .filter_map(|raw| {
            let trimmed = raw.trim_matches(|c: char| {
                c == '`' || c == '"' || c == '\'' || c == '(' || c == ')' || c == '[' || c == ']'
            });
            let trimmed = trimmed.trim_end_matches([',', '.', ':', ';']);
            let token = trimmed.strip_prefix("./").unwrap_or(trimmed);
            if token.is_empty() {
                return None;
            }
            if is_path_like(token) {
                Some(token.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Whether the token is "path-like": contains `/`, or matches `name.ext` (a dot followed
/// by one or more alphanumeric characters at the end).
fn is_path_like(token: &str) -> bool {
    if token.contains('/') {
        return true;
    }
    // Ending extension: at least one alphanumeric character after the last `.`, and the
    // dot is not at the start.
    match token.rsplit_once('.') {
        Some((stem, ext)) => {
            !stem.is_empty() && !ext.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric())
        }
        None => false,
    }
}

/// Returns whether a single skill is activated by the prompt: keyword substring OR glob
/// matches a path token.
fn skill_triggered(entry: &SkillEntry, prompt_lower: &str, path_tokens: &[String]) -> bool {
    let keyword_hit = entry
        .triggers
        .keywords
        .iter()
        .any(|kw| !kw.is_empty() && prompt_lower.contains(&kw.to_ascii_lowercase()));
    if keyword_hit {
        return true;
    }
    match &entry.triggers.globs {
        Some(set) => path_tokens.iter().any(|t| set.is_match(t)),
        None => false,
    }
}

impl StepHandler for SkillTriggersHook {
    /// In the `before_ingest` step, read the prompt text and, for each matched skill,
    /// prepend an L1 hint (a `prepend_input` verdict). Return `None` if no skill matches.
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let prompt = envelope.get("input").and_then(Value::as_str).unwrap_or("");
        let prompt_lower = prompt.to_ascii_lowercase();
        let path_tokens = extract_path_tokens(prompt);

        let hints: Vec<String> = self
            .skills
            .iter()
            .filter(|(_, e)| !e.always)
            .filter(|(_, e)| skill_triggered(e, &prompt_lower, &path_tokens))
            .map(|(name, _)| {
                format!(
                    "Detected skill `{name}` is relevant to the current task; \
                     load it with the `skill` tool when needed."
                )
            })
            .collect();

        let verdict = (!hints.is_empty()).then(|| serde_json::json!({ "prepend_input": hints }));
        Box::pin(async move { Ok(verdict) })
    }
}

// ---------------------------------------------------------------------------
// goal-gate
// ---------------------------------------------------------------------------

/// The core hook for the `--goal` goal-driven loop, **subscribing to two events**
/// (dispatched via the `hook_event` envelope):
///
/// - `after_session_enter`: Injects the goal description + `goal_done` usage contract as
///   `additional_context` into the system prompt suffix — **effective from turn 1**. This
///   lets the model know the goal and that it must actively call `goal_done` upon
///   completion from the start, avoiding an extra wasted turn waiting for the first
///   voluntary stop.
/// - `before_turn_end`: On voluntary turn stop, reads
///   [`GoalState::is_reached`](crate::session::GoalState::is_reached): if reached (model
///   called `goal_done`) → `proceed` to end; otherwise → `continue` to extend the turn +
///   inject an English prompt reminder.
///
/// The hard cap on extensions is enforced by the turn loop's
/// [`crate::session::TurnConfig::max_hook_continues`] (mapped from `--max-turns`) — this
/// hook only checks "is it done?", it does not count extensions itself.
///
/// Like [`SkillManifestHook`], this is a stateful builtin (holds `Arc<GoalState>`) and
/// cannot be constructed via [`BuiltinRegistry::defaults`]'s parameterless factory —
/// during CLI assembly, a closure capturing the state is registered for both events under
/// `--goal` (see `defect_cli::hooks`).
pub struct GoalGate {
    goal: Arc<crate::session::GoalState>,
}

impl GoalGate {
    pub fn new(goal: Arc<crate::session::GoalState>) -> Self {
        Self { goal }
    }

    /// Injected into the system prompt from turn 1 onward: goal description + `goal_done`
    /// contract.
    fn briefing(&self) -> String {
        format!(
            "## Goal\n\n\
             You are running in goal-driven mode. Your objective:\n\n{}\n\n\
             Work autonomously across as many turns as needed to achieve this goal. \
             When — and only when — the goal is genuinely and fully achieved, call the \
             `goal_done` tool to finish the run. Do not call it prematurely. If you stop \
             without calling `goal_done`, you will be prompted to keep working.",
            self.goal.objective()
        )
    }
}

impl StepHandler for GoalGate {
    /// Dispatches on the envelope's `hook_event`:
    /// - `after_session_enter` → injects goal description and contract
    ///   (`additional_context`)
    /// - `before_turn_end` → if reached, proceed; otherwise continue with a prompt
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let event = envelope
            .get("hook_event")
            .and_then(Value::as_str)
            .unwrap_or("");
        let verdict = match event {
            "after_session_enter" => {
                serde_json::json!({ "additional_context": [self.briefing()] })
            }
            // before_turn_end (and fallback): check if the goal is reached.
            _ if self.goal.is_reached() => serde_json::json!({ "control": "proceed" }),
            _ => serde_json::json!({
                "control": "continue",
                "additional_context": [format!(
                    "The goal \"{}\" is not yet complete. Keep working toward it. \
                     Once it is genuinely achieved, call the `goal_done` tool to finish.",
                    self.goal.objective()
                )],
            }),
        };
        Box::pin(async move { Ok(Some(verdict)) })
    }
}

#[cfg(test)]
mod tests;
