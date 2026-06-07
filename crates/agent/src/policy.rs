//! Sandbox policy: "Allow / Deny / Ask user" decision for tool calls.
//!
//! ## Interface with the main loop
//!
//! [`SandboxPolicy::classify`] is a pure decision; it returns a [`PolicyDecision`]:
//! - `Allow` / `Deny`: the main loop branches directly.
//! - `Ask(Ask)`: the main loop packs `Ask::options` into an ACP
//!   `RequestPermissionRequest`
//!   and waits for the user's response. When the response arrives, it calls
//!   [`SandboxPolicy::record`] so the policy can update its internal "already authorized"
//!   table.
//!
//! ## Boundary with the OS-level sandbox
//!
//! This module **only makes decisions** — OS-level isolation (landlock / seatbelt / child
//! process
//! permission dropping) is a separate trait (a future `ToolSandbox`). This module's
//! output is
//! "whether to execute", orthogonal to "how much permission to grant at execution time".

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol_schema::{PermissionOptionId, PermissionOptionKind};
use serde::{Deserialize, Serialize};

use crate::tool::SafetyClass;

/// The decision result.
///
/// `Ask::options` must be assembled by the policy itself (including the prompt text, wire
/// id, and `allows`).
/// The main loop no longer infers whether to allow for
/// [`PermissionOptionKind::AllowOnce`] / `RejectOnce`, etc. — that is the policy's
/// semantics.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PolicyDecision {
    /// Allow the action without prompting the user.
    Allow,
    /// Deny directly; the main loop feeds "denied by policy" back to the LLM as a
    /// `tool_result`.
    Deny,
    /// Requires user confirmation. The main loop fires ACP `session/request_permission`
    /// and waits for the user to pick an item from [`Ask::options`].
    Ask(Ask),
}

/// Payload for populating `Ask` options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ask {
    /// The list of options presented to the client. **An empty vector is equivalent to
    /// [`PolicyDecision::Deny`]**.
    pub options: Vec<AskOption>,
}

/// A permission option presented to the user.
///
/// `kind` is the ACP UI hint; `allows` is the policy-level "allow/deny" decision.
/// They are usually consistent (`AllowOnce` / `AllowAlways` → `allows = true`), but
/// decoupling
/// lets future partial-allow options like `AllowReadOnly` be added without breaking the
/// current shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskOption {
    pub id: PermissionOptionId,
    pub name: String,
    pub kind: PermissionOptionKind,
    /// Whether the user's selection allows (`true`) or denies (`false`) this option.
    pub allows: bool,
}

/// The "user response" that the main loop writes back to the policy.
///
/// `Selected::allows` is filled into [`AskOption`] by the policy during
/// [`SandboxPolicy::classify`]; the main loop looks it up by `option_id` and feeds it
/// back, avoiding the policy having to re-parse the option id it just sent.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedOutcome {
    Selected {
        option_id: PermissionOptionId,
        allows: bool,
    },
    /// The user cancelled the turn. The policy does not update the authorization table,
    /// but may perform auditing.
    Cancelled,
}

/// Context shared by `classify` and `record`.
#[non_exhaustive]
pub struct PolicyCtx<'a> {
    pub tool_name: &'a str,
    pub safety_hint: SafetyClass,
    pub args: &'a serde_json::Value,
    /// The working directory of the current session. Required by path-allowlist policies;
    /// implementations that do not need it may ignore this field.
    pub cwd: &'a Path,
}

impl<'a> PolicyCtx<'a> {
    pub fn new(
        tool_name: &'a str,
        safety_hint: SafetyClass,
        args: &'a serde_json::Value,
        cwd: &'a Path,
    ) -> Self {
        Self {
            tool_name,
            safety_hint,
            args,
            cwd,
        }
    }
}

/// A decision-maker for tool invocations.
///
/// The implementation must be purely functional: `classify` performs no I/O and no
/// persistence; persisting the "authorized" table is done via [`Self::record`].
pub trait SandboxPolicy: Send + Sync {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision;

    /// A write-back hook invoked after the user responds to an `Ask`.
    ///
    /// The main loop calls this once after receiving
    /// [`crate::event::PermissionResolution::Selected`] but *before* enqueuing the tool
    /// for execution or rejecting it. `outcome.allows()` has already been resolved from
    /// [`AskOption::allows`].
    fn record(&self, ctx: PolicyCtx<'_>, outcome: RecordedOutcome);
}

// Built-in policies

/// Allows everything. Equivalent to the early v0 stub; intended for testing / dev mode.
pub struct OpenPolicy;

impl SandboxPolicy for OpenPolicy {
    fn classify(&self, _ctx: PolicyCtx<'_>) -> PolicyDecision {
        PolicyDecision::Allow
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// Only allows `ReadOnly`; everything else is denied.
pub struct ReadOnlyPolicy;

impl SandboxPolicy for ReadOnlyPolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        match ctx.safety_hint {
            SafetyClass::ReadOnly => PolicyDecision::Allow,
            _ => PolicyDecision::Deny,
        }
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// Deny everything. Used for smoke testing.
pub struct DenyAllPolicy;

impl SandboxPolicy for DenyAllPolicy {
    fn classify(&self, _ctx: PolicyCtx<'_>) -> PolicyDecision {
        PolicyDecision::Deny
    }
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

/// Default policy: `ReadOnly` is directly `Allow`; `Mutating`, `Destructive`, and
/// `Network` go through `Ask`. `AllowAlways` maintains an internal whitelist of tool
/// names; a match results in an immediate `Allow`.
pub struct AskWritesPolicy {
    always_allow: Mutex<HashSet<String>>,
}

impl AskWritesPolicy {
    pub fn new() -> Self {
        Self {
            always_allow: Mutex::new(HashSet::new()),
        }
    }
}

impl Default for AskWritesPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SandboxPolicy for AskWritesPolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        if matches!(ctx.safety_hint, SafetyClass::ReadOnly) {
            return PolicyDecision::Allow;
        }
        if let Ok(table) = self.always_allow.lock()
            && table.contains(ctx.tool_name)
        {
            return PolicyDecision::Allow;
        }
        PolicyDecision::Ask(default_ask_options(ctx.tool_name))
    }

    fn record(&self, ctx: PolicyCtx<'_>, outcome: RecordedOutcome) {
        let RecordedOutcome::Selected { option_id, allows } = outcome else {
            return;
        };
        if !allows {
            return;
        }
        if option_id.0.as_ref() != ALLOW_ALWAYS_ID {
            return;
        }
        if let Ok(mut table) = self.always_allow.lock() {
            table.insert(ctx.tool_name.to_string());
        }
    }
}

/// Adapts any inner policy to a non-interactive semantics: when the inner policy returns
/// [`PolicyDecision::Ask`], it is downgraded to [`PolicyDecision::Deny`]; `Allow` /
/// `Deny` are passed through unchanged.
///
/// Used for nested turns of a subagent (`spawn_agent`) — the subagent has no human
/// present to answer permission requests. If `Ask` were allowed into the main loop, it
/// would permanently hang on [`PermissionGate`](crate::session::PermissionGate). Wrapping
/// with this policy ensures the sub-turn **never blocks and never escalates privileges**:
/// the subagent's actual authorization is always ≤ that of the wrapped parent policy
/// (what the parent would `Ask`, the child directly `Deny`s).
///
/// This is a separate gate from "tool allowlist trimming": the allowlist determines which
/// tools the subagent **sees**, while this policy determines how much access is granted
/// **at runtime** on those tools. See the `project-subagent-design` design document for
/// details.
pub struct NonInteractivePolicy {
    inner: Arc<dyn SandboxPolicy>,
}

impl NonInteractivePolicy {
    pub fn new(inner: Arc<dyn SandboxPolicy>) -> Self {
        Self { inner }
    }
}

impl SandboxPolicy for NonInteractivePolicy {
    fn classify(&self, ctx: PolicyCtx<'_>) -> PolicyDecision {
        match self.inner.classify(ctx) {
            PolicyDecision::Ask(_) => PolicyDecision::Deny,
            other => other,
        }
    }

    // This policy never returns `Ask`, so the main loop will never feed a record back to
    // it — a no-op implementation suffices.
    fn record(&self, _ctx: PolicyCtx<'_>, _outcome: RecordedOutcome) {}
}

// ----------------------------------------------------------------------------
// permission mode section
// ----------------------------------------------------------------------------

/// A permission mode entry that can be selected by an ACP client.
///
/// `defect-agent` does not know about the higher-level `SandboxMode` (that is a
/// `defect-config` concept; this crate is a dependency base of it and cannot have a
/// reverse dependency). The assembler (CLI) provides each
/// [`crate::policy::SandboxPolicy`] together with a stable `id`, a display `name`, and a
/// `description`; this crate only performs "look up by id and swap the active policy" on
/// opaque entries, corresponding one-to-one with the `SessionMode` in ACP
/// `session/set_mode`.
#[derive(Clone)]
pub struct PolicyMode {
    /// Stable identifier — the `mode_id` on the ACP wire. Convention is kebab-case (e.g.
    /// `ask-writes`), aligned with `SandboxMode::as_str()`.
    pub id: String,
    /// A human-readable name for display to clients.
    pub name: String,
    /// Optional description, shown in the client UI.
    pub description: Option<String>,
    /// The decision policy for this mode. When `set_mode` matches this entry, the policy
    /// is swapped in entirely.
    pub policy: Arc<dyn SandboxPolicy>,
}

impl std::fmt::Debug for PolicyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyMode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// A set of mutually exclusive permission modes plus the currently selected one. Maps to
/// ACP's `SessionModeState`.
///
/// Constructed once by the assembler (CLI) and flows into each session via
/// [`crate::session::AgentCore`]. Each session holds its own copy (`current` can be
/// switched independently); `set_mode` looks up the corresponding policy by id and swaps
/// it in.
#[derive(Debug, Clone)]
pub struct ModeCatalog {
    modes: Vec<PolicyMode>,
    current: String,
}

impl ModeCatalog {
    /// Constructs a catalog. `current` must match one of the `id`s in `modes`, otherwise
    /// returns `None`
    /// (assembly errors should fail loud, not silently fall back). An empty catalog also
    /// returns `None`.
    #[must_use]
    pub fn new(modes: Vec<PolicyMode>, current: impl Into<String>) -> Option<Self> {
        let current = current.into();
        if modes.is_empty() || !modes.iter().any(|m| m.id == current) {
            return None;
        }
        Some(Self { modes, current })
    }

    /// The ID of the currently selected mode.
    #[must_use]
    pub fn current_id(&self) -> &str {
        &self.current
    }

    /// The policy for the currently selected mode.
    #[must_use]
    pub fn current_policy(&self) -> Arc<dyn SandboxPolicy> {
        self.modes
            .iter()
            .find(|m| m.id == self.current)
            .map(|m| m.policy.clone())
            // Invariant: `current` always resolves to an entry (checked at construction
            // and on every `set`).
            .expect("ModeCatalog current id must always resolve to a mode")
    }

    /// All available modes, in assembly order.
    #[must_use]
    pub fn modes(&self) -> &[PolicyMode] {
        &self.modes
    }

    /// Switch the current mode. Returns `false` if `id` does not match any entry, leaving
    /// `current` unchanged.
    pub fn set_current(&mut self, id: &str) -> bool {
        if self.modes.iter().any(|m| m.id == id) {
            self.current = id.to_string();
            true
        } else {
            false
        }
    }
}

const ALLOW_ONCE_ID: &str = "allow_once";
const ALLOW_ALWAYS_ID: &str = "allow_always";
const REJECT_ONCE_ID: &str = "reject_once";

/// The default set of three `Ask` options: Allow once / Allow always / Reject once.
///
/// `RejectAlways` is not included in v0 — v0 has no need for persistent rejection; if the
/// user rejects once, the prompt will be shown again on the next invocation.
fn default_ask_options(tool_name: &str) -> Ask {
    let options = vec![
        AskOption {
            id: PermissionOptionId::new(ALLOW_ONCE_ID),
            name: format!("Allow `{tool_name}` once"),
            kind: PermissionOptionKind::AllowOnce,
            allows: true,
        },
        AskOption {
            id: PermissionOptionId::new(ALLOW_ALWAYS_ID),
            name: format!("Allow `{tool_name}` always"),
            kind: PermissionOptionKind::AllowAlways,
            allows: true,
        },
        AskOption {
            id: PermissionOptionId::new(REJECT_ONCE_ID),
            name: "Reject".to_string(),
            kind: PermissionOptionKind::RejectOnce,
            allows: false,
        },
    ];
    Ask { options }
}

#[cfg(test)]
mod tests;
