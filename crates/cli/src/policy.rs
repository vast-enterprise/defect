//! Maps a [`SandboxMode`] to a concrete [`SandboxPolicy`] instance.

use std::sync::Arc;

use defect_agent::policy::{
    AskWritesPolicy, DenyAllPolicy, ModeCatalog, OpenPolicy, PolicyMode, ReadOnlyPolicy,
    SandboxPolicy,
};
use defect_config::SandboxMode;

/// Selects the policy implementation based on `[sandbox].mode`.
pub fn build_policy(mode: SandboxMode) -> Arc<dyn SandboxPolicy> {
    match mode {
        SandboxMode::ReadOnly => Arc::new(ReadOnlyPolicy),
        SandboxMode::AskWrites => Arc::new(AskWritesPolicy::new()),
        SandboxMode::Open => Arc::new(OpenPolicy),
        SandboxMode::DenyAll => Arc::new(DenyAllPolicy),
    }
}

/// All sandbox modes in a fixed display order (read-only → ask-writes → open → deny-all).
/// `current` marks the currently selected item, mapping to the ACP `SessionModeState`.
///
/// Exposes **all** 4 modes to the client (`session/set_mode` can switch between them).
/// Mode IDs use [`SandboxMode::as_str`] — the same strings as the `[sandbox].mode` value
/// in the config file, providing a single source of truth.
pub fn build_mode_catalog(current: SandboxMode) -> ModeCatalog {
    let modes = [
        (
            SandboxMode::ReadOnly,
            "Read-only",
            "Allow read-only tools only; deny all writes, execution, and network access.",
        ),
        (
            SandboxMode::AskWrites,
            "Ask before writes",
            "Allow reads directly; ask for each write, execution, and network action, with the choice to allow once or always.",
        ),
        (
            SandboxMode::Open,
            "Open",
            "Allow everything without asking. Suitable for trusted environments / fully automated runs.",
        ),
        (
            SandboxMode::DenyAll,
            "Deny all",
            "Deny everything. For dry runs / look-but-don't-touch.",
        ),
    ]
    .into_iter()
    .map(|(mode, name, desc)| PolicyMode {
        id: mode.as_str().to_string(),
        name: name.to_string(),
        description: Some(desc.to_string()),
        policy: build_policy(mode),
    })
    .collect::<Vec<_>>();

    // Invariant: `current` must match one of the four modes above (`SandboxMode` is a
    // closed enum), so `ModeCatalog::new` always returns `Some` — if it doesn't, that's a
    // build bug; fail loud.
    ModeCatalog::new(modes, current.as_str())
        .expect("mode catalog must contain the current sandbox mode")
}
