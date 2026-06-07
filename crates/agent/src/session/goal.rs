//! Shared state for the goal-driven loop.
//!
//! In `--goal` mode, the agent runs autonomously for multiple turns until the goal is
//! reached. The mechanism:
//! - `goal_done` tool ([`crate::tool::GoalDoneTool`]) — called when the AI believes the
//!   goal is reached, sets [`GoalState::reached`].
//! - `goal-gate` hook ([`crate::hooks::builtin::GoalGate`]) — when a turn voluntarily
//!   stops (`before_turn_end`), reads [`GoalState::is_reached`]: if reached, allows the
//!   loop to end; otherwise, extends the turn (injects a "continue working" feedback) and
//!   loops back for another round.
//!
//! Both share the same `Arc<GoalState>` across turn phases: the tool writes in one turn,
//! the hook reads in a later turn.
//!
//! ## Why a named struct instead of a generic state bag
//!
//! Following the existing pattern in [`crate::session::DefaultSession`] (where
//! `background`, `compaction_slot`, etc. are all purpose-specific named structs, not a
//! catch-all `HashMap<String, Value>`). It currently has only two fields, but it's a
//! struct —
//! future additions like `summary`, `reached_at`, sub-goal lists, etc. can be added as
//! fields. Since `ToolContext` and builtins hold an `Arc<GoalState>`, adding or removing
//! fields does not break the interface.

use std::sync::atomic::{AtomicBool, Ordering};

/// Shared state for one goal-driven loop.
#[derive(Debug)]
pub struct GoalState {
    /// The objective description passed via `--goal`. Injected into the `goal-gate`
    /// keepalive feedback so the model sees the goal each round.
    objective: String,
    /// Whether the goal has been reached. Set by the `goal_done` tool; read by the
    /// `goal-gate` hook.
    reached: AtomicBool,
}

impl GoalState {
    #[must_use]
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            reached: AtomicBool::new(false),
        }
    }

    /// The objective description.
    #[must_use]
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// Mark the objective as reached (via the `goal_done` tool call).
    pub fn mark_reached(&self) {
        self.reached.store(true, Ordering::SeqCst);
    }

    /// Whether the goal has been reached (for `goal-gate` hook evaluation).
    #[must_use]
    pub fn is_reached(&self) -> bool {
        self.reached.load(Ordering::SeqCst)
    }
}
