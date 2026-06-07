//! Background compression with single-flight semantics.
//!
//! Full compression (summarization) calls an LLM, incurring several seconds of latency.
//! Running it synchronously on the turn critical path would force the user whose turn
//! triggered the overflow to wait. Background compression moves this summarization call
//! into a `tokio::spawn` task, so the turn does not block — it quietly compacts history
//! before hitting the hard watermark.
//!
//! ## Why session-level and single-flight
//!
//! - **Single-flight**: At most one compression is in flight at a time. This is a
//!   prerequisite for `History::splice_prefix`'s concurrency invariant — "no mid-section
//!   messages are added or removed while a compression is in flight" — and the only
//!   operation that removes mid-section messages is compression itself. Two concurrent
//!   compressions would invalidate each other's computed `drop_count`.
//! - **Session-level**: The compression task must outlive the turn that spawned it (the
//!   turn may end before the summary returns), so the `JoinHandle` is attached to the
//!   session, sharing the same lifetime as
//!   [`BackgroundTasks`](crate::session::BackgroundTasks).
//!
//! ## Write-back
//!
//! After the task completes, it directly calls `history.splice_prefix(drop_count,
//! summary)` to **silently rewrite history** — note the difference from
//! [`BackgroundTasks`](crate::session::BackgroundTasks): that path injects the result
//! back into the conversation as a user message, whereas compression must be silent. On
//! completion, a callback emits a `ContextCompressed` event for observability
//! consumption.

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use tokio::task::JoinHandle;

use super::compact::{self, CompactionCtx};
use crate::session::{CompactionReport, History};

/// Completion callback: emits an event after receiving a compaction report. Returns a
/// `BoxFuture` so the callback can be `await`ed (since `emit` is async) — the compaction
/// task body directly `.await`s it, so event sending completes inside the compaction task
/// under the same cancel/track constraints, without spawning a detached task (following
/// the workspace `BoxFuture` convention).
type OnDone = Arc<dyn Fn(CompactionReport) -> BoxFuture<'static, ()> + Send + Sync>;

struct SlotInner {
    /// The in-flight compaction task. `Some` means one is running; the task clears this
    /// field on completion.
    flight: Option<JoinHandle<()>>,
}

/// A session-level background compaction slot. `Clone` is cheap (internally an `Arc`).
#[derive(Clone)]
pub struct CompactionSlot {
    inner: Arc<Mutex<SlotInner>>,
}

impl Default for CompactionSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactionSlot {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SlotInner { flight: None })),
        }
    }

    /// Whether a compaction is currently in flight.
    #[must_use]
    pub(crate) fn is_in_flight(&self) -> bool {
        let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
        // Clean up any finished handle (the task body has already written itself back;
        // this only clears the bookkeeping).
        if let Some(h) = &inner.flight
            && h.is_finished()
        {
            inner.flight = None;
        }
        inner.flight.is_some()
    }

    /// Attempts to start a background compaction. If one is already in flight, returns
    /// `false` (single-flight, no duplicate spawns).
    ///
    /// `history` is `Arc<dyn History>` so the `'static` task can hold it across turns.
    /// `ctx` carries provider / model / tools / cancellation token. `on_done` is called
    /// after a successful compaction (to emit an event).
    pub(crate) fn try_spawn(
        &self,
        history: Arc<dyn History>,
        ctx: CompactionCtx,
        threshold: u64,
        on_done: OnDone,
    ) -> bool {
        let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
        // Clear any finished handles to prevent a stale completed handle from blocking
        // new ones.
        if let Some(h) = &inner.flight
            && h.is_finished()
        {
            inner.flight = None;
        }
        if inner.flight.is_some() {
            return false;
        }

        let slot = self.inner.clone();
        let handle = tokio::spawn(async move {
            run_once(history.as_ref(), &ctx, threshold, &on_done).await;
            // Cleanup: remove this task's handle placeholder (only if it still points to
            // us — i.e., the task has finished).
            if let Ok(mut inner) = slot.lock()
                && let Some(h) = &inner.flight
                && h.is_finished()
            {
                inner.flight = None;
            }
        });
        inner.flight = Some(handle);
        true
    }

    /// Waits for an in-flight compaction to complete, if any. Used as a fallback for the
    /// hard watermark: if a background compaction is already in progress when a new
    /// request is about to be constructed, it is better to wait for it to finish than to
    /// start another one synchronously. Returns immediately if no task is in flight.
    pub(crate) async fn await_in_flight(&self) {
        let handle = {
            let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
            inner.flight.take()
        };
        if let Some(handle) = handle {
            // The task already wrote back its history; we only wait for it to finish
            // here. Ignore `JoinError` (panic/abort).
            let _ = handle.await;
        }
    }
}

/// Run one background compaction cycle: snapshot → plan → summarize → write back via
/// `splice_prefix` → emit event.
/// Any best-effort skip (no boundary / summary failure) returns silently.
async fn run_once(history: &dyn History, ctx: &CompactionCtx, threshold: u64, on_done: &OnDone) {
    let messages = history.snapshot();
    let Some(plan) = compact::plan(&messages, threshold) else {
        return;
    };
    let Some(summary) = compact::summarize(ctx, &plan.head, plan.prev_summary.as_deref()).await
    else {
        return;
    };
    let summary_msg = compact::summary_message(&summary);

    // Key: `splice_prefix` drops only the first `drop_count` items from the current list,
    // keeping everything after (including tail messages appended by the foreground while
    // the summary was being computed). Single-flight ensures `drop_count` is still valid
    // for the current list.
    history.splice_prefix(plan.drop_count, summary_msg);
    let tokens_after = history.token_estimate().unwrap_or(plan.tokens_before);

    tracing::info!(
        drop_count = plan.drop_count,
        tokens_before = plan.tokens_before,
        tokens_after,
        "context compacted (background)"
    );
    on_done(CompactionReport {
        tokens_before: plan.tokens_before,
        tokens_after,
    })
    .await;
}
