//! Concrete implementation of [`History`] as [`VecHistory`]: `Vec<Message>` + token
//! accounting.
//!
//! Pure storage, no compression — compression is orchestrated in the turn main loop
//! (`session/turn/compact.rs`).
//! History — design tradeoffs for conversation history representation.
//!
//! ## Token estimation
//!
//! No tokenizer dependency (aligned with opencode: trigger uses real usage, internal
//! estimation uses
//! character heuristics). Two segments are combined:
//! - **Baseline**: the real input token count from the last LLM call
//!   (`record_input_tokens`),
//!   fed in by the turn main loop after each call. This is the most accurate segment.
//! - **Delta**: messages `append`ed after the baseline are estimated as `chars/4` and
//!   accumulated — these
//!   have not yet been sent to the LLM, so no real token count is available.
//!
//! `replace` (write-back after compression) clears the baseline: the new list's token
//! count must wait
//! for the next real call report. When the baseline is missing (session just created, or
//! just after
//! `replace`), the entire snapshot falls back to character heuristics.

use std::sync::Mutex;

use crate::llm::{Message, MessageContent};
use crate::session::History;

/// Multimodal images are counted as a fixed token cost in character estimation, aligning
/// with Claude Code microcompact's image counting (cannot estimate by characters, so a
/// conservative constant is used).
const IMAGE_TOKEN_ESTIMATE: usize = 2_000;

/// Heuristic character-to-token ratio: `chars / 4` (aligned with codex / opencode).
const CHARS_PER_TOKEN: usize = 4;

/// A [`History`] implementation backed by `Vec<Message>` + `Mutex`, with token
/// accounting.
#[derive(Default)]
pub struct VecHistory {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    messages: Vec<Message>,
    /// Real input tokens reported by the last LLM call. `None` means no real baseline
    /// exists yet (freshly created or just replaced), so `token_estimate` falls back
    /// entirely to character heuristics.
    last_real_input: Option<u64>,
    /// Accumulated character-heuristic token estimate for messages `append`ed after the
    /// last real baseline.
    est_since_baseline: u64,
}

impl VecHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                messages,
                last_real_input: None,
                est_since_baseline: 0,
            }),
        }
    }
}

impl History for VecHistory {
    fn append(&self, msg: Message) {
        let mut inner = self.inner.lock().expect("VecHistory mutex poisoned");
        // When a baseline exists, the estimate for new messages is accumulated separately
        // into the delta; when no baseline is set, no accumulation is needed (the entire
        // `token_estimate` will be recomputed).
        if inner.last_real_input.is_some() {
            inner.est_since_baseline = inner
                .est_since_baseline
                .saturating_add(estimate_message_tokens(&msg));
        }
        inner.messages.push(msg);
    }

    fn snapshot(&self) -> Vec<Message> {
        self.inner
            .lock()
            .expect("VecHistory mutex poisoned")
            .messages
            .clone()
    }

    fn replace(&self, messages: Vec<Message>) {
        let mut inner = self.inner.lock().expect("VecHistory mutex poisoned");
        inner.messages = messages;
        // The true token count of the new list is unknown; it will be reported on the
        // next LLM call.
        inner.last_real_input = None;
        inner.est_since_baseline = 0;
    }

    fn splice_prefix(&self, drop_count: usize, summary: Message) -> usize {
        let mut inner = self.inner.lock().expect("VecHistory mutex poisoned");
        // Invariant check: `drop_count` was computed from a snapshot at some earlier
        // point; by the time it is applied, the list should only have grown (via append)
        // or been replaced in place — it must not be **shorter**. If the current length
        // is less than `drop_count`, it means a mid-list deletion happened in flight,
        // violating the single-flight invariant (see `session.rs` docs). In debug builds
        // this assertion catches the bug; in release builds the `clamp` below prevents a
        // panic.
        debug_assert!(
            drop_count <= inner.messages.len(),
            "splice_prefix invariant violated: drop_count={drop_count} > current len={}; \
             history shrank mid-flight (concurrent mid-list deletion?)",
            inner.messages.len()
        );
        // Clamp to current length — concurrent tail insertion only grows the list, so
        // `drop_count` should never exceed it, but clamping is a cheap safety net (even
        // if an old snapshot is longer than the current list under extreme races, this
        // won't panic).
        let drop_count = drop_count.min(inner.messages.len());
        let tail = inner.messages.split_off(drop_count);
        inner.messages = Vec::with_capacity(tail.len() + 1);
        inner.messages.push(summary);
        inner.messages.extend(tail);
        // Same as `replace`: the true token count of the new prefix is unknown; it will
        // be reported by the next LLM call.
        inner.last_real_input = None;
        inner.est_since_baseline = 0;
        drop_count
    }

    fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("VecHistory mutex poisoned")
            .messages
            .len()
    }

    fn truncate(&self, len: usize) {
        let mut inner = self.inner.lock().expect("VecHistory mutex poisoned");
        if len >= inner.messages.len() {
            return;
        }
        inner.messages.truncate(len);
        // The dropped messages may have fed into `est_since_baseline`; the cheapest correct
        // fix is to reset the baseline, same as `replace` — the next LLM call reports the
        // true count.
        inner.last_real_input = None;
        inner.est_since_baseline = 0;
    }

    fn record_input_tokens(&self, tokens: u64) {
        let mut inner = self.inner.lock().expect("VecHistory mutex poisoned");
        inner.last_real_input = Some(tokens);
        // Baseline reset — subsequent appends count their delta from zero.
        inner.est_since_baseline = 0;
    }

    fn token_estimate(&self) -> Option<u64> {
        let inner = self.inner.lock().expect("VecHistory mutex poisoned");
        match inner.last_real_input {
            // With a real baseline: baseline + character-heuristic increment for messages
            // added after it.
            Some(real) => Some(real.saturating_add(inner.est_since_baseline)),
            // No baseline: fall back to character heuristics for the entire history.
            // Returns `None` if history is empty.
            None => {
                if inner.messages.is_empty() {
                    return None;
                }
                Some(
                    inner
                        .messages
                        .iter()
                        .map(estimate_message_tokens)
                        .fold(0u64, u64::saturating_add),
                )
            }
        }
    }
}

/// Character-based heuristic token estimate for a single message (`chars/4`, images count
/// as a constant).
///
/// `pub(crate)`: the compaction module (`session/turn/compact.rs`) reuses the same ruler
/// when selecting retention boundaries, preventing drift between two estimation sites.
pub(crate) fn estimate_message_tokens(msg: &Message) -> u64 {
    let chars: usize = msg
        .content
        .iter()
        .map(|c| match c {
            MessageContent::Text { text } => text.len() / CHARS_PER_TOKEN,
            MessageContent::Thinking { text, signature } => {
                (text.len() + signature.as_ref().map_or(0, |s| s.len())) / CHARS_PER_TOKEN
            }
            MessageContent::ToolUse { name, args, .. } => {
                (name.len() + args.to_string().len()) / CHARS_PER_TOKEN
            }
            MessageContent::ToolResult { output, .. } => {
                tool_result_chars(output) / CHARS_PER_TOKEN
            }
            MessageContent::Image { .. } => IMAGE_TOKEN_ESTIMATE,
            // The payload of a hosted activity is not persisted across processes, so it
            // is ignored in the estimate.
            MessageContent::ProviderActivity { .. } => 0,
        })
        .sum();
    chars as u64
}

fn tool_result_chars(output: &crate::llm::ToolResultBody) -> usize {
    use crate::llm::{ToolResultBody, ToolResultContent};
    match output {
        ToolResultBody::Text { text } => text.len(),
        ToolResultBody::Json { value } => value.to_string().len(),
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text.len(),
                ToolResultContent::Image { data, .. } => image_data_chars(data),
            })
            .sum(),
    }
}

/// Approximate character count for an image block: base64 string length or URL length.
/// Used for estimation and compression decisions; exact precision is not required.
fn image_data_chars(data: &crate::llm::ImageData) -> usize {
    match data {
        crate::llm::ImageData::Base64 { encoded } => encoded.len(),
        crate::llm::ImageData::Url { url } => url.len(),
    }
}

#[cfg(test)]
mod tests;
