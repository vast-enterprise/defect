//! Microcompact — cheap first line of defense for context.
//!
//! Unlike full compaction (`compact.rs`): **no LLM call**, **no message deletion**. It
//! only replaces oversized `tool_result` bodies from **older turns** with a placeholder.
//! The structure (message count, role, `tool_use`↔`tool_result` pairing) is untouched —
//! so the wire format is always valid, and the model still sees which tools it called,
//! just not the (often irrelevant) large tool output bodies.
//!
//! Pure `O(n)` data transformation, zero network round trips, safe to run synchronously
//! in the turn main loop (aligns with Claude Code's microcompact). It often cuts down the
//! biggest tool output contributors, deferring expensive full compaction.
//!
//! ## Safety constraints
//!
//! 1. **No message addition or removal** — only rewrites the `output` field of
//!    `ToolResult`. This preserves `History`'s concurrency invariants (see
//!    `splice_prefix`) and coexists with in-flight background full compaction.
//! 2. **Retention window**: tool results from the most recent [`KEEP_RECENT_TURNS`] turns
//!    are kept as-is — they're likely still in use. Only older turns are cleared.
//! 3. **Size floor**: tool results smaller than [`MIN_CLEAR_TOKENS`] are left alone (not
//!    worth it).
//! 4. **Idempotent**: the placeholder text itself acts as a sentinel; already-cleared
//!    entries are skipped on re-run.

use crate::llm::{Message, MessageContent, ToolResultBody};
use crate::session::history::estimate_message_tokens;

use super::compact::is_turn_start;

/// Keep the last N turns of tool_result intact.
const KEEP_RECENT_TURNS: usize = 3;

/// Size floor: tool results whose estimated token count does not exceed this value are
/// not worth clearing.
const MIN_CLEAR_TOKENS: u64 = 512;

/// Placeholder text inserted after clearing. Both informs the model that the output has
/// been reclaimed and serves as an idempotent sentinel.
pub(super) const CLEARED_PLACEHOLDER: &str =
    "[tool output cleared to save context — re-run the tool if its result is needed again]";

/// Microcompact report: estimated total tokens before and after clearing. `cleared` is
/// the number of `tool_result` entries actually removed.
pub(super) struct MicrocompactReport {
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub cleared: usize,
}

/// Runs micro-compaction on `messages`, returning `Some(rebuilt, report)` if any messages
/// were cleared, or `None` if nothing could be cleared (callers use this to skip
/// write-back and event emission).
///
/// Pure function: does not touch `History`; the caller decides how to write back
/// (currently via `replace`).
pub(super) fn run(messages: &[Message]) -> Option<(Vec<Message>, MicrocompactReport)> {
    // Find the retention boundary: keep all turns starting from the
    // `KEEP_RECENT_TURNS`-th turn start from the end (inclusive).
    let turn_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_turn_start(m))
        .map(|(i, _)| i)
        .collect();

    // Keep boundary index: if there are fewer turns than the retention window, there are
    // no older turns to prune, so skip directly.
    // The start of the `KEEP_RECENT_TURNS`-th turn from the end is the boundary;
    // `nth_back(K-1)` retrieves it.
    // If there are fewer than K turns, it returns `None` → skip (avoiding a potential
    // panic from raw indexing).
    let keep_from = *turn_starts.iter().rev().nth(KEEP_RECENT_TURNS - 1)?;

    let tokens_before = estimate_total(messages);
    let mut cleared = 0usize;

    let rebuilt: Vec<Message> = messages
        .iter()
        .enumerate()
        .map(|(idx, msg)| {
            // Pass through messages within the window unchanged.
            if idx >= keep_from {
                return msg.clone();
            }
            clear_oversized_results(msg, &mut cleared)
        })
        .collect();

    if cleared == 0 {
        return None;
    }

    let tokens_after = estimate_total(&rebuilt);
    Some((
        rebuilt,
        MicrocompactReport {
            tokens_before,
            tokens_after,
            cleared,
        },
    ))
}

/// Replace the body of any `ToolResult` that is both oversized and not yet cleared with a
/// placeholder; leave all other content blocks unchanged.
/// Increments `cleared` by 1 for each replacement.
fn clear_oversized_results(msg: &Message, cleared: &mut usize) -> Message {
    // First check whether this message has any ToolResult — most messages do not,
    // avoiding an unnecessary clone.
    let has_tool_result = msg
        .content
        .iter()
        .any(|c| matches!(c, MessageContent::ToolResult { .. }));
    if !has_tool_result {
        return msg.clone();
    }

    let content: Vec<MessageContent> = msg
        .content
        .iter()
        .map(|c| match c {
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error,
            } if should_clear(output) => {
                *cleared += 1;
                MessageContent::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    output: ToolResultBody::Text {
                        text: CLEARED_PLACEHOLDER.to_string(),
                    },
                    // Preserve `is_error`: the model still knows that this tool call
                    // originally failed.
                    is_error: *is_error,
                }
            }
            other => other.clone(),
        })
        .collect();

    Message {
        role: msg.role,
        content: content.into(),
    }
}

/// Whether this `tool_result` should be cleared: exceeds the size threshold and has not
/// been cleared yet (idempotent).
fn should_clear(output: &ToolResultBody) -> bool {
    if already_cleared(output) {
        return false;
    }
    estimate_tool_result_tokens(output) > MIN_CLEAR_TOKENS
}

/// Whether this is already the cleared placeholder (idempotent sentinel).
fn already_cleared(output: &ToolResultBody) -> bool {
    matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER)
}

/// Estimates tokens for a single `tool_result` by reusing `estimate_message_tokens` with
/// a temporary message containing only that result, keeping the same metric as
/// compression and trigger decisions without introducing a separate standard.
fn estimate_tool_result_tokens(output: &ToolResultBody) -> u64 {
    let probe = Message {
        role: crate::llm::Role::User,
        content: vec![MessageContent::ToolResult {
            tool_use_id: String::new(),
            output: output.clone(),
            is_error: false,
        }]
        .into(),
    };
    estimate_message_tokens(&probe)
}

fn estimate_total(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

#[cfg(test)]
mod tests;
