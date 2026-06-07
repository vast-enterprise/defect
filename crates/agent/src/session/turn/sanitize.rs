//! Fix the message sequence before sending it to the provider.
//!
//! ## Why this is needed
//!
//! The turn main loop appends the assistant message (which may contain `tool_use` blocks)
//! and the subsequent `tool_result` messages in two separate steps (see `turn.rs`: first
//! append the assistant message, then append tool_results after the tool finishes). Any
//! interruption between these two steps — a permanent LLM call failure, `session/cancel`,
//! a SIGKILL, a power outage — will leave **orphan tool_use** blocks in the persisted
//! history: a `tool_use` without a corresponding `tool_result`.
//!
//! Anthropic / Bedrock enforces that "every tool_use must be immediately followed by its
//! tool_result". Once an orphan reaches the request, it is permanently rejected
//! (`tool_use ids were found without tool_result blocks`). Every subsequent request for
//! that session will fail — the session is dead. Such interruptions are **unavoidable**
//! (the process can be killed at any time), so fault tolerance must be on the **read side
//! (before sending the request)**, not just on the write side.
//!
//! ## Placement: at request construction, not in snapshot
//!
//! The fix is applied **only at the point where the request to the provider is
//! constructed** (`build_request` in `turn.rs`, and summary sub-requests in
//! `compact.rs`). It is **not** placed in `History::snapshot()`:
//!
//! - The semantics of `snapshot` are "faithfully read the current real state of the
//!   history" — injecting synthetic results there would make "what is read" ≠ "what is
//!   stored", breaking the read-only contract.
//! - More critically: micro/full compaction goes through `snapshot() → process →
//!   replace()`, which **writes** the result back into the history. If `snapshot` already
//!   injected synthetic results, those blocks — which should only be visible to the
//!   provider — would be persisted by `replace`, leaking temporary modifications into the
//!   source of truth (a second source of truth).
//!
//! Therefore: `snapshot` / `replace` / storage records / UI replay all read the **real**
//! sequence (orphans exist as-is, the UI shows them as unclosed tool_calls — a faithful
//! representation of the real state). Only the copy sent to the provider is patched to be
//! wire-legal.

use crate::llm::{Message, MessageContent, Role, ToolResultBody};

/// Synthetic tool_result text inserted for an interrupted tool_use.
const INTERRUPTED_RESULT_TEXT: &str = "tool call interrupted; no result was recorded";

/// For each `ToolUse` that lacks a following `ToolResult`, inject a synthetic error
/// `ToolResult` so that the sequence satisfies the provider constraint that every
/// `ToolUse` is immediately followed by its corresponding `ToolResult`.
///
/// Pairing rule: the result for each `ToolUse{id}` in an assistant message should appear
/// in the **immediately following message** (in the Anthropic format, the `ToolResult` is
/// in the next user message). This function collects the set of `tool_use_id`s that are
/// already satisfied by some message, and for any unsatisfied ones, inserts a synthetic
/// `ToolResult` message with `Role::User` **after** the assistant message.
///
/// If there are no orphans, the input is returned unchanged (a single O(n) scan with zero
/// allocation changes).
pub(crate) fn sanitize_tool_pairing(messages: Vec<Message>) -> Vec<Message> {
    // First pass: which `tool_use_id` values have a corresponding `tool_result` anywhere
    // in the sequence.
    // Use "global existence" rather than "strict adjacency" — in a well-formed persisted
    // history a result always immediately follows its use, so only interrupted sequences
    // are missing entirely; global checking is more lenient and won't penalize legitimate
    // multi-block layouts.
    let mut satisfied: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in &messages {
        for content in msg.content.iter() {
            if let MessageContent::ToolResult { tool_use_id, .. } = content {
                satisfied.insert(tool_use_id.clone());
            }
        }
    }

    // Return early if there are no orphans, to avoid rebuilding the Vec.
    let has_orphan = messages.iter().any(|msg| {
        msg.content
            .iter()
            .any(|c| matches!(c, MessageContent::ToolUse { id, .. } if !satisfied.contains(id)))
    });
    if !has_orphan {
        return messages;
    }

    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 1);
    for msg in messages {
        // Collect orphan tool_use IDs from this assistant message, preserving their order
        // of appearance.
        let orphans: Vec<String> = msg
            .content
            .iter()
            .filter_map(|c| match c {
                MessageContent::ToolUse { id, .. } if !satisfied.contains(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        out.push(msg);
        if !orphans.is_empty() {
            out.push(Message {
                role: Role::User,
                content: orphans
                    .into_iter()
                    .map(|id| MessageContent::ToolResult {
                        tool_use_id: id,
                        output: ToolResultBody::Text {
                            text: INTERRUPTED_RESULT_TEXT.to_string(),
                        },
                        is_error: true,
                    })
                    .collect::<Vec<_>>()
                    .into(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests;
