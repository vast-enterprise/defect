//! Context compression orchestration.
//!
//! Compression is **not** performed inside [`crate::session::History`] — summarization
//! requires calling the LLM, and the storage abstraction has no access to the provider.
//! Instead, orchestration lives at the turn main-loop level (aligned with codex
//! `compact.rs` / opencode `compaction.ts` / Claude Code `services/compact`).
//!
//! A single compression pass:
//! 1. [`select_boundary`]: Split history into a "head to summarize" prefix and a "tail to
//!    keep as-is". The boundary **aligns to turn boundaries** (real user messages),
//!    ensuring the tail starts with a valid user turn and never splits a
//!    `tool_use`↔`tool_result` pair (neither wire codec validates pairing — we must
//!    guarantee it ourselves; see `crates/llm/src/protocol/*`).
//! 2. [`summarize`]: Run a text-only sub-request against the current provider/model for
//!    the head, requiring output in a fixed structured template; if an old summary is
//!    detected, perform incremental merging.
//! 3. Rebuild history: `[synthesized assistant summary message] ++ tail`, written back
//!    via [`History::replace`](crate::session::History::replace).
//!
//! On failure (no safe boundary / provider error / empty summary / cancellation), always
//! **best-effort** degrade: skip this compression pass, don't kill the turn — the next
//! real call will hit the context limit on its own.

use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::llm::{
    CompletionRequest, HostedCapabilities, LlmProvider, Message, MessageContent, ProviderChunk,
    Role, SamplingParams, StopReason, ThinkingConfig, ToolChoice, ToolResultBody,
    ToolResultContent,
};
use crate::session::CompactionReport;
use crate::session::history::estimate_message_tokens;
use crate::tool::ToolSchema;

/// Lower / upper bound for tail token budget (aligned with opencode's 2k–8k).
const MIN_TAIL_TOKENS: u64 = 2_000;
const MAX_TAIL_TOKENS: u64 = 8_000;

/// Maximum characters per tool_result in the head when fed to the summarizer, to prevent
/// a single oversized tool output from blowing up the summarization request (aligned with
/// opencode `toolOutputMaxChars: 2000`).
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// Self-descriptive prefix for the synthesized summary message. It gives the summarizer
/// model context and also lets **later compression** recognize that "this is a compressed
/// summary from a previous round", enabling incremental merging instead of treating it as
/// duplicate history.
pub(super) const SUMMARY_PREFIX: &str =
    "[Compacted context summary — earlier conversation was condensed to save context.]";

/// Fixed system prompt for summarization sub-requests.
const SUMMARIZER_SYSTEM: &str = "\
You are a context-summarization assistant for a coding agent session. You are given the \
earlier part of a conversation that is about to be dropped to free up context. Summarize \
ONLY what you are given. The newest turns are kept verbatim outside your summary, so focus \
on older context that still matters for continuing the work.

If a <previous-summary> block is present, treat it as the current anchored summary and UPDATE \
it: keep still-true facts, drop stale ones, merge in new facts. Always follow the exact \
section structure the user asks for, keep every section even if empty, preserve exact file \
paths / identifiers / commands / error strings, and prefer terse bullets over prose. Do not \
answer or continue the task itself, and do not mention that you are summarizing. Respond in \
the same language as the conversation.";

/// Structured summary template appended to the end of the user prompt.
const SUMMARY_TEMPLATE: &str = "\
Summarize the conversation above into the following Markdown structure. Keep every heading \
even if a section is empty (write `(none)`):

## Goal
The user's overall objective and the current concrete task.

## Constraints & Preferences
Hard requirements, user preferences, and conventions to respect.

## Progress
### Done
### In Progress
### Blocked

## Key Decisions
Important choices made and why.

## Next Steps
Concrete, ordered next actions to continue the work.

## Key Context
Critical facts, data, snippets, or references needed to continue.

## Relevant Files
`path` — why it matters (one per line).";

/// Immutable context for a compaction task — the minimal set of dependencies extracted
/// from [`super::TurnRunner`] needed to produce a single summary. All fields are owned or
/// `Arc`, so the context is `'static` and can be held by a background compaction task
/// spawned via `tokio::spawn`. The synchronous fallback path also uses this same context,
/// so both paths share the same summarization logic.
#[derive(Clone)]
pub(crate) struct CompactionCtx {
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
    pub sampling: SamplingParams,
    pub tools: Vec<ToolSchema>,
    pub cancel: CancellationToken,
}

/// A plan for one compaction: the result of selecting boundaries on a snapshot.
/// `drop_count` is the head length (= number of prefix messages to summarize and
/// discard), passed to `History::splice_prefix` when writing back.
pub(super) struct CompactionPlan {
    /// The prefix (`head`) to be summarized.
    pub head: Vec<Message>,
    /// The previous compaction summary, if found in the head, used for incremental
    /// merging.
    pub prev_summary: Option<String>,
    /// Number of prefix messages to discard.
    pub drop_count: usize,
    /// Estimated token count of the full segment (head + tail) before compaction.
    pub tokens_before: u64,
}

/// Pure computation: selects a boundary in `messages` based on `threshold` and extracts
/// the head. Returns `None` when no safe boundary exists (e.g., a single overly long turn
/// or only one turn), letting the caller skip. Does not touch `History` or call the LLM.
pub(super) fn plan(messages: &[Message], threshold: u64) -> Option<CompactionPlan> {
    let tail_budget = (threshold / 4).clamp(MIN_TAIL_TOKENS, MAX_TAIL_TOKENS);
    let Some(boundary) = select_boundary(messages, tail_budget) else {
        tracing::warn!(
            messages = messages.len(),
            tail_budget,
            "compaction skipped: no safe turn boundary to summarize before"
        );
        return None;
    };
    let (head, _tail) = messages.split_at(boundary);
    let prev_summary = extract_previous_summary(head);
    Some(CompactionPlan {
        head: head.to_vec(),
        prev_summary,
        drop_count: boundary,
        tokens_before: estimate_total(messages),
    })
}

/// Wraps the summary text as a synthetic assistant summary message (prefixed with
/// [`SUMMARY_PREFIX`]).
pub(super) fn summary_message(summary: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![MessageContent::Text {
            text: format!("{SUMMARY_PREFIX}\n{summary}"),
        }]
        .into(),
    }
}

/// Synchronous compaction (hard watermark fallback / background shutdown): runs a full
/// compaction and write-back blocking inside the turn main loop.
/// Returns `Some(report)` on success (caller emits `ContextCompressed`); `None` to
/// best-effort skip.
///
/// Uses `splice_prefix(plan.drop_count, ..)` instead of `replace`: shares the same
/// write-back primitive as the background path, keeping semantics consistent — here there
/// is no concurrent tail insertion between snapshot and write-back, so `drop_count` is
/// equivalent to the entire table prefix.
pub(super) async fn run_sync(
    history: &dyn crate::session::History,
    ctx: &CompactionCtx,
    threshold: u64,
) -> Option<CompactionReport> {
    let messages = history.snapshot();
    let plan = plan(&messages, threshold)?;
    let summary = summarize(ctx, &plan.head, plan.prev_summary.as_deref()).await?;
    let summary_msg = summary_message(&summary);

    history.splice_prefix(plan.drop_count, summary_msg);
    let tokens_after = estimate_total(&history.snapshot());

    tracing::info!(
        drop_count = plan.drop_count,
        tokens_before = plan.tokens_before,
        tokens_after,
        "context compacted (sync)"
    );
    Some(CompactionReport {
        tokens_before: plan.tokens_before,
        tokens_after,
    })
}

/// Select the retention boundary: returns the index of the **first message to keep** (the
/// start of the tail).
///
/// - A "turn start" is a message with `role == User` that contains at least one content
///   block that is not `ToolResult` (i.e., a real user input, not a tool-result
///   backfill).
/// - Walk from the newest turn backward, accumulating tail size using a character-based
///   heuristic, keeping entire turns until `tail_budget` is exceeded.
/// - The boundary must be `> 0` (so the head is non-empty and can be summarized). If
///   there is only one turn (the newest turn starts at index 0) → return `None` (no
///   earlier history to summarize).
/// - If even the newest turn exceeds the budget (a single overly long turn), still use
///   that turn's start as the boundary (do not split inside a user message) and summarize
///   everything before it — provided that start is `> 0`.
fn select_boundary(messages: &[Message], tail_budget: u64) -> Option<usize> {
    let turn_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_turn_start(m))
        .map(|(i, _)| i)
        .collect();

    let last_start = *turn_starts.last()?;
    // Only one turn (or the latest turn starts at the beginning) → no earlier history to
    // summarize.
    if last_start == 0 {
        return None;
    }

    // Accumulate from the newest turn start backward; track the oldest start that still
    // fits and is >0.
    let mut best: Option<usize> = None;
    let mut acc: u64 = 0;
    let mut next_boundary = messages.len();
    for &start in turn_starts.iter().rev() {
        acc = acc.saturating_add(estimate_range(messages, start, next_boundary));
        next_boundary = start;
        if start == 0 {
            break;
        }
        if acc <= tail_budget {
            best = Some(start);
        } else {
            break;
        }
    }

    // If `best` is set, use it; otherwise even the latest turn exceeds the budget, so
    // fall back to the start of the latest turn (`last_start` is guaranteed > 0).
    Some(best.unwrap_or(last_start))
}

/// Whether this is a "turn start": a real user input message.
///
/// `pub(super)` so that micro-compaction (`session/turn/microcompact.rs`) reuses the same
/// turn-start ruler,
/// avoiding drift between two places that determine turn starts.
pub(super) fn is_turn_start(msg: &Message) -> bool {
    msg.role == Role::User
        && msg
            .content
            .iter()
            .any(|c| !matches!(c, MessageContent::ToolResult { .. }))
}

fn estimate_range(messages: &[Message], start: usize, end: usize) -> u64 {
    messages
        .iter()
        .take(end)
        .skip(start)
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

fn estimate_total(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

/// Finds the previous round's compressed summary in `head` (assistant text starting with
/// [`SUMMARY_PREFIX`]) and returns its body with the prefix removed. Used for incremental
/// merging.
fn extract_previous_summary(head: &[Message]) -> Option<String> {
    head.iter()
        .filter(|m| m.role == Role::Assistant)
        .find_map(|m| {
            m.content.iter().find_map(|c| match c {
                MessageContent::Text { text } => text
                    .strip_prefix(SUMMARY_PREFIX)
                    .map(|rest| rest.trim_start().to_string()),
                _ => None,
            })
        })
}

/// Runs a text-only summarization sub-request on `head` and returns the summary body.
/// Any failure (cancellation, provider error, empty result) → `None` (caller degrades and
/// skips).
pub(super) async fn summarize(
    ctx: &CompactionCtx,
    head: &[Message],
    prev_summary: Option<&str>,
) -> Option<String> {
    let mut messages: Vec<Message> = head.iter().map(prepare_head_message).collect();
    messages.push(Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: build_prompt(prev_summary),
        }]
        .into(),
    });
    // The head slice may contain orphaned `tool_use` blocks left over from an
    // interruption; these must be paired before sending the summarization sub-request, or
    // the provider will reject it. This is the same step as in `build_request`.
    let messages = super::sanitize::sanitize_tool_pairing(messages);

    let req = CompletionRequest {
        model: ctx.model.clone(),
        system: Some(SUMMARIZER_SYSTEM.into()),
        messages,
        // Include the tools schema so that `tool_use`/`tool_result` history in the head
        // is valid on the wire, but set `tool_choice=None` to prevent the summarizer from
        // actually calling tools — it should only produce text.
        tools: ctx.tools.clone(),
        tool_choice: ToolChoice::None,
        sampling: SamplingParams {
            // Summarization does not need a thinking chain; disable it to save tokens.
            thinking: ThinkingConfig::Disabled,
            ..ctx.sampling.clone()
        },
        hosted_capabilities: HostedCapabilities::default(),
    };

    let mut stream = match ctx.provider.complete(req, ctx.cancel.clone()).await {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(error = %err, "compaction summarize failed: provider error");
            return None;
        }
    };

    let mut text = String::new();
    loop {
        tokio::select! {
            biased;
            () = ctx.cancel.cancelled() => {
                tracing::warn!("compaction summarize cancelled");
                return None;
            }
            next = stream.next() => match next {
                None => break,
                Some(Ok(ProviderChunk::TextDelta { text: delta })) => text.push_str(&delta),
                Some(Ok(ProviderChunk::Stop { reason })) => {
                    if matches!(reason, StopReason::Refusal) {
                        tracing::warn!("compaction summarize refused by model");
                        return None;
                    }
                    // Ignore remaining chunks (thinking / tool_use / usage /
                    // message_start).
                }
                Some(Ok(_)) => {}
                Some(Err(err)) => {
                    tracing::warn!(error = %err, "compaction summarize failed: stream error");
                    return None;
                }
            }
        }
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        tracing::warn!("compaction summarize produced empty summary");
        return None;
    }
    Some(text)
}

/// Build the user prompt for summarization: if a previous summary exists, prepend it
/// inside a `<previous-summary>` incremental block.
fn build_prompt(prev_summary: Option<&str>) -> String {
    match prev_summary {
        Some(prev) => format!(
            "Update the anchored summary below with the new conversation history.\n\n\
             <previous-summary>\n{prev}\n</previous-summary>\n\n{SUMMARY_TEMPLATE}"
        ),
        None => SUMMARY_TEMPLATE.to_string(),
    }
}

/// Prepare a single message from the head for the summarization model: truncate overly
/// long `tool_result` and strip images.
fn prepare_head_message(msg: &Message) -> Message {
    let content: Vec<MessageContent> = msg
        .content
        .iter()
        .map(|c| match c {
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => MessageContent::ToolResult {
                tool_use_id: tool_use_id.clone(),
                output: truncate_tool_output(output),
                is_error: *is_error,
            },
            // Images are irrelevant for text summarization and waste bandwidth; replace
            // with placeholder text.
            MessageContent::Image { .. } => MessageContent::Text {
                text: "[image omitted from summary]".to_string(),
            },
            other => other.clone(),
        })
        .collect();
    Message {
        role: msg.role,
        content: content.into(),
    }
}

fn truncate_tool_output(output: &ToolResultBody) -> ToolResultBody {
    match output {
        ToolResultBody::Text { text } => ToolResultBody::Text {
            text: truncate_chars(text, TOOL_RESULT_MAX_CHARS),
        },
        ToolResultBody::Json { value } => {
            let s = value.to_string();
            if s.len() <= TOOL_RESULT_MAX_CHARS {
                ToolResultBody::Json {
                    value: value.clone(),
                }
            } else {
                // When a JSON value exceeds the limit, fall back to a truncated text
                // summary — only the gist is needed, not structural fidelity.
                ToolResultBody::Text {
                    text: truncate_chars(&s, TOOL_RESULT_MAX_CHARS),
                }
            }
        }
        // Multimodal results are downgraded to plain text for summarization: text blocks
        // are kept (truncated), and image blocks are replaced with a placeholder
        // annotation — base64 data is both meaningless and expensive in a summary.
        ToolResultBody::Content { blocks } => {
            let mut text = String::new();
            for block in blocks {
                match block {
                    ToolResultContent::Text { text: t } => text.push_str(t),
                    ToolResultContent::Image { mime, .. } => {
                        text.push_str(&format!("\n[image: {mime}]"));
                    }
                }
            }
            ToolResultBody::Text {
                text: truncate_chars(&text, TOOL_RESULT_MAX_CHARS),
            }
        }
    }
}

/// Truncates at character boundaries (never in the middle of a multi-byte UTF-8
/// sequence); appends a truncation notice if the string exceeds the limit.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let kept: String = s.chars().take(max_chars).collect();
    format!("{kept}\n…[truncated for summary]")
}

#[cfg(test)]
mod tests;
