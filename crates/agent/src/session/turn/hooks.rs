//! Hook trigger logic inside the turn main loop.
//!
//! Extracted from the turn main flow: `decide_turn_end` (before turn-end continuation
//! decision), `fire_*` triggers before/after prompt ingestion and tools, and feedback
//! injection helpers, implemented as methods on [`super::TurnRunner`]. See `crate::hooks`
//! for step types and engine.

use agent_client_protocol_schema::{ContentBlock, StopReason as AcpStopReason, ToolCallId};
use serde_json::Value as JsonValue;

use crate::llm::{Message, MessageContent, Role, ToolResultBody, ToolResultContent};

use super::content::content_block_to_message_content;
use super::tools::ToolResult;
use super::{TurnRunner, TurnState};

/// The result of `fire_user_prompt_submit`.
pub(super) enum UserPromptHookFlow {
    Continue(Vec<ContentBlock>),
    Refused,
}

/// The result of `fire_pre_tool_use`.
pub(super) enum PreToolHookFlow {
    Continue { args: JsonValue },
    Block(String),
}

impl TurnRunner<'_> {
    /// `before turn-end` decision point.
    ///
    /// Called when the turn is **voluntarily stopping** (LLM said `EndTurn` / didn't
    /// request a tool). Lets the hook decide: allow the stop, or keep the turn alive
    /// (inject feedback, don't end, loop back to the top for another round).
    ///
    /// Returns `true` = keep alive (caller `continue`s back to the loop top); `false` =
    /// allow stop (end the turn normally).
    ///
    /// Keep-alive is bounded by the **hard limit** `max_stop_hook_continues` — once
    /// reached, the stop is forced to prevent infinite loops. The keep-alive feedback is
    /// injected into history as a **user message** (same pipeline as user prompts; see
    /// the final alternation fallback below).
    pub(super) async fn decide_turn_end(&self, state: &mut TurnState) -> bool {
        // Hard limit reached: stop asking the hook and force-stop.
        if !state.may_stop_hook_continue() {
            return false;
        }

        let mut step = crate::hooks::step::BeforeTurnEnd {
            stop_reason: AcpStopReason::EndTurn,
            continues_so_far: state.stop_hook_continues,
            voluntary: true,
            feedback: Vec::new(),
        };

        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        match control {
            crate::hooks::step::HookControl::Continue => {
                // Inject the feedback as a user message into the history. If the feedback
                // is empty, inject a fallback prompt to prevent the LLM from immediately
                // saying "I'm done" on the next turn, which would cause a no-op loop
                // (invariant: the next turn must always have something to act on).
                let blocks = if step.feedback.is_empty() {
                    vec![ContentBlock::from(
                        "Continue working — the stop condition is not yet satisfied.",
                    )]
                } else {
                    step.feedback
                };
                self.append_user_feedback(blocks);
                state.note_stop_hook_continue();
                true
            }
            // Proceed, Break, and Skip all mean "stop" at turn-end.
            _ => false,
        }
    }

    /// Inject a set of content blocks into the history as a user message (used for
    /// keepalive feedback).
    ///
    /// Fallback role alternation: if the history already ends with a user
    /// role, merge into the same message rather than appending an adjacent user, to
    /// prevent two wire codecs from encountering consecutive identical roles. Blocks that
    /// cannot be decoded are skipped (best effort, does not kill the turn).
    pub(super) fn append_user_feedback(&self, blocks: Vec<ContentBlock>) {
        let content: Vec<MessageContent> = blocks
            .into_iter()
            .filter_map(|b| content_block_to_message_content(b).ok())
            .flatten()
            .collect();
        if content.is_empty() {
            return;
        }
        self.history.append(Message {
            role: Role::User,
            content: content.into(),
        });
    }

    /// Triggers the `UserPromptSubmit` hook.
    ///
    /// Handles three outcomes:
    /// - `block` → rejects the turn (caller returns `Refusal`)
    /// - `patch = UserPrompt { prepend, append }` → rewrites the prompt order to
    ///   `[prepend, original, append]`; the rewritten form is used when appending to
    ///   history
    /// - `append` → not yet spliced into the system prompt (currently has no landing
    ///   point; pending `system_prompt` filled in dynamically after assembly)
    pub(super) async fn fire_user_prompt_submit(
        &self,
        prompt: Vec<ContentBlock>,
    ) -> UserPromptHookFlow {
        // Step model: `before Ingest` (before input ingestion). The hook can rewrite the
        // input or `Break` to reject the turn.
        // The source is carried by the turn — user turn = User, background continuation
        // turn = Background.
        let mut step = crate::hooks::step::BeforeIngest {
            source: self.ingest_source.clone(),
            input: prompt,
        };
        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;
        match control {
            crate::hooks::step::HookControl::Break { .. } => {
                tracing::info!("user prompt blocked by before-ingest hook");
                UserPromptHookFlow::Refused
            }
            // Proceed, Continue, and Skip all mean "continue" at the ingestion point,
            // using the hook-rewritten input.
            _ => UserPromptHookFlow::Continue(step.input),
        }
    }

    /// Fires the `before ToolApply` hook (per tool).
    pub(super) async fn fire_pre_tool_use(
        &self,
        id: &ToolCallId,
        name: &str,
        args: &JsonValue,
        safety: crate::tool::SafetyClass,
    ) -> PreToolHookFlow {
        let _ = id;
        // Step model: `before ToolApply`. The hook may modify `args`, set `result`
        // (intercepting the tool = synthetic output), or return `Break`.
        let mut step = crate::hooks::step::BeforeToolApply {
            tool_name: name.to_string(),
            safety,
            args: args.clone(),
            result: None,
        };
        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        // If `step.result` is set, the tool is blocked (synthetic output) and the turn
        // continues. This maps to the existing `Block` flow: the caller skips tool
        // execution and feeds `reason` back as the rejection text.
        if let Some(result) = step.result {
            let reason = match &result.body {
                crate::llm::ToolResultBody::Text { text } => text.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| "blocked by hook".into()),
            };
            tracing::info!(tool = %name, "tool short-circuited by before-tool-apply hook");
            return PreToolHookFlow::Block(reason);
        }
        if let crate::hooks::step::HookControl::Break { .. } = control {
            tracing::info!(tool = %name, "tool blocked by before-tool-apply hook (break)");
            return PreToolHookFlow::Block("blocked by hook".to_string());
        }
        PreToolHookFlow::Continue { args: step.args }
    }

    /// Fires the `after ToolApply` hook (per tool). Appends any `additional_context`
    /// injected by the hook to the end of `result.body`, so the next LLM turn sees the
    /// hook annotation as part of the tool output.
    pub(super) async fn fire_post_tool_hook(&self, result: &mut ToolResult) {
        // Step model: `after ToolApply`. Observable and injectable (appended to
        // `tool_result`).
        let mut step = crate::hooks::step::AfterToolApply {
            tool_name: result.name.clone(),
            is_error: result.is_error,
            output: result.body.clone(),
            additional_context: Vec::new(),
        };
        let _ = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        if step.additional_context.is_empty() {
            return;
        }

        // Append text blocks from hook-injected `ContentBlock`s to the tool_result body.
        let extra: String = step
            .additional_context
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if extra.is_empty() {
            return;
        }
        match &mut result.body {
            ToolResultBody::Text { text } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&extra);
            }
            // For multimodal results, append the extra text as a new text block at the
            // end, leaving image blocks unchanged.
            ToolResultBody::Content { blocks } => {
                blocks.push(ToolResultContent::Text { text: extra });
            }
            ToolResultBody::Json { .. } => {}
        }
    }
}
