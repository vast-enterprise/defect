//! Tool used by the AI to declare that a goal has been reached (for the `--goal`
//! goal-driven loop).
//!
//! In `--goal` mode, the agent runs autonomously for multiple turns until the goal is
//! reached. When the model decides the goal is achieved, it calls this tool, which marks
//! [`crate::session::GoalState`] as `reached`. On a later turn where the model stops
//! calling tools and voluntarily halts, the `goal-gate` hook
//! ([`crate::hooks::builtin::GoalGate`]) reads the state in `before_turn_end`: if
//! `reached`, it allows the loop to end; otherwise, it injects a "continue working"
//! message to keep the agent going.
//!
//! Design note: the turn that calls this tool **does carry tool_use**, so the turn does
//! not end immediately (see the turn loop in `session/turn.rs`). The model typically
//! calls `goal_done`, confirms there is nothing more to do, and then stops naturally on
//! the next turn, at which point `goal-gate` allows the loop to exit.
//!
//! `safety_hint = ReadOnly`: only writes an in-memory flag; does not touch files,
//! network, or subprocesses.

use std::pin::Pin;

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};

/// The name of the `goal_done` tool.
pub const GOAL_DONE_TOOL_NAME: &str = "goal_done";

/// The `goal_done` tool. Registered during `--goal` assembly; stateless (the flag lives
/// on the shared `GoalState` in [`ToolContext::goal`], this tool only calls
/// `mark_reached`).
pub struct GoalDoneTool {
    schema: ToolSchema,
}

impl Default for GoalDoneTool {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalDoneTool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: GOAL_DONE_TOOL_NAME.to_string(),
                description: "Signal that the assigned goal has been fully achieved. Call this \
                              only when you are confident the goal is genuinely complete and there \
                              is nothing left to do. After calling it, stop taking further actions \
                              so the run can finish."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of how the goal was achieved (what was done, key results)."
                        }
                    },
                    "required": []
                }),
            },
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct GoalDoneArgs {
    #[serde(default)]
    summary: Option<String>,
}

impl Tool for GoalDoneTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some("Mark goal as complete".to_string());
            fields.kind = Some(ToolKind::Think);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        // Set the flag: even in non-goal mode (goal=None), do not error — the tool should
        // not be registered, but if it is somehow invoked, silently treat it as a no-op
        // to avoid crashing the turn.
        let goal = ctx.goal.clone();
        let parsed: GoalDoneArgs = serde_json::from_value(args).unwrap_or_default();
        let fut = async move {
            if let Some(goal) = goal {
                goal.mark_reached();
            }
            let text = match parsed.summary {
                Some(s) if !s.is_empty() => {
                    format!(
                        "Goal marked as complete. The run will finish once you stop. Summary: {s}"
                    )
                }
                _ => "Goal marked as complete. The run will finish once you stop.".to_string(),
            };
            let mut fields = ToolCallUpdateFields::default();
            fields.content = Some(vec![ToolCallContent::Content(Content::new(
                ContentBlock::Text(TextContent::new(text.clone())),
            ))]);
            fields.raw_output = Some(serde_json::Value::String(text));
            ToolEvent::Completed(fields)
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}
