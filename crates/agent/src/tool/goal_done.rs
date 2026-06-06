//! `goal_done`：AI 声明目标已达成的工具（`--goal` 目标驱动循环）。
//!
//! `--goal` 模式下，agent 多轮自主跑直到目标达成。模型认为目标达成时调用本工具，
//! 它把 [`crate::session::GoalState`] 标记为 reached。之后某轮模型不再调任何工具、
//! 自愿停止时，`goal-gate` hook（[`crate::hooks::builtin::GoalGate`]）在
//! `before_turn_end` 读到 reached → 放行结束循环；未 reached → 续命注入"继续工作"。
//!
//! 设计要点（见 `docs/proposals` goal-loop）：调用本工具那轮**带 tool_use**，turn
//! 不会立即停（turn loop 见 `session/turn.rs`）——模型通常在调完 goal_done、确认无
//! 更多事可做后，下一轮自然停止，那时 goal-gate 才放行。
//!
//! `safety_hint = ReadOnly`：只写内存里的标志位，不碰文件 / 网络 / 子进程。

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

/// `goal_done` 工具的名字。
pub const GOAL_DONE_TOOL_NAME: &str = "goal_done";

/// `goal_done` 工具。`--goal` 装配期注册；无状态（标志位在 [`ToolContext::goal`]
/// 的共享 `GoalState` 上，本工具只调 `mark_reached`）。
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
        // 标志位置位：非目标模式（goal=None）下也不报错——工具不该被注册，但万一被
        // 调到，安静地当成 no-op 确认，避免 turn 崩。
        let goal = ctx.goal.clone();
        let parsed: GoalDoneArgs = serde_json::from_value(args).unwrap_or_default();
        let fut = async move {
            if let Some(goal) = goal {
                goal.mark_reached();
            }
            let text = match parsed.summary {
                Some(s) if !s.is_empty() => {
                    format!("Goal marked as complete. The run will finish once you stop. Summary: {s}")
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
