//! Background task control-surface tools: `inspect_background_task` and
//! `cancel_background_task`.
//!
//! After the main agent fire-and-forgets a subagent via `spawn_agent { run_in_background:
//! true }`, these two tools let it **inspect** and **preemptively cancel** that subagent:
//!
//! - [`InspectBackgroundTaskTool`]: without `task_id`, lists all background tasks (id /
//!   label / status / progress block count); with `task_id`, returns the task's status
//!   and **most recent blocks** (assistant text / thinking / tool-call boundaries), i.e.
//!   the subagent's current context and progress.
//! - [`CancelBackgroundTaskTool`]: preemptively cancels a single background task by
//!   `task_id`, without affecting other tasks.
//!
//! Both read the session-level task table from [`ToolContext::background`]. That handle
//! is injected only in the **top-level turn** (it is `None` in nested subagent turns), so
//! these tools—like `spawn_agent`—are capabilities of the top-level agent: they are
//! layered into the overlay but not into the subagent's tool subset, making them
//! structurally inaccessible to subagents (same reasoning as recursion prevention).
//!
//! Progress data comes from the progress forwarder attached to `spawn_agent`'s background
//! path: it subscribes to sub-turn events and feeds the "most recent blocks" into the
//! task's progress ring ([`ProgressBlock`](crate::session::ProgressBlock)). This module
//! is read-only on that ring.

use std::pin::Pin;

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::error::BoxError;
use crate::session::TaskSnapshot;
use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};

/// The name of the `inspect_background_task` tool.
pub(crate) const INSPECT_BACKGROUND_TASK_TOOL_NAME: &str = "inspect_background_task";
/// The name of the `cancel_background_task` tool.
pub(crate) const CANCEL_BACKGROUND_TASK_TOOL_NAME: &str = "cancel_background_task";

fn io_err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

/// A shared fail-loud error for both tools when `ctx.background` is `None` — do not
/// silently degrade, otherwise the model may think the query/cancel succeeded when
/// nothing actually happened.
fn no_background_err() -> ToolEvent {
    ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(
        "background tasks are not available in this context (only the top-level agent can \
         inspect or cancel background tasks)"
            .to_string(),
    ))))
}

/// Renders a task snapshot as a single line for the model (for listing, without block
/// details).
fn render_summary_line(s: &TaskSnapshot) -> String {
    format!(
        "- {} ({}) [{}] — {} progress block(s)",
        s.task_id,
        s.label,
        s.status.as_str(),
        s.block_count
    )
}

/// Render a task snapshot as multi-line text with recent block details (for peek).
fn render_detail(s: &TaskSnapshot) -> String {
    let mut out = format!(
        "background task {} ({}) [{}], {} progress block(s) total",
        s.task_id,
        s.label,
        s.status.as_str(),
        s.block_count
    );
    if s.recent.is_empty() {
        out.push_str("\n(no progress blocks recorded yet)");
    } else {
        out.push_str(&format!("\nmost recent {} block(s):", s.recent.len()));
        for b in &s.recent {
            // The body text was already truncated or cleared by `block_text_limit` when
            // writing to the progress ring, so just render it directly.
            // When the body is empty (the default bird's-eye mode with `limit=0`), show
            // only the category label without a dangling colon.
            if b.text.is_empty() {
                out.push_str(&format!("\n  [{}]", b.kind.as_str()));
            } else {
                out.push_str(&format!("\n  [{}] {}", b.kind.as_str(), b.text));
            }
        }
    }
    out
}

/// Wraps text into a `Completed` tool event, with `content` and `raw_output` sharing the
/// same source.
fn completed_text(text: String) -> ToolEvent {
    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(text.clone())),
    ))]);
    fields.raw_output = Some(serde_json::Value::String(text));
    ToolEvent::Completed(fields)
}

// ===================== inspect_background_task =====================

/// Query the status and progress of a background task. Without `task_id`, list all tasks;
/// with it, query the most recent message block for a single task.
pub struct InspectBackgroundTaskTool {
    schema: ToolSchema,
}

impl Default for InspectBackgroundTaskTool {
    fn default() -> Self {
        Self::new()
    }
}

impl InspectBackgroundTaskTool {
    #[must_use]
    pub fn new() -> Self {
        let schema = ToolSchema {
            name: INSPECT_BACKGROUND_TASK_TOOL_NAME.to_string(),
            description: "Inspect background tasks you started with `spawn_agent \
                          { run_in_background: true }`. Omit `task_id` to list all background \
                          tasks with their id, label, and status. Pass a `task_id` to see that \
                          task's status and its most recent conversation blocks — these are the \
                          subagent's committed messages (the same blocks sent to the model: its \
                          assistant text, thoughts, tool calls and tool results), NOT raw \
                          streaming fragments. Use this to check a running subagent's context \
                          and progress before deciding whether to wait, cancel, or move on."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional. The id of a background task (as returned by \
                                        spawn_agent, e.g. `bg-0`). When omitted, all background \
                                        tasks are listed instead."
                    },
                    "recent_blocks": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional. When inspecting a single task, how many of the \
                                        most recent conversation blocks to return. Defaults to a \
                                        configured value (10 unless overridden)."
                    }
                },
                "required": []
            }),
        };
        Self { schema }
    }
}

#[derive(Debug, Deserialize)]
struct InspectArgs {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    recent_blocks: Option<usize>,
}

impl Tool for InspectBackgroundTaskTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let title = match args.get("task_id").and_then(|v| v.as_str()) {
                Some(id) => format!("Inspect background task `{id}`"),
                None => "List background tasks".to_string(),
            };
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Read);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let background = ctx.background.clone();
        let fut = async move {
            let Some(bg) = background else {
                return no_background_err();
            };
            let parsed: InspectArgs = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };
            match parsed.task_id {
                None => {
                    let tasks = bg.list();
                    if tasks.is_empty() {
                        return completed_text("No background tasks.".to_string());
                    }
                    let body = tasks
                        .iter()
                        .map(render_summary_line)
                        .collect::<Vec<_>>()
                        .join("\n");
                    completed_text(format!("{} background task(s):\n{body}", tasks.len()))
                }
                Some(id) => {
                    // When `recent_blocks` is `None`, `peek` uses the configured default
                    // (10).
                    match bg.peek(&id, parsed.recent_blocks) {
                        Some(snap) => completed_text(render_detail(&snap)),
                        None => ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(
                            format!("no background task with id `{id}`"),
                        )))),
                    }
                }
            }
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

// ===================== cancel_background_task =====================

/// Cancel a single background task early.
pub struct CancelBackgroundTaskTool {
    schema: ToolSchema,
}

impl Default for CancelBackgroundTaskTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelBackgroundTaskTool {
    #[must_use]
    pub fn new() -> Self {
        let schema = ToolSchema {
            name: CANCEL_BACKGROUND_TASK_TOOL_NAME.to_string(),
            description: "Interrupt a background task you started with `spawn_agent \
                          { run_in_background: true }`, by its `task_id`. Cancellation is \
                          cooperative: the subagent is signalled to stop and the task ends \
                          shortly after; its (partial/cancelled) result still flows back to you \
                          on a later turn. Cancelling one task does not affect any other. Use \
                          `inspect_background_task` first if you need to check a task's progress \
                          before deciding to cancel it."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The id of the background task to cancel (as returned by \
                                        spawn_agent, e.g. `bg-0`)."
                    }
                },
                "required": ["task_id"]
            }),
        };
        Self { schema }
    }
}

#[derive(Debug, Deserialize)]
struct CancelArgs {
    task_id: String,
}

impl Tool for CancelBackgroundTaskTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // Cancellation is a control action with side effects (terminates a running task);
        // mark as Mutating.
        SafetyClass::Mutating
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let id = args.get("task_id").and_then(|v| v.as_str()).unwrap_or("?");
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(format!("Cancel background task `{id}`"));
            fields.kind = Some(ToolKind::Other);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let background = ctx.background.clone();
        let fut = async move {
            let Some(bg) = background else {
                return no_background_err();
            };
            let parsed: CancelArgs = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };
            match bg.cancel_task(&parsed.task_id) {
                Some(true) => completed_text(format!(
                    "Requested cancellation of background task `{}`. It will stop shortly; its \
                     result arrives on a later turn.",
                    parsed.task_id
                )),
                Some(false) => completed_text(format!(
                    "Background task `{}` has already finished — nothing to cancel.",
                    parsed.task_id
                )),
                None => ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(format!(
                    "no background task with id `{}`",
                    parsed.task_id
                ))))),
            }
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

#[cfg(test)]
mod tests;
