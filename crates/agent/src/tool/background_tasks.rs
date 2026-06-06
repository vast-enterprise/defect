//! 后台任务控制面工具：`inspect_background_task` 与 `cancel_background_task`。
//!
//! 主 agent 用 `spawn_agent { run_in_background: true }` fire-and-forget 一个 subagent
//! 后，这两个工具让它能**回头看**与**提前掐**：
//!
//! - [`InspectBackgroundTaskTool`]：不带 `task_id` 时列出所有后台任务（id / 标签 / 状态 /
//!   进度 block 数）；带 `task_id` 时返回该任务的状态与**最近几个 block**（assistant 文本 /
//!   思考 / 工具调用起止），即对应 subagent 此刻的上下文与进度。
//! - [`CancelBackgroundTaskTool`]：按 `task_id` 提前中断单个后台任务，不波及其它任务。
//!
//! 两者都从 [`ToolContext::background`] 拿 session 级任务表。该句柄只在**顶层 turn**注入
//! （子 agent 嵌套 turn 为 `None`），故这两个工具与 `spawn_agent` 一样是顶层 agent 的能力——
//! 装配时只叠进 overlay、不进子 agent 的裁子集来源，子 agent 结构性够不着（与禁递归同思路）。
//!
//! 进度数据来自 `spawn_agent` 后台路径挂的进度 forwarder：它订阅子 turn 事件、把"最近几个
//! block"喂进任务进度环（[`ProgressBlock`](crate::session::ProgressBlock)）。本模块只读这个环。

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

/// `inspect_background_task` 工具名。
pub(crate) const INSPECT_BACKGROUND_TASK_TOOL_NAME: &str = "inspect_background_task";
/// `cancel_background_task` 工具名。
pub(crate) const CANCEL_BACKGROUND_TASK_TOOL_NAME: &str = "cancel_background_task";

fn io_err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

/// 当 `ctx.background` 为 `None` 时两个工具共用的 fail-loud 错误——不静默降级，否则模型
/// 以为查/取消成功了、实际什么都没发生。
fn no_background_err() -> ToolEvent {
    ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(
        "background tasks are not available in this context (only the top-level agent can \
         inspect or cancel background tasks)"
            .to_string(),
    ))))
}

/// 把一条任务快照渲染成给模型看的一行（列举用，不带 block 明细）。
fn render_summary_line(s: &TaskSnapshot) -> String {
    format!(
        "- {} ({}) [{}] — {} progress block(s)",
        s.task_id,
        s.label,
        s.status.as_str(),
        s.block_count
    )
}

/// 把一条任务快照渲染成带最近 block 明细的多行文本（peek 用）。
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
            // 正文已在写入进度环时按 block_text_limit 收敛（截断 / 清空），这里直接渲染。
            // 正文为空（limit=0 的默认鸟瞰模式）时只标类别，不留一个空挂的冒号。
            if b.text.is_empty() {
                out.push_str(&format!("\n  [{}]", b.kind.as_str()));
            } else {
                out.push_str(&format!("\n  [{}] {}", b.kind.as_str(), b.text));
            }
        }
    }
    out
}

/// 把文本包成一个 `Completed` 工具事件（content + raw_output 同源）。
fn completed_text(text: String) -> ToolEvent {
    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(text.clone())),
    ))]);
    fields.raw_output = Some(serde_json::Value::String(text));
    ToolEvent::Completed(fields)
}

// ===================== inspect_background_task =====================

/// 查后台任务的状态与进度。无 `task_id` ⇒ 列举全部；有 ⇒ 查单个的最近消息块。
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
                    // recent_blocks=None ⇒ peek 用配置默认（默认 10）。
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

/// 提前中断单个后台任务。
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
        // 取消是个有副作用的控制动作（终止一个运行中的任务），标 Mutating。
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
