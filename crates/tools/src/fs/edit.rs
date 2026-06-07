//! `edit_file` 工具：精确字符串替换。
//!
//! Edit tool — applies a patch to an existing file.

use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, Diff, TextContent, ToolCallContent, ToolCallLocation,
    ToolCallUpdateFields, ToolKind,
};
use defect_agent::error::BoxError;
use defect_agent::fs::{FsBackend, FsError};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub struct EditFileTool {
    schema: ToolSchema,
}

impl EditFileTool {
    pub fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "edit_file".to_string(),
                description: "Replace a string in a UTF-8 text file. \
                              Performs an exact string replacement; \
                              fails if `old_string` is not found, or if it appears multiple times \
                              unless `replace_all` is true. \
                              Path must be inside the workspace root."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path or path relative to the session cwd."
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to replace. Must match a unique substring \
                                            unless `replace_all` is true. Empty string is rejected."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text. Must differ from old_string."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "When true, replace every occurrence; when false (default), \
                                            require old_string to appear exactly once.",
                            "default": false
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            },
        }
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug, Serialize)]
struct EditFileOutput {
    matches_replaced: u32,
    bytes_before: u64,
    bytes_after: u64,
}

impl Tool for EditFileTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::Mutating
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let title = if path.is_empty() {
                "Edit".to_string()
            } else {
                format!("Edit {path}")
            };
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Edit);
            if !path.is_empty() {
                fields.locations = Some(vec![ToolCallLocation::new(PathBuf::from(path))]);
            }
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cancel = ctx.cancel.clone();
        let fs = ctx.fs.clone();
        let fut = async move { run_edit(args, cancel, fs).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_edit(
    args: serde_json::Value,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn FsBackend>,
) -> ToolEvent {
    let parsed: EditArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    if parsed.old_string.is_empty() {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(arg_err(
            "old_string must not be empty",
        ))));
    }
    if parsed.old_string == parsed.new_string {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(arg_err(
            "old_string and new_string must differ",
        ))));
    }

    let path = PathBuf::from(&parsed.path);

    let read_fut = fs.read_text(path.clone(), None, None);
    let old_content = tokio::select! {
        biased;
        () = cancel.cancelled() => return ToolEvent::Failed(ToolError::Canceled),
        r = read_fut => match r {
            Ok(t) => t,
            Err(e) => return ToolEvent::Failed(map_fs_err(e)),
        },
    };

    // 紧贴 read 之后取一份"读取时的指纹"作为基线。后端要么用 mtime+size
    // （LocalFsBackend），要么走默认实现重新读全文 hash（AcpFsBackend）。
    // 任何一种方案都可以与"写前再取"的指纹比对，识别 read→write 窗口
    // 期间的并发外部修改。
    //
    // 失败时（极少见，例如 NotPermitted）放弃这层守护，正常推进——
    // v1 conflict detection 是 best-effort，不应阻塞主流程。
    let baseline_fp = fs.fingerprint(path.clone()).await.ok();

    let (new_content, matches_replaced) = match apply_edit(
        &old_content,
        &parsed.old_string,
        &parsed.new_string,
        parsed.replace_all,
    ) {
        Ok(v) => v,
        Err(EditOutcome::NotFound) => {
            return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(arg_err(
                "old_string not found",
            ))));
        }
        Err(EditOutcome::Ambiguous(n)) => {
            return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(arg_err(&format!(
                "old_string matched {n} times; add unique context or set replace_all"
            )))));
        }
    };

    let bytes_before = old_content.len() as u64;
    let bytes_after = new_content.len() as u64;

    // Conflict detection：写前再取一次指纹，与 baseline 比对。不一致即抛
    // [`FsError::Conflict`]——LLM 看到错误后应该重读再 edit，而不是覆盖。
    if let Some(baseline) = baseline_fp {
        match fs.fingerprint(path.clone()).await {
            Ok(current) if current != baseline => {
                return ToolEvent::Failed(map_fs_err(FsError::Conflict(path)));
            }
            // 取不到当前指纹时不阻塞——v1 conflict detection 是 best-effort。
            _ => {}
        }
    }

    let write_fut = fs.write_text(path.clone(), new_content.clone());
    tokio::select! {
        biased;
        () = cancel.cancelled() => return ToolEvent::Failed(ToolError::Canceled),
        r = write_fut => {
            if let Err(e) = r {
                return ToolEvent::Failed(map_fs_err(e));
            }
        }
    }

    let raw_output = serde_json::to_value(EditFileOutput {
        matches_replaced,
        bytes_before,
        bytes_after,
    })
    .unwrap_or(serde_json::Value::Null);

    let diff = Diff::new(path, new_content).old_text(Some(old_content));
    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![
        ToolCallContent::Diff(diff),
        ToolCallContent::Content(Content::new(ContentBlock::Text(TextContent::new(format!(
            "Replaced {matches_replaced} occurrence(s)"
        ))))),
    ]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

enum EditOutcome {
    NotFound,
    /// match 数（≥ 2）
    Ambiguous(u32),
}

fn apply_edit(
    text: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, u32), EditOutcome> {
    if replace_all {
        let count = text.matches(old).count() as u32;
        if count == 0 {
            return Err(EditOutcome::NotFound);
        }
        Ok((text.replace(old, new), count))
    } else {
        let count = text.matches(old).count();
        match count {
            0 => Err(EditOutcome::NotFound),
            1 => Ok((text.replacen(old, new, 1), 1)),
            n => Err(EditOutcome::Ambiguous(n as u32)),
        }
    }
}

fn map_fs_err(e: FsError) -> ToolError {
    ToolError::Execution(BoxError::new(e))
}

fn arg_err(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.to_string())
}
