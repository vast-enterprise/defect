//! `write_file` tool: overwrites a UTF-8 text file entirely.
//!
//! Write tool — writes content to a file.

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

const MAX_WRITE_BYTES: usize = 10 * 1024 * 1024;

pub struct WriteFileTool {
    schema: ToolSchema,
}

impl WriteFileTool {
    pub fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "write_file".to_string(),
                description: "Write a UTF-8 text file. \
                              Overwrites the file if it exists; creates it if it does not. \
                              Creates intermediate directories as needed. \
                              Path must be inside the workspace root."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path or path relative to the session cwd."
                        },
                        "content": {
                            "type": "string",
                            "description": "Full UTF-8 text content. Replaces the file entirely."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct WriteFileOutput {
    bytes_written: u64,
    created: bool,
    parent_existed: bool,
}

impl Tool for WriteFileTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::Mutating
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");

            let title = if path.is_empty() {
                "Write".to_string()
            } else {
                format!("Write {path}")
            };
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Edit);
            if !path.is_empty() {
                fields.locations = Some(vec![ToolCallLocation::new(PathBuf::from(path))]);

                // v1: Lightly read the old content during the `describe` phase so the
                // authorization UI can render an exact old↔new diff. On failure, fall
                // back to a "fresh" diff (old=None) — `describe` should not block
                // ToolCall delivery due to IO jitter. NotFound is equivalent to a "create
                // new file" path, where old content is None.
                let old = ctx.fs.read_text(PathBuf::from(path), None, None).await.ok();

                fields.content = Some(vec![ToolCallContent::Diff(
                    Diff::new(PathBuf::from(path), content).old_text(old),
                )]);
            }
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cancel = ctx.cancel.clone();
        let fs = ctx.fs.clone();
        let cwd = ctx.cwd.to_path_buf();
        let fut = async move { run_write(args, cancel, fs, &cwd).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_write(
    args: serde_json::Value,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn FsBackend>,
    cwd: &std::path::Path,
) -> ToolEvent {
    let parsed: WriteArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    if parsed.content.len() > MAX_WRITE_BYTES {
        return ToolEvent::Failed(ToolError::Execution(BoxError::new(FsError::TooLarge {
            bytes: parsed.content.len() as u64,
            limit: MAX_WRITE_BYTES as u64,
        })));
    }

    let path = PathBuf::from(&parsed.path);

    // Record whether the parent directory already existed before writing (best-effort,
    // used to inform the LLM).
    let abs_path = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };
    let parent_existed = abs_path.parent().is_none_or(|p| p.is_dir());

    // Best-effort read of old content for accurate diff and `created` detection
    let old = match fs.read_text(path.clone(), None, None).await {
        Ok(t) => Some(t),
        Err(FsError::NotFound(_)) => None,
        Err(_) => None, // On read failure, `created` stays `None`; the write step will report the specific error.
    };

    let bytes_written = parsed.content.len() as u64;

    let write_fut = fs.write_text(path.clone(), parsed.content.clone());
    tokio::select! {
        biased;
        () = cancel.cancelled() => return ToolEvent::Failed(ToolError::Canceled),
        r = write_fut => {
            if let Err(e) = r {
                return ToolEvent::Failed(map_fs_err(e));
            }
        }
    }

    let raw_output = serde_json::to_value(WriteFileOutput {
        bytes_written,
        created: old.is_none(),
        parent_existed,
    })
    .unwrap_or(serde_json::Value::Null);

    let diff = Diff::new(path, parsed.content).old_text(old);
    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![
        ToolCallContent::Diff(diff),
        // `turn.rs::extract_text` takes the first `Text` block as the `tool_result` —
        // feeds a short summary to the LLM.
        ToolCallContent::Content(Content::new(ContentBlock::Text(TextContent::new(format!(
            "Wrote {bytes_written} bytes"
        ))))),
    ]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

fn map_fs_err(e: FsError) -> ToolError {
    ToolError::Execution(BoxError::new(e))
}
