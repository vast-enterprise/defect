//! `read_file` 工具：读 UTF-8 文本文件。
//!
//! 设计详见 `docs/internal/tools-fs.md` §3。

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol::schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallLocation, ToolCallUpdateFields,
    ToolKind,
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

const DEFAULT_LIMIT: u32 = 2000;
const MAX_LIMIT: u32 = 5000;

pub struct ReadFileTool {
    schema: ToolSchema,
}

impl ReadFileTool {
    pub fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "read_file".to_string(),
                description: "Read a UTF-8 text file from the workspace. \
                              Optionally read a window starting at `offset` (1-based line) for `limit` lines. \
                              Returns the file content with 1-based line numbers prepended. \
                              Refuses binary files and files larger than 10 MiB."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path or path relative to the session cwd. \
                                            Must resolve inside the workspace root."
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional 1-based start line (inclusive). Defaults to 1."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_LIMIT,
                            "description": "Optional max number of lines to read. Defaults to 2000."
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default)]
    offset: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ReadFileOutput {
    bytes: u64,
    lines_returned: u32,
    /// 本窗口起始行号（offset）。为 LLM 在 chunked-read 时拼装位置。
    start_line: u32,
    /// `true` 表示后端按 limit 截断；准确判断需要二次读，v0 用近似（行数 == limit 即可能截断）。
    truncated: bool,
}

impl Tool for ReadFileTool {
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
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let offset = args
                .get("offset")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32);

            let title = if path.is_empty() {
                "Read".to_string()
            } else {
                format!("Read {path}")
            };
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Read);
            if !path.is_empty() {
                fields.locations = Some(vec![
                    ToolCallLocation::new(PathBuf::from(path)).line(offset),
                ]);
            }
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cancel = ctx.cancel.clone();
        let fs = ctx.fs.clone();
        let fut = async move { run_read(args, cancel, fs).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_read(
    args: serde_json::Value,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn FsBackend>,
) -> ToolEvent {
    let parsed: ReadArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    let limit = parsed.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);
    let offset = parsed.offset.unwrap_or(1).max(1);

    let path = PathBuf::from(&parsed.path);
    let read_fut = fs.read_text(path, Some(offset), Some(limit));
    let text = tokio::select! {
        biased;
        () = cancel.cancelled() => return ToolEvent::Failed(ToolError::Canceled),
        r = read_fut => match r {
            Ok(t) => t,
            Err(e) => return ToolEvent::Failed(map_fs_err(e)),
        },
    };

    let lines_returned = text.split_inclusive('\n').count() as u32;
    let truncated = lines_returned >= limit;
    let bytes = text.len() as u64;

    let formatted = format_with_line_numbers(&text, offset);

    let raw_output = serde_json::to_value(ReadFileOutput {
        bytes,
        lines_returned,
        start_line: offset,
        truncated,
    })
    .unwrap_or(serde_json::Value::Null);

    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(formatted)),
    ))]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

fn map_fs_err(e: FsError) -> ToolError {
    ToolError::Execution(BoxError::new(e))
}

fn format_with_line_numbers(text: &str, offset: u32) -> String {
    let mut out = String::new();
    let mut idx = offset;
    for line in text.split_inclusive('\n') {
        let display = line.strip_suffix('\n').unwrap_or(line);
        out.push_str(&format!("{idx:>4}| {display}\n"));
        idx = idx.saturating_add(1);
    }
    out
}
