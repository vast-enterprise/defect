//! `read_file` tool: reads a UTF-8 text file.
//!
//! Read tool — reads a file with an optional offset/limit window.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, ImageContent, TextContent, ToolCallContent, ToolCallLocation,
    ToolCallUpdateFields, ToolKind,
};
use base64::Engine;
use defect_agent::error::BoxError;
use defect_agent::fs::{FsBackend, FsError};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use defect_config::FsToolConfig;
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_LIMIT: u32 = 2000;
const MAX_LIMIT: u32 = 5000;

pub struct ReadFileTool {
    schema: ToolSchema,
    default_limit: u32,
    max_limit: u32,
}

impl ReadFileTool {
    pub fn new() -> Self {
        Self::from_config(&FsToolConfig {
            read_default_limit: DEFAULT_LIMIT,
            read_max_limit: MAX_LIMIT,
        })
    }

    pub fn from_config(config: &FsToolConfig) -> Self {
        let default_limit = config.read_default_limit.max(1);
        let max_limit = config.read_max_limit.max(default_limit);
        Self {
            schema: ToolSchema {
                name: "read_file".to_string(),
                description: "Read a file from the workspace. \
                              For UTF-8 text files: optionally read a window starting at `offset` (1-based line) for `limit` lines; \
                              returns the content with 1-based line numbers prepended. \
                              For image files (.png/.jpg/.jpeg/.gif/.webp): returns the image itself as visual content (offset/limit ignored). \
                              Refuses other binary files and files larger than 10 MiB."
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
                            "maximum": max_limit,
                            "description": format!(
                                "Optional max number of lines to read. Defaults to {default_limit}."
                            )
                        }
                    },
                    "required": ["path"]
                }),
            },
            default_limit,
            max_limit,
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
    /// Start line number (offset) of this window. Used by the LLM to reassemble positions
    /// during chunked reads.
    start_line: u32,
    /// `true` if the backend truncated by `limit`; exact detection requires a second
    /// read, so this uses a heuristic (lines returned == limit implies possible
    /// truncation).
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
        let default_limit = self.default_limit;
        let max_limit = self.max_limit;
        let fut = async move { run_read(args, cancel, fs, default_limit, max_limit).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

async fn run_read(
    args: serde_json::Value,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn FsBackend>,
    default_limit: u32,
    max_limit: u32,
) -> ToolEvent {
    let parsed: ReadArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    // For images, detect by extension and convert via `read_bytes` → base64 →
    // `ContentBlock::Image`.
    // `offset`/`limit` are meaningless for images and are ignored.
    if let Some(mime) = image_mime(&parsed.path) {
        return run_read_image(parsed.path, mime, cancel, fs).await;
    }

    let limit = parsed.limit.unwrap_or(default_limit).min(max_limit).max(1);
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

#[derive(Debug, Serialize)]
struct ReadImageOutput {
    bytes: u64,
    mime: String,
}

/// Reads an image: fetches raw bytes → base64 → returns as a [`ContentBlock::Image`].
///
/// Does not reject with `looks_binary` (that check is for text paths); size limits are
/// handled by the backend's own threshold in [`FsBackend::read_bytes`]. The delegated
/// backend (ACP) `read_bytes` returns `NotPermitted` by default — in that case, a
/// [`ToolError::Execution`] is raised so the model learns from the error text that the
/// delegated environment does not support reading images.
async fn run_read_image(
    path: String,
    mime: &'static str,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn FsBackend>,
) -> ToolEvent {
    let read_fut = fs.read_bytes(PathBuf::from(&path));
    let bytes = tokio::select! {
        biased;
        () = cancel.cancelled() => return ToolEvent::Failed(ToolError::Canceled),
        r = read_fut => match r {
            Ok(b) => b,
            Err(e) => return ToolEvent::Failed(map_fs_err(e)),
        },
    };

    let byte_len = bytes.len() as u64;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let raw_output = serde_json::to_value(ReadImageOutput {
        bytes: byte_len,
        mime: mime.to_string(),
    })
    .unwrap_or(serde_json::Value::Null);

    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Image(ImageContent::new(encoded, mime.to_string())),
    ))]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
}

/// Maps a file extension (case-insensitive) to an image MIME type. Returns `None` for
/// non-image extensions.
fn image_mime(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
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
