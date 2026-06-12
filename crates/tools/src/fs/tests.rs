//! Filesystem tool family unit tests.
//!
//! - #21 (ACP fake client reverse request) runs in `crates/acp/tests/fs_delegation.rs`.
//! - #22 (real LLM asks DeepSeek to write files) runs in the smoke tests under
//!   `crates/llm/examples/`.

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, ToolCallContent};
use defect_agent::fs::FsBackend;
use defect_agent::http::{HttpClient, NoopHttpClient};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use defect_agent::tool::{Tool, ToolContext, ToolError, ToolEvent};
use defect_config::FsToolConfig;
use futures::StreamExt;
use serde_json::json;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use super::{EditFileTool, LocalFsBackend, ReadFileTool, WriteFileTool};

/// A test workspace: tempdir + LocalFsBackend + cancel token.
struct Harness {
    _dir: TempDir,
    root: PathBuf,
    fs: Arc<dyn FsBackend>,
    cancel: CancellationToken,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        // Canonicalize once so that symlink chains (e.g. /var → /private/var on macOS)
        // are resolved consistently for test assertions.
        let root = std::fs::canonicalize(dir.path()).expect("canon");
        let fs: Arc<dyn FsBackend> = Arc::new(LocalFsBackend::new(root.clone()));
        Self {
            _dir: dir,
            root,
            fs,
            cancel: CancellationToken::new(),
        }
    }

    fn ctx(&self) -> ToolContext<'_> {
        let shell: Arc<dyn ShellBackend> = Arc::new(NoopShellBackend);
        let http: Arc<dyn HttpClient> = Arc::new(NoopHttpClient);
        ToolContext::new(
            &self.root,
            self.cancel.clone(),
            self.fs.clone(),
            shell,
            http,
            "test-model",
        )
    }

    fn write_file(&self, name: &str, bytes: impl AsRef<[u8]>) {
        std::fs::write(self.root.join(name), bytes).expect("write");
    }

    fn read_file(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.root.join(name)).expect("read")
    }
}

async fn drive(stream: defect_agent::tool::ToolStream) -> Vec<ToolEvent> {
    stream.collect().await
}

fn extract_text(event: &ToolEvent) -> String {
    let fields = match event {
        ToolEvent::Completed(f) => f,
        _ => panic!("expected Completed, got {event:?}"),
    };
    let content = fields.content.as_ref().expect("content");
    let mut out = String::new();
    for c in content {
        if let ToolCallContent::Content(inner) = c
            && let ContentBlock::Text(t) = &inner.content
        {
            out.push_str(&t.text);
        }
    }
    out
}

/// Extract `(mime, base64-data)` from a [`ContentBlock::Image`].
fn extract_image(event: &ToolEvent) -> (String, String) {
    let fields = match event {
        ToolEvent::Completed(f) => f,
        _ => panic!("expected Completed, got {event:?}"),
    };
    let content = fields.content.as_ref().expect("content");
    for c in content {
        if let ToolCallContent::Content(inner) = c
            && let ContentBlock::Image(img) = &inner.content
        {
            return (img.mime_type.clone(), img.data.clone());
        }
    }
    panic!("no image block in {content:?}");
}

fn extract_raw(event: &ToolEvent) -> &serde_json::Value {
    let fields = match event {
        ToolEvent::Completed(f) => f,
        _ => panic!("expected Completed, got {event:?}"),
    };
    fields.raw_output.as_ref().expect("raw_output")
}

// ---------- read_file ----------

#[tokio::test]
async fn case1_read_existing_utf8_full() {
    let h = Harness::new();
    h.write_file("hello.txt", "alpha\nbeta\ngamma\n");
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "hello.txt"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    // Line number format: right-aligned 4 digits + "| "
    assert!(text.contains("   1| alpha"), "text: {text:?}");
    assert!(text.contains("   2| beta"), "text: {text:?}");
    assert!(text.contains("   3| gamma"), "text: {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["lines_returned"], json!(3));
    assert_eq!(raw["start_line"], json!(1));
    assert_eq!(raw["truncated"], json!(false));
}

#[tokio::test]
async fn case2_read_with_offset_and_limit() {
    let h = Harness::new();
    h.write_file("nums.txt", "1\n2\n3\n4\n5\n");
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "nums.txt", "offset": 3, "limit": 2}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    assert!(text.contains("   3| 3"), "text: {text:?}");
    assert!(text.contains("   4| 4"), "text: {text:?}");
    assert!(!text.contains("   5| 5"), "text: {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["lines_returned"], json!(2));
    assert_eq!(raw["start_line"], json!(3));
}

#[tokio::test]
async fn read_file_uses_configured_line_limits() {
    let h = Harness::new();
    h.write_file("nums.txt", "1\n2\n3\n4\n");
    let tool = ReadFileTool::from_config(&FsToolConfig {
        read_default_limit: 2,
        read_max_limit: 3,
    });
    let events = drive(tool.execute(json!({"path": "nums.txt"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    assert!(text.contains("   1| 1"), "text: {text:?}");
    assert!(text.contains("   2| 2"), "text: {text:?}");
    assert!(!text.contains("   3| 3"), "text: {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["lines_returned"], json!(2));
    assert_eq!(raw["truncated"], json!(true));
}

#[tokio::test]
async fn case3_read_too_large() {
    let h = Harness::new();
    // 11 MiB exceeds the 10 MiB limit
    let big = vec![b'a'; 11 * 1024 * 1024];
    h.write_file("big.txt", &big);
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "big.txt"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("TooLarge"), "err: {err_str}");
}

#[tokio::test]
async fn case4_read_binary_refused() {
    let h = Harness::new();
    h.write_file("bin.bin", b"hello\0world");
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "bin.bin"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("binary"), "err: {err_str}");
}

#[tokio::test]
async fn case29_read_png_returns_image_block() {
    use base64::Engine;
    let h = Harness::new();
    // Arbitrary binary data (including NUL bytes) that would be rejected by
    // `looks_binary` on the text path; the image path should return it as-is.
    let raw_bytes: &[u8] = &[0x89, b'P', b'N', b'G', 0x00, 0x01, 0x02, 0xff];
    h.write_file("logo.png", raw_bytes);
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "logo.png"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    let (mime, data) = extract_image(&events[0]);
    assert_eq!(mime, "image/png");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&data)
        .expect("valid base64");
    assert_eq!(decoded, raw_bytes);
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["mime"], json!("image/png"));
    assert_eq!(raw["bytes"], json!(raw_bytes.len()));
}

#[tokio::test]
async fn case30_read_image_ignores_offset_limit_and_mime_by_ext() {
    let h = Harness::new();
    h.write_file("photo.JPEG", [0xff, 0xd8, 0xff, 0xe0]);
    let tool = ReadFileTool::new();
    // offset/limit are meaningless for images and should be silently ignored; extension
    // matching is case-insensitive.
    let events = drive(tool.execute(
        json!({"path": "photo.JPEG", "offset": 5, "limit": 1}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    let (mime, _) = extract_image(&events[0]);
    assert_eq!(mime, "image/jpeg");
}

#[tokio::test]
async fn case5_read_path_escape() {
    let h = Harness::new();
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "../../../etc/passwd"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
}

#[tokio::test]
#[cfg(unix)]
async fn case6_read_symlink_outside_workspace() {
    let h = Harness::new();
    let other = tempfile::tempdir().unwrap();
    std::fs::write(other.path().join("secret.txt"), "secret").unwrap();
    // `workspace/escape` is a symlink pointing to a directory outside the workspace
    std::os::unix::fs::symlink(other.path(), h.root.join("escape")).unwrap();

    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "escape/secret.txt"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
}

#[tokio::test]
async fn case7_read_canceled() {
    // LocalFsBackend's read is synchronous and fast, so we cancel preemptively to let the
    // `cancelled` branch of `select!` win, verifying the cancellation path exists.
    let h = Harness::new();
    h.write_file("a.txt", "hello\n");
    h.cancel.cancel();
    let tool = ReadFileTool::new();
    let events = drive(tool.execute(json!({"path": "a.txt"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Canceled)),
        "got {:?}",
        events[0]
    );
}

// --- write_file ---

#[tokio::test]
async fn case8_write_new_file() {
    let h = Harness::new();
    let tool = WriteFileTool::new();
    let events =
        drive(tool.execute(json!({"path": "new.txt", "content": "hello\n"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["created"], json!(true));
    assert_eq!(raw["bytes_written"], json!(6));
    assert_eq!(h.read_file("new.txt"), b"hello\n");
}

#[tokio::test]
async fn case9_write_overwrite_lf_keeps_lf() {
    let h = Harness::new();
    h.write_file("doc.txt", "old\nline\n");
    let tool = WriteFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "doc.txt", "content": "fresh\nbody\n"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["created"], json!(false));
    let bytes = h.read_file("doc.txt");
    assert_eq!(bytes, b"fresh\nbody\n");
    assert!(!bytes.windows(2).any(|w| w == b"\r\n"), "no CRLF");
}

#[tokio::test]
async fn case10_write_overwrite_crlf_normalizes_lf_to_crlf() {
    let h = Harness::new();
    h.write_file("crlf.txt", b"a\r\nb\r\nc\r\n");
    let tool = WriteFileTool::new();
    // The LLM provides LF; the backend should restore CRLF as the file originally had, to
    // avoid line-ending corruption.
    let events =
        drive(tool.execute(json!({"path": "crlf.txt", "content": "x\ny\nz\n"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let bytes = h.read_file("crlf.txt");
    assert_eq!(bytes, b"x\r\ny\r\nz\r\n", "must round-trip back to CRLF");
}

#[tokio::test]
async fn case11_write_parent_missing_auto_creates() {
    let h = Harness::new();
    let tool = WriteFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "no_such_dir/sub/x.txt", "content": "y"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Completed(_)),
        "expected Completed, got {:?}",
        events[0]
    );
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["created"], json!(true));
    assert_eq!(raw["parent_existed"], json!(false));
    // The file landed in the expected location
    let on_disk = std::fs::read_to_string(h.root.join("no_such_dir/sub/x.txt")).unwrap();
    assert_eq!(on_disk, "y");
    // No `.tmp` files should remain
    let stale: Vec<_> = std::fs::read_dir(&h.root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains(".defect-"))
        .collect();
    assert!(stale.is_empty(), "stale tmp files: {stale:?}");
}

#[tokio::test]
async fn case12_write_path_escape() {
    let h = Harness::new();
    let tool = WriteFileTool::new();
    let events =
        drive(tool.execute(json!({"path": "../escape.txt", "content": "y"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
}

#[tokio::test]
async fn case13_write_no_partial_file_on_success() {
    // Degenerate version of the atomic-write case: the observable outcome of tmp +
    // rename is that after writing, **only the target file** exists, with no leftover
    // `.defect-*.tmp` files.
    // The original matrix's panic injection requires hooking IO; here we use "no tmp on
    // normal path" + "no tmp when parent is missing (case11)" as a regression baseline.
    let h = Harness::new();
    let tool = WriteFileTool::new();
    let events =
        drive(tool.execute(json!({"path": "atomic.txt", "content": "data"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));

    let entries: Vec<_> = std::fs::read_dir(&h.root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.contains(&"atomic.txt".to_string()),
        "missing target: {entries:?}"
    );
    let tmp_count = entries.iter().filter(|n| n.contains(".defect-")).count();
    assert_eq!(tmp_count, 0, "tmp residue: {entries:?}");
}

// ---------- edit_file ----------

#[tokio::test]
async fn case14_edit_unique_match() {
    let h = Harness::new();
    h.write_file("e.txt", "alpha BETA gamma\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "BETA", "new_string": "delta"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["matches_replaced"], json!(1));
    assert_eq!(h.read_file("e.txt"), b"alpha delta gamma\n");
}

#[tokio::test]
async fn case15_edit_ambiguous_without_replace_all() {
    let h = Harness::new();
    h.write_file("e.txt", "x\nx\nx\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "x", "new_string": "y"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("matched 3 times"), "err: {err_str}");
    // file was not modified
    assert_eq!(h.read_file("e.txt"), b"x\nx\nx\n");
}

#[tokio::test]
async fn case16_edit_replace_all() {
    let h = Harness::new();
    h.write_file("e.txt", "x\nx\nx\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({
            "path": "e.txt",
            "old_string": "x",
            "new_string": "y",
            "replace_all": true,
        }),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["matches_replaced"], json!(3));
    assert_eq!(h.read_file("e.txt"), b"y\ny\ny\n");
}

#[tokio::test]
async fn case17_edit_not_found() {
    let h = Harness::new();
    h.write_file("e.txt", "alpha\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "ZZZ", "new_string": "y"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("not found"), "err: {err_str}");
}

#[tokio::test]
async fn case18_edit_old_equals_new() {
    let h = Harness::new();
    h.write_file("e.txt", "alpha\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "alpha", "new_string": "alpha"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("must differ"), "err: {err_str}");
}

#[tokio::test]
async fn case19_edit_empty_old_string() {
    let h = Harness::new();
    h.write_file("e.txt", "alpha\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "", "new_string": "y"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "got {:?}",
        events[0]
    );
    let err_str = format!("{:?}", events[0]);
    assert!(err_str.contains("must not be empty"), "err: {err_str}");
}

// ---------- conflict detection ----------

/// Injects an external modification between the baseline fingerprint and the "re-fetch
/// before write" fingerprint, so that `edit_file`'s conflict detection sees different
/// fingerprints. Wraps the real backend directly; immediately after the first
/// `fingerprint` call, rewrites the underlying file — ensuring the second fetch sees a
/// different mtime/size.
struct MtimeAdvancer {
    inner: Arc<dyn FsBackend>,
    bump_pending: std::sync::atomic::AtomicBool,
    target: PathBuf,
}

impl FsBackend for MtimeAdvancer {
    fn read_text(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> futures::future::BoxFuture<'_, Result<String, defect_agent::fs::FsError>> {
        self.inner.read_text(path, line, limit)
    }
    fn write_text(
        &self,
        path: PathBuf,
        content: String,
    ) -> futures::future::BoxFuture<'_, Result<(), defect_agent::fs::FsError>> {
        self.inner.write_text(path, content)
    }
    fn fingerprint(
        &self,
        path: PathBuf,
    ) -> futures::future::BoxFuture<
        '_,
        Result<defect_agent::fs::Fingerprint, defect_agent::fs::FsError>,
    > {
        let do_bump = self
            .bump_pending
            .swap(false, std::sync::atomic::Ordering::SeqCst);
        let target = self.target.clone();
        Box::pin(async move {
            let fp = self.inner.fingerprint(path).await?;
            if do_bump {
                // baseline already captured — change size to guarantee a different
                // fingerprint next time
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                std::fs::write(&target, b"someone else's edit, longer than before\n").unwrap();
            }
            Ok(fp)
        })
    }
}

#[tokio::test]
async fn case23_edit_detects_external_modification_between_read_and_write() {
    let h = Harness::new();
    h.write_file("doc.txt", "old line\n");

    let target = h.root.join("doc.txt");
    let advancer: Arc<dyn FsBackend> = Arc::new(MtimeAdvancer {
        inner: h.fs.clone(),
        bump_pending: std::sync::atomic::AtomicBool::new(true),
        target,
    });
    let shell: Arc<dyn ShellBackend> = Arc::new(NoopShellBackend);
    let http: Arc<dyn HttpClient> = Arc::new(NoopHttpClient);
    let ctx = ToolContext::new(
        &h.root,
        h.cancel.clone(),
        advancer,
        shell,
        http,
        "test-model",
    );

    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "doc.txt", "old_string": "old line", "new_string": "new line"}),
        ctx,
    ))
    .await;

    assert_eq!(events.len(), 1);
    let err_str = format!("{:?}", events[0]);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "expected Execution(Conflict), got {:?}",
        events[0]
    );
    assert!(
        err_str.contains("Conflict") || err_str.contains("changed"),
        "expected Conflict error, got: {err_str}"
    );
}

// ---------- describe-phase precise diff ----------

#[tokio::test]
async fn case27_write_describe_attaches_old_text_when_file_exists() {
    use agent_client_protocol_schema::ToolCallContent;

    let h = Harness::new();
    h.write_file("doc.txt", "old content\n");

    let tool = WriteFileTool::new();
    let args = json!({"path": "doc.txt", "content": "new content\n"});
    let desc = tool.describe(&args, h.ctx()).await;

    let content_blocks = desc.fields.content.as_ref().expect("content");
    let diff = content_blocks
        .iter()
        .find_map(|c| match c {
            ToolCallContent::Diff(d) => Some(d),
            _ => None,
        })
        .expect("expected Diff block");
    assert_eq!(diff.new_text, "new content\n");
    assert_eq!(diff.old_text.as_deref(), Some("old content\n"));
}

#[tokio::test]
async fn case28_write_describe_old_text_none_for_new_file() {
    use agent_client_protocol_schema::ToolCallContent;

    let h = Harness::new();
    let tool = WriteFileTool::new();
    let args = json!({"path": "fresh.txt", "content": "hello\n"});
    let desc = tool.describe(&args, h.ctx()).await;

    let content_blocks = desc.fields.content.as_ref().expect("content");
    let diff = content_blocks
        .iter()
        .find_map(|c| match c {
            ToolCallContent::Diff(d) => Some(d),
            _ => None,
        })
        .expect("expected Diff block");
    assert_eq!(diff.new_text, "hello\n");
    assert!(diff.old_text.is_none(), "no old_text for new file");
}

// chunked read

#[tokio::test]
async fn case25_read_window_on_oversized_file() {
    // 18 MiB > 10 MiB MAX_FS_BYTES. Windowed read should skip the overall size check and
    // stop after streaming through lines 100..110.
    let h = Harness::new();
    let path = h.root.join("big.log");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        // 200 lines, each ~100 KiB; total ~20 MiB (far exceeding the 10 MiB cap)
        let line = "x".repeat(100 * 1024);
        for i in 1..=200 {
            writeln!(f, "{i:03}-{line}").unwrap();
        }
    }

    let tool = ReadFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "big.log", "offset": 100, "limit": 10}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Completed(_)),
        "got {:?}",
        events[0]
    );
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["lines_returned"], json!(10));
    assert_eq!(raw["start_line"], json!(100));

    let text = extract_text(&events[0]);
    // Line numbers should start at 100; the "100-" line should contain this marker.
    assert!(text.contains(" 100| 100-"), "should start at line 100");
    assert!(text.contains(" 109| 109-"), "should reach line 109");
    assert!(!text.contains(" 110| "), "should stop before line 110");
}

#[tokio::test]
async fn case26_read_window_too_large_reports_too_large() {
    // Even when using the window path, a single window exceeding 10 MiB should be
    // rejected — this prevents the LLM from setting a large `limit` to effectively bypass
    // the overall threshold.
    let h = Harness::new();
    // The file is larger than `cap`; the window size (5 lines × ~3 MiB = ~15 MiB) also
    // exceeds `cap`.
    let path = h.root.join("big.log");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        let line = "y".repeat(3 * 1024 * 1024);
        for _ in 0..6 {
            writeln!(f, "{line}").unwrap();
        }
    }

    // Call the backend layer directly to avoid `ReadFileTool`'s limit clamp (default
    // 2000; 3 MiB per line × 5 = 15 MiB > cap).
    let res =
        h.fs.read_text(PathBuf::from("big.log"), Some(1), Some(5))
            .await;
    assert!(matches!(
        res,
        Err(defect_agent::fs::FsError::TooLarge { .. })
    ));
}

#[tokio::test]
async fn case24_edit_no_conflict_when_file_stable() {
    // Control group: both sides have the same fingerprint → no Conflict should be
    // reported, normal edit.
    let h = Harness::new();
    h.write_file("stable.txt", "alpha\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "stable.txt", "old_string": "alpha", "new_string": "beta"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    assert_eq!(h.read_file("stable.txt"), b"beta\n");
}

#[tokio::test]
async fn case20_edit_crlf_file_with_lf_new_string_keeps_crlf() {
    let h = Harness::new();
    h.write_file("crlf.txt", b"alpha\r\nBETA\r\ngamma\r\n");
    let tool = EditFileTool::new();
    // new_string contains LF — the edit goes through the full read → replace → write
    // pipeline, and the backend normalizes new_string to the file's original line endings
    // (CRLF) during write.
    let events = drive(tool.execute(
        json!({
            "path": "crlf.txt",
            "old_string": "BETA",
            "new_string": "x\ny",
        }),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let bytes = h.read_file("crlf.txt");
    assert_eq!(bytes, b"alpha\r\nx\r\ny\r\ngamma\r\n");
}

#[tokio::test]
async fn case29_edit_recovers_from_wrong_indentation_via_fallback() {
    // File has the block indented; old_string has the same lines but no indentation.
    // Exact matching fails, but the fault-tolerant chain locates the real (indented)
    // span and edits it. Note the documented caveat: the matched span includes the
    // line's leading indentation, so splicing the (unindented) new_string drops that
    // indentation. This is surfaced via `matched_strategy != "exact"`.
    let h = Harness::new();
    h.write_file(
        "code.rs",
        "fn main() {\n    let x = 1;\n    let y = 2;\n}\n",
    );
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({
            "path": "code.rs",
            "old_string": "let x = 1;\nlet y = 2;",
            "new_string": "let z = 3;",
        }),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Completed(_)),
        "got {:?}",
        events[0]
    );
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["matches_replaced"], json!(1));
    // The non-exact strategy is surfaced so the fuzzy match is observable.
    assert_eq!(raw["matched_strategy"], json!("line_trimmed"));
    // Indentation of the matched span is not preserved (documented fallback caveat).
    assert_eq!(h.read_file("code.rs"), b"fn main() {\nlet z = 3;\n}\n");
}

#[tokio::test]
async fn case32_edit_crlf_file_with_lf_old_string_matches() {
    // Headline CRLF fix: file is CRLF, model sends a multi-line old_string with LF (as it
    // almost always does). Before the line-ending normalization, the exact byte match
    // failed on every internal line break. Now old/new are converted to the file's ending
    // before matching, so a clean exact match succeeds and CRLF round-trips.
    let h = Harness::new();
    h.write_file("crlf.rs", b"fn f() {\r\n    a();\r\n    b();\r\n}\r\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({
            "path": "crlf.rs",
            "old_string": "    a();\n    b();",
            "new_string": "    c();",
        }),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)), "got {:?}", events[0]);
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["matched_strategy"], json!("exact"));
    assert_eq!(h.read_file("crlf.rs"), b"fn f() {\r\n    c();\r\n}\r\n");
}

#[tokio::test]
async fn case31_edit_exact_match_reports_exact_strategy() {
    // Regression guard: a clean exact match must still take the strict path and report
    // "exact", never a fuzzy fallback.
    let h = Harness::new();
    h.write_file("e.txt", "alpha BETA gamma\n");
    let tool = EditFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "e.txt", "old_string": "BETA", "new_string": "delta"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["matched_strategy"], json!("exact"));
}
