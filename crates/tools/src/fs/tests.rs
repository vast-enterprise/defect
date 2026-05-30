//! fs 工具家族单元测试。覆盖 `docs/internal/tools-fs.md` §9 的 #1–#20。
//!
//! #21（ACP fake client 反向请求）在 `crates/acp/tests/fs_delegation.rs` 跑；
//! #22（真 LLM 让 deepseek 写文件）在 `crates/llm/examples/` 的冒烟里跑。

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::{ContentBlock, ToolCallContent};
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

/// 一个跑测试用的工作区：tempdir + LocalFsBackend + cancel token。
struct Harness {
    _dir: TempDir,
    root: PathBuf,
    fs: Arc<dyn FsBackend>,
    cancel: CancellationToken,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        // canonicalize 一次：macOS 的 /var → /private/var 之类的链路要在测试断言里对齐
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
    // 行号格式：右对齐 4 位 + "| "
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
    // 11 MiB > 10 MiB 上限
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
    // workspace/escape 是 symlink，指向 workspace 外的目录
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
    // LocalFsBackend 的 read 是同步打盘，体感很快——这里直接预先 cancel
    // 让 select! 的 `cancelled` 分支胜出，验证取消通路存在。
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

// ---------- write_file ----------

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
    // LLM 给的是 LF——后端应当按文件原样还原成 CRLF，避免行末符腐蚀。
    let events =
        drive(tool.execute(json!({"path": "crlf.txt", "content": "x\ny\nz\n"}), h.ctx())).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let bytes = h.read_file("crlf.txt");
    assert_eq!(bytes, b"x\r\ny\r\nz\r\n", "must round-trip back to CRLF");
}

#[tokio::test]
async fn case11_write_parent_missing() {
    let h = Harness::new();
    let tool = WriteFileTool::new();
    let events = drive(tool.execute(
        json!({"path": "no_such_dir/x.txt", "content": "y"}),
        h.ctx(),
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
    // 不应有 .tmp 残留
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
    // §6.2 的退化版：tmp + rename 的可观察结果是——写完后**只有目标文件**，
    // 不应留 .defect-*.tmp 残留。原始矩阵的 panic 注入需要 hook IO，
    // 我们这里用「正常路径不留 tmp」+「parent 缺失不留 tmp（case11）」做回归基线。
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
    // 文件未被修改
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

// ---------- v1 conflict detection ----------

/// 在 baseline 指纹与"写前再取"指纹之间，注入一次外部改写，让
/// edit_file 的 conflict detection 看到不同的指纹。直接 wrap 真后端，
/// 第一次 `fingerprint` 调用后**立刻**改写底层文件——这样第二次取到的
/// mtime/size 必然变化。
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
                // baseline 已经取到——改 size 让下一次 fingerprint 必然不同
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

// ---------- v1 describe-phase precise diff ----------

#[tokio::test]
async fn case27_write_describe_attaches_old_text_when_file_exists() {
    use agent_client_protocol::schema::ToolCallContent;

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
    use agent_client_protocol::schema::ToolCallContent;

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

// ---------- v1 chunked read ----------

#[tokio::test]
async fn case25_read_window_on_oversized_file() {
    // 18 MiB > 10 MiB MAX_FS_BYTES。窗口读应当跳过整体 size 校验，
    // 流式扫到第 100..110 行后即停。
    let h = Harness::new();
    let path = h.root.join("big.log");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        // 200 行，每行 ~100 KiB；总大小 ~20 MiB（远超 10 MiB cap）
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
    // 行号渲染应当从 100 起；"100-" 行内应有这个标记
    assert!(text.contains(" 100| 100-"), "should start at line 100");
    assert!(text.contains(" 109| 109-"), "should reach line 109");
    assert!(!text.contains(" 110| "), "should stop before line 110");
}

#[tokio::test]
async fn case26_read_window_too_large_reports_too_large() {
    // 哪怕走窗口路径，单次窗口本身占用超过 10 MiB 时也应当拒——
    // 防止 LLM 把 `limit` 设很大变相绕过整体阈值。
    let h = Harness::new();
    // 文件比 cap 大；窗口大小（5 行 × ~3 MiB = ~15 MiB）也超过 cap
    let path = h.root.join("big.log");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        let line = "y".repeat(3 * 1024 * 1024);
        for _ in 0..6 {
            writeln!(f, "{line}").unwrap();
        }
    }

    // 直接调后端层避免 ReadFileTool 的 limit clamp（默认 2000，单行 3 MiB
    // × 5 = 15 MiB > cap）
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
    // 控制组：fingerprint 双方一致 → 不应该报 Conflict，正常 edit。
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
    // new_string 含 LF——edit 走 read → replace → write 全链路，
    // backend 在 write 时按文件原行末符（CRLF）规范化 new_string。
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
