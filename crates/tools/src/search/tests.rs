//! `SearchTool` 单元测试。覆盖 `docs/internal/tools-search.md` §10 的 #1–#24
//! （#25–#27 在 CLI 装配 / e2e 层覆盖）。

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::{ContentBlock, ToolCallContent};
use defect_agent::fs::FsBackend;
use defect_agent::http::{HttpClient, NoopHttpClient};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use defect_agent::tool::{Tool, ToolContext, ToolError, ToolEvent};
use defect_config::SearchToolConfig;
use futures::StreamExt;
use serde_json::json;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use super::SearchTool;
use crate::fs::LocalFsBackend;

struct Harness {
    _dir: TempDir,
    root: PathBuf,
    fs: Arc<dyn FsBackend>,
    cancel: CancellationToken,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
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
        )
    }

    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) {
        let path = self.root.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(path, bytes).expect("write");
    }
}

async fn drive(stream: defect_agent::tool::ToolStream) -> Vec<ToolEvent> {
    stream.collect().await
}

fn expect_completed(
    events: &[ToolEvent],
) -> (&agent_client_protocol::schema::ToolCallUpdateFields,) {
    assert_eq!(events.len(), 1, "expected exactly one event: {events:?}");
    match &events[0] {
        ToolEvent::Completed(f) => (f,),
        other => panic!("expected Completed, got {other:?}"),
    }
}

fn expect_failed(events: &[ToolEvent]) -> &ToolError {
    assert_eq!(events.len(), 1, "expected exactly one event: {events:?}");
    match &events[0] {
        ToolEvent::Failed(e) => e,
        other => panic!("expected Failed, got {other:?}"),
    }
}

fn extract_text(fields: &agent_client_protocol::schema::ToolCallUpdateFields) -> String {
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

fn extract_raw(fields: &agent_client_protocol::schema::ToolCallUpdateFields) -> serde_json::Value {
    fields.raw_output.clone().expect("raw_output")
}

// -- #1 content basic match
#[tokio::test]
async fn content_basic_matches() {
    let h = Harness::new();
    h.write("a.rs", "let x = 1;\n// TODO: fix\nlet y = 2;\n");
    h.write("b.rs", "// TODO: another\n// TODO: third\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("a.rs"), "{text}");
    assert!(text.contains("b.rs"), "{text}");
    let raw = extract_raw(fields);
    assert_eq!(raw["mode"], "content");
    assert_eq!(raw["matches_total"], 3);
    assert_eq!(raw["files_matched"], 2);
}

// -- #2 content no matches
#[tokio::test]
async fn content_no_matches() {
    let h = Harness::new();
    h.write("a.rs", "nothing relevant\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    assert_eq!(extract_text(fields), "(no matches)");
    assert_eq!(extract_raw(fields)["matches_total"], 0);
}

// -- #3 content invalid regex
#[tokio::test]
async fn content_invalid_regex() {
    let h = Harness::new();
    h.write("a.rs", "x\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "[invalid"}), h.ctx())).await;
    let err = expect_failed(&events);
    assert!(matches!(err, ToolError::InvalidArgs(_)), "{err:?}");
    assert!(format!("{err}").to_lowercase().contains("regex"));
}

// -- #4 case insensitive
#[tokio::test]
async fn content_case_insensitive() {
    let h = Harness::new();
    h.write("a.rs", "Hello world\n");
    let tool = SearchTool::new();
    let off = drive(tool.execute(json!({"pattern": "hello"}), h.ctx())).await;
    assert_eq!(extract_raw(expect_completed(&off).0)["matches_total"], 0);
    let on = drive(tool.execute(
        json!({"pattern": "hello", "case_insensitive": true}),
        h.ctx(),
    ))
    .await;
    assert_eq!(extract_raw(expect_completed(&on).0)["matches_total"], 1);
}

// -- #5 before/after context
#[tokio::test]
async fn content_with_context() {
    let h = Harness::new();
    h.write("a.rs", "line1\nline2\nMATCH\nline4\nline5\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(
        json!({"pattern": "MATCH", "before": 1, "after": 1}),
        h.ctx(),
    ))
    .await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("L2: line2"), "{text}");
    assert!(text.contains("L3: MATCH"), "{text}");
    assert!(text.contains("L4: line4"), "{text}");
}

// -- #7 content glob restricts files
#[tokio::test]
async fn content_glob_restricts() {
    let h = Harness::new();
    h.write("a.rs", "TODO rust\n");
    h.write("b.ts", "TODO ts\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(
        json!({"pattern": "TODO", "path_glob": "**/*.rs"}),
        h.ctx(),
    ))
    .await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("a.rs"), "{text}");
    assert!(!text.contains("b.ts"), "{text}");
}

// -- #7b content glob with workspace-relative directory prefix
//    AI 反馈过 `crates/**/*.rs` 完全 miss——回归这条。
#[tokio::test]
async fn content_glob_with_directory_prefix_matches_relative() {
    let h = Harness::new();
    h.write("crates/a/src/lib.rs", "pub struct Foo;\n");
    h.write("crates/b/src/main.rs", "pub struct Bar;\n");
    h.write("docs/note.md", "pub struct WrongFile;\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(
        json!({"pattern": "pub struct ", "path_glob": "crates/**/*.rs"}),
        h.ctx(),
    ))
    .await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("crates/a/src/lib.rs"), "{text}");
    assert!(text.contains("crates/b/src/main.rs"), "{text}");
    assert!(!text.contains("docs/note.md"), "{text}");
    assert_eq!(extract_raw(fields)["matches_total"], 2);
}

// -- #8 gitignore default-on
#[tokio::test]
async fn content_respects_gitignore() {
    let h = Harness::new();
    h.write(".gitignore", "vendor/\n");
    h.write("vendor/lib.rs", "TODO vendor\n");
    h.write("src/main.rs", "TODO main\n");
    let tool = SearchTool::new();
    let on = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let on_text = extract_text(expect_completed(&on).0);
    assert!(on_text.contains("src/main.rs"), "{on_text}");
    assert!(!on_text.contains("vendor"), "{on_text}");
    let off = drive(tool.execute(
        json!({"pattern": "TODO", "respect_gitignore": false}),
        h.ctx(),
    ))
    .await;
    let off_text = extract_text(expect_completed(&off).0);
    assert!(off_text.contains("vendor/lib.rs"), "{off_text}");
}

// -- #9 binary file skipped
#[tokio::test]
async fn content_skips_binary() {
    let h = Harness::new();
    let mut bin: Vec<u8> = b"prefix\0".to_vec();
    bin.extend_from_slice(b"TODO inside binary\n");
    h.write("bin.dat", bin);
    h.write("ok.txt", "TODO real\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("ok.txt"), "{text}");
    assert!(!text.contains("bin.dat"), "{text}");
}

// -- #10 oversized file skipped
#[tokio::test]
async fn content_skips_oversize() {
    let h = Harness::new();
    let mut cfg = SearchToolConfig::default();
    cfg.max_file_size_bytes = 64;
    let big = "TODO ".repeat(100);
    h.write("big.txt", &big);
    h.write("small.txt", "TODO small\n");
    let tool = SearchTool::from_config(&cfg);
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("small.txt"), "{text}");
    assert!(!text.contains("big.txt"), "{text}");
}

// -- #11 head_limit truncation
#[tokio::test]
async fn content_head_limit_truncate() {
    let h = Harness::new();
    let body = (0..20)
        .map(|i| format!("TODO line {i}\n"))
        .collect::<String>();
    h.write("a.rs", body);
    let mut cfg = SearchToolConfig::default();
    cfg.default_head_limit = 5;
    let tool = SearchTool::from_config(&cfg);
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let raw = extract_raw(fields);
    assert_eq!(raw["truncated"], true);
    let text = extract_text(fields);
    assert!(text.contains("[truncated"), "{text}");
}

// -- #14 cancellation
#[tokio::test]
async fn content_cancellation() {
    let h = Harness::new();
    for i in 0..50 {
        h.write(&format!("f{i}.txt"), "TODO\n");
    }
    h.cancel.cancel();
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let err = expect_failed(&events);
    assert!(matches!(err, ToolError::Canceled), "{err:?}");
}

// -- #15 files mode glob
#[tokio::test]
async fn files_mode_basic() {
    let h = Harness::new();
    h.write("a.rs", "x");
    h.write("b.rs", "y");
    h.write("c.ts", "z");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"mode": "files", "pattern": "**/*.rs"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("a.rs"), "{text}");
    assert!(text.contains("b.rs"), "{text}");
    assert!(!text.contains("c.ts"), "{text}");
}

// -- #16 files mode brace expansion
#[tokio::test]
async fn files_mode_brace_expansion() {
    let h = Harness::new();
    h.write("src/foo.ts", "x");
    h.write("src/foo.tsx", "y");
    h.write("src/foo.js", "z");
    let tool = SearchTool::new();
    let events = drive(tool.execute(
        json!({"mode": "files", "pattern": "src/foo.{ts,tsx}"}),
        h.ctx(),
    ))
    .await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("src/foo.ts"), "{text}");
    assert!(text.contains("src/foo.tsx"), "{text}");
    assert!(!text.contains("src/foo.js"), "{text}");
}

// -- #17 files mode invalid glob
#[tokio::test]
async fn files_mode_invalid_glob() {
    let h = Harness::new();
    h.write("a.rs", "x");
    let tool = SearchTool::new();
    let events =
        drive(tool.execute(json!({"mode": "files", "pattern": "[bad-glob"}), h.ctx())).await;
    let err = expect_failed(&events);
    assert!(matches!(err, ToolError::InvalidArgs(_)), "{err:?}");
    assert!(format!("{err}").to_lowercase().contains("glob"));
}

// -- #18 files mode no matches
#[tokio::test]
async fn files_mode_no_matches() {
    let h = Harness::new();
    h.write("a.rs", "x");
    let tool = SearchTool::new();
    let events =
        drive(tool.execute(json!({"mode": "files", "pattern": "**/*.unknown"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    assert_eq!(extract_text(fields), "(no matches)");
    assert_eq!(extract_raw(fields)["files_matched"], 0);
}

// -- #19 files mode head_limit
#[tokio::test]
async fn files_mode_head_limit() {
    let h = Harness::new();
    for i in 0..10 {
        h.write(&format!("f{i}.rs"), "x");
    }
    let mut cfg = SearchToolConfig::default();
    cfg.default_head_limit = 3;
    let tool = SearchTool::from_config(&cfg);
    let events = drive(tool.execute(json!({"mode": "files", "pattern": "**/*.rs"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let raw = extract_raw(fields);
    assert_eq!(raw["truncated"], true);
    assert_eq!(raw["files_matched"], 3);
}

// -- #21 path escape rejected
#[tokio::test]
async fn path_escape_rejected() {
    let h = Harness::new();
    h.write("a.rs", "TODO\n");
    let tool = SearchTool::new();
    let events =
        drive(tool.execute(json!({"pattern": "TODO", "path": "../../etc"}), h.ctx())).await;
    let err = expect_failed(&events);
    assert!(matches!(err, ToolError::InvalidArgs(_)), "{err:?}");
}

// -- #22 path scopes search to subdirectory
#[tokio::test]
async fn path_scopes_to_subdir() {
    let h = Harness::new();
    h.write("src/a.rs", "TODO src\n");
    h.write("docs/b.md", "TODO docs\n");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": "TODO", "path": "src"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let text = extract_text(fields);
    assert!(text.contains("a.rs"), "{text}");
    assert!(!text.contains("b.md"), "{text}");
}

// -- #23 head_limit clamped to max
#[tokio::test]
async fn head_limit_clamped() {
    let h = Harness::new();
    h.write("a.rs", "TODO\n");
    let mut cfg = SearchToolConfig::default();
    cfg.default_head_limit = 5;
    cfg.max_head_limit = 10;
    let tool = SearchTool::from_config(&cfg);
    let events = drive(tool.execute(json!({"pattern": "TODO", "head_limit": 9999}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let raw = extract_raw(fields);
    assert_eq!(raw["head_limit"], 10);
}

// -- #24 walker max_walk_files
#[tokio::test]
async fn walker_max_files_truncates() {
    let h = Harness::new();
    for i in 0..30 {
        h.write(&format!("f{i}.rs"), "TODO\n");
    }
    let mut cfg = SearchToolConfig::default();
    cfg.max_walk_files = 5;
    let tool = SearchTool::from_config(&cfg);
    let events = drive(tool.execute(json!({"pattern": "TODO"}), h.ctx())).await;
    let (fields,) = expect_completed(&events);
    let raw = extract_raw(fields);
    assert_eq!(raw["truncated"], true);
}

// -- pattern empty rejected
#[tokio::test]
async fn pattern_empty_rejected() {
    let h = Harness::new();
    h.write("a.rs", "x");
    let tool = SearchTool::new();
    let events = drive(tool.execute(json!({"pattern": ""}), h.ctx())).await;
    let err = expect_failed(&events);
    assert!(matches!(err, ToolError::InvalidArgs(_)), "{err:?}");
}
