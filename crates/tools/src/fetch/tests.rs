//! `fetch` 单元测试。覆盖 docs/internal/tools-fetch.md §10 的 #1–#19。
//! #20（真 LLM e2e）在 example 里跑。

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::{ContentBlock, ToolCallContent};
use defect_agent::http::HttpClient;
use defect_agent::tool::{SafetyClass, Tool, ToolContext, ToolError, ToolEvent, ToolStream};
use defect_config::{FetchFormat, FetchToolConfig};
use defect_http::{HttpStackConfig, ProxyConfig, build_fetch_client_arc};
use futures::StreamExt;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use defect_agent::fs::FsBackend;
use defect_agent::shell::{NoopShellBackend, ShellBackend};

use super::FetchTool;
use crate::fs::LocalFsBackend;

/// 构造一份指向真实 wiremock 的 fetch http client。
async fn fixture(config: FetchToolConfig) -> (MockServer, FetchTool, Arc<dyn HttpClient>) {
    let server = MockServer::start().await;
    let stack = HttpStackConfig {
        proxy: ProxyConfig::Disabled,
        total_timeout: Some(Duration::from_secs(5)),
        transport_retries: 0,
        ..HttpStackConfig::default()
    };
    let http = build_fetch_client_arc(&stack).expect("fetch client");
    (server, FetchTool::from_config(&config), http)
}

fn ctx<'a>(
    cwd: &'a std::path::Path,
    cancel: CancellationToken,
    http: Arc<dyn HttpClient>,
) -> ToolContext<'a> {
    let fs: Arc<dyn FsBackend> = Arc::new(LocalFsBackend::new(cwd.to_path_buf()));
    let shell: Arc<dyn ShellBackend> = Arc::new(NoopShellBackend);
    ToolContext::new(cwd, cancel, fs, shell, http, "test-model")
}

async fn drive(stream: ToolStream) -> Vec<ToolEvent> {
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

#[test]
fn safety_hint_is_readonly() {
    let tool = FetchTool::new();
    assert_eq!(
        tool.safety_hint(&json!({"url": "http://example.com"})),
        SafetyClass::ReadOnly
    );
}

#[test]
fn schema_includes_fetch_name_and_required_url() {
    let tool = FetchTool::new();
    let schema = tool.schema();
    assert_eq!(schema.name, "fetch");
    let required = schema.input_schema.get("required").unwrap();
    let arr = required.as_array().unwrap();
    assert!(arr.iter().any(|v| v.as_str() == Some("url")));
}

// ─── §10 #1 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case1_text_markdown_passthrough() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/200"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("# Hello\nbody\n", "text/markdown"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/200", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let text = extract_text(&events[0]);
    assert!(text.contains("# Hello"), "got: {text}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["status"], 200);
}

// ─── §10 #2 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case2_html_to_markdown_rendered() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("<h1>Title</h1><p>Body text</p>", "text/html; charset=utf-8"),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/page", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    let text = extract_text(&events[0]);
    assert!(text.contains("# Title"), "got: {text}");
    assert!(text.contains("Body text"), "got: {text}");
    assert!(!text.contains("<h1>"));
}

// ─── §10 #3 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case3_format_html_with_markdown_content_type_fails() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/md"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("# x", "text/markdown"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/md", server.uri());
    let events = drive(tool.execute(
        json!({"url": url, "format": "html"}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got {:?}",
        events[0]
    );
    let msg = format!("{:?}", events[0]);
    assert!(msg.contains("not HTML"), "got: {msg}");
}

// ─── §10 #4 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case4_format_text_strips_html_tags() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw("<p>plain <em>text</em></p>", "text/html"),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/page", server.uri());
    let events = drive(tool.execute(
        json!({"url": url, "format": "text"}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    let text = extract_text(&events[0]);
    assert!(!text.contains('<'));
    assert!(text.contains("plain"), "got: {text}");
}

// ─── §10 #5 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case5_404_is_completed_with_status_marker() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_raw("nope", "text/plain"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/missing", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let text = extract_text(&events[0]);
    assert!(text.contains("[http status: 404]"), "got: {text}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["status"], 404);
}

// ─── §10 #6 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case6_500_is_completed_with_status_marker() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/oops"))
        .respond_with(ResponseTemplate::new(500).set_body_raw("err", "text/plain"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/oops", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    let text = extract_text(&events[0]);
    assert!(text.contains("[http status: 500]"), "got: {text}");
}

// ─── §10 #7 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case7_file_scheme_rejected() {
    let (_server, tool, http) = fixture(FetchToolConfig::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let events = drive(tool.execute(
        json!({"url": "file:///etc/passwd"}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "got: {:?}",
        events[0]
    );
}

// ─── §10 #9 ────────────────────────────────────────────────────────────────
#[tokio::test]
async fn case9_timeout_yields_failed_execution() {
    let mut config = FetchToolConfig::default();
    config.default_timeout_secs = 1;
    let (server, tool, http) = fixture(config).await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("late", "text/plain")
                .set_delay(Duration::from_secs(3)),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/slow", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    let msg = format!("{:?}", events[0]);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got: {msg}"
    );
    assert!(msg.contains("timed out"), "got: {msg}");
}

// ─── §10 #10 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case10_response_truncation() {
    let mut config = FetchToolConfig::default();
    config.max_response_bytes = 64;
    let (server, tool, http) = fixture(config).await;
    let big = "a".repeat(1024);
    Mock::given(method("GET"))
        .and(path("/big"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(big, "text/plain"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/big", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    let text = extract_text(&events[0]);
    assert!(text.contains("[response truncated"), "got: {text}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["truncated"], true);
}

// ─── §10 #11 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case11_redirect_followed() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    let final_path = "/dest";
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", final_path))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(final_path))
        .respond_with(ResponseTemplate::new(200).set_body_raw("arrived", "text/plain"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/start", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["redirects"], 1);
    assert_eq!(raw["status"], 200);
    let text = extract_text(&events[0]);
    assert!(text.contains("arrived"), "got: {text}");
}

// ─── §10 #13 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case13_no_follow_returns_3xx() {
    let mut config = FetchToolConfig::default();
    config.follow_redirects = false;
    let (server, tool, http) = fixture(config).await;
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", "/dest"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/start", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["status"], 302);
    assert_eq!(raw["redirects"], 0);
}

// ─── §10 #14 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case14_html_to_markdown_disabled_returns_raw_html() {
    let mut config = FetchToolConfig::default();
    config.html_to_markdown = false;
    let (server, tool, http) = fixture(config).await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("<h1>X</h1>", "text/html"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/page", server.uri());
    let events = drive(tool.execute(
        json!({"url": url, "format": "markdown"}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    let text = extract_text(&events[0]);
    assert!(text.contains("<h1>X</h1>"), "got: {text}");
    assert!(text.contains("html_to_markdown disabled"), "got: {text}");
}

// ─── §10 #16 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case16_cancel_yields_failed_canceled() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("late", "text/plain")
                .set_delay(Duration::from_secs(3)),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let cancel2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel2.cancel();
    });

    let url = format!("{}/slow", server.uri());
    let events = drive(tool.execute(json!({"url": url}), ctx(dir.path(), cancel, http))).await;
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Canceled)),
        "got: {:?}",
        events[0]
    );
}

// ─── §10 #18 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case18_clamp_timeout_records_clamped_from() {
    let mut config = FetchToolConfig::default();
    config.default_timeout_secs = 1;
    config.max_timeout_secs = 2;
    let (server, tool, http) = fixture(config).await;
    Mock::given(method("GET"))
        .and(path("/quick"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("ok", "text/plain"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/quick", server.uri());
    let events = drive(tool.execute(
        json!({"url": url, "timeout_secs": 999}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["timeout_clamped_from"], 999);
}

// ─── §10 #19 ───────────────────────────────────────────────────────────────
#[tokio::test]
async fn case19_binary_content_type_rejected() {
    let (server, tool, http) = fixture(FetchToolConfig::default()).await;
    Mock::given(method("GET"))
        .and(path("/img"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(vec![0u8, 1, 2, 3], "image/png"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::new();
    let url = format!("{}/img", server.uri());
    let events = drive(tool.execute(
        json!({"url": url, "format": "text"}),
        ctx(dir.path(), cancel, http),
    ))
    .await;
    let msg = format!("{:?}", events[0]);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Execution(_))),
        "got: {msg}"
    );
    assert!(msg.contains("binary content-type"), "got: {msg}");
}

#[test]
fn default_format_baked_into_schema() {
    let mut config = FetchToolConfig::default();
    config.default_format = FetchFormat::Html;
    let tool = FetchTool::from_config(&config);
    let schema = tool.schema();
    // The format property's description mentions the configured default.
    let format_prop = &schema.input_schema["properties"]["format"]["description"];
    let desc = format_prop.as_str().unwrap_or("");
    assert!(desc.contains("html"), "got: {desc}");
}
