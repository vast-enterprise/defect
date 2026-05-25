use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::AcpAgent;
use agent_client_protocol::schema::{
    ContentBlock, EnvVariable, InitializeRequest, McpServer, McpServerSse, McpServerStdio,
    NewSessionRequest, PromptRequest, ProtocolVersion, SessionNotification, SessionUpdate,
    StopReason, ToolCallContent, ToolCallStatus,
};
use serde_json::Value;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const TEST_OPENAI_API_KEY: &str = "test-openai-key";
const TEST_OPENAI_AUTH_HEADER: &str = "Bearer test-openai-key";
const DONE: &str = "[DONE]";

#[tokio::test]
async fn stdio_mcp_tool_round_trip() {
    let openai = MockServer::start().await;
    let state_root = tempfile::tempdir().expect("state tempdir");
    let cwd = tempfile::tempdir().expect("cwd tempdir");

    let round1 = openai_sse_body(&[
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_echo","type":"function","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"message\":\"hello from mcp\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    ]);
    let round2 = openai_sse_body(&[
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"content":"done after mcp"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", TEST_OPENAI_AUTH_HEADER))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).expect("body json");
            let has_tool_result = body
                .get("messages")
                .and_then(Value::as_array)
                .map(|messages| {
                    messages
                        .iter()
                        .any(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
                })
                .unwrap_or(false);
            let payload = if has_tool_result {
                round2.clone()
            } else {
                round1.clone()
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(payload, "text/event-stream")
        })
        .expect(2)
        .mount(&openai)
        .await;

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_defect"));
    let agent = AcpAgent::from_args([
        format!("XDG_STATE_HOME={}", state_root.path().display()),
        format!("OPENAI_API_KEY={TEST_OPENAI_API_KEY}"),
        format!("OPENAI_BASE_URL={}", openai.uri()),
        binary.display().to_string(),
        "--provider".to_string(),
        "openai".to_string(),
    ])
    .expect("valid defect command");

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = Arc::clone(&updates);
    let mcp_server = McpServer::Stdio(
        McpServerStdio::new(
            "mcp-echo",
            PathBuf::from(env!("CARGO_BIN_EXE_defect-mcp-test-server")),
        )
        .env(vec![EnvVariable::new("MCP_TEST_VALUE", "from-env")]),
    );

    let stop_reason = agent_client_protocol::Client
        .builder()
        .name("stdio-mcp-smoke-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let session = cx
                .send_request(NewSessionRequest::new(cwd.path()).mcp_servers(vec![mcp_server]))
                .block_task()
                .await?;

            let response = cx
                .send_request(PromptRequest::new(
                    session.session_id,
                    vec![ContentBlock::from("please use the mcp tool")],
                ))
                .block_task()
                .await?;

            Ok(response.stop_reason)
        })
        .await
        .expect("client connection completed");

    assert_eq!(stop_reason, StopReason::EndTurn);

    let updates = updates.lock().expect("updates mutex");
    let tool_completion = updates.iter().find_map(|update| match update {
        SessionUpdate::ToolCallUpdate(tool_update)
            if tool_update.fields.status == Some(ToolCallStatus::Completed) =>
        {
            tool_update.fields.content.as_ref()
        }
        _ => None,
    });
    let Some(tool_completion) = tool_completion else {
        panic!("expected completed tool update; updates={updates:?}");
    };
    assert!(
        tool_completion.iter().any(|content| matches!(
            content,
            ToolCallContent::Content(block)
                if matches!(
                    &block.content,
                    ContentBlock::Text(text)
                        if text.text.contains(r#""echo":"hello from mcp""#)
                            && text.text.contains(r#""env":"from-env""#)
                )
        )),
        "tool completion should contain MCP response; updates={updates:?}",
    );

    let assistant_chunks: String = updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => Some(&chunk.content),
            _ => None,
        })
        .filter_map(|content| match content {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        assistant_chunks.contains("done after mcp"),
        "assistant updates should include second-round text; got {assistant_chunks:?}",
    );
}

fn openai_sse_body(chunks: &[&str]) -> Vec<u8> {
    let mut body = Vec::new();
    for chunk in chunks {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(chunk.as_bytes());
        body.extend_from_slice(b"\n\n");
    }
    body.extend_from_slice(b"data: ");
    body.extend_from_slice(DONE.as_bytes());
    body.extend_from_slice(b"\n\n");
    body
}

#[tokio::test]
async fn sse_mcp_tool_round_trip() {
    let state_root = tempfile::tempdir().expect("state tempdir");
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let server = spawn_streamable_http_server().await;

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_defect"));
    let agent = AcpAgent::from_args([
        format!("XDG_STATE_HOME={}", state_root.path().display()),
        format!("OPENAI_API_KEY={TEST_OPENAI_API_KEY}"),
        format!("OPENAI_BASE_URL={}", server.openai.uri()),
        binary.display().to_string(),
        "--provider".to_string(),
        "openai".to_string(),
    ])
    .expect("valid defect command");

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = Arc::clone(&updates);

    let stop_reason = agent_client_protocol::Client
        .builder()
        .name("sse-mcp-smoke-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let mcp_server = McpServer::Sse(
                McpServerSse::new("mcp-sse", format!("{}/mcp", server.mcp_base_url)).headers(vec![
                    agent_client_protocol::schema::HttpHeader::new("x-mcp-test", "enabled"),
                ]),
            );
            let session = cx
                .send_request(NewSessionRequest::new(cwd.path()).mcp_servers(vec![mcp_server]))
                .block_task()
                .await?;
            let response = cx
                .send_request(PromptRequest::new(
                    session.session_id,
                    vec![ContentBlock::from("please use the sse mcp tool")],
                ))
                .block_task()
                .await?;
            Ok(response.stop_reason)
        })
        .await
        .expect("client connection completed");

    assert_eq!(stop_reason, StopReason::EndTurn);

    let updates = updates.lock().expect("updates mutex");
    let tool_completion = updates.iter().find_map(|update| match update {
        SessionUpdate::ToolCallUpdate(tool_update)
            if tool_update.fields.status == Some(ToolCallStatus::Completed) =>
        {
            tool_update.fields.content.as_ref()
        }
        _ => None,
    });
    let Some(tool_completion) = tool_completion else {
        panic!("expected completed tool update; updates={updates:?}");
    };
    assert!(
        tool_completion.iter().any(|content| matches!(
            content,
            ToolCallContent::Content(block)
                if matches!(
                    &block.content,
                    ContentBlock::Text(text)
                        if text.text.contains(r#""echo":"hello from mcp""#)
                            && text.text.contains(r#""env":"from-env""#)
                )
        )),
        "tool completion should contain MCP response; updates={updates:?}",
    );
}

struct StreamableHttpServerHandle {
    child: tokio::process::Child,
    mcp_base_url: String,
    openai: MockServer,
}

impl Drop for StreamableHttpServerHandle {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

async fn spawn_streamable_http_server() -> StreamableHttpServerHandle {
    let openai = MockServer::start().await;
    let round1 = openai_sse_body(&[
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_echo","type":"function","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"message\":\"hello from mcp\"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    ]);
    let round2 = openai_sse_body(&[
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"content":"done after mcp"},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", TEST_OPENAI_AUTH_HEADER))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).expect("body json");
            let has_tool_result = body
                .get("messages")
                .and_then(Value::as_array)
                .map(|messages| {
                    messages
                        .iter()
                        .any(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
                })
                .unwrap_or(false);
            let payload = if has_tool_result {
                round2.clone()
            } else {
                round1.clone()
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(payload, "text/event-stream")
        })
        .expect(2)
        .mount(&openai)
        .await;

    let addr_file = tempfile::NamedTempFile::new().expect("addr file");
    let addr_path = addr_file.path().to_path_buf();
    let child =
        tokio::process::Command::new(env!("CARGO_BIN_EXE_defect-mcp-streamable-http-test-server"))
            .env("MCP_STREAMABLE_HTTP_BOUND_ADDR_FILE", addr_path.as_os_str())
            .env("MCP_TEST_VALUE", "from-env")
            .spawn()
            .expect("streamable http MCP server should spawn");

    let mcp_base_url = wait_for_bound_addr(&addr_path).await;
    StreamableHttpServerHandle {
        child,
        mcp_base_url: format!("http://{mcp_base_url}"),
        openai,
    }
}

async fn wait_for_bound_addr(path: &std::path::Path) -> String {
    const MAX_ATTEMPTS: usize = 100;
    const SLEEP_MS: u64 = 50;

    for _ in 0..MAX_ATTEMPTS {
        if let Ok(bound_addr) = std::fs::read_to_string(path) {
            let trimmed = bound_addr.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        tokio::time::sleep(Duration::from_millis(SLEEP_MS)).await;
    }

    panic!("timed out waiting for streamable http MCP server address at {path:?}");
}
