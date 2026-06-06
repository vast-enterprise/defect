use agent_client_protocol_schema::{
    Content as AcpContent, McpServer, McpServerSse, McpServerStdio, ToolCallContent,
};
use rmcp::model::{CallToolResult, Content};
use serde_json::json;

use crate::{build_call_params, completed_event, merge_mcp_servers, registered_mcp_tool_name};

#[test]
fn build_call_params_accepts_object_args() {
    let params = build_call_params("echo".to_string(), json!({"message": "hi"}))
        .expect("object args should be accepted");

    assert_eq!(params.name.as_ref(), "echo");
    assert_eq!(
        params.arguments.expect("arguments should exist"),
        serde_json::Map::from_iter([(String::from("message"), json!("hi"))]),
    );
}

#[test]
fn completed_event_uses_structured_content_when_text_missing() {
    let mut call = CallToolResult::success(Vec::new());
    call.structured_content = Some(json!({"echo": "hello"}));
    let event = completed_event(call);

    let defect_agent::tool::ToolEvent::Completed(fields) = event else {
        panic!("expected completed event");
    };
    let content = fields.content.expect("content should exist");
    assert_eq!(content.len(), 1);
    assert_eq!(
        content[0],
        ToolCallContent::Content(AcpContent::new(r#"{"echo":"hello"}"#))
    );
    assert!(fields.raw_output.is_some());
}

#[test]
fn completed_event_collects_text_content() {
    let event = completed_event(CallToolResult::success(vec![
        Content::text("hello"),
        Content::text(" world"),
    ]));

    let defect_agent::tool::ToolEvent::Completed(fields) = event else {
        panic!("expected completed event");
    };
    let content = fields.content.expect("content should exist");
    assert_eq!(
        content,
        vec![
            ToolCallContent::Content(AcpContent::new("hello")),
            ToolCallContent::Content(AcpContent::new(" world")),
        ]
    );
}

#[test]
fn completed_event_ignores_non_text_content() {
    let event = completed_event(CallToolResult::success(vec![Content::image(
        "aGVsbG8=",
        "image/png",
    )]));

    let defect_agent::tool::ToolEvent::Completed(fields) = event else {
        panic!("expected completed event");
    };
    assert!(fields.content.is_none());
    assert!(fields.raw_output.is_some());
}

#[test]
fn registered_name_prefixes_server_and_tool() {
    // 形态固定为 `mcp.<server>.<tool>`——既覆盖 search/fetch 撞名，也覆盖普通
    // MCP 工具，且没有任何条件分支：所有 MCP 工具一律 namespace。
    assert_eq!(
        registered_mcp_tool_name("docs", "search"),
        "mcp.docs.search"
    );
    assert_eq!(
        registered_mcp_tool_name("notion", "fetch"),
        "mcp.notion.fetch"
    );
    assert_eq!(
        registered_mcp_tool_name("private", "create_page"),
        "mcp.private.create_page"
    );
}

#[test]
fn session_mcp_servers_override_config_defaults_by_name() {
    let merged = merge_mcp_servers(
        &[
            McpServer::Stdio(McpServerStdio::new("echo", "/usr/bin/default-echo")),
            McpServer::Sse(McpServerSse::new("docs", "http://127.0.0.1:3000/mcp")),
        ],
        &[McpServer::Stdio(McpServerStdio::new(
            "echo",
            "/usr/bin/session-echo",
        ))],
    );

    assert_eq!(merged.len(), 2);
    assert!(matches!(
        &merged[0],
        McpServer::Sse(server) if server.name == "docs"
    ));
    assert!(matches!(
        &merged[1],
        McpServer::Stdio(server)
            if server.name == "echo" && server.command == std::path::PathBuf::from("/usr/bin/session-echo")
    ));
}
