use agent_client_protocol::schema::{Content as AcpContent, ToolCallContent};
use rmcp::model::{CallToolResult, Content};
use serde_json::json;

use crate::{build_call_params, completed_event};

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
