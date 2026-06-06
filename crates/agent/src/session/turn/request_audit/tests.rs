use serde_json::json;

use super::{RequestAuditDelta, RequestAuditSnapshot};
use crate::llm::{
    CompletionRequest, HostedCapabilities, Message, MessageContent, Role, SamplingParams,
    ToolChoice,
};
use crate::tool::ToolSchema;

#[test]
fn snapshot_counts_thinking_tool_use_and_tool_result_blocks() {
    let req = CompletionRequest {
        model: "deepseek-v4-flash".to_string(),
        system: Some("system".into()),
        messages: vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    MessageContent::Thinking {
                        text: "reason".to_string(),
                        signature: None,
                    },
                    MessageContent::Text {
                        text: "answer".to_string(),
                    },
                    MessageContent::ToolUse {
                        id: "call_1".to_string(),
                        name: "read_file".to_string(),
                        args: json!({"path": "Cargo.toml"}),
                    },
                ]
                .into(),
            },
            Message {
                role: Role::User,
                content: vec![MessageContent::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    output: crate::llm::ToolResultBody::Text {
                        text: "ok".to_string(),
                    },
                    is_error: false,
                }]
                .into(),
            },
        ],
        tools: vec![ToolSchema {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
        }],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: HostedCapabilities::default(),
    };

    let snapshot = RequestAuditSnapshot::from_request(&req);
    assert_eq!(snapshot.assistant_messages, 1);
    assert_eq!(snapshot.user_messages, 1);
    assert_eq!(snapshot.thinking_blocks, 1);
    assert_eq!(snapshot.text_blocks, 1);
    assert_eq!(snapshot.tool_use_blocks, 1);
    assert_eq!(snapshot.tool_result_blocks, 1);
    assert_eq!(snapshot.total_tool_result_bytes, 2);
}

#[test]
fn delta_reports_message_and_tool_changes() {
    let base = RequestAuditSnapshot::from_request(&request_with_text("hello"));
    let changed = RequestAuditSnapshot::from_request(&request_with_text("hello again"));
    let delta = RequestAuditDelta::between(Some(&base), &changed);

    assert!(delta.changed.contains(&"messages"));
    assert!(delta.changed.contains(&"text_bytes"));
    assert_eq!(delta.changed_count(), 2);
}

#[test]
fn delta_reports_none_when_snapshot_is_identical() {
    let base = RequestAuditSnapshot::from_request(&request_with_text("hello"));
    let changed = RequestAuditSnapshot::from_request(&request_with_text("hello"));
    let delta = RequestAuditDelta::between(Some(&base), &changed);

    assert_eq!(delta.changed, vec!["none"]);
    assert_eq!(delta.changed_count(), 0);
}

fn request_with_text(text: &str) -> CompletionRequest {
    CompletionRequest {
        model: "gpt-4.1-nano".to_string(),
        system: Some("system".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text {
                text: text.to_string(),
            }]
            .into(),
        }],
        tools: vec![ToolSchema {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
        }],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: HostedCapabilities::default(),
    }
}
