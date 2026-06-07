use super::*;

fn msg(role: Role, content: Vec<MessageContent>) -> Message {
    Message {
        role,
        content: content.into(),
    }
}

fn tool_use(id: &str) -> MessageContent {
    MessageContent::ToolUse {
        id: id.to_string(),
        name: "do_thing".to_string(),
        args: serde_json::json!({}),
    }
}

fn tool_result(id: &str) -> MessageContent {
    MessageContent::ToolResult {
        tool_use_id: id.to_string(),
        output: ToolResultBody::Text {
            text: "ok".to_string(),
        },
        is_error: false,
    }
}

fn text(t: &str) -> MessageContent {
    MessageContent::Text {
        text: t.to_string(),
    }
}

/// Valid paired sequence: returned as-is (same Arc, no rebuild).
#[test]
fn paired_sequence_unchanged() {
    let input = vec![
        msg(Role::User, vec![text("hi")]),
        msg(Role::Assistant, vec![tool_use("a")]),
        msg(Role::User, vec![tool_result("a")]),
    ];
    let out = sanitize_tool_pairing(input.clone());
    assert_eq!(out, input);
}

/// Orphan tool_use (no matching result): inject an error tool_result after the assistant
/// message.
#[test]
fn orphan_tool_use_gets_synthetic_result() {
    let input = vec![
        msg(Role::User, vec![text("hi")]),
        msg(Role::Assistant, vec![tool_use("a")]),
        // No result for `a` — the turn was interrupted mid-tool-execution.
    ];
    let out = sanitize_tool_pairing(input);
    assert_eq!(out.len(), 3);
    // The synthetic entry follows immediately after the assistant message.
    let MessageContent::ToolResult {
        tool_use_id,
        is_error,
        ..
    } = &out[2].content[0]
    else {
        panic!("expected synthetic tool_result");
    };
    assert_eq!(tool_use_id, "a");
    assert!(is_error);
    assert_eq!(out[2].role, Role::User);
}

/// Multiple orphans in the same assistant message: merge into a single user message
/// containing multiple results.
#[test]
fn multiple_orphans_in_one_assistant() {
    let input = vec![msg(Role::Assistant, vec![tool_use("a"), tool_use("b")])];
    let out = sanitize_tool_pairing(input);
    assert_eq!(out.len(), 2);
    assert_eq!(out[1].content.len(), 2);
    for (i, id) in ["a", "b"].iter().enumerate() {
        let MessageContent::ToolResult { tool_use_id, .. } = &out[1].content[i] else {
            panic!("expected tool_result");
        };
        assert_eq!(tool_use_id, id);
    }
}

/// Partial pairing: only fill the missing one; leave already paired ones unchanged.
#[test]
fn only_missing_one_is_filled() {
    let input = vec![
        msg(Role::Assistant, vec![tool_use("a")]),
        msg(Role::User, vec![tool_result("a")]),
        msg(Role::Assistant, vec![tool_use("b")]),
        // b has no result.
    ];
    let out = sanitize_tool_pairing(input);
    assert_eq!(out.len(), 4);
    let MessageContent::ToolResult { tool_use_id, .. } = &out[3].content[0] else {
        panic!("expected synthetic result for b");
    };
    assert_eq!(tool_use_id, "b");
}

/// A result is considered satisfied if it exists anywhere in the sequence (even if not
/// strictly adjacent); no duplicate is inserted.
#[test]
fn globally_satisfied_not_duplicated() {
    let input = vec![
        msg(Role::Assistant, vec![tool_use("a")]),
        msg(Role::User, vec![tool_result("a")]),
    ];
    let out = sanitize_tool_pairing(input.clone());
    assert_eq!(out, input);
    assert_eq!(out.len(), 2);
}

/// Empty sequence / no tool_use: returned unchanged.
#[test]
fn no_tool_use_unchanged() {
    let input = vec![
        msg(Role::User, vec![text("hi")]),
        msg(Role::Assistant, vec![text("hello")]),
    ];
    let out = sanitize_tool_pairing(input.clone());
    assert_eq!(out, input);

    let empty: Vec<Message> = Vec::new();
    assert!(sanitize_tool_pairing(empty).is_empty());
}
