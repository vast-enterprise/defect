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

/// 合法配对序列：原样返回（同一 Arc，不重建）。
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

/// 孤儿 tool_use（无对应 result）：在该 assistant 之后补一条 error tool_result。
#[test]
fn orphan_tool_use_gets_synthetic_result() {
    let input = vec![
        msg(Role::User, vec![text("hi")]),
        msg(Role::Assistant, vec![tool_use("a")]),
        // 没有 a 的 result —— turn 在工具执行中途被中断。
    ];
    let out = sanitize_tool_pairing(input);
    assert_eq!(out.len(), 3);
    // 补的那条紧跟在 assistant 之后。
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

/// 多个孤儿在同一条 assistant 里：合并补一条含多个 result 的 user 消息。
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

/// 部分配对：只补缺的那个，已配对的不动。
#[test]
fn only_missing_one_is_filled() {
    let input = vec![
        msg(Role::Assistant, vec![tool_use("a")]),
        msg(Role::User, vec![tool_result("a")]),
        msg(Role::Assistant, vec![tool_use("b")]),
        // b 无 result。
    ];
    let out = sanitize_tool_pairing(input);
    assert_eq!(out.len(), 4);
    let MessageContent::ToolResult { tool_use_id, .. } = &out[3].content[0] else {
        panic!("expected synthetic result for b");
    };
    assert_eq!(tool_use_id, "b");
}

/// result 在序列中"全局存在"（即便不严格紧邻）也算已满足，不重复补。
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

/// 空序列 / 无 tool_use：原样返回。
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
