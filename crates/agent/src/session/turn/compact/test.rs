use super::*;
use crate::llm::{Message, MessageContent, Role, ToolResultBody};

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }]
        .into(),
    }
}

fn assistant(text: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }]
        .into(),
    }
}

fn assistant_tool_use(id: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![MessageContent::ToolUse {
            id: id.to_string(),
            name: "read".to_string(),
            args: serde_json::json!({}),
        }]
        .into(),
    }
}

/// 工具结果回填消息（role=User 但只含 ToolResult）——**不是**轮次起点。
fn tool_result(id: &str, text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::ToolResult {
            tool_use_id: id.to_string(),
            output: ToolResultBody::Text {
                text: text.to_string(),
            },
            is_error: false,
        }]
        .into(),
    }
}

#[test]
fn turn_start_requires_non_tool_result_user_content() {
    assert!(is_turn_start(&user("hi")));
    assert!(!is_turn_start(&assistant("hello")));
    assert!(!is_turn_start(&tool_result("t1", "out")));
}

#[test]
fn single_turn_has_no_earlier_history_to_summarize() {
    // 只有一个用户轮次（开头即唯一轮次起点）→ 无更早历史，返回 None。
    let messages = vec![user("only"), assistant("reply")];
    assert_eq!(select_boundary(&messages, 8_000), None);
}

#[test]
fn empty_history_returns_none() {
    assert_eq!(select_boundary(&[], 8_000), None);
}

#[test]
fn boundary_keeps_recent_turns_within_budget() {
    // 三个轮次，每个很小。预算够大 → 保留尽量多但 head 非空：边界落在第二个
    // 轮次起点（index 2），head = 第一个轮次。
    let messages = vec![
        user("turn1 user"),       // 0  turn start
        assistant("turn1 reply"), // 1
        user("turn2 user"),       // 2  turn start
        assistant("turn2 reply"), // 3
        user("turn3 user"),       // 4  turn start
        assistant("turn3 reply"), // 5
    ];
    // 预算极大 → 想全保留，但 last_start>0 时仍不能让 head 空：
    // 算法从最新往旧累加，start==0 时 break 不记入 best，故 best 最小到 index 2。
    let boundary = select_boundary(&messages, 1_000_000).expect("boundary");
    assert_eq!(boundary, 2);
    let (head, tail) = messages.split_at(boundary);
    assert_eq!(head.len(), 2);
    assert!(is_turn_start(tail.first().expect("tail non-empty")));
}

#[test]
fn tiny_budget_keeps_only_last_turn() {
    let messages = vec![
        user("turn1"),
        assistant("r1"),
        user("turn2"),
        assistant("r2"),
        user("turn3"),
        assistant("r3"),
    ];
    // 预算极小：连最新轮次都装不下 → 回退到最新轮次起点 index 4。
    let boundary = select_boundary(&messages, 1).expect("boundary");
    assert_eq!(boundary, 4);
}

#[test]
fn boundary_never_splits_tool_use_result_pair() {
    // 轮次2 含 tool_use(assistant) + tool_result(user)。边界必须落在轮次起点，
    // 绝不落在 tool_result 上，确保 tail 不出现孤儿 tool_result。
    let messages = vec![
        user("turn1"),                // 0 start
        assistant("r1"),              // 1
        user("turn2"),                // 2 start
        assistant_tool_use("call_a"), // 3
        tool_result("call_a", "out"), // 4  <- 不是轮次起点
        assistant("r2"),              // 5
    ];
    let boundary = select_boundary(&messages, 1_000_000).expect("boundary");
    // 唯一安全的非零轮次起点是 index 2。
    assert_eq!(boundary, 2);
    let (_head, tail) = messages.split_at(boundary);
    // tail 第一条是真实 user 轮次起点，不是孤儿 tool_result。
    assert!(is_turn_start(tail.first().expect("tail non-empty")));
}

#[test]
fn extract_previous_summary_strips_prefix() {
    let summary_body = "## Goal\nbuild a thing";
    let head = vec![
        user("earlier"),
        Message {
            role: Role::Assistant,
            content: vec![MessageContent::Text {
                text: format!("{SUMMARY_PREFIX}\n{summary_body}"),
            }]
            .into(),
        },
    ];
    assert_eq!(
        extract_previous_summary(&head).as_deref(),
        Some(summary_body)
    );
}

#[test]
fn extract_previous_summary_none_when_absent() {
    let head = vec![user("a"), assistant("regular reply")];
    assert_eq!(extract_previous_summary(&head), None);
}

#[test]
fn truncate_chars_respects_multibyte_boundary() {
    let s = "héllo wörld"; // multibyte
    let out = truncate_chars(s, 5);
    assert!(out.starts_with("héllo"));
    assert!(out.contains("truncated"));
    // 不截断短串。
    assert_eq!(truncate_chars("short", 100), "short");
}

#[test]
fn prepare_head_message_truncates_tool_output_and_strips_image() {
    let long = "x".repeat(TOOL_RESULT_MAX_CHARS + 500);
    let msg = Message {
        role: Role::User,
        content: vec![
            MessageContent::ToolResult {
                tool_use_id: "t1".to_string(),
                output: ToolResultBody::Text { text: long },
                is_error: false,
            },
            MessageContent::Image {
                mime: "image/png".to_string(),
                data: crate::llm::ImageData::Base64 {
                    encoded: "AAAA".to_string(),
                },
            },
        ]
        .into(),
    };
    let prepared = prepare_head_message(&msg);
    match prepared.content.first().expect("tool result content") {
        MessageContent::ToolResult { output, .. } => match output {
            ToolResultBody::Text { text } => {
                assert!(text.chars().count() <= TOOL_RESULT_MAX_CHARS + 40);
                assert!(text.contains("truncated"));
            }
            _ => panic!("expected text output"),
        },
        _ => panic!("expected tool result"),
    }
    assert!(matches!(
        prepared.content.get(1).expect("image placeholder"),
        MessageContent::Text { text } if text.contains("image omitted")
    ));
}

#[test]
fn build_prompt_wraps_previous_summary() {
    let with_prev = build_prompt(Some("old summary"));
    assert!(with_prev.contains("<previous-summary>"));
    assert!(with_prev.contains("old summary"));
    let without = build_prompt(None);
    assert!(!without.contains("<previous-summary>"));
}
