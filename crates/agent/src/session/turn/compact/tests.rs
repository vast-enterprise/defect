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

/// Tool result backfill message (role=User but only contains ToolResult) — **not** a turn
/// start.
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
fn single_turn_falls_back_to_assistant_boundary() {
    // A single user turn (the only turn start is at index 0) → the user-turn ruler yields
    // nothing, so we fall back to the assistant boundary so the tail can still be split off
    // and the head summarized. `[user, assistant]` → boundary at the assistant (index 1).
    let messages = vec![user("only"), assistant("reply")];
    assert_eq!(select_boundary(&messages, 8_000), Some(1));
}

#[test]
fn single_turn_long_autonomous_loop_compacts_via_assistant_boundary() {
    // Reproduces the real-world goal/autonomous case: one user input drives many tool
    // round-trips with no further user message. The old user-turn-only ruler returned None
    // here (compaction silently skipped, context grew unbounded). The assistant-boundary
    // fallback must produce a non-zero boundary so the head can be summarized.
    let mut messages = vec![user("do the whole task")]; // 0: the only turn start
    for i in 0..20 {
        messages.push(assistant_tool_use(&format!("call_{i}")));
        messages.push(tool_result(&format!("call_{i}"), "output"));
    }
    let boundary = select_boundary(&messages, 8_000).expect("must compact, not skip");
    assert!(boundary > 0, "head must be non-empty");
    // The cut lands on an assistant message — never on a tool_result, which would orphan it.
    assert_eq!(messages[boundary].role, Role::Assistant);
    // And that assistant is a real round-trip start, not a dangling tool_result.
    let (_head, tail) = messages.split_at(boundary);
    assert_eq!(tail.first().expect("tail non-empty").role, Role::Assistant);
}

#[test]
fn empty_history_returns_none() {
    assert_eq!(select_boundary(&[], 8_000), None);
}

#[test]
fn boundary_keeps_recent_turns_within_budget() {
    // Three turns, each very small. Budget large enough → keep as many as possible but
    // head non-empty: boundary lands at the second turn's start (index 2), head = first
    // turn.
    let messages = vec![
        user("turn1 user"),       // 0  turn start
        assistant("turn1 reply"), // 1
        user("turn2 user"),       // 2  turn start
        assistant("turn2 reply"), // 3
        user("turn3 user"),       // 4  turn start
        assistant("turn3 reply"), // 5
    ];
    // Even with a huge budget, we cannot leave `head` empty when `last_start > 0`:
    // the algorithm accumulates from newest to oldest, breaking at `start == 0` without
    // counting it in `best`, so `best` is at least index 2.
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
    // Budget too small to fit even the latest turn; fall back to the start of the latest
    // turn at index 4.
    let boundary = select_boundary(&messages, 1).expect("boundary");
    assert_eq!(boundary, 4);
}

#[test]
fn boundary_never_splits_tool_use_result_pair() {
    // Turn 2 contains a tool_use (assistant) + tool_result (user). The boundary must fall
    // on a turn start, never on a tool_result, to ensure the tail does not contain an
    // orphan tool_result.
    let messages = vec![
        user("turn1"),                // 0 start
        assistant("r1"),              // 1
        user("turn2"),                // turn 2 start
        assistant_tool_use("call_a"), // 3
        tool_result("call_a", "out"), // 4  <- not a turn start
        assistant("r2"),              // 5
    ];
    let boundary = select_boundary(&messages, 1_000_000).expect("boundary");
    // The only safe non-zero turn start is index 2.
    assert_eq!(boundary, 2);
    let (_head, tail) = messages.split_at(boundary);
    // The first element of `tail` is a real user turn start, not an orphaned
    // `tool_result`.
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
    let s = "héllo wörld"; // Contains multibyte characters
    let out = truncate_chars(s, 5);
    assert!(out.starts_with("héllo"));
    assert!(out.contains("truncated"));
    // Do not truncate short strings.
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
