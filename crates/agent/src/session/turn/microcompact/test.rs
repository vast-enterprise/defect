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

fn assistant_tool_use(id: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![MessageContent::ToolUse {
            id: id.to_string(),
            name: "read_file".to_string(),
            args: serde_json::json!({}),
        }]
        .into(),
    }
}

/// 一条 tool_result 消息（user role，仅含 ToolResult，非轮次起点）。
fn tool_result(id: &str, body_chars: usize) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::ToolResult {
            tool_use_id: id.to_string(),
            output: ToolResultBody::Text {
                text: "x".repeat(body_chars),
            },
            is_error: false,
        }]
        .into(),
    }
}

/// 构造 N 个完整轮次：每轮 = [user, assistant(tool_use), tool_result(大)]。
fn turns(n: usize, body_chars: usize) -> Vec<Message> {
    let mut v = Vec::new();
    for i in 0..n {
        v.push(user(&format!("turn {i}")));
        v.push(assistant_tool_use(&format!("t{i}")));
        v.push(tool_result(&format!("t{i}"), body_chars));
    }
    v
}

#[test]
fn skips_when_too_few_turns() {
    // 仅 KEEP_RECENT_TURNS 个轮次（或更少）→ 无更老轮次可清 → None。
    let msgs = turns(KEEP_RECENT_TURNS, 4_000);
    assert!(run(&msgs).is_none());
}

#[test]
fn clears_oversized_results_in_old_turns_only() {
    // KEEP_RECENT_TURNS + 2 个轮次，每个 tool_result 4000 chars → 1000 token > 地板。
    let n = KEEP_RECENT_TURNS + 2;
    let msgs = turns(n, 4_000);
    let (rebuilt, report) = run(&msgs).expect("should clear old oversized results");

    // 只清最老的 2 个轮次的 tool_result。
    assert_eq!(report.cleared, 2);
    assert!(report.tokens_after < report.tokens_before);
    assert_eq!(rebuilt.len(), msgs.len()); // 不增删消息

    // 最老 2 轮的 tool_result 已是占位符。
    for turn_idx in 0..2 {
        let tr = &rebuilt[turn_idx * 3 + 2];
        let MessageContent::ToolResult { output, .. } = &tr.content[0] else {
            panic!("expected tool_result");
        };
        assert!(matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER));
    }
    // 保留窗口内（最后 KEEP_RECENT_TURNS 轮）的 tool_result 原样。
    let kept = &rebuilt[(n - 1) * 3 + 2];
    let MessageContent::ToolResult { output, .. } = &kept.content[0] else {
        panic!("expected tool_result");
    };
    assert!(matches!(output, ToolResultBody::Text { text } if text.len() == 4_000));
}

#[test]
fn respects_size_floor() {
    // 老轮次的 tool_result 很小（40 chars → 10 token < 地板）→ 不清 → None。
    let msgs = turns(KEEP_RECENT_TURNS + 2, 40);
    assert!(run(&msgs).is_none());
}

#[test]
fn idempotent() {
    let msgs = turns(KEEP_RECENT_TURNS + 2, 4_000);
    let (once, r1) = run(&msgs).expect("first pass clears");
    assert_eq!(r1.cleared, 2);
    // 第二遍：占位符是 sentinel，已清的不再清 → None。
    assert!(run(&once).is_none());
}

#[test]
fn preserves_tool_use_id_and_is_error() {
    let mut msgs = turns(KEEP_RECENT_TURNS + 1, 4_000);
    // 把最老轮次的 tool_result 标成 error。
    if let MessageContent::ToolResult {
        tool_use_id,
        is_error,
        ..
    } = &mut std::sync::Arc::make_mut(&mut msgs[2].content)[0]
    {
        *is_error = true;
        assert_eq!(tool_use_id, "t0");
    }

    let (rebuilt, _) = run(&msgs).expect("clears");
    let MessageContent::ToolResult {
        tool_use_id,
        is_error,
        output,
    } = &rebuilt[2].content[0]
    else {
        panic!("expected tool_result");
    };
    assert_eq!(tool_use_id, "t0"); // 配对 id 保住
    assert!(*is_error); // error 标记保住
    assert!(matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER));
}
