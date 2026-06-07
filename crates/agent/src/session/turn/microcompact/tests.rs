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

/// A `tool_result` message (user role, contains only a `ToolResult`, not a turn start).
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

/// Construct N complete turns: each turn = [user, assistant(tool_use),
/// tool_result(large)].
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
    // Only KEEP_RECENT_TURNS turns (or fewer) → no older turns to clear → None.
    let msgs = turns(KEEP_RECENT_TURNS, 4_000);
    assert!(run(&msgs).is_none());
}

#[test]
fn clears_oversized_results_in_old_turns_only() {
    // KEEP_RECENT_TURNS + 2 turns, each with a 4000-char tool_result → 1000 tokens >
    // floor.
    let n = KEEP_RECENT_TURNS + 2;
    let msgs = turns(n, 4_000);
    let (rebuilt, report) = run(&msgs).expect("should clear old oversized results");

    // Only clear the tool_result of the oldest 2 turns.
    assert_eq!(report.cleared, 2);
    assert!(report.tokens_after < report.tokens_before);
    assert_eq!(rebuilt.len(), msgs.len()); // does not add or remove messages

    // The oldest 2 turns' tool_result are already placeholders.
    for turn_idx in 0..2 {
        let tr = &rebuilt[turn_idx * 3 + 2];
        let MessageContent::ToolResult { output, .. } = &tr.content[0] else {
            panic!("expected tool_result");
        };
        assert!(matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER));
    }
    // Keep the tool_result within the window (the last KEEP_RECENT_TURNS turns)
    // unchanged.
    let kept = &rebuilt[(n - 1) * 3 + 2];
    let MessageContent::ToolResult { output, .. } = &kept.content[0] else {
        panic!("expected tool_result");
    };
    assert!(matches!(output, ToolResultBody::Text { text } if text.len() == 4_000));
}

#[test]
fn respects_size_floor() {
    // Old turn's tool_result is tiny (40 chars → 10 tokens < floor) → not cleared → None.
    let msgs = turns(KEEP_RECENT_TURNS + 2, 40);
    assert!(run(&msgs).is_none());
}

#[test]
fn idempotent() {
    let msgs = turns(KEEP_RECENT_TURNS + 2, 4_000);
    let (once, r1) = run(&msgs).expect("first pass clears");
    assert_eq!(r1.cleared, 2);
    // Second pass: placeholders are sentinel; already-cleared entries are not cleared
    // again → None.
    assert!(run(&once).is_none());
}

#[test]
fn preserves_tool_use_id_and_is_error() {
    let mut msgs = turns(KEEP_RECENT_TURNS + 1, 4_000);
    // Mark the oldest turn's tool_result as an error.
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
    assert_eq!(tool_use_id, "t0"); // ensures the paired id matches
    assert!(*is_error); // error flag is preserved
    assert!(matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER));
}
