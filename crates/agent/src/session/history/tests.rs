use super::*;
use crate::llm::{MessageContent, Role};

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }]
        .into(),
    }
}

#[test]
fn append_then_snapshot() {
    let h = VecHistory::new();
    h.append(user("hi"));
    h.append(user("there"));
    let snap = h.snapshot();
    assert_eq!(snap.len(), 2);
}

#[test]
fn token_estimate_none_when_empty() {
    let h = VecHistory::new();
    assert!(h.token_estimate().is_none());
}

#[test]
fn token_estimate_char_heuristic_without_baseline() {
    // No real baseline: entire snapshot falls back to chars/4.
    let h = VecHistory::new();
    h.append(user(&"a".repeat(40))); // 40 chars → 10 tokens
    assert_eq!(h.token_estimate(), Some(10));
}

#[test]
fn record_input_tokens_becomes_baseline_plus_increment() {
    let h = VecHistory::new();
    h.append(user("seed"));
    // Baseline is 1000; subsequent messages use character-based incremental accumulation.
    h.record_input_tokens(1_000);
    assert_eq!(h.token_estimate(), Some(1_000));
    h.append(user(&"b".repeat(40))); // +10 tokens
    assert_eq!(h.token_estimate(), Some(1_010));
}

#[test]
fn record_input_tokens_refreshes_baseline_and_resets_increment() {
    let h = VecHistory::new();
    h.record_input_tokens(1_000);
    h.append(user(&"b".repeat(40))); // +10
    assert_eq!(h.token_estimate(), Some(1_010));
    // A new real round-trip: baseline refreshed, increment reset to zero.
    h.record_input_tokens(2_000);
    assert_eq!(h.token_estimate(), Some(2_000));
}

#[test]
fn replace_swaps_messages_and_clears_baseline() {
    let h = VecHistory::new();
    h.append(user("old one"));
    h.append(user("old two"));
    h.record_input_tokens(5_000);
    assert_eq!(h.token_estimate(), Some(5_000));

    h.replace(vec![user(&"c".repeat(80))]); // 80 chars → 20 tokens
    let snap = h.snapshot();
    assert_eq!(snap.len(), 1);
    // Baseline cleared → full character heuristic.
    assert_eq!(h.token_estimate(), Some(20));
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

#[test]
fn splice_prefix_replaces_head_keeps_tail() {
    let h = VecHistory::new();
    h.append(user("turn one"));
    h.append(assistant("reply one"));
    h.append(user("turn two"));
    h.append(assistant("reply two"));

    // Drop the first 2 entries (turn one + reply one), replace them with a summary, and
    // keep the remaining 2.
    let dropped = h.splice_prefix(2, assistant("[summary]"));
    assert_eq!(dropped, 2);

    let snap = h.snapshot();
    assert_eq!(snap.len(), 3); // summary + 2 retained
    assert_eq!(snap[0].role, Role::Assistant);
    assert!(matches!(
        &snap[0].content[0],
        MessageContent::Text { text } if text == "[summary]"
    ));
    assert!(matches!(
        &snap[1].content[0],
        MessageContent::Text { text } if text == "turn two"
    ));
}

#[test]
fn splice_prefix_preserves_tail_appended_during_flight() {
    // Simulate background compaction: drop_count=2 was computed on the old snapshot
    // (len=2), but the frontend appended 2 tail messages before the write-back —
    // splice_prefix must preserve them.
    let h = VecHistory::new();
    h.append(user("old one"));
    h.append(assistant("old reply"));
    // Append two messages from the frontend while compaction is in flight.
    h.append(user("new one"));
    h.append(assistant("new reply"));

    let dropped = h.splice_prefix(2, assistant("[summary]"));
    assert_eq!(dropped, 2);

    let snap = h.snapshot();
    // The summary and the two new trailing messages added during that time must never be
    // lost.
    assert_eq!(snap.len(), 3);
    assert!(matches!(
        &snap[1].content[0],
        MessageContent::Text { text } if text == "new one"
    ));
    assert!(matches!(
        &snap[2].content[0],
        MessageContent::Text { text } if text == "new reply"
    ));
}

#[test]
#[should_panic(expected = "splice_prefix invariant violated")]
fn splice_prefix_overlong_drop_count_trips_invariant_in_debug() {
    // A `drop_count` exceeding the current length means someone deleted a middle message
    // mid-flight, violating the single-flight invariant. In debug builds the
    // `debug_assert` fires (this test asserts it does); in release builds `clamp` handles
    // it.
    let h = VecHistory::new();
    h.append(user("only one"));
    let _ = h.splice_prefix(99, assistant("[summary]"));
}

#[test]
fn splice_prefix_clears_baseline() {
    let h = VecHistory::new();
    h.append(user("seed one"));
    h.append(user("seed two"));
    h.record_input_tokens(5_000);
    assert_eq!(h.token_estimate(), Some(5_000));

    h.splice_prefix(1, assistant(&"c".repeat(80))); // summary: 80 chars → 20 tokens
    // Baseline cleared → full character heuristic (summary 20 + "seed two" 8 chars → 2).
    assert_eq!(h.token_estimate(), Some(22));
}
