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
    // 无真实基线：整份 snapshot 走 chars/4 兜底。
    let h = VecHistory::new();
    h.append(user(&"a".repeat(40))); // 40 chars → 10 token
    assert_eq!(h.token_estimate(), Some(10));
}

#[test]
fn record_input_tokens_becomes_baseline_plus_increment() {
    let h = VecHistory::new();
    h.append(user("seed"));
    // 真实基线 1000；其后追加的消息走字符增量叠加。
    h.record_input_tokens(1_000);
    assert_eq!(h.token_estimate(), Some(1_000));
    h.append(user(&"b".repeat(40))); // +10 token
    assert_eq!(h.token_estimate(), Some(1_010));
}

#[test]
fn record_input_tokens_refreshes_baseline_and_resets_increment() {
    let h = VecHistory::new();
    h.record_input_tokens(1_000);
    h.append(user(&"b".repeat(40))); // +10
    assert_eq!(h.token_estimate(), Some(1_010));
    // 新一轮真实回报：基线刷新、增量归零。
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

    h.replace(vec![user(&"c".repeat(80))]); // 80 chars → 20 token
    let snap = h.snapshot();
    assert_eq!(snap.len(), 1);
    // 基线清空 → 整份字符启发式。
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

    // 丢掉前 2 条（turn one + reply one），换成摘要，保留后 2 条。
    let dropped = h.splice_prefix(2, assistant("[summary]"));
    assert_eq!(dropped, 2);

    let snap = h.snapshot();
    assert_eq!(snap.len(), 3); // summary + 保留的 2 条
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
    // 模拟后台压缩：在旧 snapshot（len=2）上算出 drop_count=2，
    // 但回写前前台又 append 了 2 条尾部消息——splice_prefix 必须保住它们。
    let h = VecHistory::new();
    h.append(user("old one"));
    h.append(assistant("old reply"));
    // 飞行期间前台尾插。
    h.append(user("new one"));
    h.append(assistant("new reply"));

    let dropped = h.splice_prefix(2, assistant("[summary]"));
    assert_eq!(dropped, 2);

    let snap = h.snapshot();
    // summary + 期间新增的 2 条尾部，绝不能丢。
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
    // drop_count 超过当前长度意味着飞行期间有人删了中段消息——违反 single-flight
    // 不变式。debug 下 debug_assert 炸出来（本测试断言它确实炸）；release 下 clamp 兜底。
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

    h.splice_prefix(1, assistant(&"c".repeat(80))); // summary 80 chars → 20
    // 基线清空 → 整份字符启发式（summary 20 + "seed two" 8 chars→2）。
    assert_eq!(h.token_estimate(), Some(22));
}
