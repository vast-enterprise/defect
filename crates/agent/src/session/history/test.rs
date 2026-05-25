use super::*;
use crate::llm::{MessageContent, Role};

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }],
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
fn token_estimate_is_none_in_v0() {
    let h = VecHistory::new();
    assert!(h.token_estimate().is_none());
}

#[tokio::test]
async fn compact_is_noop_in_v0() {
    let h = VecHistory::new();
    let report = h.compact().await.expect("compact");
    assert_eq!(report.tokens_before, 0);
    assert_eq!(report.tokens_after, 0);
}
