use agent_client_protocol_schema::ToolCallId;

use super::{ToolResult, oversized_rejection_text, reject_oversized_results};
use crate::llm::ToolResultBody;

/// Build a minimal `ToolResult` whose text body is `chars` characters long. The token
/// estimate is `chars / 4` (the shared `CHARS_PER_TOKEN` heuristic).
fn result_with_chars(chars: usize) -> ToolResult {
    ToolResult {
        id: ToolCallId::new("call-1".to_string()),
        name: "search".to_string(),
        tool_use_id: "tu-1".to_string(),
        body: ToolResultBody::Text {
            text: "x".repeat(chars),
        },
        is_error: false,
        fields: None,
        error: None,
    }
}

fn body_text(r: &ToolResult) -> &str {
    match &r.body {
        ToolResultBody::Text { text } => text,
        _ => panic!("expected text body"),
    }
}

#[test]
fn rejects_result_exceeding_context_window() {
    // 4000 chars ≈ 1000 tokens, well over a 100-token window.
    let mut results = vec![result_with_chars(4000)];
    let rejected = reject_oversized_results(&mut results, Some(100));
    assert_eq!(rejected, 1);
    assert!(
        results[0].is_error,
        "oversized result must be flagged as error"
    );
    assert!(
        body_text(&results[0]).contains("exceeds the model context window"),
        "body should be replaced with the actionable rejection message"
    );
}

#[test]
fn keeps_result_within_context_window() {
    // 40 chars ≈ 10 tokens, under a 100-token window.
    let mut results = vec![result_with_chars(40)];
    let rejected = reject_oversized_results(&mut results, Some(100));
    assert_eq!(rejected, 0);
    assert!(!results[0].is_error);
    assert_eq!(
        body_text(&results[0]).len(),
        40,
        "body must be left untouched"
    );
}

#[test]
fn no_window_means_no_ceiling() {
    // Even a huge result is left alone when the context window is unknown.
    let mut results = vec![result_with_chars(40_000)];
    let rejected = reject_oversized_results(&mut results, None);
    assert_eq!(rejected, 0);
    assert!(!results[0].is_error);
}

#[test]
fn rejects_only_the_oversized_ones_in_a_batch() {
    let mut results = vec![
        result_with_chars(40),   // ~10 tokens, fits
        result_with_chars(4000), // ~1000 tokens, rejected
        result_with_chars(80),   // ~20 tokens, fits
    ];
    let rejected = reject_oversized_results(&mut results, Some(100));
    assert_eq!(rejected, 1);
    assert!(!results[0].is_error);
    assert!(results[1].is_error);
    assert!(!results[2].is_error);
}

#[test]
fn rejection_message_names_the_numbers() {
    let msg = oversized_rejection_text(1234, 100);
    assert!(msg.contains("1234"));
    assert!(msg.contains("100"));
}
