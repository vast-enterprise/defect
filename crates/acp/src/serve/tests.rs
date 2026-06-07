use super::*;
use agent_client_protocol::schema::ErrorCode;
use defect_agent::error::BoxError;
use defect_agent::llm::{ProviderError, ProviderErrorKind};

/// `TurnError::Provider` must propagate the inner `Display` into the wire `message`.
/// The previous implementation used [`Wire::internal_error()`], so the message was always
/// the literal
/// "Internal error" — the client UI received no identifying information, and acpx
/// displayed
/// `RUNTIME: Internal error`.
#[test]
fn turn_provider_error_carries_message_on_wire() {
    let provider_err = ProviderError::new(ProviderErrorKind::ModelNotFound {
        model: "deepseek-v4-pro".into(),
    });
    let acp_err = AcpError::Turn(TurnError::Provider(provider_err));
    let wire = acp_err.into_wire_error();

    assert_eq!(wire.code, ErrorCode::InternalError);
    assert!(
        wire.message.contains("model not found") && wire.message.contains("deepseek-v4-pro"),
        "expected provider Display text in wire message, got: {:?}",
        wire.message
    );
    // data.kind still distinguishes provider from internal, aiding verbose-mode
    // debugging.
    let data = wire.data.expect("wire data");
    assert_eq!(data.get("kind").and_then(|v| v.as_str()), Some("provider"));
}

#[test]
fn turn_internal_error_carries_message_on_wire() {
    let acp_err = AcpError::Turn(TurnError::Internal(BoxError::new(std::io::Error::other(
        "history backend exploded",
    ))));
    let wire = acp_err.into_wire_error();

    assert_eq!(wire.code, ErrorCode::InternalError);
    assert!(
        wire.message.contains("history backend exploded"),
        "expected inner io Display in wire message, got: {:?}",
        wire.message
    );
}

#[test]
fn turn_in_progress_uses_invalid_request_code() {
    let acp_err = AcpError::Turn(TurnError::TurnInProgress);
    let wire = acp_err.into_wire_error();
    assert_eq!(wire.code, ErrorCode::InvalidRequest);
    assert!(wire.message.contains("turn already in progress"));
}

use agent_client_protocol::schema::FileSystemCapabilities;

/// Regression test for ACP filesystem delegation decisions — if any fs capability bit is
/// false, the entire group falls back to local, no mixing.
#[test]
fn decide_fs_mode_full_caps_is_delegated() {
    let caps = ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(true)
        .write_text_file(true));
    assert_eq!(decide_fs_mode(&caps), FsMode::Delegated);
}

#[test]
fn decide_fs_mode_read_only_falls_back_to_local() {
    let caps = ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(true)
        .write_text_file(false));
    assert_eq!(decide_fs_mode(&caps), FsMode::Local);
}

#[test]
fn decide_fs_mode_write_only_falls_back_to_local() {
    let caps = ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(false)
        .write_text_file(true));
    assert_eq!(decide_fs_mode(&caps), FsMode::Local);
}

#[test]
fn decide_fs_mode_default_caps_is_local() {
    let caps = ClientCapabilities::new();
    assert_eq!(decide_fs_mode(&caps), FsMode::Local);
}

fn text_prompt(s: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text(TextContent::new(s.to_string()))]
}

#[test]
fn parse_slash_command_recognizes_bare_command() {
    let (name, args) = parse_slash_command(&text_prompt("/compact")).expect("command");
    assert_eq!(name, "compact");
    assert_eq!(args, "");
}

#[test]
fn parse_slash_command_splits_args_and_trims() {
    let (name, args) =
        parse_slash_command(&text_prompt("/context   extra stuff ")).expect("command");
    assert_eq!(name, "context");
    assert_eq!(args, "extra stuff");
}

#[test]
fn parse_slash_command_tolerates_leading_whitespace() {
    let (name, _) = parse_slash_command(&text_prompt("   /compact")).expect("command");
    assert_eq!(name, "compact");
}

#[test]
fn parse_slash_command_ignores_plain_text() {
    assert!(parse_slash_command(&text_prompt("hello there")).is_none());
    // A slash mid-sentence is not a command — only a leading slash counts.
    assert!(parse_slash_command(&text_prompt("what about a/b testing")).is_none());
}

#[test]
fn parse_slash_command_ignores_empty_prompt() {
    assert!(parse_slash_command(&[]).is_none());
}

#[test]
fn fmt_tokens_humanizes_thousands() {
    assert_eq!(fmt_tokens(999), "999");
    assert_eq!(fmt_tokens(1000), "1.0k");
    assert_eq!(fmt_tokens(12_345), "12.3k");
}
