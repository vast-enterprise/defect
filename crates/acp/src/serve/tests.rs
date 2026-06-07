use super::*;
use agent_client_protocol::schema::ErrorCode;
use defect_agent::error::BoxError;
use defect_agent::llm::{ProviderError, ProviderErrorKind};

/// `TurnError::Provider` 必须把内层 Display 灌进 wire `message`。
/// 之前的实现用 [`Wire::internal_error()`]，message 永远是字面量
/// "Internal error"——客户端 UI 拿不到任何辨识信息，acpx 显示成
/// `RUNTIME: Internal error`。
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
    // data.kind 仍然区分 provider vs internal，方便 verbose 模式排障。
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

/// Regression test for ACP filesystem delegation decisions —
/// 任一 fs 能力位 false → 整组退回本地，不混用。
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
