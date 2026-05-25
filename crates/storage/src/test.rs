use std::fs;
use std::path::PathBuf;

use agent_client_protocol::schema::{ContentBlock, SessionId, StopReason, TextContent};
use defect_agent::event::AgentEvent;
use defect_agent::llm::Usage;
use tempfile::tempdir;

use crate::{SessionMeta, SessionStore, StorageError, StoredEvent};

fn user_text_event(text: &str) -> AgentEvent {
    AgentEvent::AssistantText {
        content: ContentBlock::Text(TextContent::new(text)),
    }
}

#[test]
fn init_creates_meta_and_event_files() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-1");
    let store = SessionStore::for_session(dir.path(), &session_id);
    let meta = SessionMeta::new(
        session_id.clone(),
        PathBuf::from("/tmp/project"),
        Vec::new(),
    );

    store.init(&meta).expect("init store");

    assert!(store.meta_path().exists());
    assert!(store.events_path().exists());
    assert_eq!(store.load_meta().expect("load meta"), meta);
}

#[test]
fn append_then_replay_preserves_order() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-2");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let first = StoredEvent::new(0, AgentEvent::TurnStarted);
    let second = StoredEvent::new(1, user_text_event("hello"));
    let third = StoredEvent::new(
        2,
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
    );

    store.append_event(&first).expect("append first");
    store.append_event(&second).expect("append second");
    store.append_event(&third).expect("append third");

    let replayed = store.replay().expect("replay");
    assert_eq!(replayed, vec![first, second, third]);
}

#[test]
fn replay_rejects_sequence_gaps() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-3");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    store
        .append_event(&StoredEvent::new(1, AgentEvent::TurnStarted))
        .expect("append event");

    let err = store.replay().expect_err("should reject sequence gap");
    assert!(matches!(
        err,
        StorageError::SequenceGap {
            expected: 0,
            actual: 1
        }
    ));
}

#[test]
fn replay_reports_invalid_jsonl_line_number() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-4");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");
    fs::write(store.events_path(), b"{not json}\n").expect("write bad line");

    let err = store.replay().expect_err("should reject invalid line");
    assert!(matches!(
        err,
        StorageError::InvalidEventLine { line: 1, .. }
    ));
}
