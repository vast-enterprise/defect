use std::fs;
use std::path::PathBuf;

use agent_client_protocol_schema::{
    Content, ContentBlock, SessionId, StopReason, TextContent, ToolCallContent, ToolCallStatus,
    ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::{Message, MessageContent, Role, ToolResultBody, Usage};
use serde_json::json;
use tempfile::tempdir;

use crate::{
    RecordProjector, SessionMeta, SessionRecord, SessionStore, SnapshotState, StorageError,
    StoredRecord, StoredSnapshot,
};

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
    assert!(store.journal_path().exists());
    assert_eq!(store.load_meta().expect("load meta"), meta);
}

#[test]
fn append_record_then_replay_preserves_order() {
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

    let first = StoredRecord::new(0, SessionRecord::TurnStarted);
    let second = StoredRecord::new(
        1,
        SessionRecord::Message {
            message: Message {
                role: Role::User,
                content: vec![MessageContent::Text {
                    text: "hello".to_string(),
                }]
                .into(),
            },
        },
    );
    let third = StoredRecord::new(
        2,
        SessionRecord::Message {
            message: Message {
                role: Role::Assistant,
                content: vec![MessageContent::Text {
                    text: "world".to_string(),
                }]
                .into(),
            },
        },
    );
    let fourth = StoredRecord::new(
        3,
        SessionRecord::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
    );

    store.append_record(&first).expect("append first");
    store.append_record(&second).expect("append second");
    store.append_record(&third).expect("append third");
    store.append_record(&fourth).expect("append fourth");

    let replayed = store.replay_records().expect("replay");
    assert_eq!(replayed, vec![first, second, third, fourth]);
}

#[test]
fn replay_records_rejects_sequence_gaps() {
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
        .append_record(&StoredRecord::new(1, SessionRecord::TurnStarted))
        .expect("append record");

    let err = store
        .replay_records()
        .expect_err("should reject sequence gap");
    assert!(matches!(
        err,
        StorageError::SequenceGap {
            expected: 0,
            actual: 1
        }
    ));
}

#[test]
fn replay_records_reports_invalid_jsonl_line_number() {
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
    fs::write(store.journal_path(), b"{not json}\n").expect("write bad line");

    let err = store
        .replay_records()
        .expect_err("should reject invalid line");
    assert!(matches!(
        err,
        StorageError::InvalidEventLine { line: 1, .. }
    ));
}

#[test]
fn write_then_load_snapshot_roundtrips() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-snapshot-roundtrip");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let snapshot = StoredSnapshot::new(
        2,
        SnapshotState {
            history: vec![Message {
                role: Role::User,
                content: vec![MessageContent::Text {
                    text: "compacted".to_string(),
                }]
                .into(),
            }],
            turn_count: 1,
            last_turn_ended: true,
        },
    );

    store.write_snapshot(&snapshot).expect("write snapshot");

    assert_eq!(
        store.load_snapshot().expect("load snapshot"),
        Some(snapshot)
    );
}

#[test]
fn replay_state_uses_snapshot_and_journal_tail() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-snapshot-tail");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let pre_snapshot = [
        StoredRecord::new(
            0,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "old user".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(1, SessionRecord::TurnStarted),
    ];
    for record in pre_snapshot {
        store.append_record(&record).expect("append pre snapshot");
    }

    store
        .write_snapshot(&StoredSnapshot::new(
            2,
            SnapshotState {
                history: vec![Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "snapshot user".to_string(),
                    }]
                    .into(),
                }],
                turn_count: 1,
                last_turn_ended: false,
            },
        ))
        .expect("write snapshot");

    store
        .append_record(&StoredRecord::new(
            2,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text {
                        text: "tail assistant".to_string(),
                    }]
                    .into(),
                },
            },
        ))
        .expect("append tail message");
    store
        .append_record(&StoredRecord::new(
            3,
            SessionRecord::TurnEnded {
                reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ))
        .expect("append tail boundary");

    let replay = store.replay_state().expect("replay state");

    assert_eq!(replay.turn_count, 1);
    assert!(replay.last_turn_ended);
    assert_eq!(
        replay.history,
        vec![
            Message {
                role: Role::User,
                content: vec![MessageContent::Text {
                    text: "snapshot user".to_string(),
                }]
                .into(),
            },
            Message {
                role: Role::Assistant,
                content: vec![MessageContent::Text {
                    text: "tail assistant".to_string(),
                }]
                .into(),
            },
        ]
    );
}

#[test]
fn snapshot_tail_rejects_sequence_gap() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-snapshot-gap");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    store
        .write_snapshot(&StoredSnapshot::new(
            2,
            SnapshotState {
                history: Vec::new(),
                turn_count: 0,
                last_turn_ended: true,
            },
        ))
        .expect("write snapshot");
    store
        .append_record(&StoredRecord::new(
            3,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text {
                        text: "gap".to_string(),
                    }]
                    .into(),
                },
            },
        ))
        .expect("append gapped tail");

    let err = store
        .replay_state()
        .expect_err("should reject snapshot tail gap");

    assert!(matches!(
        err,
        StorageError::SequenceGap {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn next_record_seq_continues_after_snapshot_tail() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-next-seq");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    store
        .append_record(&StoredRecord::new(
            0,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "before".to_string(),
                    }]
                    .into(),
                },
            },
        ))
        .expect("append before snapshot");
    store
        .write_snapshot(&StoredSnapshot::new(
            1,
            SnapshotState {
                history: vec![Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "before".to_string(),
                    }]
                    .into(),
                }],
                turn_count: 0,
                last_turn_ended: true,
            },
        ))
        .expect("write snapshot");
    store
        .append_record(&StoredRecord::new(
            1,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text {
                        text: "after".to_string(),
                    }]
                    .into(),
                },
            },
        ))
        .expect("append tail");

    assert_eq!(store.next_record_seq().expect("next seq"), 2);
}

#[test]
fn next_record_seq_allows_compacted_empty_tail() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-empty-tail");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    store
        .write_snapshot(&StoredSnapshot::new(
            4,
            SnapshotState {
                history: Vec::new(),
                turn_count: 2,
                last_turn_ended: true,
            },
        ))
        .expect("write snapshot");

    assert_eq!(store.next_record_seq().expect("next seq"), 4);
    assert_eq!(store.replay_state().expect("replay").turn_count, 2);
}

#[test]
fn compact_journal_to_snapshot_removes_covered_prefix() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-compact");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let records = [
        StoredRecord::new(
            0,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "hello".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(1, SessionRecord::TurnStarted),
        StoredRecord::new(
            2,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text {
                        text: "world".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(
            3,
            SessionRecord::TurnEnded {
                reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ),
    ];
    for record in records {
        store.append_record(&record).expect("append record");
    }

    let snapshot = store.compact_journal_to_snapshot().expect("compact");

    assert_eq!(snapshot.next_seq, 4);
    assert!(
        store
            .replay_records_from(snapshot.next_seq)
            .expect("replay tail")
            .is_empty()
    );
    assert_eq!(store.next_record_seq().expect("next seq"), 4);

    let replay = store.replay_state().expect("replay state");
    assert_eq!(replay.history.len(), 2);
    assert_eq!(replay.turn_count, 1);
    assert!(replay.last_turn_ended);
}

#[test]
fn replay_state_rebuilds_user_and_assistant_history() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-5");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let records = [
        StoredRecord::new(
            0,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "hello".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(1, SessionRecord::TurnStarted),
        StoredRecord::new(
            2,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text {
                        text: "world".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(
            3,
            SessionRecord::TurnEnded {
                reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ),
    ];
    for record in records {
        store.append_record(&record).expect("append record");
    }

    let replay = store.replay_state().expect("replay state");

    assert_eq!(replay.turn_count, 1);
    assert!(replay.last_turn_ended);
    assert_eq!(replay.history.len(), 2);
    assert_eq!(replay.history[0].role, Role::User);
    assert_eq!(
        replay.history[0].content,
        Into::<std::sync::Arc<[_]>>::into(vec![MessageContent::Text {
            text: "hello".to_string()
        }])
    );
    assert_eq!(replay.history[1].role, Role::Assistant);
    assert_eq!(
        replay.history[1].content,
        Into::<std::sync::Arc<[_]>>::into(vec![MessageContent::Text {
            text: "world".to_string()
        }])
    );
}

#[test]
fn replay_state_rebuilds_tool_use_and_tool_result_history() {
    let dir = tempdir().expect("tempdir");
    let session_id = SessionId::new("sess-6");
    let store = SessionStore::for_session(dir.path(), &session_id);
    store
        .init(&SessionMeta::new(
            session_id,
            PathBuf::from("/tmp/project"),
            Vec::new(),
        ))
        .expect("init store");

    let records = [
        StoredRecord::new(
            0,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::Text {
                        text: "hello".to_string(),
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(1, SessionRecord::TurnStarted),
        StoredRecord::new(
            2,
            SessionRecord::Message {
                message: Message {
                    role: Role::Assistant,
                    content: vec![
                        MessageContent::Text {
                            text: "calling tool".to_string(),
                        },
                        MessageContent::ToolUse {
                            id: "call-1".to_string(),
                            name: "echo".to_string(),
                            args: json!({ "msg": "hi" }),
                        },
                    ]
                    .into(),
                },
            },
        ),
        StoredRecord::new(
            3,
            SessionRecord::Message {
                message: Message {
                    role: Role::User,
                    content: vec![MessageContent::ToolResult {
                        tool_use_id: "call-1".to_string(),
                        output: ToolResultBody::Text {
                            text: "hi".to_string(),
                        },
                        is_error: false,
                    }]
                    .into(),
                },
            },
        ),
        StoredRecord::new(
            4,
            SessionRecord::TurnEnded {
                reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ),
    ];
    for record in records {
        store.append_record(&record).expect("append record");
    }

    let replay = store.replay_state().expect("replay state");

    assert_eq!(replay.history.len(), 3);
    assert_eq!(replay.history[0].role, Role::User);
    assert_eq!(replay.history[1].role, Role::Assistant);
    assert_eq!(replay.history[2].role, Role::User);
    assert_eq!(
        replay.history[1].content,
        Into::<std::sync::Arc<[_]>>::into(vec![
            MessageContent::Text {
                text: "calling tool".to_string(),
            },
            MessageContent::ToolUse {
                id: "call-1".to_string(),
                name: "echo".to_string(),
                args: json!({ "msg": "hi" }),
            },
        ])
    );
    assert_eq!(
        replay.history[2].content,
        Into::<std::sync::Arc<[_]>>::into(vec![MessageContent::ToolResult {
            tool_use_id: "call-1".to_string(),
            output: ToolResultBody::Text {
                text: "hi".to_string(),
            },
            is_error: false,
        }])
    );
}

#[test]
fn projector_rebuilds_tool_loop_as_messages() {
    let mut projector = RecordProjector::default();
    let mut tool_started = ToolCallUpdateFields::default();
    tool_started.raw_input = Some(json!({ "msg": "hi" }));

    let mut tool_finished = ToolCallUpdateFields::default();
    tool_finished.status = Some(ToolCallStatus::Completed);
    tool_finished.content = Some(vec![ToolCallContent::Content(Content::new("hi"))]);

    let events = [
        AgentEvent::UserPromptCommitted {
            content: vec![ContentBlock::Text(TextContent::new("hello"))],
        },
        AgentEvent::TurnStarted,
        AgentEvent::AssistantText {
            content: ContentBlock::Text(TextContent::new("calling tool")),
        },
        AgentEvent::ToolCallStarted {
            id: "call-1".into(),
            name: "echo".to_string(),
            fields: tool_started,
        },
        AgentEvent::ToolCallFinished {
            id: "call-1".into(),
            fields: tool_finished,
        },
        AgentEvent::AssistantText {
            content: ContentBlock::Text(TextContent::new("done")),
        },
        AgentEvent::TurnEnded {
            reason: StopReason::EndTurn,
            usage: Usage::default(),
        },
    ];

    let records = events
        .into_iter()
        .flat_map(|event| projector.project(event))
        .collect::<Vec<_>>();

    let mut records = records.into_iter();
    assert!(matches!(
        records.next(),
        Some(SessionRecord::Message { .. })
    ));
    assert_eq!(records.next(), Some(SessionRecord::TurnStarted));

    let SessionRecord::Message {
        message: assistant_call,
    } = records.next().expect("assistant tool call message")
    else {
        panic!("expected assistant tool call message");
    };
    assert_eq!(assistant_call.role, Role::Assistant);
    assert_eq!(
        assistant_call.content,
        Into::<std::sync::Arc<[_]>>::into(vec![
            MessageContent::Text {
                text: "calling tool".to_string(),
            },
            MessageContent::ToolUse {
                id: "call-1".to_string(),
                name: "echo".to_string(),
                args: json!({ "msg": "hi" }),
            },
        ])
    );

    let SessionRecord::Message {
        message: tool_result,
    } = records.next().expect("tool result message")
    else {
        panic!("expected tool result message");
    };
    assert_eq!(tool_result.role, Role::User);
    let Some(MessageContent::ToolResult { tool_use_id, .. }) = tool_result.content.first() else {
        panic!("expected tool result content");
    };
    assert_eq!(tool_use_id, "call-1");

    let SessionRecord::Message {
        message: assistant_done,
    } = records.next().expect("final assistant message")
    else {
        panic!("expected final assistant message");
    };
    assert_eq!(assistant_done.role, Role::Assistant);
    assert_eq!(
        assistant_done.content,
        Into::<std::sync::Arc<[_]>>::into(vec![MessageContent::Text {
            text: "done".to_string(),
        }])
    );
    assert!(matches!(
        records.next(),
        Some(SessionRecord::TurnEnded { .. })
    ));
    assert_eq!(records.next(), None);
}

#[test]
fn projector_ignores_non_replay_events() {
    let mut projector = RecordProjector::default();
    let records = [
        AgentEvent::ToolCallProgress {
            id: "call-1".into(),
            fields: ToolCallUpdateFields::default(),
        },
        AgentEvent::LlmCallStarted {
            model: "m".to_string(),
            attempt: 1,
            request: Default::default(),
        },
        AgentEvent::LlmCallFinished {
            model: "m".to_string(),
            attempt: 1,
            usage: Usage::default(),
            error: None,
        },
        AgentEvent::ContextCompressed {
            tokens_before: 100,
            tokens_after: 10,
        },
    ]
    .into_iter()
    .flat_map(|event| projector.project(event))
    .collect::<Vec<_>>();

    assert!(records.is_empty());
}
