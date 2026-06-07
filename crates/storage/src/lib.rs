//! Session persistence.
//!
//! Currently writes session recovery logs as JSONL on disk, supporting append and
//! replay; later evolves on demand to snapshot + sqlite or other indexed storage.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_client_protocol_schema::{
    McpServer, SessionId, StopReason as AcpStopReason, ToolCallContent, ToolCallStatus,
    ToolCallUpdateFields,
};
use defect_agent::error::BoxError;
use defect_agent::event::AgentEvent;
use defect_agent::llm::{Message, MessageContent, Role, ToolResultBody, Usage};
use defect_agent::session::{
    LoadedSession, Session, SessionCreateInfo, SessionLoader, SessionObserver,
};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

const META_FILENAME: &str = "meta.json";
const JOURNAL_FILENAME: &str = "journal.jsonl";
const SNAPSHOT_FILENAME: &str = "snapshot.json";
const STORAGE_SCHEMA_VERSION: u32 = 1;

/// Observer for session persistence after creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageObserver {
    sessions_root: PathBuf,
}

impl StorageObserver {
    #[must_use]
    pub fn new(sessions_root: PathBuf) -> Self {
        Self { sessions_root }
    }

    #[must_use]
    pub fn sessions_root(&self) -> &Path {
        &self.sessions_root
    }

    /// Find the most recently active session id under `cwd` (used when `--resume` is
    /// given without an id).
    ///
    /// Scans each session directory under `sessions_root`, reads `meta.json` to get the
    /// `cwd`, and among candidates matching the given `cwd` (canonicalized, byte-for-byte
    /// comparison) picks the one with the latest `journal.jsonl` modification time.
    /// Returns `Ok(None)` if no match is found.
    ///
    /// If a single session directory is corrupted (meta unreadable or parse failure), it
    /// is skipped rather than failing the entire operation — a bad archive should not
    /// make resume completely unusable.
    ///
    /// # Errors
    ///
    /// Returns an error if `sessions_root` exists but cannot be enumerated; if the
    /// directory does not exist, it is treated as "no candidates" (`Ok(None)`).
    pub fn latest_session_id_for_cwd(&self, cwd: &Path) -> Result<Option<SessionId>, StorageError> {
        let target = fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
        let entries = match fs::read_dir(&self.sessions_root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(StorageError::Io(err)),
        };

        let mut best: Option<(std::time::SystemTime, SessionId)> = None;
        for entry in entries {
            let entry = entry?;
            if !entry.path().is_dir() {
                continue;
            }
            let store = SessionStore::new(entry.path());
            let Ok(meta) = store.load_meta() else {
                continue; // Corrupted or missing meta; skip
            };
            let meta_cwd = fs::canonicalize(&meta.cwd).unwrap_or_else(|_| meta.cwd.clone());
            if meta_cwd != target {
                continue;
            }
            // Activity is ordered by the journal's last modification time, falling back
            // to UNIX_EPOCH if the time cannot be obtained.
            let mtime = fs::metadata(store.journal_path())
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best
                .as_ref()
                .is_none_or(|(best_mtime, _)| mtime >= *best_mtime)
            {
                best = Some((mtime, meta.session_id));
            }
        }
        Ok(best.map(|(_, id)| id))
    }
}

impl SessionObserver for StorageObserver {
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError> {
        let store = SessionStore::for_session(&self.sessions_root, &info.id);
        let meta = SessionMeta::new(info.id.clone(), info.cwd, info.mcp_servers);
        store.init(&meta).map_err(BoxError::new)?;
        let start_seq = store.next_record_seq().map_err(BoxError::new)?;
        let replay = store.replay_state().map_err(BoxError::new)?;

        let session_id = info.id.clone();
        tokio::spawn(async move {
            let mut events = session.subscribe();
            let mut seq = start_seq;
            let mut state = replay;
            let mut projector = RecordProjector::default();
            while let Some(event) = events.next().await {
                for projected in projector.project(event) {
                    let record = StoredRecord::new(seq, projected.clone());
                    if let Err(err) = store.append_record(&record) {
                        tracing::warn!(
                            session_id = %session_id,
                            error = %err,
                            "failed to append session journal record"
                        );
                        return;
                    }
                    state.apply_record(projected);
                    seq = seq.saturating_add(1);
                    if let SessionRecord::TurnEnded { .. } = record.record {
                        let snapshot = StoredSnapshot::new(seq, state.to_snapshot_state());
                        if let Err(err) = store.write_snapshot(&snapshot) {
                            tracing::warn!(
                                session_id = %session_id,
                                error = %err,
                                "failed to write session snapshot"
                            );
                        }
                    }
                }
            }
        });

        Ok(())
    }
}

impl SessionLoader for StorageObserver {
    fn load_session(&self, id: SessionId) -> BoxFuture<'_, Result<LoadedSession, BoxError>> {
        Box::pin(async move {
            let store = SessionStore::for_session(&self.sessions_root, &id);
            if !store.meta_path().exists() {
                return Err(BoxError::new(StorageError::SessionNotFound(id)));
            }
            let meta = store.load_meta().map_err(BoxError::new)?;
            let replay = store.replay_state().map_err(BoxError::new)?;
            Ok(LoadedSession {
                info: SessionCreateInfo {
                    id: meta.session_id,
                    cwd: meta.cwd,
                    mcp_servers: meta.mcp_servers,
                },
                history: replay.history,
            })
        })
    }
}

/// Directory on disk for a single session.
///
/// The source of truth is `meta.json` + `journal.jsonl`; `snapshot.json` is kept as a
/// future optimization for faster resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    /// Binds a session directory.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Creates a store under `sessions_root/<session_id>/`.
    #[must_use]
    pub fn for_session(sessions_root: impl AsRef<Path>, session_id: &SessionId) -> Self {
        Self::new(sessions_root.as_ref().join(session_id.0.as_ref()))
    }

    /// Returns the session directory path.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to `meta.json`.
    #[must_use]
    pub fn meta_path(&self) -> PathBuf {
        self.root.join(META_FILENAME)
    }

    /// Path to `journal.jsonl`.
    #[must_use]
    pub fn journal_path(&self) -> PathBuf {
        self.root.join(JOURNAL_FILENAME)
    }

    /// Path to `snapshot.json`.
    #[must_use]
    pub fn snapshot_path(&self) -> PathBuf {
        self.root.join(SNAPSHOT_FILENAME)
    }

    /// Writes the current recovery state to `snapshot.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation, serialization, temporary file writing, or
    /// atomic replacement fails.
    pub fn write_snapshot(&self, snapshot: &StoredSnapshot) -> Result<(), StorageError> {
        ensure_supported_schema(snapshot.schema_version)?;
        fs::create_dir_all(&self.root)?;

        let mut temp_file = NamedTempFile::new_in(&self.root)?;
        let encoded = serde_json::to_string_pretty(snapshot)?;
        temp_file.as_file_mut().write_all(encoded.as_bytes())?;
        temp_file.as_file_mut().write_all(b"\n")?;
        temp_file.as_file_mut().flush()?;
        temp_file
            .persist(self.snapshot_path())
            .map_err(|err| err.error)?;
        Ok(())
    }

    /// Initializes the session directory and metadata file.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation, serialization, or file writing fails.
    pub fn init(&self, meta: &SessionMeta) -> Result<(), StorageError> {
        fs::create_dir_all(&self.root)?;
        let encoded = serde_json::to_vec_pretty(meta)?;
        fs::write(self.meta_path(), encoded)?;
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_path())?;
        Ok(())
    }

    /// Appends a recovery journal record.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory does not exist, the file cannot be opened,
    /// serialization fails, or writing/flushing fails.
    pub fn append_record(&self, record: &StoredRecord) -> Result<(), StorageError> {
        ensure_supported_schema(record.schema_version)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_path())?;
        let encoded = serde_json::to_string(record)?;
        file.write_all(encoded.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }

    /// Reads `meta.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, the content is not valid JSON, or the
    /// schema is unsupported.
    pub fn load_meta(&self) -> Result<SessionMeta, StorageError> {
        let bytes = fs::read(self.meta_path())?;
        let meta = serde_json::from_slice::<SessionMeta>(&bytes)?;
        ensure_supported_schema(meta.schema_version)?;
        Ok(meta)
    }

    /// Reads `snapshot.json`, returning `Ok(None)` if the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but its contents are invalid, or if the schema
    /// is unsupported.
    pub fn load_snapshot(&self) -> Result<Option<StoredSnapshot>, StorageError> {
        let path = self.snapshot_path();
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(StorageError::Io(err)),
        };
        let snapshot = serde_json::from_slice::<StoredSnapshot>(&bytes)?;
        ensure_supported_schema(snapshot.schema_version)?;
        Ok(Some(snapshot))
    }

    /// Reads the available journal records sequentially.
    ///
    /// If a `snapshot.json` exists, reads only the tail after the snapshot; otherwise
    /// reads from 0.
    /// If a crash leaves a partial line at the end of the file, currently returns
    /// an error directly; this may later evolve to automatic truncation.
    ///
    /// # Errors
    ///
    /// File not found, line parsing failure, unsupported schema, or non-contiguous record
    /// sequence numbers.
    pub fn replay_records(&self) -> Result<Vec<StoredRecord>, StorageError> {
        let start_seq = self
            .load_snapshot()?
            .map_or(0, |snapshot| snapshot.next_seq);
        self.replay_records_from(start_seq)
    }

    /// Replays the recovery journal tail sequentially, starting from the given sequence
    /// number.
    ///
    /// # Errors
    ///
    /// File not found, line parsing failure, unsupported schema, or non‑contiguous record
    /// sequence numbers.
    pub fn replay_records_from(&self, start_seq: u64) -> Result<Vec<StoredRecord>, StorageError> {
        let file = File::open(self.journal_path())?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut expected_seq = start_seq;

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str::<StoredRecord>(&line).map_err(|source| {
                StorageError::InvalidEventLine {
                    line: line_no + 1,
                    source,
                }
            })?;
            ensure_supported_schema(record.schema_version)?;
            if record.seq < start_seq {
                continue;
            }
            if record.seq != expected_seq {
                return Err(StorageError::SequenceGap {
                    expected: expected_seq,
                    actual: record.seq,
                });
            }
            expected_seq = expected_seq.saturating_add(1);
            records.push(record);
        }

        Ok(records)
    }

    /// Computes the next sequence number to use for a recovery journal entry.
    ///
    /// If a `snapshot.json` exists, only validates and scans the journal tail after the
    /// snapshot;
    /// otherwise requires the journal to be contiguous from 0.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, line parsing fails, the schema is
    /// unsupported, or the tail sequence numbers are not contiguous.
    pub fn next_record_seq(&self) -> Result<u64, StorageError> {
        let start_seq = self
            .load_snapshot()?
            .map_or(0, |snapshot| snapshot.next_seq);
        let records = self.replay_records_from(start_seq)?;
        let Some(last) = records.last() else {
            return Ok(start_seq);
        };
        Ok(last.seq.saturating_add(1))
    }

    /// Write the current recovery state as a snapshot.
    ///
    /// # Errors
    ///
    /// Replaying the state, reading the next sequence number, or writing the snapshot
    /// fails.
    pub fn refresh_snapshot(&self) -> Result<StoredSnapshot, StorageError> {
        let state = self.replay_state()?;
        let snapshot = StoredSnapshot::new(self.next_record_seq()?, state.to_snapshot_state());
        self.write_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    /// Overwrite the journal prefix already covered by the latest snapshot.
    ///
    /// This method must not be called concurrently with event writes for the same
    /// session; the caller must explicitly trigger compaction when the session is
    /// idle.
    ///
    /// # Errors
    ///
    /// Replay failure, snapshot write failure, journal tail read failure, or journal
    /// rewrite failure.
    pub fn compact_journal_to_snapshot(&self) -> Result<StoredSnapshot, StorageError> {
        let snapshot = self.refresh_snapshot()?;
        let tail = self.replay_records_from(snapshot.next_seq)?;
        self.write_journal_records(&tail)?;
        Ok(snapshot)
    }

    /// Replays the recovery journal and folds it into a recoverable historical state.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying replay fails, or if the record sequence cannot
    /// be folded into a semantically consistent history.
    pub fn replay_state(&self) -> Result<ReplayState, StorageError> {
        let Some(snapshot) = self.load_snapshot()? else {
            let mut state = ReplayState::default();
            for record in self.replay_records()? {
                state.apply_record(record.record);
            }
            return Ok(state);
        };

        let mut state: ReplayState = snapshot.state.into();
        for record in self.replay_records_from(snapshot.next_seq)? {
            state.apply_record(record.record);
        }
        Ok(state)
    }

    fn write_journal_records(&self, records: &[StoredRecord]) -> Result<(), StorageError> {
        fs::create_dir_all(&self.root)?;

        let mut temp_file = NamedTempFile::new_in(&self.root)?;
        for record in records {
            ensure_supported_schema(record.schema_version)?;
            let encoded = serde_json::to_string(record)?;
            temp_file.as_file_mut().write_all(encoded.as_bytes())?;
            temp_file.as_file_mut().write_all(b"\n")?;
        }
        temp_file.as_file_mut().flush()?;
        temp_file
            .persist(self.journal_path())
            .map_err(|err| err.error)?;
        Ok(())
    }
}

/// Stable metadata for `meta.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub cwd: PathBuf,
    pub mcp_servers: Vec<McpServer>,
}

impl SessionMeta {
    /// Constructs metadata.
    #[must_use]
    pub fn new(session_id: SessionId, cwd: PathBuf, mcp_servers: Vec<McpServer>) -> Self {
        Self {
            schema_version: STORAGE_SCHEMA_VERSION,
            session_id,
            cwd,
            mcp_servers,
        }
    }
}

/// A single recovery record in `journal.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRecord {
    pub schema_version: u32,
    pub seq: u64,
    pub record: SessionRecord,
}

/// Persistent snapshot of `snapshot.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredSnapshot {
    pub schema_version: u32,
    pub next_seq: u64,
    pub state: SnapshotState,
}

/// Snapshot-able recovery state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SnapshotState {
    pub history: Vec<Message>,
    pub turn_count: u64,
    pub last_turn_ended: bool,
}

/// Records for session recovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionRecord {
    TurnStarted,
    TurnEnded {
        reason: AcpStopReason,
        usage: Usage,
    },
    Message {
        message: Message,
    },
    Snapshot {
        history: Vec<Message>,
        turn_count: u64,
        last_turn_ended: bool,
    },
}

/// Session recovery state replayed from the event stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayState {
    pub history: Vec<Message>,
    pub turn_count: u64,
    pub last_turn_ended: bool,
}

impl ReplayState {
    fn apply_record(&mut self, record: SessionRecord) {
        match record {
            SessionRecord::TurnStarted => {
                self.turn_count = self.turn_count.saturating_add(1);
                self.last_turn_ended = false;
            }
            SessionRecord::TurnEnded { .. } => {
                self.last_turn_ended = true;
            }
            SessionRecord::Message { message } => {
                self.history.push(message);
            }
            SessionRecord::Snapshot {
                history,
                turn_count,
                last_turn_ended,
            } => {
                self.history = history;
                self.turn_count = turn_count;
                self.last_turn_ended = last_turn_ended;
            }
        }
    }

    /// Project the current replay state into a persistent snapshot.
    #[must_use]
    pub fn to_snapshot_state(&self) -> SnapshotState {
        self.into()
    }
}

impl From<SnapshotState> for ReplayState {
    fn from(state: SnapshotState) -> Self {
        Self {
            history: state.history,
            turn_count: state.turn_count,
            last_turn_ended: state.last_turn_ended,
        }
    }
}

impl From<&ReplayState> for SnapshotState {
    fn from(state: &ReplayState) -> Self {
        Self {
            history: state.history.clone(),
            turn_count: state.turn_count,
            last_turn_ended: state.last_turn_ended,
        }
    }
}

impl StoredSnapshot {
    /// Constructs a persistent snapshot using the current schema.
    #[must_use]
    pub fn new(next_seq: u64, state: SnapshotState) -> Self {
        Self {
            schema_version: STORAGE_SCHEMA_VERSION,
            next_seq,
            state,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct AssistantReplay {
    text: String,
    thinking_text: String,
    thinking_signature: Option<String>,
    tool_uses: Vec<MessageContent>,
}

#[derive(Debug, Default)]
struct RecordProjector {
    current_assistant: Option<AssistantReplay>,
    pending_tool_results: Vec<MessageContent>,
}

impl RecordProjector {
    fn project(&mut self, event: AgentEvent) -> Vec<SessionRecord> {
        let mut records = Vec::new();
        match event {
            AgentEvent::UserPromptCommitted { content } => {
                append_if_some(&mut records, self.flush_assistant());
                append_if_some(&mut records, self.flush_tool_results());
                records.push(SessionRecord::Message {
                    message: Message {
                        role: Role::User,
                        content: content
                            .into_iter()
                            .map(content_block_to_message_content)
                            .collect(),
                    },
                });
            }
            AgentEvent::AssistantText { content } => {
                append_if_some(&mut records, self.flush_tool_results());
                if let agent_client_protocol_schema::ContentBlock::Text(text) = content {
                    self.current_assistant().text.push_str(text.text.as_str());
                }
            }
            AgentEvent::AssistantThought { content } => {
                append_if_some(&mut records, self.flush_tool_results());
                if let agent_client_protocol_schema::ContentBlock::Text(text) = content {
                    self.current_assistant()
                        .thinking_text
                        .push_str(text.text.as_str());
                }
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                append_if_some(&mut records, self.flush_tool_results());
                // A `ToolCallStarted` with an empty `name` is a failure marker event (a
                // wire signal for tool-not-found / denied, etc.), not a real tool call.
                // Persisting it as a `ToolUse` would leave `name` empty, which the
                // provider rejects on resume because `tool_use.name` must be at least 1
                // character. Such events are excluded from history.
                if !name.is_empty() {
                    self.current_assistant()
                        .tool_uses
                        .push(MessageContent::ToolUse {
                            id: id.to_string(),
                            name,
                            args: fields
                                .raw_input
                                .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
                        });
                }
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                append_if_some(&mut records, self.flush_assistant());
                let status = fields.status.unwrap_or(ToolCallStatus::Completed);
                let output = tool_call_output(&fields);
                self.pending_tool_results.push(MessageContent::ToolResult {
                    tool_use_id: id.to_string(),
                    output,
                    is_error: status == ToolCallStatus::Failed,
                });
            }
            AgentEvent::TurnStarted => {
                append_if_some(&mut records, self.flush_assistant());
                append_if_some(&mut records, self.flush_tool_results());
                records.push(SessionRecord::TurnStarted);
            }
            AgentEvent::TurnEnded { reason, usage } => {
                append_if_some(&mut records, self.flush_assistant());
                append_if_some(&mut records, self.flush_tool_results());
                records.push(SessionRecord::TurnEnded { reason, usage });
            }
            AgentEvent::ToolCallProgress { .. }
            | AgentEvent::PolicyDecision { .. }
            | AgentEvent::PermissionResolved { .. }
            | AgentEvent::LlmCallStarted { .. }
            | AgentEvent::LlmCallFinished { .. }
            | AgentEvent::ContextCompressed { .. } => {}
            _ => {}
        }
        records
    }

    fn current_assistant(&mut self) -> &mut AssistantReplay {
        self.current_assistant
            .get_or_insert_with(AssistantReplay::default)
    }

    fn flush_tool_results(&mut self) -> Option<SessionRecord> {
        if self.pending_tool_results.is_empty() {
            return None;
        }
        Some(SessionRecord::Message {
            message: Message {
                role: Role::User,
                content: std::mem::take(&mut self.pending_tool_results).into(),
            },
        })
    }

    fn flush_assistant(&mut self) -> Option<SessionRecord> {
        let assistant = self.current_assistant.take()?;
        assistant
            .into_message()
            .map(|message| SessionRecord::Message { message })
    }
}

impl AssistantReplay {
    fn into_message(self) -> Option<Message> {
        let mut content = Vec::new();
        if !self.thinking_text.is_empty() || self.thinking_signature.is_some() {
            content.push(MessageContent::Thinking {
                text: self.thinking_text,
                signature: self.thinking_signature,
            });
        }
        if !self.text.is_empty() {
            content.push(MessageContent::Text { text: self.text });
        }
        content.extend(self.tool_uses);
        if content.is_empty() {
            return None;
        }
        Some(Message {
            role: Role::Assistant,
            content: content.into(),
        })
    }
}

fn append_if_some(records: &mut Vec<SessionRecord>, record: Option<SessionRecord>) {
    if let Some(record) = record {
        records.push(record);
    }
}

fn content_block_to_message_content(
    block: agent_client_protocol_schema::ContentBlock,
) -> MessageContent {
    match block {
        agent_client_protocol_schema::ContentBlock::Text(text) => {
            MessageContent::Text { text: text.text }
        }
        _ => MessageContent::Text {
            text: String::new(),
        },
    }
}

fn tool_call_output(fields: &ToolCallUpdateFields) -> ToolResultBody {
    let Some(content) = &fields.content else {
        return ToolResultBody::Text {
            text: String::new(),
        };
    };
    let text = content
        .iter()
        .filter_map(|item| match item {
            ToolCallContent::Content(inner) => match &inner.content {
                agent_client_protocol_schema::ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect::<String>();
    ToolResultBody::Text { text }
}

impl StoredRecord {
    /// Constructs a recovery log record using the current schema.
    #[must_use]
    pub fn new(seq: u64, record: SessionRecord) -> Self {
        Self {
            schema_version: STORAGE_SCHEMA_VERSION,
            seq,
            record,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unsupported storage schema version: {0}")]
    UnsupportedSchema(u32),

    #[error("invalid event line {line}: {source}")]
    InvalidEventLine {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("event sequence gap: expected {expected}, got {actual}")]
    SequenceGap { expected: u64, actual: u64 },

    #[error("session not found in storage: {0}")]
    SessionNotFound(SessionId),
}

fn ensure_supported_schema(schema_version: u32) -> Result<(), StorageError> {
    if schema_version == STORAGE_SCHEMA_VERSION {
        return Ok(());
    }
    Err(StorageError::UnsupportedSchema(schema_version))
}

#[cfg(test)]
mod tests;
