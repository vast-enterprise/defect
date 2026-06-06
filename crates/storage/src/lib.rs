//! 会话持久化。
//!
//! v0 起步以 jsonl 形式落盘会话恢复日志，支持 append 与回放；后续按需演进到
//! snapshot + sqlite 等带索引的存储。

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]

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

/// Session 创建后的落盘观察器。
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

    /// 找出 `cwd` 下最近活跃的 session id（用于 `--resume` 不带 id 时）。
    ///
    /// 扫 `sessions_root` 下每个 session 目录，读 `meta.json` 取 `cwd`，匹配
    /// 给定 `cwd`（规整后逐字节比较）的候选里按 `journal.jsonl` 最后修改时间
    /// 取最新者。没有匹配返回 `Ok(None)`。
    ///
    /// 单个 session 目录损坏（meta 读不出 / 解析失败）跳过而非整体失败——
    /// 一份坏存档不该让 resume 完全不可用。
    ///
    /// # Errors
    ///
    /// `sessions_root` 存在但无法枚举时返回错误；目录不存在按"无候选"
    /// （`Ok(None)`）处理。
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
                continue; // 坏存档 / 无 meta：跳过
            };
            let meta_cwd = fs::canonicalize(&meta.cwd).unwrap_or_else(|_| meta.cwd.clone());
            if meta_cwd != target {
                continue;
            }
            // 活跃度按 journal 最后修改时间排；取不到时间的退到 UNIX_EPOCH。
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

/// 单个 session 的落盘目录。
///
/// 真相源是 `meta.json` + `journal.jsonl`；`snapshot.json` 保留为后续
/// resume 加速口子。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    /// 绑定一个 session 目录。
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// 在 `sessions_root/<session_id>/` 下创建一个 store。
    #[must_use]
    pub fn for_session(sessions_root: impl AsRef<Path>, session_id: &SessionId) -> Self {
        Self::new(sessions_root.as_ref().join(session_id.0.as_ref()))
    }

    /// session 目录路径。
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `meta.json` 路径。
    #[must_use]
    pub fn meta_path(&self) -> PathBuf {
        self.root.join(META_FILENAME)
    }

    /// `journal.jsonl` 路径。
    #[must_use]
    pub fn journal_path(&self) -> PathBuf {
        self.root.join(JOURNAL_FILENAME)
    }

    /// `snapshot.json` 路径。
    #[must_use]
    pub fn snapshot_path(&self) -> PathBuf {
        self.root.join(SNAPSHOT_FILENAME)
    }

    /// 将当前恢复态写入 `snapshot.json`。
    ///
    /// # Errors
    ///
    /// 目录创建失败、序列化失败、临时文件写入失败、或原子替换失败。
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

    /// 初始化 session 目录与元数据文件。
    ///
    /// # Errors
    ///
    /// 创建目录失败、序列化失败、已存在且覆盖写入失败等。
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

    /// 追加一条恢复日志记录。
    ///
    /// # Errors
    ///
    /// 目录不存在、打开文件失败、序列化失败、或写入/flush 失败。
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

    /// 读取 `meta.json`。
    ///
    /// # Errors
    ///
    /// 文件不存在、内容不是合法 JSON、或 schema 不支持。
    pub fn load_meta(&self) -> Result<SessionMeta, StorageError> {
        let bytes = fs::read(self.meta_path())?;
        let meta = serde_json::from_slice::<SessionMeta>(&bytes)?;
        ensure_supported_schema(meta.schema_version)?;
        Ok(meta)
    }

    /// 读取 `snapshot.json`，文件不存在时返回 `Ok(None)`。
    ///
    /// # Errors
    ///
    /// 文件存在但内容不合法、或 schema 不支持时返回错误。
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

    /// 顺序读取当前可用的 journal 记录。
    ///
    /// 如果存在 `snapshot.json`，只读取 snapshot 之后的 tail；否则从 0 开始读取。
    /// 如果文件尾有崩溃残留的半行，v0 直接返回错误；后续可演进成自动截尾。
    ///
    /// # Errors
    ///
    /// 文件不存在、逐行解析失败、schema 不支持、或记录序号不连续。
    pub fn replay_records(&self) -> Result<Vec<StoredRecord>, StorageError> {
        let start_seq = self
            .load_snapshot()?
            .map_or(0, |snapshot| snapshot.next_seq);
        self.replay_records_from(start_seq)
    }

    /// 顺序回放从指定序号开始的恢复日志 tail。
    ///
    /// # Errors
    ///
    /// 文件不存在、逐行解析失败、schema 不支持、或记录序号不连续。
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

    /// 计算下一条恢复日志应使用的序号。
    ///
    /// 如果存在 `snapshot.json`，只校验并扫描 snapshot 之后的 journal tail；
    /// 否则要求 journal 从 0 开始连续。
    ///
    /// # Errors
    ///
    /// 文件不存在、逐行解析失败、schema 不支持、或 tail 序号不连续。
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

    /// 将当前恢复态写成 snapshot。
    ///
    /// # Errors
    ///
    /// 回放失败、读取下一序号失败、或写入 snapshot 失败。
    pub fn refresh_snapshot(&self) -> Result<StoredSnapshot, StorageError> {
        let state = self.replay_state()?;
        let snapshot = StoredSnapshot::new(self.next_record_seq()?, state.to_snapshot_state());
        self.write_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    /// 用最新 snapshot 覆盖已被快照包含的 journal 前缀。
    ///
    /// 该方法不应和同一个 session 的事件写入并发调用；当前 v0 调用方需要在
    /// session 空闲时显式触发 compaction。
    ///
    /// # Errors
    ///
    /// 回放失败、snapshot 写入失败、journal tail 读取失败、或 journal 重写失败。
    pub fn compact_journal_to_snapshot(&self) -> Result<StoredSnapshot, StorageError> {
        let snapshot = self.refresh_snapshot()?;
        let tail = self.replay_records_from(snapshot.next_seq)?;
        self.write_journal_records(&tail)?;
        Ok(snapshot)
    }

    /// 回放恢复日志并折叠出可恢复的历史状态。
    ///
    /// # Errors
    ///
    /// 底层 replay 失败、或记录序列无法折叠成语义一致的历史时返回错误。
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

/// `meta.json` 的稳定元数据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub cwd: PathBuf,
    pub mcp_servers: Vec<McpServer>,
}

impl SessionMeta {
    /// 构造 v0 元数据。
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

/// `journal.jsonl` 的单行恢复记录。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRecord {
    pub schema_version: u32,
    pub seq: u64,
    pub record: SessionRecord,
}

/// `snapshot.json` 的持久化快照。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredSnapshot {
    pub schema_version: u32,
    pub next_seq: u64,
    pub state: SnapshotState,
}

/// 可快照化的恢复态。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SnapshotState {
    pub history: Vec<Message>,
    pub turn_count: u64,
    pub last_turn_ended: bool,
}

/// session 恢复日志记录。
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

/// 由事件流回放出的 session 恢复态。
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

    /// 将当前恢复态投影为持久化 snapshot。
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
    /// 以当前 schema 构造一个持久化快照。
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
                // 空 name 的 ToolCallStarted 是失败标记事件（tool-not-found /
                // denied 等的 wire 信号），不是真实工具调用——持久化成 ToolUse
                // 会让 name 为空，resume 时被 provider 拒（`tool_use.name` 至少 1
                // 字符）。这类事件不入历史。
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
    /// 以当前 schema 构造一条恢复日志记录。
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
mod test;
