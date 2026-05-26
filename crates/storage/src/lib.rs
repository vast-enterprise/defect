//! 会话持久化。
//!
//! v0 起步以 jsonl 形式落盘会话事件，支持 append 与回放；后续按需演进到
//! sqlite 等带索引的存储。

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Error, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_client_protocol::schema::{
    McpServer, SessionId, ToolCallContent, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::error::BoxError;
use defect_agent::event::AgentEvent;
use defect_agent::llm::{Message, MessageContent, Role, ToolResultBody};
use defect_agent::session::{
    LoadedSession, Session, SessionCreateInfo, SessionLoader, SessionObserver,
};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

const META_FILENAME: &str = "meta.json";
const EVENTS_FILENAME: &str = "events.jsonl";
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

        let session_id = info.id.clone();
        tokio::spawn(async move {
            let mut events = session.subscribe();
            let mut seq = 0_u64;
            while let Some(event) = events.next().await {
                let record = StoredEvent::new(seq, event);
                if let Err(err) = store.append_event(&record) {
                    tracing::warn!(
                        session_id = %session_id,
                        error = %err,
                        "failed to append session event"
                    );
                    return;
                }
                seq += 1;
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
/// 真相源是 `meta.json` + `events.jsonl`；`snapshot.json` 仅保留为后续
/// resume 加速口子，v0 不写入也不读取。
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

    /// `events.jsonl` 路径。
    #[must_use]
    pub fn events_path(&self) -> PathBuf {
        self.root.join(EVENTS_FILENAME)
    }

    /// `snapshot.json` 路径。
    #[must_use]
    pub fn snapshot_path(&self) -> PathBuf {
        self.root.join(SNAPSHOT_FILENAME)
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
            .open(self.events_path())?;
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

    /// 追加一条事件。
    ///
    /// # Errors
    ///
    /// 目录不存在、打开文件失败、序列化失败、或写入/flush 失败。
    pub fn append_event(&self, record: &StoredEvent) -> Result<(), StorageError> {
        ensure_supported_schema(record.schema_version)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.events_path())?;
        let encoded = serde_json::to_string(record)?;
        file.write_all(encoded.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }

    /// 顺序回放全部事件。
    ///
    /// 如果文件尾有崩溃残留的半行，v0 直接返回错误；后续可演进成自动截尾。
    ///
    /// # Errors
    ///
    /// 文件不存在、逐行解析失败、schema 不支持、或事件序号不连续。
    pub fn replay(&self) -> Result<Vec<StoredEvent>, StorageError> {
        let path = self.events_path();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        let mut expected_seq = 0_u64;

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str::<StoredEvent>(&line).map_err(|source| {
                StorageError::InvalidEventLine {
                    line: line_no + 1,
                    source,
                }
            })?;
            ensure_supported_schema(record.schema_version)?;
            if record.seq != expected_seq {
                return Err(StorageError::SequenceGap {
                    expected: expected_seq,
                    actual: record.seq,
                });
            }
            expected_seq += 1;
            events.push(record);
        }

        Ok(events)
    }

    /// 回放事件并折叠出可恢复的历史状态。
    ///
    /// # Errors
    ///
    /// 底层 replay 失败、或事件序列无法折叠成语义一致的历史时返回错误。
    pub fn replay_state(&self) -> Result<ReplayState, StorageError> {
        let mut state = ReplayState::default();
        for record in self.replay()? {
            state.apply(record.event)?;
        }
        Ok(state)
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

/// `events.jsonl` 的单行记录。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredEvent {
    pub schema_version: u32,
    pub seq: u64,
    pub event: AgentEvent,
}

/// 由事件流回放出的 session 恢复态。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayState {
    pub history: Vec<Message>,
    pub turn_count: u64,
    pub last_turn_ended: bool,
    current_assistant: Option<AssistantReplay>,
    pending_tool_results: Vec<MessageContent>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct AssistantReplay {
    text: String,
    thinking_text: String,
    thinking_signature: Option<String>,
    tool_uses: Vec<MessageContent>,
}

impl ReplayState {
    fn apply(&mut self, event: AgentEvent) -> Result<(), StorageError> {
        match event {
            AgentEvent::UserPromptCommitted { content } => {
                self.flush_assistant();
                self.flush_tool_results();
                self.history.push(Message {
                    role: Role::User,
                    content: content
                        .into_iter()
                        .map(content_block_to_message_content)
                        .collect(),
                });
                self.last_turn_ended = false;
                Ok(())
            }
            AgentEvent::AssistantText { content } => {
                let replay = self
                    .current_assistant
                    .get_or_insert_with(AssistantReplay::default);
                if let agent_client_protocol::schema::ContentBlock::Text(text) = content {
                    replay.text.push_str(&text.text);
                }
                Ok(())
            }
            AgentEvent::AssistantThought { content } => {
                let replay = self
                    .current_assistant
                    .get_or_insert_with(AssistantReplay::default);
                if let agent_client_protocol::schema::ContentBlock::Text(text) = content {
                    replay.thinking_text.push_str(&text.text);
                }
                Ok(())
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                let replay = self
                    .current_assistant
                    .get_or_insert_with(AssistantReplay::default);
                replay.tool_uses.push(MessageContent::ToolUse {
                    id: id.to_string(),
                    name,
                    args: fields
                        .raw_input
                        .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
                });
                Ok(())
            }
            AgentEvent::ToolCallProgress { .. }
            | AgentEvent::PolicyDecision { .. }
            | AgentEvent::PermissionResolved { .. }
            | AgentEvent::LlmCallStarted { .. }
            | AgentEvent::LlmCallFinished { .. }
            | AgentEvent::ContextCompressed { .. } => Ok(()),
            AgentEvent::ToolCallFinished { id, fields } => {
                let status = fields.status.unwrap_or(ToolCallStatus::Completed);
                let output = tool_call_output(&fields);
                self.pending_tool_results.push(MessageContent::ToolResult {
                    tool_use_id: id.to_string(),
                    output,
                    is_error: status == ToolCallStatus::Failed,
                });
                Ok(())
            }
            AgentEvent::TurnStarted => {
                self.turn_count = self.turn_count.saturating_add(1);
                self.last_turn_ended = false;
                self.current_assistant = None;
                Ok(())
            }
            AgentEvent::TurnEnded { .. } => {
                self.flush_assistant();
                self.flush_tool_results();
                self.last_turn_ended = true;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn flush_tool_results(&mut self) {
        if self.pending_tool_results.is_empty() {
            return;
        }
        self.flush_assistant();
        self.history.push(Message {
            role: Role::User,
            content: std::mem::take(&mut self.pending_tool_results),
        });
    }

    fn flush_assistant(&mut self) {
        let Some(assistant) = self.current_assistant.take() else {
            return;
        };
        let mut content = Vec::new();
        if !assistant.thinking_text.is_empty() || assistant.thinking_signature.is_some() {
            content.push(MessageContent::Thinking {
                text: assistant.thinking_text,
                signature: assistant.thinking_signature,
            });
        }
        if !assistant.text.is_empty() {
            content.push(MessageContent::Text {
                text: assistant.text,
            });
        }
        content.extend(assistant.tool_uses);
        if !content.is_empty() {
            self.history.push(Message {
                role: Role::Assistant,
                content,
            });
        }
    }
}

fn content_block_to_message_content(
    block: agent_client_protocol::schema::ContentBlock,
) -> MessageContent {
    match block {
        agent_client_protocol::schema::ContentBlock::Text(text) => {
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
                agent_client_protocol::schema::ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect::<String>();
    ToolResultBody::Text { text }
}

impl StoredEvent {
    /// 以当前 schema 构造一条落盘事件。
    #[must_use]
    pub fn new(seq: u64, event: AgentEvent) -> Self {
        Self {
            schema_version: STORAGE_SCHEMA_VERSION,
            seq,
            event,
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
