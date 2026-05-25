//! 会话持久化。
//!
//! v0 起步以 jsonl 形式落盘会话事件，支持 append 与回放；后续按需演进到
//! sqlite 等带索引的存储。

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use agent_client_protocol::schema::{McpServer, SessionId};
use defect_agent::event::AgentEvent;
use serde::{Deserialize, Serialize};

const META_FILENAME: &str = "meta.json";
const EVENTS_FILENAME: &str = "events.jsonl";
const SNAPSHOT_FILENAME: &str = "snapshot.json";
const STORAGE_SCHEMA_VERSION: u32 = 1;

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
    Io(#[from] std::io::Error),

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
}

fn ensure_supported_schema(schema_version: u32) -> Result<(), StorageError> {
    if schema_version == STORAGE_SCHEMA_VERSION {
        return Ok(());
    }
    Err(StorageError::UnsupportedSchema(schema_version))
}

#[cfg(test)]
mod test;
