//! Langfuse ingestion API 的 wire 结构体。
//!
//! 对应 `POST /api/public/ingestion`。请求体是一个 `{ "batch": [ <event>, ... ] }`，
//! 每个 event 是一个「信封」（envelope）：`{ id, type, timestamp, body }`。
//!
//! - 信封的 `id` 用于**去重**（每请求唯一）；
//! - `body.id` 才是 trace / observation 的真实 id（同 id 不同 envelope id = 更新）。
//! - 字段命名全 camelCase（`#[serde(rename_all = "camelCase")]`）。
//!
//! Data model mirrors
//! <https://langfuse.com/docs/observability/data-model>。
//!
//! 我们只覆盖接入需要的事件类型与字段；Langfuse 的完整 schema 比这宽，
//! 未用到的字段一律不发（`skip_serializing_if = "Option::is_none"`）。

use serde::{Deserialize, Serialize};

/// 一次 ingestion 请求体。
#[derive(Debug, Clone, Serialize)]
pub struct IngestionBatch {
    pub batch: Vec<IngestionEvent>,
}

/// ingestion 端点的响应体。
///
/// **注意**：该端点对批量请求**始终返回 207 Multi-Status**（即便逐条全部成功）；
/// 逐条结果分散在 `successes`（含各自 HTTP status，成功为 201）与 `errors` 里。
/// 因此“是否真出错”只能看 `errors` 是否非空，**不能**凭 207 状态码判断。
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionResponse {
    #[serde(default)]
    pub successes: Vec<IngestionSuccess>,
    #[serde(default)]
    pub errors: Vec<IngestionError>,
}

/// 单条成功结果。
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionSuccess {
    pub id: String,
    pub status: u16,
}

/// 单条失败结果。字段尽量宽松——只用于诊断日志，未知字段忽略。
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionError {
    pub id: String,
    pub status: u16,
    #[serde(default)]
    pub message: Option<String>,
}

/// 单个事件信封。`type` 是 oneOf 判别字段，`body` 随类型而异。
///
/// 用扁平的 `kind`（映射到 `"type"`）+ 泛化 `body: serde_json::Value` 而非
/// 给每种类型一个 enum 变体：投影器（projector）按需构造 body JSON，
/// model 层只负责包信封。这样新增字段不必动 model 层。
#[derive(Debug, Clone, Serialize)]
pub struct IngestionEvent {
    /// 信封 id，每请求唯一——Langfuse 用它去重。
    pub id: String,
    /// 事件类型判别字符串，如 `trace-create` / `generation-create`。
    #[serde(rename = "type")]
    pub kind: EventKind,
    /// 事件产生时刻（ISO-8601 / RFC3339）。
    pub timestamp: String,
    /// 类型特定的载荷。`body.id` 是被操作的 trace / observation id。
    pub body: serde_json::Value,
}

/// Ingestion 事件类型判别值。
///
/// 取值是 Langfuse ingestion 的公开契约。`-create` 与 `-update` 共享同一
/// body 形状（同 `body.id` = upsert/合并），我们靠它实现「先建后补 endTime」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EventKind {
    TraceCreate,
    GenerationCreate,
    GenerationUpdate,
    SpanCreate,
    SpanUpdate,
    EventCreate,
}

/// observation 的 level（决定 UI 里的状态着色 / 过滤）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ObservationLevel {
    Debug,
    Default,
    Warning,
    Error,
}

/// trace 的 body（`trace-create` / 更新共用）。
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceBody {
    /// trace id（我们的 turn 级 UUID）。同 id 二次发送 = 更新。
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 把同一 defect session 的多个 turn-trace 串起来。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// 部署环境（如 `production`）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// trace 开始时间（RFC3339）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// observation（generation / span）的 body。
///
/// generation 与 span 共用一份 body 字段集——区别只在事件 `type` 与是否带
/// `model` / `usageDetails`（span 通常不带）。
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationBody {
    /// observation id。同 id 二次发送 = 合并（先 create 后 update endTime）。
    pub id: String,
    /// 所属 trace id。
    pub trace_id: String,
    /// 父 observation；None 表示直接挂在 trace 下。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_observation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<ObservationLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    // ---- generation 专有 ----
    /// 模型名（仅 generation）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// 自由形态的 token 用量明细：键名自定，`total` 不填则由后端推导。
    /// 我们填 `input` / `output` / `cache_read_input_tokens` /
    /// `cache_creation_input_tokens`（对齐 [`defect_agent::llm::Usage`]）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl IngestionEvent {
    /// 包一个 trace 事件。
    pub fn trace(
        envelope_id: String,
        timestamp: String,
        kind: EventKind,
        body: &TraceBody,
    ) -> Self {
        Self {
            id: envelope_id,
            kind,
            timestamp,
            body: serde_json::to_value(body).unwrap_or(serde_json::Value::Null),
        }
    }

    /// 包一个 observation 事件（generation / span / event）。
    pub fn observation(
        envelope_id: String,
        timestamp: String,
        kind: EventKind,
        body: &ObservationBody,
    ) -> Self {
        Self {
            id: envelope_id,
            kind,
            timestamp,
            body: serde_json::to_value(body).unwrap_or(serde_json::Value::Null),
        }
    }
}
