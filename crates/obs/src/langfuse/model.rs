//! Wire types for the Langfuse ingestion API.
//!
//! Corresponds to `POST /api/public/ingestion`. The request body is `{ "batch": [
//! <event>, ... ] }`,
//! where each event is an envelope: `{ id, type, timestamp, body }`.
//!
//! - The envelope `id` is used for **deduplication** (unique per request).
//! - `body.id` is the real trace / observation id (same id with different envelope id =
//!   update).
//! - All fields use camelCase (`#[serde(rename_all = "camelCase")]`).
//!
//! Data model mirrors
//! <https://langfuse.com/docs/observability/data-model>.
//!
//! We only cover the event types and fields needed for ingestion; the full Langfuse
//! schema is wider.
//! Unused fields are omitted (`skip_serializing_if = "Option::is_none"`).

use serde::{Deserialize, Serialize};

/// The body of an ingestion request.
#[derive(Debug, Clone, Serialize)]
pub struct IngestionBatch {
    pub batch: Vec<IngestionEvent>,
}

/// Response body for the ingestion endpoint.
///
/// **Note**: This endpoint **always returns 207 Multi-Status** for batch requests (even
/// if every individual item succeeds); per-item results are split between `successes`
/// (each with its own HTTP status, 201 on success) and `errors`. Therefore, whether an
/// error actually occurred can only be determined by checking whether `errors` is
/// non-empty — the 207 status code **cannot** be used for that purpose.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionResponse {
    #[serde(default)]
    pub successes: Vec<IngestionSuccess>,
    #[serde(default)]
    pub errors: Vec<IngestionError>,
}

/// A single successful ingestion result.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionSuccess {
    pub id: String,
    pub status: u16,
}

/// A single failure result. Fields are intentionally lenient — only used for diagnostic
/// logging; unknown fields are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionError {
    pub id: String,
    pub status: u16,
    #[serde(default)]
    pub message: Option<String>,
}

/// A single event envelope. `type` is the oneOf discriminator; `body` varies by type.
///
/// Uses a flat `kind` (mapped to `"type"`) + generic `body: serde_json::Value` instead of
/// an enum variant per type: the projector constructs the body JSON as needed, and the
/// model layer only wraps the envelope. This way, adding new fields does not require
/// changes to the model layer.
#[derive(Debug, Clone, Serialize)]
pub struct IngestionEvent {
    /// Envelope ID, unique per request — Langfuse uses it for deduplication.
    pub id: String,
    /// A discriminant string for the event type, e.g. `trace-create` /
    /// `generation-create`.
    #[serde(rename = "type")]
    pub kind: EventKind,
    /// Timestamp when the event was produced (ISO-8601 / RFC3339).
    pub timestamp: String,
    /// Type-specific payload. `body.id` is the trace or observation id being operated on.
    pub body: serde_json::Value,
}

/// Discriminant for ingestion event types.
///
/// Values follow the public Langfuse ingestion contract. `-create` and `-update` share
/// the same body shape (same `body.id` = upsert/merge), which we rely on to implement
/// "create first, fill in endTime later".
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

/// The level of an observation, determining UI state coloring and filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ObservationLevel {
    Debug,
    Default,
    Warning,
    Error,
}

/// Body for a trace (shared by `trace-create` and update).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceBody {
    /// Trace ID (our turn-level UUID). Sending the same ID again acts as an update.
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Links together multiple turn-traces belonging to the same defect session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Deployment environment (e.g. `production`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Trace start time (RFC3339).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Body of an observation (generation / span).
///
/// Generation and span share the same set of body fields — they differ only in the event
/// `type` and whether `model` / `usageDetails` are present (span usually does not carry
/// them).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationBody {
    /// Observation id. Sending the same id twice results in a merge (first create, then
    /// update endTime).
    pub id: String,
    /// The trace ID this observation belongs to.
    pub trace_id: String,
    /// Parent observation; `None` means it is directly attached to the trace.
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
    // ---- generation-specific ----
    /// The model name (generation only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Free-form token usage details: keys are arbitrary; if `total` is omitted, the
    /// backend infers it.
    /// We populate `input` / `output` / `cache_read_input_tokens` /
    /// `cache_creation_input_tokens` (aligned with [`defect_agent::llm::Usage`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl IngestionEvent {
    /// Wraps a trace event.
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

    /// Wrap an observation event (generation / span / event).
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
