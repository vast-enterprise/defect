//! `AgentEvent` → Langfuse ingestion 事件的翻译。
//!
//! [`TraceProjector`] 是**有状态、逐 session** 的投影器（与 `defect-storage` 的
//! `RecordProjector` 同构）。主循环每收到一个 [`AgentEvent`] 调一次
//! [`TraceProjector::project`]，拿回 0..N 个 [`IngestionEvent`] 交给上报器。
//!
//! ## 映射（每个 turn 一个 trace）
//!
//! - `TurnStarted` → `trace-create`（新 turn 级 trace_id）
//! - `UserPromptCommitted` → 暂存为 trace `input`
//! - `LlmCallStarted` → `generation-create`
//! - `AssistantText` / `AssistantThought` → 累积进当前 generation 的 output
//! - `LlmCallFinished` → `generation-update`（endTime + usageDetails + level）
//! - `ToolCallStarted` → `span-create`；`ToolCallFinished` → `span-update`
//! - `ContextCompressed` → trace 上一个 `event-create`
//! - `TurnEnded` → `trace-update`（endTime + output + stopReason）
//!
//! 设计详见 `docs/internal/observability-langfuse.md` §3。
//!
//! ## id 策略
//!
//! - **traceId**：`TurnStarted` 时生成一次 `Uuid::new_v4()`，turn 内复用。
//!   **不可**用 `{session}-turn-{seq}` 自增——resume 后会撞 id（见设计文档 §3.5）。
//! - **generation / span id**：派生自 trace_id + 序号 / `ToolCallId`，turn 内唯一。
//! - **信封 id**：每个 ingestion 事件一个 `Uuid::new_v4()`，供 Langfuse 去重。
//!
//! ## 时间戳
//!
//! `AgentEvent` 不带时间，调用方传入 `now`（RFC3339 字符串）。projector 不自己
//! 读时钟，便于测试与确定性。

use std::collections::HashMap;

use agent_client_protocol::schema::{
    ContentBlock, StopReason, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::Usage;

use super::model::{EventKind, IngestionEvent, ObservationBody, ObservationLevel, TraceBody};

/// 部署环境标签（写进 trace / observation 的 `environment`）。
const DEFAULT_ENVIRONMENT: &str = "production";

/// 逐 session 的投影状态。
pub struct TraceProjector {
    session_id: String,
    /// 当前 turn 的状态；`None` 表示不在 turn 内（TurnStarted 之前）。
    turn: Option<TurnState>,
}

/// 单个 turn（= 一个 langfuse trace）的累积状态。
struct TurnState {
    trace_id: String,
    /// 用户 prompt 文本，TurnStarted 后由 UserPromptCommitted 填，写进 trace input。
    input: Option<String>,
    /// 第几次 LLM 调用——派生 generation id，并跟踪“当前” generation。
    call_seq: u32,
    /// 当前进行中的 generation id（最近一次 LlmCallStarted），用于累积助手输出。
    current_gen_id: Option<String>,
    /// 当前 generation 累积的助手文本（含 thinking 单独累积）。
    gen_output: String,
    /// 整个 turn 的最终助手文本（写进 trace output）。
    final_output: String,
    /// 工具调用 id → 已分配的 span id（Started/Finished 跨事件配对）。
    tool_spans: HashMap<String, String>,
}

impl TurnState {
    fn new(trace_id: String) -> Self {
        Self {
            trace_id,
            input: None,
            call_seq: 0,
            current_gen_id: None,
            gen_output: String::new(),
            final_output: String::new(),
            tool_spans: HashMap::new(),
        }
    }
}

impl TraceProjector {
    /// 新建逐 session 投影器。
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn: None,
        }
    }

    /// 翻译一个事件为 0..N 个 ingestion 事件。`now` 是 RFC3339 时间戳。
    /// `new_id` 提供唯一 id（信封 id / trace id）——注入以便测试确定性。
    pub fn project(
        &mut self,
        event: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        match event {
            AgentEvent::TurnStarted => self.on_turn_started(now, new_id),
            AgentEvent::UserPromptCommitted { content } => {
                self.on_user_prompt(&content);
                Vec::new()
            }
            AgentEvent::LlmCallStarted { model, attempt } => {
                self.on_llm_started(model, attempt, now, new_id)
            }
            AgentEvent::AssistantText { content } => {
                self.accumulate_output(&content);
                Vec::new()
            }
            AgentEvent::AssistantThought { .. } => {
                // thinking 不计入 generation output（避免污染对话内容）；
                // 如需展示可后续单独建 observation。本期忽略。
                Vec::new()
            }
            AgentEvent::LlmCallFinished {
                model,
                attempt,
                usage,
                error,
            } => self.on_llm_finished(&model, attempt, usage, error, now, new_id),
            AgentEvent::ToolCallStarted { id, name, fields } => {
                self.on_tool_started(id.to_string(), name, fields.raw_input, now, new_id)
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                self.on_tool_finished(&id.to_string(), &fields, now, new_id)
            }
            AgentEvent::ContextCompressed {
                tokens_before,
                tokens_after,
            } => self.on_context_compressed(tokens_before, tokens_after, now, new_id),
            AgentEvent::TurnEnded { reason, usage } => {
                self.on_turn_ended(reason, usage, now, new_id)
            }
            // 不上报：进度增量、权限审计（本期不入 langfuse）。
            AgentEvent::ToolCallProgress { .. }
            | AgentEvent::PolicyDecision { .. }
            | AgentEvent::PermissionResolved { .. } => Vec::new(),
            _ => Vec::new(),
        }
    }

    // ---- 各事件处理 ----

    fn on_turn_started(
        &mut self,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let trace_id = new_id();
        self.turn = Some(TurnState::new(trace_id.clone()));
        let body = TraceBody {
            id: trace_id,
            name: Some("turn".into()),
            session_id: Some(self.session_id.clone()),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        vec![IngestionEvent::trace(
            new_id(),
            now.to_string(),
            EventKind::TraceCreate,
            &body,
        )]
    }

    fn on_user_prompt(&mut self, content: &[ContentBlock]) {
        let text = content_text(content);
        if let Some(turn) = self.turn.as_mut()
            && !text.is_empty()
        {
            turn.input = Some(text);
        }
    }

    fn on_llm_started(
        &mut self,
        model: String,
        attempt: u32,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        turn.call_seq += 1;
        let gen_id = format!("{}-gen-{}", turn.trace_id, turn.call_seq);
        turn.current_gen_id = Some(gen_id.clone());
        turn.gen_output.clear();

        let mut meta = serde_json::Map::new();
        meta.insert("attempt".into(), attempt.into());

        let body = ObservationBody {
            id: gen_id,
            trace_id: turn.trace_id.clone(),
            name: Some("llm_call".into()),
            model: Some(model),
            start_time: Some(now.to_string()),
            metadata: Some(serde_json::Value::Object(meta)),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            ..Default::default()
        };
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::GenerationCreate,
            &body,
        )]
    }

    fn accumulate_output(&mut self, content: &ContentBlock) {
        if let Some(turn) = self.turn.as_mut()
            && let ContentBlock::Text(text) = content
        {
            turn.gen_output.push_str(&text.text);
            turn.final_output.push_str(&text.text);
        }
    }

    fn on_llm_finished(
        &mut self,
        model: &str,
        _attempt: u32,
        usage: Usage,
        error: Option<String>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        // 用当前 generation id；若没有（异常），跳过。
        let Some(gen_id) = turn.current_gen_id.clone() else {
            return Vec::new();
        };
        let output = std::mem::take(&mut turn.gen_output);

        let body = ObservationBody {
            id: gen_id,
            trace_id: turn.trace_id.clone(),
            model: Some(model.to_string()),
            end_time: Some(now.to_string()),
            output: (!output.is_empty()).then_some(serde_json::Value::String(output)),
            usage_details: usage_to_details(&usage),
            level: error.as_ref().map(|_| ObservationLevel::Error),
            status_message: error,
            ..Default::default()
        };
        turn.current_gen_id = None;
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::GenerationUpdate,
            &body,
        )]
    }

    fn on_tool_started(
        &mut self,
        tool_call_id: String,
        name: String,
        raw_input: Option<serde_json::Value>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        let span_id = format!("{}-tool-{}", turn.trace_id, tool_call_id);
        turn.tool_spans.insert(tool_call_id, span_id.clone());

        let body = ObservationBody {
            id: span_id,
            trace_id: turn.trace_id.clone(),
            name: Some(name),
            start_time: Some(now.to_string()),
            input: raw_input,
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            ..Default::default()
        };
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::SpanCreate,
            &body,
        )]
    }

    fn on_tool_finished(
        &mut self,
        tool_call_id: &str,
        fields: &ToolCallUpdateFields,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        // 取回 Started 时分配的 span id；缺失（乱序）则现派生一个。
        let span_id = turn
            .tool_spans
            .remove(tool_call_id)
            .unwrap_or_else(|| format!("{}-tool-{}", turn.trace_id, tool_call_id));

        let failed = matches!(
            fields.status,
            Some(ToolCallStatus::Failed)
        );
        let body = ObservationBody {
            id: span_id,
            trace_id: turn.trace_id.clone(),
            end_time: Some(now.to_string()),
            output: fields.raw_output.clone(),
            level: failed.then_some(ObservationLevel::Error),
            ..Default::default()
        };
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::SpanUpdate,
            &body,
        )]
    }

    fn on_context_compressed(
        &mut self,
        tokens_before: u64,
        tokens_after: u64,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_ref() else {
            return Vec::new();
        };
        let mut meta = serde_json::Map::new();
        meta.insert("tokens_before".into(), tokens_before.into());
        meta.insert("tokens_after".into(), tokens_after.into());

        let body = ObservationBody {
            id: new_id(),
            trace_id: turn.trace_id.clone(),
            name: Some("context_compaction".into()),
            start_time: Some(now.to_string()),
            metadata: Some(serde_json::Value::Object(meta)),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            ..Default::default()
        };
        vec![IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::EventCreate,
            &body,
        )]
    }

    fn on_turn_ended(
        &mut self,
        reason: StopReason,
        usage: Usage,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.take() else {
            return Vec::new();
        };
        let mut meta = serde_json::Map::new();
        meta.insert(
            "stop_reason".into(),
            serde_json::to_value(reason).unwrap_or(serde_json::Value::Null),
        );
        if let Some(details) = usage_to_details(&usage) {
            meta.insert("usage".into(), serde_json::Value::Object(details));
        }

        let body = TraceBody {
            id: turn.trace_id,
            session_id: Some(self.session_id.clone()),
            input: turn.input.map(serde_json::Value::String),
            output: (!turn.final_output.is_empty())
                .then_some(serde_json::Value::String(turn.final_output)),
            metadata: Some(serde_json::Value::Object(meta)),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        vec![IngestionEvent::trace(
            new_id(),
            now.to_string(),
            // 同 trace_id 二次发送 = 更新（合并 endTime/output/metadata）。
            EventKind::TraceCreate,
            &body,
        )]
    }
}

/// 把 [`Usage`] 转成 langfuse `usageDetails` map。全 None 时返回 None（不上报）。
fn usage_to_details(usage: &Usage) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut map = serde_json::Map::new();
    if let Some(v) = usage.input_tokens {
        map.insert("input".into(), v.into());
    }
    if let Some(v) = usage.output_tokens {
        map.insert("output".into(), v.into());
    }
    if let Some(v) = usage.cache_read_input_tokens {
        map.insert("cache_read_input_tokens".into(), v.into());
    }
    if let Some(v) = usage.cache_creation_input_tokens {
        map.insert("cache_creation_input_tokens".into(), v.into());
    }
    (!map.is_empty()).then_some(map)
}

/// 拼接 ContentBlock 列表里的文本（忽略非文本块）。
fn content_text(content: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in content {
        if let ContentBlock::Text(text) = block {
            out.push_str(&text.text);
        }
    }
    out
}
