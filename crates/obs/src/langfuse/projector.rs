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

use agent_client_protocol_schema::{
    ContentBlock, StopReason, ToolCallStatus, ToolCallUpdateFields,
};
use defect_agent::event::{AgentEvent, LlmRequestSnapshot};
use defect_agent::llm::{Message, MessageContent, Role, Usage};

use super::model::{EventKind, IngestionEvent, ObservationBody, ObservationLevel, TraceBody};

/// 部署环境标签（写进 trace / observation 的 `environment`）。
const DEFAULT_ENVIRONMENT: &str = "production";

/// 逐 session 的投影状态。
pub struct TraceProjector {
    session_id: String,
    /// 当前 turn 的状态；`None` 表示不在 turn 内（TurnStarted 之前）。
    turn: Option<TurnState>,
    /// 暂存的用户 prompt 文本。主循环**先发 `UserPromptCommitted` 再发
    /// `TurnStarted`**（turn.rs:197 先于 :212），所以收到 prompt 时 turn 还没建——
    /// 先存这里，`TurnStarted` 建 `TurnState` 时再取进去。
    pending_input: Option<String>,
}

/// 单个 turn（= 一个 langfuse trace）的累积状态。
struct TurnState {
    trace_id: String,
    /// 用户 prompt 文本，TurnStarted 后由 UserPromptCommitted 填，写进 trace input。
    input: Option<String>,
    /// 第几次 LLM 调用——派生 generation id。
    call_seq: u32,
    /// 当前进行中的 generation（最近一次 LlmCallStarted 建立）。
    ///
    /// 关键：generation 的 output / thinking 是在 `LlmCallStarted` 之后才流式到达的
    /// （`LlmCallFinished` 也早于流），所以 generation 的收尾（generation-update）
    /// **延迟**到下一次 `LlmCallStarted` 或 `TurnEnded` 时才发——保证 create 永远
    /// 先于 update（修复 “observation not found”）。
    current_gen: Option<PendingGeneration>,
    /// 整个 turn 的最终助手文本（写进 trace output）。
    final_output: String,
    /// 工具调用 id → 已分配的 span id（Started/Finished 跨事件配对）。
    tool_spans: HashMap<String, String>,
    /// 进行中的 subagent 子 turn 状态：父 `spawn_agent` 工具调用 id → 嵌套状态。
    /// 一次 fanout 同时跑多个 subagent 时各占一项，互不串扰。
    subagents: HashMap<String, SubagentState>,
}

/// 一个 subagent 子 turn 的嵌套投影状态。它的 generation / 工具 span 都挂在父
/// `spawn_agent` 工具 span（`parent_span_id`）之下、复用父 trace_id，但 id 命名空间
/// 用父 tool_call_id 隔开，避免与父 / 兄弟 subagent 撞 id。
struct SubagentState {
    /// 父 `spawn_agent` 工具 span id——子 observation 的 `parent_observation_id`。
    parent_span_id: String,
    /// 子 agent profile 名（进子 observation 的 metadata）。
    agent_type: String,
    /// 子 turn 第几次 LLM 调用——派生子 generation id。
    call_seq: u32,
    /// 子 turn 进行中的 generation。
    current_gen: Option<PendingGeneration>,
    /// 子 turn 的工具调用 id → span id。
    tool_spans: HashMap<String, String>,
}

impl SubagentState {
    fn new(parent_span_id: String, agent_type: String) -> Self {
        Self {
            parent_span_id,
            agent_type,
            call_seq: 0,
            current_gen: None,
            tool_spans: HashMap::new(),
        }
    }
}

/// 进行中的 generation 累积状态。收尾时一次性 flush 成 generation-update。
struct PendingGeneration {
    id: String,
    model: String,
    /// 累积的助手回复正文。
    output: String,
    /// 累积的 thinking 文本（放进 generation 的 metadata.reasoning，不进 output）。
    thinking: String,
    /// 本次调用的 token 用量（来自 LlmCallFinished.usage，流 drain 后到达）。
    usage: Usage,
    /// 失败信息（来自 LlmCallFinished.error）。
    error: Option<String>,
}

impl TurnState {
    fn new(trace_id: String) -> Self {
        Self {
            trace_id,
            input: None,
            call_seq: 0,
            current_gen: None,
            final_output: String::new(),
            tool_spans: HashMap::new(),
            subagents: HashMap::new(),
        }
    }
}

impl TraceProjector {
    /// 新建逐 session 投影器。
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn: None,
            pending_input: None,
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
            AgentEvent::LlmCallStarted {
                model,
                attempt,
                request,
            } => self.on_llm_started(model, attempt, request.as_ref(), now, new_id),
            AgentEvent::AssistantText { content } => {
                self.accumulate_text(&content);
                Vec::new()
            }
            AgentEvent::AssistantThought { content } => {
                self.accumulate_thinking(&content);
                Vec::new()
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                // LlmCallFinished 现在由 run_inner 在流 drain 之后发，带**本次调用**的
                // 真 usage（非 turn 累计）。记到当前 generation，收尾时写入 usageDetails。
                // generation 收尾（output/thinking/endTime）仍延迟到下次 LlmCallStarted
                // 或 TurnEnded——保证 generation-create 先于 update。
                self.note_llm_finished(usage, error);
                Vec::new()
            }
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
            AgentEvent::Subagent {
                parent_tool_call_id,
                agent_type,
                inner,
            } => self.on_subagent(
                &parent_tool_call_id.to_string(),
                agent_type,
                *inner,
                now,
                new_id,
            ),
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
        let mut state = TurnState::new(trace_id.clone());
        // 取走 TurnStarted 之前暂存的用户 prompt。
        state.input = self.pending_input.take();
        let body = TraceBody {
            id: trace_id,
            name: Some("turn".into()),
            session_id: Some(self.session_id.clone()),
            // trace-create 时就带上 input，UI 立刻能看到用户输入（不必等 TurnEnded）。
            input: state.input.clone().map(serde_json::Value::String),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        self.turn = Some(state);
        vec![IngestionEvent::trace(
            new_id(),
            now.to_string(),
            EventKind::TraceCreate,
            &body,
        )]
    }

    fn on_user_prompt(&mut self, content: &[ContentBlock]) {
        let text = content_text(content);
        if !text.is_empty() {
            // 主循环先发 UserPromptCommitted 再发 TurnStarted，此刻 turn 尚未建立——
            // 暂存，等 on_turn_started 取走。
            self.pending_input = Some(text);
        }
    }

    fn on_llm_started(
        &mut self,
        model: String,
        attempt: u32,
        request: &LlmRequestSnapshot,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        // 先收尾上一个 generation（若有）——保证它的 generation-update 在本次
        // generation-create 之前发出。
        let mut events = self.flush_generation(now, new_id);

        let Some(turn) = self.turn.as_mut() else {
            return events;
        };
        turn.call_seq += 1;
        let gen_id = format!("{}-gen-{}", turn.trace_id, turn.call_seq);
        turn.current_gen = Some(PendingGeneration {
            id: gen_id.clone(),
            model: model.clone(),
            output: String::new(),
            thinking: String::new(),
            usage: Usage::default(),
            error: None,
        });

        let mut meta = serde_json::Map::new();
        meta.insert("attempt".into(), attempt.into());

        let body = ObservationBody {
            id: gen_id,
            trace_id: turn.trace_id.clone(),
            name: Some("llm_call".into()),
            model: Some(model),
            start_time: Some(now.to_string()),
            // input = 标准 chat messages 数组（system 作为第一条 {role:"system"}）。
            input: Some(request_to_input(request)),
            metadata: Some(serde_json::Value::Object(meta)),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            ..Default::default()
        };
        events.push(IngestionEvent::observation(
            new_id(),
            now.to_string(),
            EventKind::GenerationCreate,
            &body,
        ));
        events
    }

    /// 助手回复正文增量 → 累积到当前 generation 的 output + turn 的 final_output。
    fn accumulate_text(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(turn) = self.turn.as_mut()
        {
            turn.final_output.push_str(&text.text);
            if let Some(pg) = turn.current_gen.as_mut() {
                pg.output.push_str(&text.text);
            }
        }
    }

    /// thinking 增量 → 累积到当前 generation 的 thinking（最终进 metadata.reasoning）。
    fn accumulate_thinking(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(turn) = self.turn.as_mut()
            && let Some(pg) = turn.current_gen.as_mut()
        {
            pg.thinking.push_str(&text.text);
        }
    }

    /// 记录 LLM 调用结束（LlmCallFinished）的 usage / error 到当前 generation，
    /// 收尾时写出。usage 是**本次调用**的（非 turn 累计）。
    fn note_llm_finished(&mut self, usage: Usage, error: Option<String>) {
        if let Some(turn) = self.turn.as_mut()
            && let Some(pg) = turn.current_gen.as_mut()
        {
            pg.usage = usage;
            if error.is_some() {
                pg.error = error;
            }
        }
    }

    /// 收尾当前 generation：把累积的 output / thinking / error / endTime 一次性
    /// flush 成 generation-update。无进行中 generation 时是 no-op。
    ///
    /// 在“下一次 LlmCallStarted”和“TurnEnded”两处调用——保证 update 永远在
    /// 对应的 create 之后、且数据已流式到齐。
    fn flush_generation(
        &mut self,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        let Some(pg) = turn.current_gen.take() else {
            return Vec::new();
        };

        let mut meta = serde_json::Map::new();
        if !pg.thinking.is_empty() {
            // thinking/reasoning 没有 ingestion 专用字段——放 metadata，不污染 output。
            meta.insert("reasoning".into(), serde_json::Value::String(pg.thinking));
        }

        let body = ObservationBody {
            id: pg.id,
            trace_id: turn.trace_id.clone(),
            model: Some(pg.model),
            end_time: Some(now.to_string()),
            output: (!pg.output.is_empty()).then_some(serde_json::Value::String(pg.output)),
            // 本次调用的 usage（非 turn 累计）——这才是单次 generation 的真用量。
            usage_details: usage_to_details(&pg.usage),
            metadata: (!meta.is_empty()).then_some(serde_json::Value::Object(meta)),
            level: pg.error.as_ref().map(|_| ObservationLevel::Error),
            status_message: pg.error,
            ..Default::default()
        };
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

        let failed = matches!(fields.status, Some(ToolCallStatus::Failed));
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
        // 先收尾仍进行中的 generation（其 update 须在 trace 收尾前发出）。
        let mut events = self.flush_generation(now, new_id);

        let Some(turn) = self.turn.take() else {
            return events;
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
        events.push(IngestionEvent::trace(
            new_id(),
            now.to_string(),
            // 同 trace_id 二次发送 = 更新（合并 endTime/output/metadata）。
            EventKind::TraceCreate,
            &body,
        ));
        events
    }

    /// 处理一个 subagent 子 turn 事件（`AgentEvent::Subagent` 解包后的 `inner`）。
    ///
    /// 把子 turn 的 generation / 工具 span 投成挂在父 `spawn_agent` 工具 span
    /// （`parent_observation_id`）之下、复用父 trace_id 的嵌套 observation。id
    /// 命名空间用父 tool_call_id 隔开，避免与父 / 兄弟 subagent 撞 id。
    ///
    /// 找不到父 tool span（父 trace 已结束 / 后台 subagent 无张开的父 span）时丢弃——
    /// 不凭空造孤儿 observation。
    fn on_subagent(
        &mut self,
        parent_tool_call_id: &str,
        agent_type: String,
        inner: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(turn) = self.turn.as_mut() else {
            return Vec::new();
        };
        // 父 tool span 必须已存在（ToolCallStarted 先于子事件串行发出）。缺失则
        // 说明父 span 已收尾或从未建立——丢弃子事件，不造孤儿。
        let Some(parent_span_id) = turn.tool_spans.get(parent_tool_call_id).cloned() else {
            return Vec::new();
        };
        let trace_id = turn.trace_id.clone();
        let sub = turn
            .subagents
            .entry(parent_tool_call_id.to_string())
            .or_insert_with(|| SubagentState::new(parent_span_id.clone(), agent_type));

        match inner {
            AgentEvent::LlmCallStarted { model, attempt, .. } => {
                let mut events = flush_sub_generation(&trace_id, sub, now, new_id);
                sub.call_seq += 1;
                let gen_id =
                    format!("{trace_id}-sub-{parent_tool_call_id}-gen-{}", sub.call_seq);
                sub.current_gen = Some(PendingGeneration {
                    id: gen_id.clone(),
                    model: model.clone(),
                    output: String::new(),
                    thinking: String::new(),
                    usage: Usage::default(),
                    error: None,
                });
                let mut meta = serde_json::Map::new();
                meta.insert("attempt".into(), attempt.into());
                meta.insert("agent_type".into(), sub.agent_type.clone().into());
                let body = ObservationBody {
                    id: gen_id,
                    trace_id,
                    parent_observation_id: Some(sub.parent_span_id.clone()),
                    name: Some("llm_call".into()),
                    model: Some(model),
                    start_time: Some(now.to_string()),
                    metadata: Some(serde_json::Value::Object(meta)),
                    environment: Some(DEFAULT_ENVIRONMENT.into()),
                    ..Default::default()
                };
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::GenerationCreate,
                    &body,
                ));
                events
            }
            AgentEvent::AssistantText { content } => {
                if let (ContentBlock::Text(text), Some(pg)) =
                    (&content, sub.current_gen.as_mut())
                {
                    pg.output.push_str(&text.text);
                }
                Vec::new()
            }
            AgentEvent::AssistantThought { content } => {
                if let (ContentBlock::Text(text), Some(pg)) =
                    (&content, sub.current_gen.as_mut())
                {
                    pg.thinking.push_str(&text.text);
                }
                Vec::new()
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                if let Some(pg) = sub.current_gen.as_mut() {
                    pg.usage = usage;
                    if error.is_some() {
                        pg.error = error;
                    }
                }
                Vec::new()
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                let tool_id = id.to_string();
                let span_id =
                    format!("{trace_id}-sub-{parent_tool_call_id}-tool-{tool_id}");
                sub.tool_spans.insert(tool_id, span_id.clone());
                let body = ObservationBody {
                    id: span_id,
                    trace_id,
                    parent_observation_id: Some(sub.parent_span_id.clone()),
                    name: Some(name),
                    start_time: Some(now.to_string()),
                    input: fields.raw_input,
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
            AgentEvent::ToolCallFinished { id, fields } => {
                let tool_id = id.to_string();
                let span_id = sub.tool_spans.remove(&tool_id).unwrap_or_else(|| {
                    format!("{trace_id}-sub-{parent_tool_call_id}-tool-{tool_id}")
                });
                let failed = matches!(fields.status, Some(ToolCallStatus::Failed));
                let body = ObservationBody {
                    id: span_id,
                    trace_id,
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
            // 子 turn 结束：收尾仍进行中的子 generation，清掉子状态。
            AgentEvent::TurnEnded { .. } => {
                let events = flush_sub_generation(&trace_id, sub, now, new_id);
                turn.subagents.remove(parent_tool_call_id);
                events
            }
            // 子 turn 的其余事件（TurnStarted / UserPromptCommitted / 进度 / 审计 /
            // 嵌套 Subagent——结构性不会发生）不单独上报。
            _ => Vec::new(),
        }
    }
}

/// 收尾一个 subagent 子 generation：把累积 output / thinking / usage / endTime
/// flush 成 generation-update。无进行中 generation 时 no-op。与父级 `flush_generation`
/// 同形，但写 `parent_observation_id` 让 update 也挂在父 tool span 下。
fn flush_sub_generation(
    trace_id: &str,
    sub: &mut SubagentState,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let Some(pg) = sub.current_gen.take() else {
        return Vec::new();
    };
    let mut meta = serde_json::Map::new();
    if !pg.thinking.is_empty() {
        meta.insert("reasoning".into(), serde_json::Value::String(pg.thinking));
    }
    let body = ObservationBody {
        id: pg.id,
        trace_id: trace_id.to_string(),
        parent_observation_id: Some(sub.parent_span_id.clone()),
        model: Some(pg.model),
        end_time: Some(now.to_string()),
        output: (!pg.output.is_empty()).then_some(serde_json::Value::String(pg.output)),
        usage_details: usage_to_details(&pg.usage),
        metadata: (!meta.is_empty()).then_some(serde_json::Value::Object(meta)),
        level: pg.error.as_ref().map(|_| ObservationLevel::Error),
        status_message: pg.error,
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::GenerationUpdate,
        &body,
    )]
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

/// 把请求快照还原成 langfuse generation 的标准 `input`：chat messages 数组。
///
/// system prompt 作为第一条 `{role:"system"}`，随后是完整 messages 历史。
/// 这是 Langfuse SDK 的标准格式（见 observation-types 文档）——UI 能渲染成
/// 对话气泡、支持 playground 重放。
fn request_to_input(request: &LlmRequestSnapshot) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = &request.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    for msg in &request.messages {
        messages.push(message_to_value(msg));
    }
    serde_json::Value::Array(messages)
}

/// 单条 [`Message`] → langfuse `{role, content}`。content 把多模态块降级成
/// 文本 / 结构化片段（langfuse input 接受任意 JSON，UI 尽力渲染）。
fn message_to_value(msg: &Message) -> serde_json::Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let parts: Vec<serde_json::Value> = msg.content.iter().map(content_to_value).collect();
    // 单条纯文本时直接用字符串 content（最常见、最易读）；否则用数组。
    let content = match parts.as_slice() {
        [serde_json::Value::String(s)] => serde_json::Value::String(s.clone()),
        _ => serde_json::Value::Array(parts),
    };
    serde_json::json!({ "role": role, "content": content })
}

/// [`MessageContent`] → langfuse content 片段。
fn content_to_value(content: &MessageContent) -> serde_json::Value {
    match content {
        MessageContent::Text { text } => serde_json::Value::String(text.clone()),
        MessageContent::Thinking { text, .. } => {
            serde_json::json!({ "type": "thinking", "text": text })
        }
        MessageContent::ToolUse { id, name, args } => {
            serde_json::json!({ "type": "tool_use", "id": id, "name": name, "input": args })
        }
        MessageContent::ToolResult {
            tool_use_id,
            is_error,
            ..
        } => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }),
        MessageContent::Image { mime, .. } => {
            serde_json::json!({ "type": "image", "mime": mime })
        }
        MessageContent::ProviderActivity {
            provider_id, kind, ..
        } => serde_json::json!({
            "type": "provider_activity",
            "provider_id": provider_id,
            "kind": format!("{kind:?}"),
        }),
        // MessageContent 是 #[non_exhaustive]：未来新增的块降级成一个标记。
        _ => serde_json::json!({ "type": "unknown" }),
    }
}
