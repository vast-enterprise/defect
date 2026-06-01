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
/// 每个 agent turn 对应的 Langfuse trace 名称。
const TRACE_NAME: &str = "turn";
/// LLM 调用对应的 Langfuse generation 名称。
const GENERATION_NAME: &str = "llm_call";
/// `spawn_agent` 工具名（wire 上的字符串）。真相源是
/// `defect_agent::tool::spawn_agent::SPAWN_AGENT_TOOL_NAME`（`pub(crate)` 不可跨 crate 引），
/// 这里按 wire 名复制一份——projector 据它把 spawn_agent 工具调用登记成 subagent 锚点。
const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";
/// subagent 独立 span 的名称前缀（与发起它的工具 span 分开的那一层）。
const SUBAGENT_SPAN_NAME: &str = "subagent";

/// 一个 `spawn_agent` 工具调用的 langfuse 坐标——**session 级、跨 turn 存活**。
///
/// 前台 subagent 的子事件在父 turn trace 内到达（`turn.tool_spans` 也有），但**后台**
/// subagent 的子事件在**发起 turn 结束之后**才到，那时 `turn` 已被覆盖。锚点把
/// `(trace_id, tool_span_id)` 留住，让后台子事件仍能把 subagent span 挂回原 tool span 下。
#[derive(Clone)]
struct SubagentAnchor {
    trace_id: String,
    /// 发起它的 `spawn_agent` 工具 span id——subagent span 的 `parent_observation_id`。
    tool_span_id: String,
}

/// 逐 session 的投影状态。
pub struct TraceProjector {
    session_id: String,
    /// 当前 turn 的状态；`None` 表示不在 turn 内（TurnStarted 之前）。
    turn: Option<TurnState>,
    /// 暂存的用户 prompt 文本。主循环**先发 `UserPromptCommitted` 再发
    /// `TurnStarted`**（turn.rs:197 先于 :212），所以收到 prompt 时 turn 还没建——
    /// 先存这里，`TurnStarted` 建 `TurnState` 时再取进去。
    pending_input: Option<String>,
    /// `spawn_agent` 工具调用 id → 锚点。`on_tool_started` 见到 spawn_agent 时登记，
    /// subagent 子 turn `TurnEnded` 时清除。session 级，使后台子事件跨 turn 也能找回父。
    subagent_anchors: HashMap<String, SubagentAnchor>,
    /// 进行中的 subagent 投影状态：`spawn_agent` 工具调用 id → 状态。**session 级**
    /// （非 turn 级）——后台 subagent 的事件可能跨 turn 边界陆续到达。
    subagents: HashMap<String, SubagentState>,
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
}

/// 一个 subagent 子 turn 的投影状态。
///
/// 关键：subagent 有**自己独立的 span**（`subagent_span_id`），它是发起它的
/// `spawn_agent` 工具 span 的子节点；子 turn 的 generation / 工具 span 再挂在这个
/// subagent span 之下。这样工具 span 与 subagent span 解耦——
/// - **前台**：工具 span 全程张开，subagent span 嵌在其内；
/// - **后台**：工具 span 早早正常关闭（"已启动"），subagent span 作为它的**相邻子节点**
///   自行张开到子 turn 真正结束，时间轴各自真实、不互相牵连。
///
/// id 命名空间用父 tool_call_id 隔开，避免与父 / 兄弟 subagent 撞 id。
struct SubagentState {
    /// 本 subagent 自己的 span id——子 turn 的 generation / 工具 span 的 `parent_observation_id`。
    /// （trace_id 不存这里——每次 `on_subagent` 从 session 级锚点取，单一真相源。）
    subagent_span_id: String,
    /// 子 turn 第几次 LLM 调用——派生子 generation id。
    call_seq: u32,
    /// 子 turn 进行中的 generation。
    current_gen: Option<PendingGeneration>,
    /// 子 turn 的工具调用 id → span id。
    tool_spans: HashMap<String, String>,
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
            subagent_anchors: HashMap::new(),
            subagents: HashMap::new(),
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
            name: Some(TRACE_NAME.into()),
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
            name: Some(GENERATION_NAME.into()),
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
            name: Some(GENERATION_NAME.into()),
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
        turn.tool_spans.insert(tool_call_id.clone(), span_id.clone());

        // spawn_agent 工具：额外登记一个 session 级锚点。subagent 子事件（尤其**后台**，
        // 在发起 turn 结束后才到）据它把 subagent span 挂回这个工具 span 下。
        if name == SPAWN_AGENT_TOOL_NAME {
            self.subagent_anchors.insert(
                tool_call_id,
                SubagentAnchor {
                    trace_id: turn.trace_id.clone(),
                    tool_span_id: span_id.clone(),
                },
            );
        }

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
            name: Some(TRACE_NAME.into()),
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
    /// 工具 span 与 subagent span **分两层**（用户构想）：首次见到该 subagent 的事件时
    /// 懒创建一个独立 subagent span（父 = 发起它的 `spawn_agent` 工具 span，坐标取自
    /// session 级 `subagent_anchors`），子 turn 的 generation / 工具 span 再挂在 subagent
    /// span 之下。这样工具 span 可独立关闭：
    /// - **前台**：工具 span 全程张开，subagent span 嵌在其内（视觉嵌套）；
    /// - **后台**：工具 span 早早关闭（"已启动"），subagent span 作为它的相邻子节点自行
    ///   张开到子 turn `TurnEnded`，两者时间轴各自真实、互不牵连。
    ///
    /// 锚点是 session 级、跨 turn 存活，故后台子事件在发起 turn 结束后到达也能挂回。找不到
    /// 锚点（从未见过该 spawn_agent 工具调用）时丢弃——不造孤儿。
    fn on_subagent(
        &mut self,
        parent_tool_call_id: &str,
        agent_type: String,
        inner: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        // 锚点是 session 级、跨 turn 存活——前台（发起 turn 仍在）与后台（发起 turn 已结束）
        // 都从它找父 tool span 坐标。缺失说明从未见过这个 spawn_agent 工具调用——丢弃，不造孤儿。
        let Some(anchor) = self.subagent_anchors.get(parent_tool_call_id).cloned() else {
            return Vec::new();
        };
        let trace_id = anchor.trace_id.clone();

        // 首次见到该 subagent 的事件：懒创建一个**独立 subagent span**（父 = 工具 span）。
        // 前台时工具 span 仍张开（嵌套）；后台时工具 span 已关闭，本 span 作为其相邻子节点
        // 自行张开。无论哪种，子 turn 的 gen / tool 都挂在这个 subagent span 之下。
        let mut events = Vec::new();
        if !self.subagents.contains_key(parent_tool_call_id) {
            let subagent_span_id = format!("{trace_id}-sub-{parent_tool_call_id}");
            let mut meta = serde_json::Map::new();
            meta.insert("agent_type".into(), agent_type.clone().into());
            let body = ObservationBody {
                id: subagent_span_id.clone(),
                trace_id: trace_id.clone(),
                parent_observation_id: Some(anchor.tool_span_id.clone()),
                name: Some(format!("{SUBAGENT_SPAN_NAME}:{agent_type}")),
                start_time: Some(now.to_string()),
                metadata: Some(serde_json::Value::Object(meta)),
                environment: Some(DEFAULT_ENVIRONMENT.into()),
                ..Default::default()
            };
            events.push(IngestionEvent::observation(
                new_id(),
                now.to_string(),
                EventKind::SpanCreate,
                &body,
            ));
            self.subagents.insert(
                parent_tool_call_id.to_string(),
                SubagentState {
                    subagent_span_id,
                    call_seq: 0,
                    current_gen: None,
                    tool_spans: HashMap::new(),
                },
            );
        }
        let sub = self
            .subagents
            .get_mut(parent_tool_call_id)
            .expect("subagent state just ensured");

        match inner {
            AgentEvent::LlmCallStarted { model, attempt, .. } => {
                events.extend(flush_sub_generation(&trace_id, sub, now, new_id));
                sub.call_seq += 1;
                let gen_id = format!("{trace_id}-sub-{parent_tool_call_id}-gen-{}", sub.call_seq);
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
                let body = ObservationBody {
                    id: gen_id,
                    trace_id,
                    parent_observation_id: Some(sub.subagent_span_id.clone()),
                    name: Some(GENERATION_NAME.into()),
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
                if let (ContentBlock::Text(text), Some(pg)) = (&content, sub.current_gen.as_mut()) {
                    pg.output.push_str(&text.text);
                }
                events
            }
            AgentEvent::AssistantThought { content } => {
                if let (ContentBlock::Text(text), Some(pg)) = (&content, sub.current_gen.as_mut()) {
                    pg.thinking.push_str(&text.text);
                }
                events
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                if let Some(pg) = sub.current_gen.as_mut() {
                    pg.usage = usage;
                    if error.is_some() {
                        pg.error = error;
                    }
                }
                events
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                let tool_id = id.to_string();
                let span_id = format!("{trace_id}-sub-{parent_tool_call_id}-tool-{tool_id}");
                sub.tool_spans.insert(tool_id, span_id.clone());
                let body = ObservationBody {
                    id: span_id,
                    trace_id,
                    parent_observation_id: Some(sub.subagent_span_id.clone()),
                    name: Some(name),
                    start_time: Some(now.to_string()),
                    input: fields.raw_input,
                    environment: Some(DEFAULT_ENVIRONMENT.into()),
                    ..Default::default()
                };
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::SpanCreate,
                    &body,
                ));
                events
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
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::SpanUpdate,
                    &body,
                ));
                events
            }
            // 子 turn 结束：收尾进行中的子 generation，**关闭 subagent span**（写 end_time），
            // 清掉 session 级子状态与锚点。subagent span 的 end_time 标记子 turn 真正结束——
            // 后台时它晚于工具 span 的关闭，时间轴各自真实。
            AgentEvent::TurnEnded { .. } => {
                events.extend(flush_sub_generation(&trace_id, sub, now, new_id));
                let subagent_span_id = sub.subagent_span_id.clone();
                let body = ObservationBody {
                    id: subagent_span_id,
                    trace_id,
                    end_time: Some(now.to_string()),
                    ..Default::default()
                };
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::SpanUpdate,
                    &body,
                ));
                self.subagents.remove(parent_tool_call_id);
                self.subagent_anchors.remove(parent_tool_call_id);
                events
            }
            // 子 turn 的其余事件（TurnStarted / UserPromptCommitted / 进度 / 审计 /
            // 嵌套 Subagent——结构性不会发生）不单独上报。
            _ => events,
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
        parent_observation_id: Some(sub.subagent_span_id.clone()),
        name: Some(GENERATION_NAME.into()),
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
