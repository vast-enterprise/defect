//! `AgentEvent` → Langfuse ingestion 事件的翻译。
//!
//! [`TraceProjector`] 是**有状态、逐 session** 的投影器（与 `defect-storage` 的
//! `RecordProjector` 同构）。主循环每收到一个 [`AgentEvent`] 调一次
//! [`TraceProjector::project`]，拿回 0..N 个 [`IngestionEvent`] 交给上报器。
//!
//! ## 层级（每个 turn 一个 trace）
//!
//! ```text
//! trace (turn)
//! └── step (span)                 每轮一个：一次 llm_call + 它触发的工具
//!     ├── llm_call (generation)
//!     └── tool (span)             与 llm_call 同为 step 的子节点（兄弟）
//!         └── (spawn_agent) → subagent (span)
//!             └── step (span)     子 agent 内同构递归
//!                 ├── llm_call
//!                 └── tool / spawn_agent → subagent → ...
//! ```
//!
//! 关键：**step span** 是每轮的容器，`llm_call`（generation）与该轮触发的工具
//! 都挂在它下面，互为兄弟。这样 generation 的时长回归**纯 LLM 调用**（在
//! `LlmCallFinished` 收尾，不再延迟到下轮、不再包住工具执行时间）；工具时长落在
//! step 里。
//!
//! ## 递归 subagent（扁平化 ancestor_path）
//!
//! [`AgentEvent::Subagent`] 携带一条 `ancestor_path`（顶层 `spawn_agent` 工具调用到
//! 当前层的 `ToolCallId` 链），`inner` 永远是叶子事件。projector 用这条链**确定性**
//! 派生所有 span id，故无需逐层 anchor，任意深度同构处理。每个 subagent 层是一个
//! 独立 **scope**（与顶层 turn 共用同一套 step/gen/tool 投影逻辑）。
//!
//! ## id 策略
//!
//! - **traceId**：`TurnStarted` 时生成一次 `Uuid::new_v4()`，turn 内复用。
//!   **不可**用 `{session}-turn-{seq}` 自增——resume 后会撞 id（见设计文档 §3.5）。
//! - **scope 前缀**：顶层 = `{trace}`；subagent 路径 `[A,B]` = `{trace}-sub-A-sub-B`。
//!   subagent span 的 id **就是**它的 scope 前缀。
//! - **step / generation / tool span id**：派生自 scope 前缀 + 序号 / `ToolCallId`，
//!   全局唯一且确定性——subagent span 的父（发起它的 tool span）由路径直接算出。
//! - **anchor**：唯一需要存的状态是**顶层 `spawn_agent` 工具调用 id → trace_id**
//!   （trace_id 随机不可推导）。同一 turn 的所有 subagent 共享该 trace_id，故只锚顶层。
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
/// 每轮（一次 llm_call + 它触发的工具）的容器 span 名称。
const STEP_NAME: &str = "step";
/// LLM 调用对应的 Langfuse generation 名称。
const GENERATION_NAME: &str = "llm_call";
/// `spawn_agent` 工具名（wire 上的字符串）。真相源是
/// `defect_agent::tool::spawn_agent::SPAWN_AGENT_TOOL_NAME`（`pub(crate)` 不可跨 crate 引），
/// 这里按 wire 名复制一份——projector 据它把**顶层** spawn_agent 工具调用锚定到 trace_id。
const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";
/// subagent 独立 span 的名称前缀（与发起它的工具 span 分开的那一层）。
const SUBAGENT_SPAN_NAME: &str = "subagent";

/// 逐 session 的投影状态。
pub struct TraceProjector {
    session_id: String,
    /// 当前**顶层** turn 的元信息；`None` 表示不在 turn 内（TurnStarted 之前 /
    /// TurnEnded 之后）。注意 subagent 的事件可能在顶层 turn 结束后仍到达（后台），
    /// 那时 `turn` 已 `None`，但 subagent scope 仍存活、trace_id 经 [`Self::anchors`] 取回。
    turn: Option<TurnMeta>,
    /// 暂存的用户 prompt 文本。主循环**先发 `UserPromptCommitted` 再发 `TurnStarted`**，
    /// 所以收到 prompt 时 turn 还没建——先存这里，`TurnStarted` 建 turn 时取走。
    pending_input: Option<String>,
    /// **顶层** `spawn_agent` 工具调用 id → 其所属 trace_id。subagent 事件据
    /// `ancestor_path[0]` 查它取回 trace_id（trace_id 随机不可推导；同一 turn 的所有
    /// 嵌套 subagent 共享该 trace_id，故只锚顶层这一跳）。subagent（路径长度 1）结束时清除。
    anchors: HashMap<String, String>,
    /// 所有进行中的 scope：`scope 前缀` → 状态。**session 级**——顶层 turn scope（前缀
    /// = trace_id）与各 subagent scope（前缀 = `{trace}-sub-...`）共存。subagent scope
    /// 可能跨 turn 边界存活（后台），故不随 turn 清空，各自在对应 `TurnEnded` 时移除。
    scopes: HashMap<String, ScopeState>,
}

/// 当前顶层 turn 的元信息（trace 级，不含 step/gen/tool 投影状态——那些在
/// `scopes[trace_id]` 这个 scope 里，与 subagent scope 同构）。
struct TurnMeta {
    trace_id: String,
    /// 用户 prompt 文本，写进 trace input。
    input: Option<String>,
    /// 整个 turn 的最终助手文本（写进 trace output）。
    final_output: String,
}

/// 一个 scope（顶层 turn 或某 subagent 层）的 step/generation/tool 投影状态。
///
/// 顶层与 subagent 共用本结构——这正是"subagent 不过是有亲代的 agent"在 observability
/// 侧的体现：同一套 step 容器 + generation + tool span 逻辑，仅挂载点（`step_parent`）
/// 与 id 前缀（`prefix`）不同。
struct ScopeState {
    /// id 派生前缀：顶层 = `{trace}`；subagent = `{trace}-sub-A-sub-B`。
    /// subagent scope 的 `prefix` 同时**就是**其 subagent span 的 id。
    prefix: String,
    /// 本 scope 里 step span 的父 observation：顶层 = `None`（直接挂 trace）；
    /// subagent = `Some(subagent span id)` = `Some(prefix)`。
    step_parent: Option<String>,
    /// 当前进行中的 step span id（`None` = 尚无 llm_call）。
    current_step_id: Option<String>,
    /// 第几个 step——派生 step id。
    step_seq: u32,
    /// 当前 step 里进行中的 generation。
    current_gen: Option<PendingGeneration>,
    /// 工具调用 id → 已分配的 span id（Started/Finished 跨事件配对）。
    tool_spans: HashMap<String, String>,
}

/// 进行中的 generation 累积状态。收尾（`LlmCallFinished`）时一次性 flush 成
/// generation-update。
struct PendingGeneration {
    id: String,
    parent_step_id: String,
    model: String,
    /// 累积的助手回复正文。
    output: String,
    /// 累积的 thinking 文本（放进 generation 的 metadata.reasoning，不进 output）。
    thinking: String,
    /// 本次调用的 token 用量（来自 LlmCallFinished.usage）。
    usage: Usage,
    /// 失败信息（来自 LlmCallFinished.error）。
    error: Option<String>,
}

impl ScopeState {
    fn new(prefix: String, step_parent: Option<String>) -> Self {
        Self {
            prefix,
            step_parent,
            current_step_id: None,
            step_seq: 0,
            current_gen: None,
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
            anchors: HashMap::new(),
            scopes: HashMap::new(),
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
            } => self.on_top_llm_started(model, attempt, request.as_ref(), now, new_id),
            AgentEvent::AssistantText { content } => {
                self.accumulate_top_text(&content);
                Vec::new()
            }
            AgentEvent::AssistantThought { content } => {
                self.accumulate_top_thinking(&content);
                Vec::new()
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                self.on_top_llm_finished(usage, error, now, new_id)
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                self.on_top_tool_started(id.to_string(), name, fields.raw_input, now, new_id)
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                self.on_top_tool_finished(&id.to_string(), &fields, now, new_id)
            }
            AgentEvent::ContextCompressed {
                tokens_before,
                tokens_after,
            } => self.on_context_compressed(tokens_before, tokens_after, None, now, new_id),
            AgentEvent::ContextMicrocompacted {
                tokens_before,
                tokens_after,
                cleared,
            } => {
                self.on_context_compressed(tokens_before, tokens_after, Some(cleared), now, new_id)
            }
            AgentEvent::TurnEnded { reason, usage } => {
                self.on_turn_ended(reason, usage, now, new_id)
            }
            AgentEvent::Subagent {
                ancestor_path,
                agent_type,
                inner,
            } => {
                let path: Vec<String> = ancestor_path.iter().map(ToString::to_string).collect();
                self.on_subagent(&path, agent_type, *inner, now, new_id)
            }
            // 不上报：进度增量、权限审计（本期不入 langfuse）。
            AgentEvent::ToolCallProgress { .. }
            | AgentEvent::PolicyDecision { .. }
            | AgentEvent::PermissionResolved { .. } => Vec::new(),
            _ => Vec::new(),
        }
    }

    // ---- 顶层 turn 事件 ----

    fn on_turn_started(
        &mut self,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let trace_id = new_id();
        let input = self.pending_input.take();
        let body = TraceBody {
            id: trace_id.clone(),
            name: Some(TRACE_NAME.into()),
            session_id: Some(self.session_id.clone()),
            // trace-create 时就带上 input，UI 立刻能看到用户输入（不必等 TurnEnded）。
            input: input.clone().map(serde_json::Value::String),
            environment: Some(DEFAULT_ENVIRONMENT.into()),
            timestamp: Some(now.to_string()),
            ..Default::default()
        };
        // 顶层 scope：前缀 = trace_id，step 直接挂 trace（step_parent = None）。
        self.scopes
            .insert(trace_id.clone(), ScopeState::new(trace_id.clone(), None));
        self.turn = Some(TurnMeta {
            trace_id: trace_id.clone(),
            input,
            final_output: String::new(),
        });
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
            self.pending_input = Some(text);
        }
    }

    fn on_top_llm_started(
        &mut self,
        model: String,
        attempt: u32,
        request: &LlmRequestSnapshot,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_llm_started(scope, &trace_id, model, attempt, request, now, new_id)
    }

    fn accumulate_top_text(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(turn) = self.turn.as_mut()
        {
            turn.final_output.push_str(&text.text);
            let trace_id = turn.trace_id.clone();
            if let Some(scope) = self.scopes.get_mut(&trace_id)
                && let Some(pg) = scope.current_gen.as_mut()
            {
                pg.output.push_str(&text.text);
            }
        }
    }

    fn accumulate_top_thinking(&mut self, content: &ContentBlock) {
        if let ContentBlock::Text(text) = content
            && let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone())
            && let Some(scope) = self.scopes.get_mut(&trace_id)
            && let Some(pg) = scope.current_gen.as_mut()
        {
            pg.thinking.push_str(&text.text);
        }
    }

    fn on_top_llm_finished(
        &mut self,
        usage: Usage,
        error: Option<String>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        note_llm_finished(scope, usage, error);
        flush_generation(scope, &trace_id, now, new_id)
    }

    fn on_top_tool_started(
        &mut self,
        tool_call_id: String,
        name: String,
        raw_input: Option<serde_json::Value>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        // 顶层 spawn_agent 工具调用：锚定 trace_id，供后续（含后台、跨 turn）subagent
        // 事件经 ancestor_path[0] 取回 trace_id。
        if name == SPAWN_AGENT_TOOL_NAME {
            self.anchors.insert(tool_call_id.clone(), trace_id.clone());
        }
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_tool_started(
            scope,
            &trace_id,
            &tool_call_id,
            name,
            raw_input,
            now,
            new_id,
        )
    }

    fn on_top_tool_finished(
        &mut self,
        tool_call_id: &str,
        fields: &ToolCallUpdateFields,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let Some(scope) = self.scopes.get_mut(&trace_id) else {
            return Vec::new();
        };
        scope_tool_finished(scope, &trace_id, tool_call_id, fields, now, new_id)
    }

    /// `cleared` 为 `Some` 表示微压缩（清理 tool_result，无 LLM）；`None` 表示全量
    /// 摘要压缩。两者投成同形观测，仅 name/metadata 区分。压缩是 turn 级、跨 step 的
    /// 操作（不属某次 llm_call），故直接挂 trace（无 parent）。
    fn on_context_compressed(
        &mut self,
        tokens_before: u64,
        tokens_after: u64,
        cleared: Option<usize>,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(trace_id) = self.turn.as_ref().map(|t| t.trace_id.clone()) else {
            return Vec::new();
        };
        let mut meta = serde_json::Map::new();
        meta.insert("tokens_before".into(), tokens_before.into());
        meta.insert("tokens_after".into(), tokens_after.into());
        if let Some(cleared) = cleared {
            meta.insert("cleared_tool_results".into(), cleared.into());
        }
        let name = if cleared.is_some() {
            "context_microcompaction"
        } else {
            "context_compaction"
        };
        let body = ObservationBody {
            id: new_id(),
            trace_id,
            name: Some(name.into()),
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
        let trace_id = turn.trace_id.clone();
        let mut events = Vec::new();
        // 收尾顶层 scope：flush 仍进行中的 generation + 关闭当前 step，然后移除 scope。
        if let Some(mut scope) = self.scopes.remove(&trace_id) {
            events.extend(flush_generation(&mut scope, &trace_id, now, new_id));
            events.extend(close_current_step(&mut scope, &trace_id, now, new_id));
        }

        let mut meta = serde_json::Map::new();
        meta.insert(
            "stop_reason".into(),
            serde_json::to_value(reason).unwrap_or(serde_json::Value::Null),
        );
        if let Some(details) = usage_to_details(&usage) {
            meta.insert("usage".into(), serde_json::Value::Object(details));
        }
        let body = TraceBody {
            id: trace_id,
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

    // ---- subagent 事件（任意深度，同构）----

    /// 处理一个 subagent 子 turn 的**叶子**事件（`AgentEvent::Subagent` 解包后的 `inner`）。
    ///
    /// `path` = 从顶层 `spawn_agent` 工具调用到当前层的 `ToolCallId` 链。trace_id 经
    /// `anchors[path[0]]` 取回；scope 前缀与 subagent span 的父（发起它的 tool span）由
    /// `path` 确定性派生。首次见到某 path 时懒创建其 subagent span + scope；之后把 inner
    /// 投到该 scope（与顶层共用 step/gen/tool 逻辑）。
    ///
    /// 找不到顶层 anchor（从未见过那个 spawn_agent 工具调用）时丢弃——不造孤儿。
    fn on_subagent(
        &mut self,
        path: &[String],
        agent_type: String,
        inner: AgentEvent,
        now: &str,
        new_id: &mut dyn FnMut() -> String,
    ) -> Vec<IngestionEvent> {
        let Some(first) = path.first() else {
            return Vec::new();
        };
        let Some(trace_id) = self.anchors.get(first).cloned() else {
            // 没有顶层锚点：从未见过这个 spawn_agent 工具调用——丢弃，不造孤儿。
            return Vec::new();
        };
        let prefix = scope_prefix(&trace_id, path);

        let mut events = Vec::new();
        // 首次见到该 path：懒创建独立 subagent span（父 = 发起它的 tool span，由 path 派生）。
        if !self.scopes.contains_key(&prefix) {
            let parent_tool = parent_tool_span_id(&trace_id, path);
            let mut meta = serde_json::Map::new();
            meta.insert("agent_type".into(), agent_type.clone().into());
            let body = ObservationBody {
                id: prefix.clone(),
                trace_id: trace_id.clone(),
                parent_observation_id: Some(parent_tool),
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
            // subagent scope：step 挂在这个 subagent span（= prefix）下。
            self.scopes.insert(
                prefix.clone(),
                ScopeState::new(prefix.clone(), Some(prefix.clone())),
            );
        }
        let scope = self
            .scopes
            .get_mut(&prefix)
            .expect("subagent scope just ensured");

        match inner {
            AgentEvent::LlmCallStarted {
                model,
                attempt,
                request,
            } => {
                events.extend(scope_llm_started(
                    scope,
                    &trace_id,
                    model,
                    attempt,
                    request.as_ref(),
                    now,
                    new_id,
                ));
            }
            AgentEvent::AssistantText { content } => {
                if let (ContentBlock::Text(text), Some(pg)) = (&content, scope.current_gen.as_mut())
                {
                    pg.output.push_str(&text.text);
                }
            }
            AgentEvent::AssistantThought { content } => {
                if let (ContentBlock::Text(text), Some(pg)) = (&content, scope.current_gen.as_mut())
                {
                    pg.thinking.push_str(&text.text);
                }
            }
            AgentEvent::LlmCallFinished { usage, error, .. } => {
                note_llm_finished(scope, usage, error);
                events.extend(flush_generation(scope, &trace_id, now, new_id));
            }
            AgentEvent::ToolCallStarted { id, name, fields } => {
                events.extend(scope_tool_started(
                    scope,
                    &trace_id,
                    &id.to_string(),
                    name,
                    fields.raw_input,
                    now,
                    new_id,
                ));
            }
            AgentEvent::ToolCallFinished { id, fields } => {
                events.extend(scope_tool_finished(
                    scope,
                    &trace_id,
                    &id.to_string(),
                    &fields,
                    now,
                    new_id,
                ));
            }
            // 子 turn 结束：收尾进行中的 generation + 关闭当前 step + 关闭 subagent span，
            // 清掉 session 级 scope；顶层那一跳（path 长度 1）的 anchor 也清掉。
            AgentEvent::TurnEnded { .. } => {
                events.extend(flush_generation(scope, &trace_id, now, new_id));
                events.extend(close_current_step(scope, &trace_id, now, new_id));
                let subagent_span_id = scope.prefix.clone();
                let body = ObservationBody {
                    id: subagent_span_id,
                    trace_id: trace_id.clone(),
                    end_time: Some(now.to_string()),
                    ..Default::default()
                };
                events.push(IngestionEvent::observation(
                    new_id(),
                    now.to_string(),
                    EventKind::SpanUpdate,
                    &body,
                ));
                self.scopes.remove(&prefix);
                if path.len() == 1 {
                    self.anchors.remove(first);
                }
            }
            // 子 turn 的其余事件（TurnStarted / UserPromptCommitted / 进度 / 审计）不单独上报。
            _ => {}
        }
        events
    }
}

// ---- scope 通用投影（顶层 turn 与 subagent 共用）----

/// 一次 LLM 调用开始：收尾上一个 step（若有）→ 开新 step → 在新 step 下建 generation。
fn scope_llm_started(
    scope: &mut ScopeState,
    trace_id: &str,
    model: String,
    attempt: u32,
    request: &LlmRequestSnapshot,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    // 防御：上一个 generation 理应已在它的 LlmCallFinished 收尾（gen 时长 = 纯 LLM）；
    // 若仍在则先 flush，保证 create 先于 update。
    let mut events = flush_generation(scope, trace_id, now, new_id);
    // 收尾上一个 step（它含上一次 llm_call + 那轮触发的工具）。
    events.extend(close_current_step(scope, trace_id, now, new_id));

    // 开新 step。
    scope.step_seq += 1;
    let step_id = format!("{}-step-{}", scope.prefix, scope.step_seq);
    scope.current_step_id = Some(step_id.clone());
    let step_body = ObservationBody {
        id: step_id.clone(),
        trace_id: trace_id.to_string(),
        parent_observation_id: scope.step_parent.clone(),
        name: Some(STEP_NAME.into()),
        start_time: Some(now.to_string()),
        environment: Some(DEFAULT_ENVIRONMENT.into()),
        ..Default::default()
    };
    events.push(IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanCreate,
        &step_body,
    ));

    // generation 挂在新 step 下。
    let gen_id = format!("{step_id}-gen");
    scope.current_gen = Some(PendingGeneration {
        id: gen_id.clone(),
        parent_step_id: step_id.clone(),
        model: model.clone(),
        output: String::new(),
        thinking: String::new(),
        usage: Usage::default(),
        error: None,
    });
    let mut meta = serde_json::Map::new();
    meta.insert("attempt".into(), attempt.into());
    let gen_body = ObservationBody {
        id: gen_id,
        trace_id: trace_id.to_string(),
        parent_observation_id: Some(step_id),
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
        &gen_body,
    ));
    events
}

/// 记录 LlmCallFinished 的 usage / error 到当前 generation（收尾时写出）。
fn note_llm_finished(scope: &mut ScopeState, usage: Usage, error: Option<String>) {
    if let Some(pg) = scope.current_gen.as_mut() {
        pg.usage = usage;
        if error.is_some() {
            pg.error = error;
        }
    }
}

/// 收尾当前 generation：output / thinking / usage / endTime → generation-update。
/// 无进行中 generation 时 no-op。在 `LlmCallFinished`（成功路径流已 drain，output/
/// thinking 已到齐）调用——generation 时长 = 纯 LLM 调用，不含工具执行。
fn flush_generation(
    scope: &mut ScopeState,
    trace_id: &str,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let Some(pg) = scope.current_gen.take() else {
        return Vec::new();
    };
    let mut meta = serde_json::Map::new();
    if !pg.thinking.is_empty() {
        // thinking/reasoning 没有 ingestion 专用字段——放 metadata，不污染 output。
        meta.insert("reasoning".into(), serde_json::Value::String(pg.thinking));
    }
    let body = ObservationBody {
        id: pg.id,
        trace_id: trace_id.to_string(),
        parent_observation_id: Some(pg.parent_step_id),
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

/// 关闭当前 step span（写 end_time）。无进行中 step 时 no-op。
fn close_current_step(
    scope: &mut ScopeState,
    trace_id: &str,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let Some(step_id) = scope.current_step_id.take() else {
        return Vec::new();
    };
    let body = ObservationBody {
        id: step_id,
        trace_id: trace_id.to_string(),
        end_time: Some(now.to_string()),
        ..Default::default()
    };
    vec![IngestionEvent::observation(
        new_id(),
        now.to_string(),
        EventKind::SpanUpdate,
        &body,
    )]
}

/// 工具调用开始 → span-create，挂在当前 step 下（与 llm_call 互为兄弟）。
fn scope_tool_started(
    scope: &mut ScopeState,
    trace_id: &str,
    tool_call_id: &str,
    name: String,
    raw_input: Option<serde_json::Value>,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    let span_id = format!("{}-tool-{}", scope.prefix, tool_call_id);
    scope
        .tool_spans
        .insert(tool_call_id.to_string(), span_id.clone());
    let body = ObservationBody {
        id: span_id,
        trace_id: trace_id.to_string(),
        // 工具挂在当前 step 下；理论上工具调用恒在某次 llm_call 之后，故 step 必存在。
        // 防御性地允许 None（乱序 / 无 step）——退化为直接挂 trace。
        parent_observation_id: scope.current_step_id.clone(),
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

/// 工具调用结束 → span-update（endTime + output + level）。
fn scope_tool_finished(
    scope: &mut ScopeState,
    trace_id: &str,
    tool_call_id: &str,
    fields: &ToolCallUpdateFields,
    now: &str,
    new_id: &mut dyn FnMut() -> String,
) -> Vec<IngestionEvent> {
    // 取回 Started 时分配的 span id；缺失（乱序）则现派生一个。
    let span_id = scope
        .tool_spans
        .remove(tool_call_id)
        .unwrap_or_else(|| format!("{}-tool-{}", scope.prefix, tool_call_id));
    let failed = matches!(fields.status, Some(ToolCallStatus::Failed));
    let body = ObservationBody {
        id: span_id,
        trace_id: trace_id.to_string(),
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

// ---- id 派生 ----

/// 某 scope 的 id 前缀：顶层（path 空）= `{trace}`；subagent 路径 `[A,B]` =
/// `{trace}-sub-A-sub-B`。subagent scope 的前缀同时就是其 subagent span 的 id。
fn scope_prefix(trace_id: &str, path: &[String]) -> String {
    let mut s = trace_id.to_string();
    for id in path {
        s.push_str("-sub-");
        s.push_str(id);
    }
    s
}

/// 一个 subagent span 的父 observation（发起它的 `spawn_agent` 工具 span）id。
/// = `{父 scope 前缀}-tool-{该 subagent 的发起 tool_call_id}`。`path` 非空。
fn parent_tool_span_id(trace_id: &str, path: &[String]) -> String {
    let (last, parent_path) = path.split_last().expect("path is non-empty");
    format!("{}-tool-{}", scope_prefix(trace_id, parent_path), last)
}

// ---- 数据转换 helper ----

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
