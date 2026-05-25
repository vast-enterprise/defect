//! [`AgentEvent`] → ACP wire 形态的投影。
//!
//! 翻译表见 `docs/inbound/acp-bridge.md` §3。把内部事件分流为：
//! - [`Projection::Update`]：直接 `cx.send_notification(SessionNotification {..})`
//! - [`Projection::Permission`]：触发 `session/request_permission` 反向请求
//! - [`Projection::EndTurn`]：驱动 `PromptResponse::stop_reason`
//! - [`Projection::Ignore`]：仅审计，不入 wire

use agent_client_protocol::schema::{
    ContentChunk, PermissionOption, SessionId, SessionNotification, SessionUpdate, ToolCall,
    ToolCallId, ToolCallUpdate, ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::policy::PolicyDecision;

/// 一次投影的产物。
#[allow(clippy::large_enum_variant)]
pub(crate) enum Projection {
    /// 普通 `session/update` 通知。
    Update(SessionNotification),
    /// 需要发反向 `session/request_permission`。
    Permission(PermissionAsk),
    /// turn 终态——驱动 `PromptResponse`。事件本身仅作哨兵，权威 stop_reason
    /// 由 [`Session::run_turn`] 的返回值给出。
    EndTurn,
    /// 仅审计，不入 wire。
    Ignore,
}

/// 一次 `session/request_permission` 请求所需的上下文。
pub(crate) struct PermissionAsk {
    pub tool_call_id: ToolCallId,
    pub fields: ToolCallUpdateFields,
    pub options: Vec<PermissionOption>,
}

/// 翻译单条 [`AgentEvent`]。
pub(crate) fn project(session_id: &SessionId, event: AgentEvent) -> Projection {
    match event {
        AgentEvent::TurnStarted => Projection::Ignore,
        AgentEvent::TurnEnded { .. } => Projection::EndTurn,

        AgentEvent::AssistantText { content } => Projection::Update(notification(
            session_id,
            SessionUpdate::AgentMessageChunk(ContentChunk::new(content)),
        )),
        AgentEvent::AssistantThought { content } => Projection::Update(notification(
            session_id,
            SessionUpdate::AgentThoughtChunk(ContentChunk::new(content)),
        )),

        AgentEvent::ToolCallStarted { id, fields } => {
            let tool_call = tool_call_from_fields(id, fields);
            Projection::Update(notification(session_id, SessionUpdate::ToolCall(tool_call)))
        }
        AgentEvent::ToolCallProgress { id, fields }
        | AgentEvent::ToolCallFinished { id, fields } => Projection::Update(notification(
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields)),
        )),

        AgentEvent::PolicyDecision { id, decision } => match decision {
            PolicyDecision::Ask(ask) => {
                let options = ask
                    .options
                    .into_iter()
                    .map(|opt| PermissionOption::new(opt.id, opt.name, opt.kind))
                    .collect();
                Projection::Permission(PermissionAsk {
                    tool_call_id: id,
                    fields: ToolCallUpdateFields::default(),
                    options,
                })
            }
            PolicyDecision::Allow | PolicyDecision::Deny => Projection::Ignore,
            // `PolicyDecision` 同样 `#[non_exhaustive]`——未识别变体当作仅审计。
            _ => Projection::Ignore,
        },

        AgentEvent::PermissionResolved { .. }
        | AgentEvent::LlmCallStarted { .. }
        | AgentEvent::LlmCallFinished { .. }
        | AgentEvent::ContextCompressed { .. } => Projection::Ignore,

        // `AgentEvent` 是 #[non_exhaustive]——新增变体走 Ignore，避免桥接层
        // 因为新事件而拒签代码；真正需要上 wire 的变体在上面显式分支。
        _ => Projection::Ignore,
    }
}

fn notification(session_id: &SessionId, update: SessionUpdate) -> SessionNotification {
    SessionNotification::new(session_id.clone(), update)
}

/// 从 [`ToolCallUpdateFields`] 拼出一个完整 [`ToolCall`]。
///
/// `ToolCall::new` 需要 id + title 两个必填字段；其它通过 `update` 注入。
fn tool_call_from_fields(id: ToolCallId, fields: ToolCallUpdateFields) -> ToolCall {
    let title = fields.title.clone().unwrap_or_default();
    let mut call = ToolCall::new(id, title);
    call.update(fields);
    call
}
