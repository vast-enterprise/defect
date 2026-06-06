//! [`AgentEvent`] → ACP wire 形态的投影。
//!
//! 翻译表见 `docs/inbound/acp-bridge.md` §3。把内部事件分流为：
//! - [`Projection::Update`]：直接 `cx.send_notification(SessionNotification {..})`
//! - [`Projection::Permission`]：触发 `session/request_permission` 反向请求
//! - [`Projection::EndTurn`]：驱动 `PromptResponse::stop_reason`
//! - [`Projection::Ignore`]：仅审计，不入 wire

use agent_client_protocol::schema::{
    Content, ContentChunk, EmbeddedResource, EmbeddedResourceResource, ImageContent,
    PermissionOption, SessionId, SessionNotification, SessionUpdate, TextContent, ToolCall,
    ToolCallContent, ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::{
    ImageData, Message, MessageContent, Role, ToolResultBody, ToolResultContent,
};
use defect_agent::policy::PolicyDecision;

const REPLAY_TOOL_RESULT_TITLE: &str = "Tool result";
const REPLAY_RESOURCE_URI: &str = "defect://session-replay/text";

/// 一次投影的产物。
#[allow(clippy::large_enum_variant)]
pub(crate) enum Projection {
    /// 普通 `session/update` 通知。
    Update(SessionNotification),
    /// 需要发反向 `session/request_permission`。
    Permission(PermissionAsk),
    /// turn 终态——驱动 `PromptResponse`。事件本身仅作哨兵，权威 stop_reason
    /// 由 [`Session::run_turn`](defect_agent::session::Session::run_turn) 的返回值给出。
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

        AgentEvent::ToolCallStarted { id, fields, .. } => {
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

/// 将恢复出的 message history 投影成 ACP transcript replay 通知。
pub(crate) fn replay_notifications(
    session_id: &SessionId,
    history: &[Message],
) -> Vec<SessionNotification> {
    let mut notifications = Vec::new();
    for message in history {
        match message.role {
            Role::User => replay_user_message(session_id, message, &mut notifications),
            Role::Assistant => replay_assistant_message(session_id, message, &mut notifications),
        }
    }
    notifications
}

fn notification(session_id: &SessionId, update: SessionUpdate) -> SessionNotification {
    SessionNotification::new(session_id.clone(), update)
}

fn replay_user_message(
    session_id: &SessionId,
    message: &Message,
    notifications: &mut Vec<SessionNotification>,
) {
    for content in message.content.iter() {
        match content {
            MessageContent::Text { text } => notifications.push(notification(
                session_id,
                SessionUpdate::UserMessageChunk(ContentChunk::new(text_block(text.clone()))),
            )),
            MessageContent::Image { mime, data } => notifications.push(notification(
                session_id,
                SessionUpdate::UserMessageChunk(ContentChunk::new(image_block(mime, data))),
            )),
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => replay_tool_result(session_id, tool_use_id, output, *is_error, notifications),
            MessageContent::Thinking { .. }
            | MessageContent::ToolUse { .. }
            | MessageContent::ProviderActivity { .. } => {}
            _ => {}
        }
    }
}

fn replay_assistant_message(
    session_id: &SessionId,
    message: &Message,
    notifications: &mut Vec<SessionNotification>,
) {
    for content in message.content.iter() {
        match content {
            MessageContent::Thinking { text, .. } => notifications.push(notification(
                session_id,
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(text_block(text.clone()))),
            )),
            MessageContent::Text { text } => notifications.push(notification(
                session_id,
                SessionUpdate::AgentMessageChunk(ContentChunk::new(text_block(text.clone()))),
            )),
            MessageContent::ToolUse { id, name, args } => {
                let fields = ToolCallUpdateFields::new()
                    .title(name.clone())
                    .raw_input(args.clone());
                let tool_call = tool_call_from_fields(ToolCallId::new(id.clone()), fields);
                notifications.push(notification(session_id, SessionUpdate::ToolCall(tool_call)));
            }
            MessageContent::Image { .. }
            | MessageContent::ToolResult { .. }
            | MessageContent::ProviderActivity { .. } => {}
            _ => {}
        }
    }
}

fn replay_tool_result(
    session_id: &SessionId,
    tool_use_id: &str,
    output: &ToolResultBody,
    is_error: bool,
    notifications: &mut Vec<SessionNotification>,
) {
    let status = if is_error {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    };
    let blocks = tool_result_blocks(output);
    let fields = ToolCallUpdateFields::new()
        .title(REPLAY_TOOL_RESULT_TITLE.to_string())
        .status(status)
        .content(blocks);
    notifications.push(notification(
        session_id,
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            ToolCallId::new(tool_use_id.to_string()),
            fields,
        )),
    ));
}

/// 把 [`ToolResultBody`] 还原成给客户端 UI 重放的 ACP content 块。
/// 多模态结果逐块还原成文本/图片块；文本/JSON 仍是单个文本块。
fn tool_result_blocks(output: &ToolResultBody) -> Vec<ToolCallContent> {
    let blocks = match output {
        ToolResultBody::Text { text } => vec![text_block(text.clone())],
        ToolResultBody::Json { value } => vec![text_block(value.to_string())],
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text_block(text.clone()),
                ToolResultContent::Image { mime, data } => image_block(mime, data),
                _ => text_block(String::new()),
            })
            .collect(),
        _ => vec![text_block(String::new())],
    };
    blocks
        .into_iter()
        .map(|b| ToolCallContent::Content(Content::new(b)))
        .collect()
}

fn text_block(text: String) -> agent_client_protocol::schema::ContentBlock {
    agent_client_protocol::schema::ContentBlock::Text(TextContent::new(text))
}

fn image_block(mime: &str, data: &ImageData) -> agent_client_protocol::schema::ContentBlock {
    match data {
        ImageData::Base64 { encoded } => agent_client_protocol::schema::ContentBlock::Image(
            ImageContent::new(encoded.clone(), mime.to_string()),
        ),
        ImageData::Url { url } => agent_client_protocol::schema::ContentBlock::ResourceLink(
            agent_client_protocol::schema::ResourceLink::new(url.clone(), url.clone())
                .mime_type(mime.to_string()),
        ),
        _ => agent_client_protocol::schema::ContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(
                agent_client_protocol::schema::TextResourceContents::new(
                    "unsupported replay image data",
                    REPLAY_RESOURCE_URI,
                ),
            ),
        )),
    }
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
