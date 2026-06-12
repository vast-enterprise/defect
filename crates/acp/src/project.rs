//! [`AgentEvent`] → ACP wire projection.
//!
//! Translation table for ACP events. Splits internal events into:
//! - [`Projection::Update`]: directly calls `cx.send_notification(SessionNotification
//!   {..})`
//! - [`Projection::Permission`]: triggers a `session/request_permission` reverse request
//! - [`Projection::EndTurn`]: drives `PromptResponse::stop_reason`
//! - [`Projection::Ignore`]: audit-only, not sent over the wire

use agent_client_protocol::schema::{
    Content, ContentChunk, ImageContent, PermissionOption, SessionId, SessionNotification,
    SessionUpdate, TextContent, ToolCall, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::{
    ImageData, Message, MessageContent, Role, ToolResultBody, ToolResultContent,
};
use defect_agent::policy::PolicyDecision;

const REPLAY_TOOL_RESULT_TITLE: &str = "Tool result";

/// A single projection's output.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Projection {
    /// A normal `session/update` notification.
    Update(SessionNotification),
    /// Needs to send a reverse `session/request_permission`.
    Permission(PermissionAsk),
    /// Terminal turn state – drives a `PromptResponse`. The event itself is only a
    /// sentinel; the authoritative `stop_reason` is given by the return value of
    /// [`Session::run_turn`](defect_agent::session::Session::run_turn).
    EndTurn,
    /// Audit-only, not sent over the wire.
    Ignore,
}

/// Context required for a `session/request_permission` request.
pub(crate) struct PermissionAsk {
    pub tool_call_id: ToolCallId,
    pub fields: ToolCallUpdateFields,
    pub options: Vec<PermissionOption>,
}

/// Translates a single [`AgentEvent`].
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
            // `PolicyDecision` is also `#[non_exhaustive]` — treat unrecognized variants
            // as audit-only.
            _ => Projection::Ignore,
        },

        AgentEvent::PermissionResolved { .. }
        | AgentEvent::LlmCallStarted { .. }
        | AgentEvent::LlmCallFinished { .. }
        | AgentEvent::ContextCompressed { .. } => Projection::Ignore,

        // `AgentEvent` is `#[non_exhaustive]` — new variants fall through to `Ignore`,
        // preventing the bridge layer from rejecting code due to new events; variants
        // that actually need to go on the wire are explicitly handled above.
        _ => Projection::Ignore,
    }
}

/// Projects the recovered message history into ACP transcript replay notifications.
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

/// Reconstruct [`ToolResultBody`] into ACP content blocks for replay in the client UI.
/// Multimodal results are decomposed into text/image blocks; text/JSON remain as a single
/// text block.
fn tool_result_blocks(output: &ToolResultBody) -> Vec<ToolCallContent> {
    let blocks = match output {
        ToolResultBody::Text { text } => vec![text_block(text.clone())],
        ToolResultBody::Json { value } => vec![text_block(value.to_string())],
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text_block(text.clone()),
                ToolResultContent::Image { mime, data } => image_block(mime, data),
            })
            .collect(),
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
    }
}

/// Construct a complete [`ToolCall`] from [`ToolCallUpdateFields`].
///
/// `ToolCall::new` requires `id` and `title`; all other fields are injected via `update`.
fn tool_call_from_fields(id: ToolCallId, fields: ToolCallUpdateFields) -> ToolCall {
    let title = fields.title.clone().unwrap_or_default();
    let mut call = ToolCall::new(id, title);
    call.update(fields);
    call
}
