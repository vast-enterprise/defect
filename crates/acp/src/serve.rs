//! `defect-acp` 的对外入口。
//!
//! 起 stdio JSON-RPC 服务，注册 ACP v1 的 client→agent 方法处理器，
//! 把 [`AgentCore`] / [`Session`] 暴露在线上。
//!
//! 设计详见 `docs/inbound/acp-bridge.md`。

use std::sync::Arc;

use agent_client_protocol::schema::{
    AgentCapabilities, AuthenticateRequest, CancelNotification, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, SessionId, StopReason as AcpStopReason,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Stdio};
use defect_agent::event::{AgentEvent, PermissionResolution};
use defect_agent::llm::ProviderError;
use defect_agent::session::{AgentCore, AgentError, Session, TurnError};
use futures::StreamExt;
use serde_json::json;

use crate::project::{project, PermissionAsk, Projection};

/// `defect-acp` 公共错误类型。
///
/// 划线规则：每个 variant 对应一种 wire 上能稳定区分的错误形态——
/// session 是否存在、会话创建是否成功、turn 是否跑完。下游 LLM /
/// 工具失败由 [`TurnError`] 自己分类承载（这一层不再细拆）。
///
/// 投影规则见 [`AcpError::into_wire_error`]：variant → JSON-RPC ErrorCode +
/// 结构化 `data` 字段。诊断字段（`session_id` / `request_id` 等）走 `data`，
/// 不糊在 `message` 里。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AcpError {
    /// JSON-RPC / stdio 传输层失败。仅 [`serve_on`] 的顶层 `?` 用得上；
    /// handler 内部任何地方都不会构造这个 variant。
    #[error("acp transport error: {0}")]
    Transport(agent_client_protocol::Error),

    /// `session/prompt` / `session/cancel` 引用的 session 在 agent 侧不存在。
    #[error("session not found: {session_id}")]
    SessionNotFound { session_id: String },

    /// `session/new` 创建 session 失败（cwd 不存在 / MCP 启动失败等）。
    #[error("create_session failed: {0}")]
    CreateSession(#[source] AgentError),

    /// `session/prompt` 跑 turn 时失败（重试用尽的 provider 错误 / 主循环
    /// invariant 被破坏）。
    #[error("turn failed: {0}")]
    Turn(#[source] TurnError),

    /// turn task 在返回 stop reason 之前被 drop（理应不可达，留作安全网）。
    #[error("turn task dropped before completion")]
    TurnDropped,

    /// 客户端请求 `authenticate`，但 v0 不支持。
    #[error("authentication not supported")]
    AuthNotSupported,
}

impl From<agent_client_protocol::Error> for AcpError {
    fn from(err: agent_client_protocol::Error) -> Self {
        AcpError::Transport(err)
    }
}

impl AcpError {
    /// 投影成 ACP wire `Error`：选 ErrorCode + 在 `data` 里挂结构化诊断字段。
    ///
    /// 调用方（handler）在 [`agent_client_protocol::Responder::respond_with_error`]
    /// 处用它替代手搓的 [`agent_client_protocol::util::internal_error`] +
    /// `format!`，让客户端能稳定 match `code` / 读 `data.kind` 而非解析字符串。
    pub fn into_wire_error(self) -> agent_client_protocol::Error {
        use agent_client_protocol::Error as Wire;
        use agent_client_protocol::schema::ErrorCode;
        match self {
            AcpError::Transport(err) => err,

            AcpError::SessionNotFound { session_id } => {
                // 用 ResourceNotFound 而不是 InternalError——这是"客户端引用了
                // 不存在的资源"，是客户端可恢复的 4xx 类语义。
                Wire::resource_not_found(Some(session_id))
            }

            AcpError::CreateSession(err) => {
                // 把内层 Display 放到 wire `message`——客户端 UI（acpx 等）
                // 渲染时直接读 message，默认占位 "Internal error" 把诊断信息
                // 全埋在 `data` 里，导致用户只看见 "RUNTIME: Internal error"。
                Wire::new(ErrorCode::InternalError.into(), err.to_string()).data(json!({
                    "kind": "create_session_failed",
                    "message": err.to_string(),
                }))
            }

            AcpError::Turn(err) => {
                // 把内层 Display 灌进 wire `message`——客户端 UI 默认只读
                // message 字段；占位 "Internal error" 把实际信息埋在 `data` 里
                // 会让用户只看见 "RUNTIME: Internal error" 这种无意义占位。
                // 注意：code 选择有坑——acpx 把 -32001/-32002 映射成 NO_SESSION
                // （会议会话误判），所以 Provider 也走 InternalError，由 message
                // 自身的文本（"rate limit" / "model not found"）让 acpx 的
                // text-error-rules 命中合适的 hint。
                let code = match &err {
                    TurnError::TurnInProgress => ErrorCode::InvalidRequest,
                    _ => ErrorCode::InternalError,
                };
                Wire::new(code.into(), err.to_string()).data(turn_error_data(&err))
            }

            AcpError::TurnDropped => Wire::new(
                ErrorCode::InternalError.into(),
                "turn task dropped before completion",
            )
            .data(json!({
                "kind": "turn_task_dropped",
                "message": "turn task dropped before completion",
            })),

            // method_not_found 比 internal_error 更对位"未实现的方法"
            AcpError::AuthNotSupported => Wire::method_not_found().data(json!({
                "kind": "auth_not_supported",
                "message": "authentication not supported",
            })),
        }
    }
}

/// 把 [`TurnError`] 拍成 wire `data` 字段。区分两个 sub-kind：
/// - `provider` —— 重试用尽后仍失败的 provider 错误，附 `retry_hint` /
///   `request_id`，让客户端能据此提示用户"换模型 / 等一会再试"
/// - `internal` —— 主循环 invariant 被破坏，纯诊断用
fn turn_error_data(err: &TurnError) -> serde_json::Value {
    match err {
        TurnError::TurnInProgress => json!({
            "kind": "turn_in_progress",
            "message": err.to_string(),
        }),
        TurnError::Provider(provider_err) => provider_error_data(provider_err),
        TurnError::Internal(_) => json!({
            "kind": "internal",
            "message": err.to_string(),
        }),
        // TurnError 是 #[non_exhaustive]：未来新 variant 落到这里走 internal
        // 兜底，不阻塞编译；新增分类时优先把它提到上面写专门 arm。
        _ => json!({
            "kind": "internal",
            "message": err.to_string(),
        }),
    }
}

fn provider_error_data(err: &ProviderError) -> serde_json::Value {
    let mut data = json!({
        "kind": "provider",
        "message": err.to_string(),
        "retryable": err.is_retryable(),
    });
    if let Some(req_id) = &err.request_id
        && let Some(map) = data.as_object_mut()
    {
        map.insert("request_id".into(), json!(req_id));
    }
    data
}

/// 启动 stdio ACP 服务，阻塞到对端断开。
///
/// `agent` 由 `defect-cli` 装配（含 provider / 工具 / 配置）后注入。
pub async fn serve(agent: Arc<dyn AgentCore>) -> Result<(), AcpError> {
    serve_on(agent, Stdio::new()).await
}

/// 在自定义 transport 上跑同一套 ACP handler。
///
/// 公共入口 [`serve`] 用 stdio；集成测试用 `Channel` 在进程内对接。
pub async fn serve_on<T>(agent: Arc<dyn AgentCore>, transport: T) -> Result<(), AcpError>
where
    T: ConnectTo<Agent> + 'static,
{
    let agent_init = agent.clone();
    let agent_session_new = agent.clone();
    let agent_prompt = agent.clone();
    let agent_cancel = agent.clone();

    Agent
        .builder()
        .name("defect-agent")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx| {
                tracing::debug!(version = ?req.protocol_version, "initialize");
                let _ = &agent_init;
                responder.respond(
                    InitializeResponse::new(req.protocol_version)
                        .agent_capabilities(AgentCapabilities::new()),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_req: AuthenticateRequest, responder, _cx| {
                // v0 不开 auth；任何客户端发起的 auth 请求都按未实现拒绝。
                responder.respond_with_error(AcpError::AuthNotSupported.into_wire_error())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent_session_new.clone();
                async move |req: NewSessionRequest, responder, _cx| {
                    let agent = agent.clone();
                    let cwd_for_log = req.cwd.clone();
                    match agent.create_session(req.cwd, req.mcp_servers).await {
                        Ok(session) => {
                            tracing::info!(
                                session_id = %short_session_id(session.id()),
                                cwd = %cwd_for_log.display(),
                                "session created"
                            );
                            responder.respond(NewSessionResponse::new(session.id().clone()))
                        }
                        Err(err) => {
                            let acp_err = AcpError::CreateSession(err);
                            tracing::warn!(error = %acp_err, "create_session failed");
                            responder.respond_with_error(acp_err.into_wire_error())
                        }
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent_prompt.clone();
                async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    let agent = agent.clone();
                    let session_id = req.session_id.clone();
                    let Some(session) = agent.session(&session_id) else {
                        return responder.respond_with_error(
                            AcpError::SessionNotFound {
                                session_id: session_id.0.to_string(),
                            }
                            .into_wire_error(),
                        );
                    };
                    // 把 turn 的执行扔到 spawn 任务里，handler 立即返回，
                    // 让 dispatch loop 不被阻塞——这样后续 cancel / resolve
                    // 等消息能在 turn 跑的同时被处理。
                    cx.spawn({
                        let cx = cx.clone();
                        async move {
                            run_prompt_turn(session, session_id, req.prompt, cx, responder).await
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let agent = agent_cancel.clone();
                async move |notif: CancelNotification, _cx| {
                    if let Some(session) = agent.session(&notif.session_id) {
                        session.cancel_turn();
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(transport)
        .await?;

    Ok(())
}

/// 一次 `session/prompt` 的完整 turn：订阅事件、跑 turn、把事件投射到 wire、
/// 在 turn 结束时 respond `PromptResponse`。
#[tracing::instrument(
    name = "acp_prompt_turn",
    skip_all,
    fields(session_id = %short_session_id(&session_id))
)]
async fn run_prompt_turn(
    session: Arc<dyn Session>,
    session_id: SessionId,
    prompt: Vec<agent_client_protocol::schema::ContentBlock>,
    cx: ConnectionTo<Client>,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> Result<(), agent_client_protocol::Error> {
    // 必须在 run_turn 启动前订阅，否则事件先到没人接。
    let mut events = session.subscribe();

    // 把 turn future spawn 到独立任务，stop_reason 通过 oneshot 回流。
    let (turn_tx, mut turn_rx) = tokio::sync::oneshot::channel::<Result<AcpStopReason, TurnError>>();
    let session_for_turn = session.clone();
    tokio::spawn(async move {
        let result = session_for_turn.run_turn(prompt).await;
        let _ = turn_tx.send(result);
    });

    let mut stop_reason: Option<AcpStopReason> = None;
    loop {
        tokio::select! {
            biased;
            next = events.next() => {
                match next {
                    Some(event) => {
                        if matches!(event, AgentEvent::TurnEnded { .. }) {
                            // 取出 reason 后 break——run_turn 返回值才是权威。
                            if let AgentEvent::TurnEnded { reason, .. } = event {
                                stop_reason.get_or_insert(reason);
                            }
                            break;
                        }
                        if let Err(err) = handle_event(&session, &session_id, event, &cx) {
                            tracing::warn!(?err, "failed to project agent event");
                        }
                    }
                    None => break,
                }
            }
            run_result = &mut turn_rx => {
                match run_result {
                    Ok(Ok(reason)) => {
                        stop_reason.get_or_insert(reason);
                    }
                    Ok(Err(err)) => {
                        let acp_err = AcpError::Turn(err);
                        tracing::warn!(error = %acp_err, "turn failed; responding with wire error");
                        return responder
                            .respond_with_error(acp_err.into_wire_error());
                    }
                    Err(_) => {
                        tracing::warn!("turn task dropped; responding with wire error");
                        return responder
                            .respond_with_error(AcpError::TurnDropped.into_wire_error());
                    }
                }
                // turn 已结束，drain 剩余事件，确保 ToolCallFinished 等都上 wire。
                while let Some(event) = events.next().await {
                    if matches!(event, AgentEvent::TurnEnded { .. }) {
                        break;
                    }
                    if let Err(err) = handle_event(&session, &session_id, event, &cx) {
                        tracing::warn!(?err, "failed to project trailing event");
                    }
                }
                break;
            }
        }
    }

    // turn 已结束（或事件流提前关闭），等待 turn future 给出权威 stop_reason。
    let stop = match stop_reason {
        Some(r) => r,
        None => match (&mut turn_rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(err)) => {
                let acp_err = AcpError::Turn(err);
                tracing::warn!(error = %acp_err, "turn failed; responding with wire error");
                return responder
                    .respond_with_error(acp_err.into_wire_error());
            }
            Err(_) => AcpStopReason::Cancelled,
        },
    };

    responder.respond(PromptResponse::new(stop))
}

fn handle_event(
    session: &Arc<dyn Session>,
    session_id: &SessionId,
    event: AgentEvent,
    cx: &ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    match project(session_id, event) {
        Projection::Update(notif) => cx.send_notification(notif),
        Projection::Permission(ask) => {
            spawn_permission_request(session.clone(), session_id.clone(), ask, cx.clone());
            Ok(())
        }
        Projection::EndTurn | Projection::Ignore => Ok(()),
    }
}

/// 给 tracing span / log 用的 session id 短形：按字符取前 12 个。仅诊断用。
fn short_session_id(id: &SessionId) -> &str {
    let s: &str = id.0.as_ref();
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// 反向请求 `session/request_permission`，等客户端响应后回写到 [`Session`]。
fn spawn_permission_request(
    session: Arc<dyn Session>,
    session_id: SessionId,
    ask: PermissionAsk,
    cx: ConnectionTo<Client>,
) {
    let req = RequestPermissionRequest::new(
        session_id,
        agent_client_protocol::schema::ToolCallUpdate::new(ask.tool_call_id.clone(), ask.fields),
        ask.options,
    );
    let tool_call_id = ask.tool_call_id;
    let cx_for_task = cx.clone();
    let _ = cx.spawn(async move {
        let response = cx_for_task.send_request(req).block_task().await;
        let outcome = match response {
            Ok(resp) => match resp.outcome {
                RequestPermissionOutcome::Selected(selected) => PermissionResolution::Selected {
                    option_id: selected.option_id,
                },
                RequestPermissionOutcome::Cancelled => PermissionResolution::Cancelled,
                _ => PermissionResolution::Cancelled,
            },
            Err(err) => {
                tracing::warn!(?err, "request_permission failed; treating as cancelled");
                PermissionResolution::Cancelled
            }
        };
        session.resolve_permission(tool_call_id, outcome);
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::ErrorCode;
    use defect_agent::error::BoxError;
    use defect_agent::llm::{ProviderError, ProviderErrorKind};

    /// `TurnError::Provider` 必须把内层 Display 灌进 wire `message`。
    /// 之前的实现用 [`Wire::internal_error()`]，message 永远是字面量
    /// "Internal error"——客户端 UI 拿不到任何辨识信息，acpx 显示成
    /// `RUNTIME: Internal error`。
    #[test]
    fn turn_provider_error_carries_message_on_wire() {
        let provider_err = ProviderError::new(ProviderErrorKind::ModelNotFound {
            model: "deepseek-v4-pro".into(),
        });
        let acp_err = AcpError::Turn(TurnError::Provider(provider_err));
        let wire = acp_err.into_wire_error();

        assert_eq!(wire.code, ErrorCode::InternalError);
        assert!(
            wire.message.contains("model not found")
                && wire.message.contains("deepseek-v4-pro"),
            "expected provider Display text in wire message, got: {:?}",
            wire.message
        );
        // data.kind 仍然区分 provider vs internal，方便 verbose 模式排障。
        let data = wire.data.expect("wire data");
        assert_eq!(data.get("kind").and_then(|v| v.as_str()), Some("provider"));
    }

    #[test]
    fn turn_internal_error_carries_message_on_wire() {
        let acp_err = AcpError::Turn(TurnError::Internal(BoxError::new(std::io::Error::other(
            "history backend exploded",
        ))));
        let wire = acp_err.into_wire_error();

        assert_eq!(wire.code, ErrorCode::InternalError);
        assert!(
            wire.message.contains("history backend exploded"),
            "expected inner io Display in wire message, got: {:?}",
            wire.message
        );
    }

    #[test]
    fn turn_in_progress_uses_invalid_request_code() {
        let acp_err = AcpError::Turn(TurnError::TurnInProgress);
        let wire = acp_err.into_wire_error();
        assert_eq!(wire.code, ErrorCode::InvalidRequest);
        assert!(wire.message.contains("turn already in progress"));
    }
}
