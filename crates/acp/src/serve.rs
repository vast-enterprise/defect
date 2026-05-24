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
use defect_agent::session::{AgentCore, Session, TurnError};
use futures::StreamExt;

use crate::project::{project, PermissionAsk, Projection};

/// `defect-acp` 公共错误类型。
#[derive(Debug, thiserror::Error)]
pub enum AcpError {
    #[error("acp transport error: {0}")]
    Transport(agent_client_protocol::Error),
}

impl From<agent_client_protocol::Error> for AcpError {
    fn from(err: agent_client_protocol::Error) -> Self {
        AcpError::Transport(err)
    }
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
                responder.respond_with_error(agent_client_protocol::util::internal_error(
                    "authentication not supported",
                ))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent_session_new.clone();
                async move |req: NewSessionRequest, responder, _cx| {
                    let agent = agent.clone();
                    match agent.create_session(req.cwd, req.mcp_servers).await {
                        Ok(session) => {
                            responder.respond(NewSessionResponse::new(session.id().clone()))
                        }
                        Err(err) => responder.respond_with_error(
                            agent_client_protocol::util::internal_error(format!(
                                "create_session failed: {err}"
                            )),
                        ),
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
                            agent_client_protocol::util::internal_error(format!(
                                "session not found: {}",
                                session_id.0
                            )),
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
                        return responder.respond_with_error(
                            agent_client_protocol::util::internal_error(format!("turn failed: {err}")),
                        );
                    }
                    Err(_) => {
                        return responder.respond_with_error(
                            agent_client_protocol::util::internal_error(
                                "turn task dropped before completion",
                            ),
                        );
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
                return responder.respond_with_error(
                    agent_client_protocol::util::internal_error(format!("turn failed: {err}")),
                );
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
