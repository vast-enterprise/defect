//! `defect-acp` 的对外入口。
//!
//! 起 stdio JSON-RPC 服务，注册 ACP v1 的 client→agent 方法处理器，
//! 把 [`AgentCore`] / [`Session`] 暴露在线上。
//!
//! 设计详见 `docs/inbound/acp-bridge.md`。

use std::sync::{Arc, RwLock};

use agent_client_protocol::schema::{
    AgentCapabilities, AuthenticateRequest, CancelNotification, ClientCapabilities,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse, ModelId,
    ModelInfo as AcpModelInfo, NewSessionRequest, NewSessionResponse, PromptRequest,
    PromptResponse, RequestPermissionOutcome, RequestPermissionRequest, SessionId,
    SessionModelState, SetSessionModelRequest, SetSessionModelResponse,
    StopReason as AcpStopReason,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Stdio};
use defect_agent::event::{AgentEvent, PermissionResolution};
use defect_agent::fs::FsBackend;
use defect_agent::llm::{ModelCandidate, ModelInfo, ProviderError, ProviderInfo};
use defect_agent::session::{AgentCore, AgentError, Session, TurnError, new_session_id};
use defect_agent::shell::ShellBackend;
use defect_tools::{LocalFsBackend, LocalShellBackend};
use futures::StreamExt;
use serde_json::json;

use crate::fs::AcpFsBackend;
use crate::project::{PermissionAsk, Projection, project, replay_notifications};
use crate::shell::AcpShellBackend;

/// 客户端 fs 能力协商结果（连接级）。
///
/// 在 `initialize` handler 里读 [`ClientCapabilities::fs`] 后写入，
/// `session/new` handler 据此选 [`AcpFsBackend`] / [`LocalFsBackend`]。
/// 设计详见 `docs/inbound/acp-fs.md` §1。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FsMode {
    /// 客户端同时声明 `read_text_file` 与 `write_text_file`：完全委托。
    Delegated,
    /// 任一能力缺失或 fs 字段未声明：整组退回本地（不混用，§1.2 决策表）。
    Local,
}

fn decide_fs_mode(client_caps: &ClientCapabilities) -> FsMode {
    if client_caps.fs.read_text_file && client_caps.fs.write_text_file {
        FsMode::Delegated
    } else {
        FsMode::Local
    }
}

/// 客户端 terminal 能力协商结果（连接级）。
///
/// `initialize` handler 读 [`ClientCapabilities::terminal`] 后写入，
/// `session/new` / `session/load` 据此选 [`AcpShellBackend`] /
/// [`LocalShellBackend`]。设计详见 `docs/inbound/acp-shell.md` §1。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellMode {
    /// 客户端声明完整 `terminal/*` 支持：完全委托。
    Delegated,
    /// 字段为 false 或缺失：整组退回本地（不混用，§1.2 决策表）。
    Local,
}

fn decide_shell_mode(client_caps: &ClientCapabilities) -> ShellMode {
    if client_caps.terminal {
        ShellMode::Delegated
    } else {
        ShellMode::Local
    }
}

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

    #[error("load_session failed: {0}")]
    LoadSession(#[source] AgentError),

    #[error("set_model failed: {0}")]
    SetModel(#[source] ProviderError),

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

            AcpError::LoadSession(err) => {
                Wire::new(ErrorCode::InternalError.into(), err.to_string()).data(json!({
                    "kind": "load_session_failed",
                    "message": err.to_string(),
                }))
            }

            AcpError::SetModel(err) => {
                let code = match err.kind {
                    defect_agent::llm::ProviderErrorKind::ModelNotFound { .. } => {
                        ErrorCode::InvalidParams
                    }
                    _ => ErrorCode::InternalError,
                };
                Wire::new(code.into(), err.to_string()).data(provider_error_data(&err))
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

async fn session_model_state(session: &dyn Session) -> Option<SessionModelState> {
    let current_model = session.current_model().to_string();
    let current_provider = session.provider_info();
    let candidates = match session.list_candidates().await {
        Ok(models) => models,
        Err(err) => {
            tracing::warn!(
                provider = %current_provider.vendor,
                model = %current_model,
                error = %err,
                "failed to load ACP session model candidates; falling back to current model only"
            );
            Vec::new()
        }
    };

    Some(SessionModelState::new(
        ModelId::new(current_model.clone()),
        acp_model_candidates(&current_provider, &current_model, candidates),
    ))
}

fn acp_model_candidates(
    current_provider: &ProviderInfo,
    current_model: &str,
    candidates: Vec<ModelCandidate>,
) -> Vec<AcpModelInfo> {
    let mut acp_candidates = candidates
        .into_iter()
        .map(|candidate| acp_model_info(&candidate.provider, candidate.model))
        .collect::<Vec<_>>();

    let has_current_model = acp_candidates
        .iter()
        .any(|candidate| candidate.model_id.0.as_ref() == current_model);
    if !has_current_model {
        // 兜底：registry 没有声明当前 model（理论不会发生——session 启动
        // 时 model 必须落在某个 entry 上）。仍然把当前 model 渲染出来以便
        // 客户端 UI 不至于看到一个空 dropdown。
        acp_candidates.insert(
            0,
            AcpModelInfo::new(current_model.to_string(), current_model.to_string()).description(
                Some(provider_model_description(current_provider, None, false)),
            ),
        );
    }

    acp_candidates
}

fn acp_model_info(provider: &ProviderInfo, model: ModelInfo) -> AcpModelInfo {
    let name = model
        .display_name
        .clone()
        .unwrap_or_else(|| model.id.clone());
    AcpModelInfo::new(model.id.clone(), name).description(Some(provider_model_description(
        provider,
        Some(&model),
        model.deprecated,
    )))
}

fn provider_model_description(
    provider: &ProviderInfo,
    model: Option<&ModelInfo>,
    deprecated: bool,
) -> String {
    let mut parts = vec![format!("provider: {}", provider.display_name)];
    if let Some(model) = model {
        if let Some(context_window) = model.context_window {
            parts.push(format!("context_window={context_window}"));
        }
        if let Some(max_output_tokens) = model.max_output_tokens {
            parts.push(format!("max_output_tokens={max_output_tokens}"));
        }
    }
    if deprecated {
        parts.push("deprecated".to_string());
    }
    parts.join(", ")
}

/// 连接级共享状态。`serve_on` 给每个 handler 克隆一份 `Arc<ServeState>`。
///
/// `agent` 是注入的核心，连接生命周期内只读。`fs_mode` / `shell_mode` 由
/// `initialize` 写入、后续 `session/new` 与 `session/load` 读取——RwLock 在
/// 读多写少的前提下保护这一次握手结果。
struct ServeState {
    agent: Arc<dyn AgentCore>,
    /// 客户端按 ACP 规范应当先 initialize 再 session/new。Default = `Local`
    /// 是回归保守值——initialize 还没到时就 session/new 是协议违规，但即便如此
    /// 我们也宁可走本地盘也不裸调反向请求。
    fs_mode: RwLock<FsMode>,
    /// shell 后端选择。Default = `Local`——同 [`Self::fs_mode`] 取保守降级。
    shell_mode: RwLock<ShellMode>,
}

impl ServeState {
    fn new(agent: Arc<dyn AgentCore>) -> Self {
        Self {
            agent,
            fs_mode: RwLock::new(FsMode::Local),
            shell_mode: RwLock::new(ShellMode::Local),
        }
    }

    fn current_fs_mode(&self) -> FsMode {
        self.fs_mode.read().map(|g| *g).unwrap_or(FsMode::Local)
    }

    fn current_shell_mode(&self) -> ShellMode {
        self.shell_mode
            .read()
            .map(|g| *g)
            .unwrap_or(ShellMode::Local)
    }

    /// 在 connection 级 fs_mode 与 session 级 cwd 之间组装 fs 后端。
    fn fs_backend(
        &self,
        cx: &ConnectionTo<Client>,
        session_id: &SessionId,
        cwd: &std::path::Path,
    ) -> Arc<dyn FsBackend> {
        match self.current_fs_mode() {
            FsMode::Delegated => Arc::new(AcpFsBackend::new(
                cx.clone(),
                session_id.clone(),
                cwd.to_path_buf(),
            )),
            FsMode::Local => Arc::new(LocalFsBackend::new(cwd.to_path_buf())),
        }
    }

    /// 在 connection 级 shell_mode 与 session 级 cwd 之间组装 shell 后端。
    fn shell_backend(
        &self,
        cx: &ConnectionTo<Client>,
        session_id: &SessionId,
        cwd: &std::path::Path,
    ) -> Arc<dyn ShellBackend> {
        match self.current_shell_mode() {
            ShellMode::Delegated => Arc::new(AcpShellBackend::new(
                cx.clone(),
                session_id.clone(),
                cwd.to_path_buf(),
            )),
            ShellMode::Local => Arc::new(LocalShellBackend::new()),
        }
    }

    async fn on_initialize(
        &self,
        req: InitializeRequest,
        responder: agent_client_protocol::Responder<InitializeResponse>,
    ) -> Result<(), agent_client_protocol::Error> {
        let fs_mode = decide_fs_mode(&req.client_capabilities);
        let shell_mode = decide_shell_mode(&req.client_capabilities);
        tracing::debug!(
            version = ?req.protocol_version,
            fs_mode = ?fs_mode,
            shell_mode = ?shell_mode,
            "initialize"
        );
        if let Ok(mut guard) = self.fs_mode.write() {
            *guard = fs_mode;
        }
        if let Ok(mut guard) = self.shell_mode.write() {
            *guard = shell_mode;
        }
        responder.respond(
            InitializeResponse::new(req.protocol_version)
                .agent_capabilities(AgentCapabilities::new().load_session(true)),
        )
    }

    async fn on_authenticate(
        &self,
        responder: agent_client_protocol::Responder<
            agent_client_protocol::schema::AuthenticateResponse,
        >,
    ) -> Result<(), agent_client_protocol::Error> {
        // v0 不开 auth；任何客户端发起的 auth 请求都按未实现拒绝。
        responder.respond_with_error(AcpError::AuthNotSupported.into_wire_error())
    }

    async fn on_session_new(
        &self,
        req: NewSessionRequest,
        responder: agent_client_protocol::Responder<NewSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), agent_client_protocol::Error> {
        let cwd_for_log = req.cwd.clone();
        let session_id = SessionId::new(new_session_id());
        let fs = self.fs_backend(&cx, &session_id, &req.cwd);
        let shell = self.shell_backend(&cx, &session_id, &req.cwd);
        match self
            .agent
            .create_session(session_id, req.cwd, req.mcp_servers, fs, shell)
            .await
        {
            Ok(session) => {
                let models = session_model_state(session.as_ref()).await;
                tracing::info!(
                    session_id = %short_session_id(session.id()),
                    cwd = %cwd_for_log.display(),
                    "session created"
                );
                responder.respond(NewSessionResponse::new(session.id().clone()).models(models))
            }
            Err(err) => {
                let acp_err = AcpError::CreateSession(err);
                tracing::warn!(error = %acp_err, "create_session failed");
                responder.respond_with_error(acp_err.into_wire_error())
            }
        }
    }

    async fn on_session_load(
        &self,
        req: LoadSessionRequest,
        responder: agent_client_protocol::Responder<LoadSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), agent_client_protocol::Error> {
        let session_id = req.session_id.clone();
        let cwd_for_log = req.cwd.clone();
        let fs = self.fs_backend(&cx, &session_id, &req.cwd);
        let shell = self.shell_backend(&cx, &session_id, &req.cwd);
        match self.agent.load_session(session_id.clone(), fs, shell).await {
            Ok(session) => {
                let models = session_model_state(session.as_ref()).await;
                for notification in replay_notifications(&session_id, &session.history_snapshot()) {
                    if let Err(err) = cx.send_notification(notification) {
                        tracing::warn!(?err, "failed to replay loaded session transcript");
                    }
                }
                tracing::info!(
                    session_id = %short_session_id(session.id()),
                    cwd = %cwd_for_log.display(),
                    "session loaded"
                );
                responder.respond(LoadSessionResponse::new().models(models))
            }
            Err(err) => {
                let acp_err = match err {
                    AgentError::SessionNotFound(id) => AcpError::SessionNotFound {
                        session_id: id.0.to_string(),
                    },
                    other => AcpError::LoadSession(other),
                };
                tracing::warn!(error = %acp_err, "load_session failed");
                responder.respond_with_error(acp_err.into_wire_error())
            }
        }
    }

    async fn on_set_model(
        &self,
        req: SetSessionModelRequest,
        responder: agent_client_protocol::Responder<SetSessionModelResponse>,
    ) -> Result<(), agent_client_protocol::Error> {
        let session_id = req.session_id.clone();
        let Some(session) = self.agent.session(&session_id) else {
            return responder.respond_with_error(
                AcpError::SessionNotFound {
                    session_id: session_id.0.to_string(),
                }
                .into_wire_error(),
            );
        };

        match session.set_model(req.model_id.0.to_string()).await {
            Ok(()) => responder.respond(SetSessionModelResponse::new()),
            Err(err) => {
                let acp_err = AcpError::SetModel(err);
                tracing::warn!(
                    session_id = %short_session_id(session.id()),
                    error = %acp_err,
                    "session/set_model failed"
                );
                responder.respond_with_error(acp_err.into_wire_error())
            }
        }
    }

    async fn on_prompt(
        &self,
        req: PromptRequest,
        responder: agent_client_protocol::Responder<PromptResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), agent_client_protocol::Error> {
        let session_id = req.session_id.clone();
        let Some(session) = self.agent.session(&session_id) else {
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
            async move { run_prompt_turn(session, session_id, req.prompt, cx, responder).await }
        })
    }

    async fn on_cancel(
        &self,
        notif: CancelNotification,
    ) -> Result<(), agent_client_protocol::Error> {
        if let Some(session) = self.agent.session(&notif.session_id) {
            session.cancel_turn();
        }
        Ok(())
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
    let state = Arc::new(ServeState::new(agent));

    Agent
        .builder()
        .name("defect-agent")
        .on_receive_request(
            {
                let state = state.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    state.on_initialize(req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                async move |_req: AuthenticateRequest, responder, _cx| {
                    state.on_authenticate(responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                async move |req: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                    state.on_session_new(req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                    state.on_session_load(req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                async move |req: SetSessionModelRequest, responder, _cx| {
                    state.on_set_model(req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    state.on_prompt(req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let state = state.clone();
                async move |notif: CancelNotification, _cx| state.on_cancel(notif).await
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
    let (turn_tx, mut turn_rx) =
        tokio::sync::oneshot::channel::<Result<AcpStopReason, TurnError>>();
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
                return responder.respond_with_error(acp_err.into_wire_error());
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
mod test;
