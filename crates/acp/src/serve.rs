//! `defect-acp` 的对外入口。
//!
//! 起 stdio JSON-RPC 服务，注册 ACP v1 的 client→agent 方法处理器，
//! 把 [`AgentCore`] / [`Session`] 暴露在线上。
//!
//! 设计详见 `docs/inbound/acp-bridge.md`。

use std::sync::{Arc, RwLock};

use agent_client_protocol::schema::{
    AgentCapabilities, AuthenticateRequest, CancelNotification, ClientCapabilities,
    InitializeRequest, InitializeResponse, LoadSessionRequest, LoadSessionResponse,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, RequestPermissionOutcome,
    RequestPermissionRequest, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionConfigValueId, SessionId, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Stdio};
use defect_agent::event::{AgentEvent, PermissionResolution};
use defect_agent::fs::FsBackend;
use defect_agent::llm::{ModelCandidate, ProviderError, ReasoningEffort};
use defect_agent::session::{AgentCore, AgentError, Frontend, Session, TurnError, new_session_id};
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

    /// `session/set_config_option` 收到未知 config_id 或非法 value。
    #[error("invalid session config option: {0}")]
    InvalidConfigOption(String),

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

            AcpError::InvalidConfigOption(message) => {
                Wire::new(ErrorCode::InvalidParams.into(), message.clone()).data(json!({
                    "kind": "invalid_config_option",
                    "message": message,
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

/// `session/set_config_option` 里 thought-level 选择器的稳定 config id。
const THOUGHT_LEVEL_CONFIG_ID: &str = "reasoning_effort";

/// `session/set_config_option` 里权限模式选择器的稳定 config id。
///
/// Session Config Options 取代了旧的 Session Modes API：现代客户端（如 Zed
/// ≥ 1.4）只读 `configOptions`、忽略响应里 deprecated 的 `modes` 字段。故权限
/// 模式必须**也**作为一个 `category = Mode` 的 config option 暴露，否则客户端
/// 不渲染模式选择器。`modes` 字段仍保留以兼容老客户端。
const MODE_CONFIG_ID: &str = "permission_mode";

/// `session/set_config_option` 里模型选择器的稳定 config id。
///
/// 与 [`MODE_CONFIG_ID`] 同理——现代客户端只读 config_options，故模型也必须
/// 作为 `category = Model` 的 config option 暴露，否则不渲染模型选择器。响应里
/// deprecated 的 `models` 字段仍保留以兼容老客户端。
const MODEL_CONFIG_ID: &str = "model";

/// thought-level 的 "不设置"（沿用 provider 默认）档位的 value id。
/// 其余档位用 [`ReasoningEffort`] 的 wire token（`minimal` / `low` / …）。
const REASONING_DEFAULT_VALUE: &str = "default";

/// 把 ACP value id 解析成 [`ReasoningEffort`] 覆盖。`"default"` → `None`
/// （清除覆盖）；其余按 wire token 匹配；未知 token 返回 `Err`。
fn parse_reasoning_value(value: &str) -> Result<Option<ReasoningEffort>, ()> {
    match value {
        REASONING_DEFAULT_VALUE => Ok(None),
        "none" => Ok(Some(ReasoningEffort::None)),
        "minimal" => Ok(Some(ReasoningEffort::Minimal)),
        "low" => Ok(Some(ReasoningEffort::Low)),
        "medium" => Ok(Some(ReasoningEffort::Medium)),
        "high" => Ok(Some(ReasoningEffort::High)),
        "xhigh" => Ok(Some(ReasoningEffort::Xhigh)),
        _ => Err(()),
    }
}

/// 当前 [`ReasoningEffort`] 覆盖对应的 ACP value id。`None` → `"default"`。
fn reasoning_value_id(effort: Option<ReasoningEffort>) -> &'static str {
    match effort {
        None => REASONING_DEFAULT_VALUE,
        Some(ReasoningEffort::None) => "none",
        Some(ReasoningEffort::Minimal) => "minimal",
        Some(ReasoningEffort::Low) => "low",
        Some(ReasoningEffort::Medium) => "medium",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::Xhigh) => "xhigh",
    }
}

/// 构造 session 的配置项列表（ACP `config_options`）。
///
/// 含三个 select：模型（`category = Model`）、权限模式（`category = Mode`，来自
/// session 的模式目录）、thought-level（`category = ThoughtLevel`，6 档
/// `reasoning_effort` + "default"）。
///
/// **三者都必须经 config option 暴露**：Session Config Options 取代了旧的
/// Session Modes / Models API，现代客户端（如 Zed ≥ 1.4）只渲染 config_options
/// 里的选择器、忽略响应里 deprecated 的 `models` / `modes` 字段（后两者仍保留
/// 以兼容老客户端）。见 [`MODE_CONFIG_ID`] / [`MODEL_CONFIG_ID`]。
async fn session_config_options(session: &dyn Session) -> Vec<SessionConfigOption> {
    let mut out = Vec::new();

    // 0) 模型选择器。候选来自 registry（不发网络请求，恒可解析）。拿不到候选
    //    时退而只列当前模型，保证 dropdown 非空。
    {
        let current_model = session.current_model();
        let candidates = session.list_candidates().await.unwrap_or_default();
        let mut model_options = candidates
            .into_iter()
            .map(|c| {
                let name = c
                    .model
                    .display_name
                    .clone()
                    .unwrap_or_else(|| c.model.id.clone());
                let description = model_option_description(&c);
                SessionConfigSelectOption::new(SessionConfigValueId::new(c.model.id), name)
                    .description(Some(description))
            })
            .collect::<Vec<_>>();
        if !model_options
            .iter()
            .any(|o| o.value.0.as_ref() == current_model)
        {
            // 兜底：候选里没有当前模型（理论不该发生）。仍列出它，避免空 dropdown。
            model_options.insert(
                0,
                SessionConfigSelectOption::new(
                    SessionConfigValueId::new(current_model.clone()),
                    current_model.clone(),
                ),
            );
        }
        out.push(
            SessionConfigOption::select(
                MODEL_CONFIG_ID,
                "Model",
                SessionConfigValueId::new(current_model),
                model_options,
            )
            .category(Some(SessionConfigOptionCategory::Model))
            .description(Some("本会话使用的模型".to_string())),
        );
    }

    // 1) 权限模式选择器（仅当 session 装配了模式目录）。
    if let Some(current_mode) = session.current_mode() {
        let mode_options = session
            .available_modes()
            .into_iter()
            .map(|m| {
                let opt = SessionConfigSelectOption::new(SessionConfigValueId::new(m.id), m.name);
                match m.description {
                    Some(desc) => opt.description(Some(desc)),
                    None => opt,
                }
            })
            .collect::<Vec<_>>();
        out.push(
            SessionConfigOption::select(
                MODE_CONFIG_ID,
                "Permission mode",
                SessionConfigValueId::new(current_mode),
                mode_options,
            )
            .category(Some(SessionConfigOptionCategory::Mode))
            .description(Some(
                "工具调用的放行策略：只读 / 写前询问 / 全放行 / 全拒绝".to_string(),
            )),
        );
    }

    // 2) thought-level 选择器。顺序：default 在最前，其余按强度递增——与
    //    OpenAI wire 枚举一致。
    let current_effort = reasoning_value_id(session.current_reasoning_effort());
    let effort_options = vec![
        SessionConfigSelectOption::new(
            SessionConfigValueId::new(REASONING_DEFAULT_VALUE),
            "Default",
        )
        .description(Some(
            "沿用 provider 默认，不下发 reasoning_effort".to_string(),
        )),
        SessionConfigSelectOption::new(SessionConfigValueId::new("none"), "None"),
        SessionConfigSelectOption::new(SessionConfigValueId::new("minimal"), "Minimal"),
        SessionConfigSelectOption::new(SessionConfigValueId::new("low"), "Low"),
        SessionConfigSelectOption::new(SessionConfigValueId::new("medium"), "Medium"),
        SessionConfigSelectOption::new(SessionConfigValueId::new("high"), "High"),
        SessionConfigSelectOption::new(SessionConfigValueId::new("xhigh"), "Extra high"),
    ];
    out.push(
        SessionConfigOption::select(
            THOUGHT_LEVEL_CONFIG_ID,
            "Reasoning effort",
            SessionConfigValueId::new(current_effort),
            effort_options,
        )
        .category(Some(SessionConfigOptionCategory::ThoughtLevel))
        .description(Some(
            "OpenAI 兼容协议的思考强度等级；不支持的 provider 忽略".to_string(),
        )),
    );

    out
}

/// 一个模型候选在 config-option 选择器里的描述串：`provider: X,
/// context_window=…, max_output_tokens=…, deprecated`（缺的字段省略）。
fn model_option_description(candidate: &ModelCandidate) -> String {
    let mut parts = vec![format!("provider: {}", candidate.provider.display_name)];
    if let Some(context_window) = candidate.model.context_window {
        parts.push(format!("context_window={context_window}"));
    }
    if let Some(max_output_tokens) = candidate.model.max_output_tokens {
        parts.push(format!("max_output_tokens={max_output_tokens}"));
    }
    if candidate.model.deprecated {
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
    /// `--resume` 目标。ACP 客户端驱动会话生命周期，CLI 无法直接发起 load，
    /// 故把目标 id 暂存于此：**首个** `session/new` 透明改走 load_session 并
    /// 回放该会话，之后清空（一次性）。`None` = 不 resume。
    resume_target: RwLock<Option<SessionId>>,
}

impl ServeState {
    fn with_resume(agent: Arc<dyn AgentCore>, resume_target: Option<SessionId>) -> Self {
        Self {
            agent,
            fs_mode: RwLock::new(FsMode::Local),
            shell_mode: RwLock::new(ShellMode::Local),
            resume_target: RwLock::new(resume_target),
        }
    }

    /// 取出并清空一次性 resume 目标。第二次起返回 `None`。
    fn take_resume_target(&self) -> Option<SessionId> {
        self.resume_target.write().ok().and_then(|mut g| g.take())
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

    /// 由当前协商出的 fs / shell mode 组装 [`Frontend::Acp`]——agent 据此在
    /// system prompt 的 `# Environment` 段标明文件 / 命令执行是本地还是委托。
    fn frontend(&self) -> Frontend {
        Frontend::Acp {
            fs_delegated: self.current_fs_mode() == FsMode::Delegated,
            shell_delegated: self.current_shell_mode() == ShellMode::Delegated,
        }
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
        // `--resume`：首个 session/new 透明改走 load_session（一次性）。客户端
        // 拿回的是被恢复的旧 session id，并在响应前收到回放的历史 transcript。
        if let Some(target) = self.take_resume_target() {
            return self.resume_on_session_new(target, req, responder, cx).await;
        }

        let cwd_for_log = req.cwd.clone();
        let session_id = SessionId::new(new_session_id());
        let fs = self.fs_backend(&cx, &session_id, &req.cwd);
        let shell = self.shell_backend(&cx, &session_id, &req.cwd);
        let frontend = self.frontend();
        match self
            .agent
            .create_session(session_id, req.cwd, req.mcp_servers, fs, shell, frontend)
            .await
        {
            Ok(session) => {
                let config_options = session_config_options(session.as_ref()).await;
                // 起持久 event pump：把本 session 全生命周期的事件（含 driver 自主续转
                // turn）转成 session/update。§5.3。
                spawn_session_pump(session.clone(), session.id().clone(), cx);
                tracing::info!(
                    session_id = %short_session_id(session.id()),
                    cwd = %cwd_for_log.display(),
                    "session created"
                );
                responder.respond(
                    NewSessionResponse::new(session.id().clone()).config_options(config_options),
                )
            }
            Err(err) => {
                let acp_err = AcpError::CreateSession(err);
                tracing::warn!(error = %acp_err, "create_session failed");
                responder.respond_with_error(acp_err.into_wire_error())
            }
        }
    }

    /// `--resume` 路径下的 `session/new`：加载目标 session，回放 transcript，
    /// 把恢复出的（旧）session id 作为 `NewSessionResponse` 回给客户端。
    ///
    /// fs/shell 后端按本次 `session/new` 请求的 cwd 协商（resume 是在“此处此刻”
    /// 继续旧对话，运行环境用当前连接的协商结果，而非旧会话落盘的 cwd）。
    async fn resume_on_session_new(
        &self,
        target: SessionId,
        req: NewSessionRequest,
        responder: agent_client_protocol::Responder<NewSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), agent_client_protocol::Error> {
        let cwd_for_log = req.cwd.clone();
        let fs = self.fs_backend(&cx, &target, &req.cwd);
        let shell = self.shell_backend(&cx, &target, &req.cwd);
        let frontend = self.frontend();
        match self
            .agent
            .load_session(target.clone(), fs, shell, frontend)
            .await
        {
            Ok(session) => {
                let config_options = session_config_options(session.as_ref()).await;
                for notification in replay_notifications(session.id(), &session.history_snapshot())
                {
                    if let Err(err) = cx.send_notification(notification) {
                        tracing::warn!(?err, "failed to replay resumed session transcript");
                    }
                }
                spawn_session_pump(session.clone(), session.id().clone(), cx);
                tracing::info!(
                    session_id = %short_session_id(session.id()),
                    cwd = %cwd_for_log.display(),
                    "session resumed via session/new"
                );
                responder.respond(
                    NewSessionResponse::new(session.id().clone()).config_options(config_options),
                )
            }
            Err(err) => {
                let acp_err = match err {
                    AgentError::SessionNotFound(id) => AcpError::SessionNotFound {
                        session_id: id.0.to_string(),
                    },
                    other => AcpError::LoadSession(other),
                };
                tracing::warn!(error = %acp_err, "resume load_session failed");
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
        let frontend = self.frontend();
        match self
            .agent
            .load_session(session_id.clone(), fs, shell, frontend)
            .await
        {
            Ok(session) => {
                let config_options = session_config_options(session.as_ref()).await;
                for notification in replay_notifications(&session_id, &session.history_snapshot()) {
                    if let Err(err) = cx.send_notification(notification) {
                        tracing::warn!(?err, "failed to replay loaded session transcript");
                    }
                }
                // 起持久 event pump（同 session/new）——replay 之后，新事件由它接力。
                spawn_session_pump(session.clone(), session_id.clone(), cx);
                tracing::info!(
                    session_id = %short_session_id(session.id()),
                    cwd = %cwd_for_log.display(),
                    "session loaded"
                );
                responder.respond(LoadSessionResponse::new().config_options(config_options))
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

    async fn on_set_config_option(
        &self,
        req: SetSessionConfigOptionRequest,
        responder: agent_client_protocol::Responder<SetSessionConfigOptionResponse>,
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

        let config_id = req.config_id.0.to_string();
        let value = req.value.0.to_string();

        // 设值成功后统一回带刷新过的完整配置项集合——协议要求
        // SetSessionConfigOptionResponse 携带全量 config_options。
        let invalid_value = || {
            AcpError::InvalidConfigOption(format!(
                "unknown value `{value}` for config option `{config_id}`"
            ))
        };
        let apply_result = match config_id.as_str() {
            // 模型：转调 session.set_model（与 deprecated `session/set_model`
            // 同一后端）。未知 / 越界 model id → InvalidConfigOption。
            MODEL_CONFIG_ID => session
                .set_model(value.clone())
                .await
                .map_err(|_| invalid_value()),
            // 权限模式：转调 session.set_mode（与 deprecated `session/set_mode`
            // 同一后端）。未知 mode id → InvalidConfigOption。
            MODE_CONFIG_ID => session.set_mode(value.clone()).map_err(|_| invalid_value()),
            // thought-level：解析成 ReasoningEffort 覆盖。
            THOUGHT_LEVEL_CONFIG_ID => match parse_reasoning_value(&value) {
                Ok(effort) => {
                    session.set_reasoning_effort(effort);
                    Ok(())
                }
                Err(()) => Err(invalid_value()),
            },
            _ => Err(AcpError::InvalidConfigOption(format!(
                "unknown config option `{config_id}`"
            ))),
        };

        match apply_result {
            Ok(()) => responder.respond(SetSessionConfigOptionResponse::new(
                session_config_options(session.as_ref()).await,
            )),
            Err(acp_err) => {
                tracing::warn!(error = %acp_err, "session/set_config_option failed");
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
        // 等消息能在 turn 跑的同时被处理。事件投射不在这里做：由 session 级
        // 持久 event pump（session/new · load 时起）统一转发，含 driver 自发的
        // 自主续转 turn。详见 docs/proposals/task-arrange.md §5.3。
        cx.spawn(async move { run_prompt_turn(session, req.prompt, responder).await })
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

/// 同 [`serve`]，但带一次性 `--resume` 目标：首个 `session/new` 透明改走
/// load_session 恢复该会话（见 [`ServeState::resume_on_session_new`]）。
/// `resume = None` 时与 [`serve`] 等价。
pub async fn serve_with_resume(
    agent: Arc<dyn AgentCore>,
    resume: Option<SessionId>,
) -> Result<(), AcpError> {
    serve_on_with_resume(agent, Stdio::new(), resume).await
}

/// 在自定义 transport 上跑同一套 ACP handler。
///
/// 公共入口 [`serve`] 用 stdio；集成测试用 `Channel` 在进程内对接。
pub async fn serve_on<T>(agent: Arc<dyn AgentCore>, transport: T) -> Result<(), AcpError>
where
    T: ConnectTo<Agent> + 'static,
{
    serve_on_with_resume(agent, transport, None).await
}

/// [`serve_on`] + 一次性 resume 目标。见 [`serve_with_resume`]。
pub async fn serve_on_with_resume<T>(
    agent: Arc<dyn AgentCore>,
    transport: T,
    resume: Option<SessionId>,
) -> Result<(), AcpError>
where
    T: ConnectTo<Agent> + 'static,
{
    let state = Arc::new(ServeState::with_resume(agent, resume));

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
                async move |req: SetSessionConfigOptionRequest, responder, _cx| {
                    state.on_set_config_option(req, responder).await
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

/// 一次 `session/prompt`：跑 turn、按 run_turn 返回值 respond `PromptResponse`。
///
/// **不**在这里订阅 / 投射事件——`session/update` 通知由 session 级**持久 event pump**
/// （[`spawn_session_pump`]，session/new · load 时起）统一转发，含 driver 自发的自主续转
/// turn。本函数只负责把这一条 prompt 的 turn 结果回成 JSON-RPC 响应。
///
/// **排队（§5.2）**：撞上 `TurnInProgress`（driver 的自主续转 turn 恰在跑，或并发 prompt）
/// 时不立刻报错，短暂退避重试，让这条 prompt 排队等到 slot 空出——`session/prompt` 在
/// 协议语义上期望被处理。
#[tracing::instrument(name = "acp_prompt_turn", skip_all)]
async fn run_prompt_turn(
    session: Arc<dyn Session>,
    prompt: Vec<agent_client_protocol::schema::ContentBlock>,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> Result<(), agent_client_protocol::Error> {
    // 排队重试：自主续转 turn 通常很短，退避几次即可拿到 slot。退避上限兜底防止
    // 卡死（极端情况下退化为报错，让客户端重发）。
    const MAX_RETRIES: u32 = 100;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(20);

    let mut attempt = 0;
    let result = loop {
        match session.run_turn(prompt.clone()).await {
            Err(TurnError::TurnInProgress) if attempt < MAX_RETRIES => {
                attempt += 1;
                tokio::time::sleep(BACKOFF).await;
            }
            other => break other,
        }
    };

    match result {
        Ok(stop) => responder.respond(PromptResponse::new(stop)),
        Err(err) => {
            let acp_err = AcpError::Turn(err);
            tracing::warn!(error = %acp_err, "turn failed; responding with wire error");
            responder.respond_with_error(acp_err.into_wire_error())
        }
    }
}

/// Session 级**持久 event pump**：订阅一次该 session 的事件流，把每个事件投射到 wire，
/// 跨所有 turn 存活（含 driver 自发的自主续转 turn）。session/new · load 时 spawn。
///
/// 这是阶段二补齐的关键一环：原先事件只在一次 `session/prompt` 期间被订阅转发，自主续转
/// turn 的事件无人接收。详见 docs/proposals/task-arrange.md §5.3。
///
/// 生命周期：`session.subscribe()` 的事件流在 session drop（EventEmitter 析构）时结束，
/// `events.next()` 返回 `None`，pump 自然退出。pump 持 `Arc<dyn Session>`，与 AgentCore
/// 的 sessions 表同样强引用——v0 session 随进程存活，pump 亦然。
fn spawn_session_pump(session: Arc<dyn Session>, session_id: SessionId, cx: ConnectionTo<Client>) {
    let mut events = session.subscribe();
    let cx_for_pump = cx.clone();
    let _ = cx.spawn(async move {
        while let Some(event) = events.next().await {
            // TurnStarted / TurnEnded 是 turn 边界标记，wire 上由 PromptResponse 表达，
            // 不投射成 session/update（project 已把它们归到 EndTurn/Ignore）。
            if let Err(err) = handle_event(&session, &session_id, event, &cx_for_pump) {
                tracing::warn!(?err, "session pump failed to project agent event");
            }
        }
        tracing::debug!("session event pump exited (stream closed)");
        Ok(())
    });
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
