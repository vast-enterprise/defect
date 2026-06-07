//! Public entry point for `defect-acp`.
//!
//! Starts a stdio JSON-RPC service, registers ACP v1 client-to-agent method handlers,
//! and exposes [`AgentCore`] / [`Session`] over the wire.
//!
//! ACP server — serves the agent over stdin/stdout or a Unix socket.

use std::sync::{Arc, RwLock};

use agent_client_protocol::schema::{
    AgentCapabilities, AuthenticateRequest, AvailableCommand, AvailableCommandsUpdate,
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PromptRequest, PromptResponse, RequestPermissionOutcome,
    RequestPermissionRequest, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionConfigValueId, SessionId, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, StopReason, TextContent,
};
use agent_client_protocol::{Agent, Client, ConnectTo, ConnectionTo, Stdio};
use defect_agent::event::{AgentEvent, PermissionResolution};
use defect_agent::fs::FsBackend;
use defect_agent::llm::{ModelCandidate, ProviderError, ReasoningEffort};
use defect_agent::session::{
    AgentCore, AgentError, Frontend, ModelSelection, Session, TurnError, new_session_id,
};
use defect_agent::shell::ShellBackend;
use defect_tools::{LocalFsBackend, LocalShellBackend};
use futures::StreamExt;
use serde_json::json;

use crate::fs::AcpFsBackend;
use crate::project::{PermissionAsk, Projection, project, replay_notifications};
use crate::shell::AcpShellBackend;

/// Negotiated client fs capabilities (connection-level).
///
/// Read from [`ClientCapabilities::fs`] in the `initialize` handler and stored; the
/// `session/new` handler uses this to select [`AcpFsBackend`] or [`LocalFsBackend`].
/// See ACP filesystem design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FsMode {
    /// Client declares both `read_text_file` and `write_text_file`: full delegation.
    Delegated,
    /// If either capability is missing or the `fs` field is not declared, the entire
    /// group falls back to local (no mixing).
    Local,
}

fn decide_fs_mode(client_caps: &ClientCapabilities) -> FsMode {
    if client_caps.fs.read_text_file && client_caps.fs.write_text_file {
        FsMode::Delegated
    } else {
        FsMode::Local
    }
}

/// Result of terminal capability negotiation at the connection level.
///
/// Written by the `initialize` handler after reading [`ClientCapabilities::terminal`];
/// used by `session/new` / `session/load` to select [`AcpShellBackend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellMode {
    /// Client declares full `terminal/*` support: fully delegated.
    Delegated,
    /// Field is false or missing: entire group falls back to local (no mixing).
    Local,
}

fn decide_shell_mode(client_caps: &ClientCapabilities) -> ShellMode {
    if client_caps.terminal {
        ShellMode::Delegated
    } else {
        ShellMode::Local
    }
}

/// Public error type for `defect-acp`.
///
/// Demarcation rule: each variant corresponds to a wire-stable error shape — whether the
/// session exists, session creation succeeded, or a turn completed. Downstream LLM / tool
/// failures are classified by [`TurnError`] itself (this layer does not further decompose
/// them).
///
/// Projection rule: see [`AcpError::into_wire_error`]; variant → JSON-RPC ErrorCode +
/// structured `data` field. Diagnostic fields (`session_id` / `request_id` etc.) go into
/// `data`, not smeared into `message`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AcpError {
    /// JSON-RPC / stdio transport layer failure. Only used by the top-level `?` in
    /// [`serve_on`]; this variant is never constructed inside any handler.
    #[error("acp transport error: {0}")]
    Transport(agent_client_protocol::Error),

    /// The session referenced by `session/prompt` or `session/cancel` does not exist on
    /// the agent side.
    #[error("session not found: {session_id}")]
    SessionNotFound { session_id: String },

    /// `session/new` failed to create a session (e.g., cwd does not exist, MCP startup
    /// failed).
    #[error("create_session failed: {0}")]
    CreateSession(#[source] AgentError),

    #[error("load_session failed: {0}")]
    LoadSession(#[source] AgentError),

    /// `session/set_config_option` received an unknown config_id or invalid value.
    #[error("invalid session config option: {0}")]
    InvalidConfigOption(String),

    /// `session/prompt` failed while running a turn (provider error after retries
    /// exhausted / main loop invariant broken).
    #[error("turn failed: {0}")]
    Turn(#[source] TurnError),

    /// The turn task was dropped before returning a stop reason (should be unreachable;
    /// kept as a safety net).
    #[error("turn task dropped before completion")]
    TurnDropped,

    /// Client requested `authenticate`, but it is not currently supported.
    #[error("authentication not supported")]
    AuthNotSupported,
}

impl From<agent_client_protocol::Error> for AcpError {
    fn from(err: agent_client_protocol::Error) -> Self {
        AcpError::Transport(err)
    }
}

impl AcpError {
    /// Project into an ACP wire `Error`: selects an `ErrorCode` and attaches structured
    /// diagnostic fields in `data`.
    ///
    /// Callers (handlers) use this in
    /// [`agent_client_protocol::Responder::respond_with_error`]
    /// instead of hand-rolling [`agent_client_protocol::util::internal_error`] +
    /// `format!`, so clients can reliably match on `code` / read `data.kind` rather than
    /// parsing strings.
    pub fn into_wire_error(self) -> agent_client_protocol::Error {
        use agent_client_protocol::Error as Wire;
        use agent_client_protocol::schema::ErrorCode;
        match self {
            AcpError::Transport(err) => err,

            AcpError::SessionNotFound { session_id } => {
                // Use `ResourceNotFound` instead of `InternalError` — this is "client
                // referenced a non-existent resource", a client-recoverable 4xx-class
                // semantic.
                Wire::resource_not_found(Some(session_id))
            }

            AcpError::CreateSession(err) => {
                // Place the inner Display impl into the wire `message` — client UIs
                // (acpx, etc.) read `message` directly for rendering. The default
                // placeholder "Internal error" buries all diagnostic information inside
                // `data`, so users only see "RUNTIME: Internal error".
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
                // Use the inner `Display` as the wire `message` — client UIs only read
                // the `message` field by default. Putting "Internal error" as a
                // placeholder and burying the real detail in `data` would show users
                // meaningless text like "RUNTIME: Internal error".
                // Note: choosing the right code is tricky — acpx maps -32001/-32002 to
                // NO_SESSION (misidentifying a meeting session), so the Provider also
                // uses `InternalError`. The message text itself ("rate limit" / "model
                // not found") lets acpx's text-error-rules match the appropriate hint.
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

            // method_not_found is more appropriate than internal_error for "unimplemented
            // method"
            AcpError::AuthNotSupported => Wire::method_not_found().data(json!({
                "kind": "auth_not_supported",
                "message": "authentication not supported",
            })),
        }
    }
}

/// Serializes a [`TurnError`] into the wire `data` field. Distinguishes two sub-kinds:
/// - `provider` — a provider error that still fails after retries are exhausted, includes
///   `retry_hint` / `request_id` so the client can prompt the user to "switch models /
///   try again later"
/// - `internal` — a main-loop invariant was violated, for diagnostics only
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
        // `TurnError` is `#[non_exhaustive]`; future variants fall through to this
        // internal catch-all, keeping compilation unblocked. When adding a new variant,
        // prefer writing a dedicated arm above this one.
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

/// The stable config id for the thought-level selector in `session/set_config_option`.
const THOUGHT_LEVEL_CONFIG_ID: &str = "reasoning_effort";

/// Stable config id for the permission mode selector in `session/set_config_option`.
///
/// Session Config Options replace the old Session Modes API: modern clients (e.g. Zed
/// ≥ 1.4) only read `configOptions` and ignore the deprecated `modes` field in the
/// response.
/// Therefore the permission mode must **also** be exposed as a config option with
/// `category = Mode`, otherwise the client will not render the mode selector. The `modes`
/// field is kept for backward compatibility with older clients.
const MODE_CONFIG_ID: &str = "permission_mode";

/// Stable config id for the model selector in `session/set_config_option`.
///
/// Analogous to [`MODE_CONFIG_ID`] — modern clients only read `config_options`, so the
/// model must also be exposed as a config option with `category = Model`, otherwise the
/// model selector is not rendered. The deprecated `models` field in the response is kept
/// for backward compatibility with older clients.
const MODEL_CONFIG_ID: &str = "model";

/// The value id for the "not set" tier at the thought level (uses the provider default).
/// Other tiers use [`ReasoningEffort`] wire tokens (`minimal` / `low` / …).
const REASONING_DEFAULT_VALUE: &str = "default";

/// Parses an ACP value id into a [`ReasoningEffort`] override. `"default"` → `None`
/// (clears the override); otherwise matches wire tokens; unknown tokens return `Err`.
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

/// Returns the ACP value id corresponding to the given [`ReasoningEffort`] override.
/// `None` maps to `"default"`.
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

/// Build the session's config option list (ACP `config_options`).
///
/// Contains three selects: model (`category = Model`), permission mode (`category =
/// Mode`, from the session's mode directory), and thought-level (`category =
/// ThoughtLevel`, 6 levels of `reasoning_effort` + "default").
///
/// **All three must be exposed via config options**: Session Config Options replace the
/// old Session Modes / Models API. Modern clients (e.g. Zed ≥ 1.4) only render selectors
/// from `config_options` and ignore the deprecated `models` / `modes` fields in the
/// response (those are kept for backward compatibility with older clients). See
/// [`MODE_CONFIG_ID`] / [`MODEL_CONFIG_ID`].
async fn session_config_options(session: &dyn Session) -> Vec<SessionConfigOption> {
    let mut out = Vec::new();

    // 0) Model selector. Candidates come from the registry (no network request, always
    // resolvable). If no candidates are available, fall back to listing only the current
    // model to ensure the dropdown is non-empty.
    {
        let current_model = session.current_model();
        let current_vendor = session.provider_info().vendor;
        // The selection key is a `(vendor, model)` pair — models with the same name can
        // come from multiple providers. The value ID is encoded as `vendor::model`
        // (vendor is a TOML section name and never contains `::`; model may contain
        // arbitrary characters, so parsing splits on the first `::`). The current value
        // is encoded the same way.
        let current_value = encode_model_value(&current_vendor, &current_model);
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
                let value = encode_model_value(&c.provider.vendor, &c.model.id);
                SessionConfigSelectOption::new(SessionConfigValueId::new(value), name)
                    .description(Some(description))
            })
            .collect::<Vec<_>>();
        if !model_options
            .iter()
            .any(|o| o.value.0.as_ref() == current_value)
        {
            // Fallback: the current model is not in the candidates (should not happen in
            // theory). Still list it to avoid an empty dropdown.
            model_options.insert(
                0,
                SessionConfigSelectOption::new(
                    SessionConfigValueId::new(current_value.clone()),
                    current_model.clone(),
                ),
            );
        }
        out.push(
            SessionConfigOption::select(
                MODEL_CONFIG_ID,
                "Model",
                SessionConfigValueId::new(current_value),
                model_options,
            )
            .category(Some(SessionConfigOptionCategory::Model))
            .description(Some("The model this session uses".to_string())),
        );
    }

    // 1) Permission mode selector (only when the session has a mode directory).
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
                "Approval policy for tool calls: read-only / ask before writes / allow all / deny all".to_string(),
            )),
        );
    }

    // 2) thought-level selector. Order: default first, then increasing intensity —
    // matching the OpenAI wire enum.
    let current_effort = reasoning_value_id(session.current_reasoning_effort());
    let effort_options = vec![
        SessionConfigSelectOption::new(
            SessionConfigValueId::new(REASONING_DEFAULT_VALUE),
            "Default",
        )
        .description(Some(
            "Follow the provider default; do not send reasoning_effort".to_string(),
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
            "Reasoning-effort level for the OpenAI-compatible protocol; ignored by providers that do not support it".to_string(),
        )),
    );

    out
}

/// A description string for a model candidate in the config-option selector: `provider:
/// X, context_window=…, max_output_tokens=…, deprecated` (omitted fields are skipped).
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

/// Separator for model-selector value IDs. The vendor is a TOML `[providers.<name>]`
/// section name, which never contains `::`; the model ID may contain arbitrary characters
/// (e.g. `us.anthropic.claude:1`), so [`decode_model_value`] splits only on the **first**
/// `::` to preserve the model side intact.
const MODEL_VALUE_SEP: &str = "::";

/// Encodes a `(vendor, model)` selection pair into a single ACP value id.
fn encode_model_value(vendor: &str, model: &str) -> String {
    format!("{vendor}{MODEL_VALUE_SEP}{model}")
}

/// Decode a value id produced by [`encode_model_value`] back into a [`ModelSelection`].
/// Returns `None` if the separator is missing (invalid/legacy format).
fn decode_model_value(value: &str) -> Option<ModelSelection> {
    let (vendor, model) = value.split_once(MODEL_VALUE_SEP)?;
    Some(ModelSelection {
        provider: vendor.to_string(),
        model: model.to_string(),
    })
}

/// Connection-level shared state. `serve_on` clones an `Arc<ServeState>` for each
/// handler.
///
/// `agent` is the injected core, read-only for the connection's lifetime. `fs_mode` /
/// `shell_mode` are written by `initialize` and read by subsequent `session/new` and
/// `session/load` — the `RwLock` protects this one-time handshake result under a
/// read-heavy, write-light workload.
struct ServeState {
    agent: Arc<dyn AgentCore>,
    /// Per the ACP spec, clients must call `initialize` before `session/new`. The default
    /// `Local` is a conservative fallback — calling `session/new` before `initialize` is
    /// a protocol violation, but even so we prefer to use the local disk rather than
    /// making a raw reverse request.
    fs_mode: RwLock<FsMode>,
    /// Shell backend selection. Defaults to `Local` — same conservative fallback as
    /// [`Self::fs_mode`].
    shell_mode: RwLock<ShellMode>,
    /// `--resume` target. The ACP client drives the session lifecycle; the CLI cannot
    /// directly initiate a load, so the target id is stored here: the **first**
    /// `session/new` transparently switches to `load_session` and replays that session,
    /// then is cleared (one-shot). `None` = no resume.
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

    /// Takes and clears the one-shot resume target. Returns `None` on subsequent calls.
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

    /// Assemble the negotiated fs / shell modes into a [`Frontend::Acp`] — the agent uses
    /// this to indicate in the `# Environment` section of the system prompt whether
    /// file/command execution is local or delegated.
    fn frontend(&self) -> Frontend {
        Frontend::Acp {
            fs_delegated: self.current_fs_mode() == FsMode::Delegated,
            shell_delegated: self.current_shell_mode() == ShellMode::Delegated,
        }
    }

    /// Assemble the fs backend from the connection-level `fs_mode` and the session-level
    /// `cwd`.
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

    /// Assemble a shell backend from the connection-level shell mode and session-level
    /// cwd.
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
        // Auth is not currently enabled; any auth request from a client is rejected as
        // unimplemented.
        responder.respond_with_error(AcpError::AuthNotSupported.into_wire_error())
    }

    async fn on_session_new(
        &self,
        req: NewSessionRequest,
        responder: agent_client_protocol::Responder<NewSessionResponse>,
        cx: ConnectionTo<Client>,
    ) -> Result<(), agent_client_protocol::Error> {
        // `--resume`: transparently redirects the first `session/new` to `load_session`
        // (one-shot). The client receives the restored old session ID and gets the
        // replayed history transcript before the response.
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
                // Spawn a persistent event pump that forwards all events for the lifetime
                // of this session (including driver-initiated turn continuations) as
                // `session/update`.
                spawn_session_pump(session.clone(), session.id().clone(), cx.clone());
                announce_commands(session.id(), &cx);
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

    /// `session/new` on the `--resume` path: loads the target session, replays the
    /// transcript,
    /// and returns the recovered (old) session id as a `NewSessionResponse` to the
    /// client.
    ///
    /// The fs/shell backends are negotiated based on the cwd of this `session/new`
    /// request
    /// (resume continues the old conversation "here and now"; the runtime environment
    /// uses
    /// the current connection's negotiation result, not the cwd persisted in the old
    /// session).
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
                spawn_session_pump(session.clone(), session.id().clone(), cx.clone());
                announce_commands(session.id(), &cx);
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
                // Start a persistent event pump (same as session/new) — after replay, it
                // takes over new events.
                spawn_session_pump(session.clone(), session_id.clone(), cx.clone());
                announce_commands(&session_id, &cx);
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

        // After a successful set, the response must carry the full refreshed set of
        // config options — the protocol requires that `SetSessionConfigOptionResponse`
        // includes all `config_options`.
        let invalid_value = || {
            AcpError::InvalidConfigOption(format!(
                "unknown value `{value}` for config option `{config_id}`"
            ))
        };
        let apply_result = match config_id.as_str() {
            // Model: delegates to `session.set_model` (same backend as the deprecated
            // `session/set_model`). Unknown or out-of-range model id →
            // `InvalidConfigOption`.
            MODEL_CONFIG_ID => match decode_model_value(&value) {
                Some(selection) => session
                    .set_model(selection)
                    .await
                    .map_err(|_| invalid_value()),
                None => Err(invalid_value()),
            },
            // Permission mode: delegates to `session.set_mode` (same backend as the
            // deprecated `session/set_mode`). Unknown mode id → `InvalidConfigOption`.
            MODE_CONFIG_ID => session.set_mode(value.clone()).map_err(|_| invalid_value()),
            // thought-level: parse into a `ReasoningEffort` override.
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
        // Slash command interception: a leading `/compact` / `/context` is run
        // out-of-band (no LLM turn) and replied to as an agent message. Anything else is
        // a normal prompt.
        if let Some((name, _args)) = parse_slash_command(&req.prompt)
            && matches!(name.as_str(), CMD_COMPACT | CMD_CONTEXT)
        {
            let cx_for_cmd = cx.clone();
            return cx.spawn(async move {
                let stop = run_slash_command(&session, &session_id, &name, &cx_for_cmd).await;
                responder.respond(PromptResponse::new(stop))
            });
        }

        // Spawn the turn execution into a background task so the handler returns
        // immediately and the dispatch loop is not blocked; this allows subsequent
        // cancel/resolve messages to be processed while the turn runs. Event projection
        // is not done here — it is handled uniformly by the session-level persistent
        // event pump (started on session/new and load), including the driver's autonomous
        // continuation to the next turn.
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

/// Starts the stdio ACP service, blocking until the peer disconnects.
///
/// `agent` is assembled by `defect-cli` (with provider, tools, and configuration) and
/// then injected.
pub async fn serve(agent: Arc<dyn AgentCore>) -> Result<(), AcpError> {
    serve_on(agent, Stdio::new()).await
}

/// Like [`serve`], but with a one-shot `--resume` target: the first `session/new`
/// transparently calls `load_session` to restore that session (see
/// `ServeState::resume_on_session_new`). When `resume = None`, behaves identically to
/// [`serve`].
pub async fn serve_with_resume(
    agent: Arc<dyn AgentCore>,
    resume: Option<SessionId>,
) -> Result<(), AcpError> {
    serve_on_with_resume(agent, Stdio::new(), resume).await
}

/// Runs the same ACP handler on a custom transport.
///
/// The public entry point [`serve`] uses stdio; integration tests use `Channel` for
/// in-process communication.
pub async fn serve_on<T>(agent: Arc<dyn AgentCore>, transport: T) -> Result<(), AcpError>
where
    T: ConnectTo<Agent> + 'static,
{
    serve_on_with_resume(agent, transport, None).await
}

/// [`serve_on`] with a one-shot resume target. See [`serve_with_resume`].
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

/// A single `session/prompt`: runs a turn and responds with a `PromptResponse` based on
/// the `run_turn` return value.
///
/// **Does not** subscribe to or emit events here — `session/update` notifications are
/// forwarded uniformly by the session-level **persistent event pump**
/// ([`spawn_session_pump`], started at session/new · load time), including
/// driver-initiated autonomous turn continuations. This function only returns the turn
/// result for this single prompt as a JSON-RPC response.
///
/// **Queuing**: When a `TurnInProgress` is encountered (the driver's autonomous
/// continuation turn is running, or a concurrent prompt is in progress), instead of
/// immediately returning an error, briefly back off and retry, allowing this prompt to
/// queue until a slot becomes available — `session/prompt` is expected to be processed
/// per the protocol semantics.
#[tracing::instrument(name = "acp_prompt_turn", skip_all)]
async fn run_prompt_turn(
    session: Arc<dyn Session>,
    prompt: Vec<agent_client_protocol::schema::ContentBlock>,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> Result<(), agent_client_protocol::Error> {
    // Retry with backoff: a self-continued turn is usually short, so a few backoff
    // attempts should acquire the slot. The backoff cap prevents deadlock (in extreme
    // cases it degrades to an error, letting the client retry).
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

/// A persistent event pump at the session level: subscribes once to the session's event
/// stream and projects each event onto the wire, surviving across all turns (including
/// driver-initiated automatic turn renewal). Spawned on `session/new` and `load`.
///
/// This is a key piece of phase-two completion: previously events were only subscribed
/// and forwarded during a single `session/prompt`; automatic turn renewal had no consumer
/// for turn events — in background mode, events were dropped.
///
/// Lifecycle: the event stream from `session.subscribe()` ends when the session is
/// dropped (the `EventEmitter` is destroyed), at which point `events.next()` returns
/// `None` and the pump exits naturally. The pump holds an `Arc<dyn Session>`, the same
/// strong reference used by `AgentCore`'s sessions table — sessions currently live for
/// the process lifetime, and so does the pump.
fn spawn_session_pump(session: Arc<dyn Session>, session_id: SessionId, cx: ConnectionTo<Client>) {
    let mut events = session.subscribe();
    let cx_for_pump = cx.clone();
    let _ = cx.spawn(async move {
        while let Some(event) = events.next().await {
            // TurnStarted / TurnEnded are turn boundary markers; on the wire they are
            // represented as PromptResponse and are not projected into session/update
            // (the project already classifies them as EndTurn/Ignore).
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

/// Short session ID for tracing spans / logs: first 12 characters. Diagnostic use only.
fn short_session_id(id: &SessionId) -> &str {
    let s: &str = id.0.as_ref();
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Sends a reverse request `session/request_permission` and writes the client's response
/// back to [`Session`].
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

// ===== Slash commands =====
//
// ACP models slash commands as agent-declared `AvailableCommand`s: the agent broadcasts
// the list via `session/update`, and when the user types `/<name> …` the client sends it
// back as an ordinary text prompt. We therefore (1) announce the command list right after
// a session is created/loaded, and (2) intercept a leading `/name` in `on_prompt`,
// running it out-of-band instead of feeding it to the LLM.

const CMD_COMPACT: &str = "compact";
const CMD_CONTEXT: &str = "context";

/// Build the `session/update` that advertises our slash commands to the client.
fn available_commands_notification(session_id: &SessionId) -> SessionNotification {
    let commands = vec![
        AvailableCommand::new(
            CMD_COMPACT,
            "Compact the conversation now: summarize older turns to free up context.",
        ),
        AvailableCommand::new(
            CMD_CONTEXT,
            "Show current context usage (tokens used vs. the model's context window).",
        ),
    ];
    SessionNotification::new(
        session_id.clone(),
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
    )
}

/// Announce the available slash commands for a freshly created / loaded / resumed session.
/// Best-effort: a failed send is logged but does not abort session setup.
fn announce_commands(session_id: &SessionId, cx: &ConnectionTo<Client>) {
    if let Err(err) = cx.send_notification(available_commands_notification(session_id)) {
        tracing::warn!(?err, "failed to announce available commands");
    }
}

/// If the prompt is a slash command we recognize, returns `(name, args)` with the leading
/// `/` stripped; otherwise `None` (the prompt is a normal message). Only the first text
/// block is inspected, and only when it starts with `/` after trimming leading whitespace.
fn parse_slash_command(prompt: &[ContentBlock]) -> Option<(String, String)> {
    let ContentBlock::Text(first) = prompt.first()? else {
        return None;
    };
    let trimmed = first.text.trim_start();
    let rest = trimmed.strip_prefix('/')?;
    // Split the command name from its argument text (everything after the first run of
    // whitespace). `/compact` → ("compact", ""); `/context  ` → ("context", "").
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((name, args)) => (name, args.trim()),
        None => (rest, ""),
    };
    Some((name.to_string(), args.to_string()))
}

/// Send a one-line agent message back to the client as the command's visible result.
/// Slash commands have no dedicated response shape in ACP, so we surface the outcome the
/// same way the agent would speak — every client can already render it.
fn send_command_reply(session_id: &SessionId, text: String, cx: &ConnectionTo<Client>) {
    let notif = SessionNotification::new(
        session_id.clone(),
        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(TextContent::new(
            text,
        )))),
    );
    if let Err(err) = cx.send_notification(notif) {
        tracing::warn!(?err, "failed to send slash command reply");
    }
}

/// Format a token count compactly: `1234` → `1.2k`, `999` → `999`.
fn fmt_tokens(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Run a recognized slash command out-of-band and reply to the client. Returns the
/// `PromptResponse` stop reason to complete the original `session/prompt` (always
/// `EndTurn` — the command does not start an LLM turn).
async fn run_slash_command(
    session: &Arc<dyn Session>,
    session_id: &SessionId,
    name: &str,
    cx: &ConnectionTo<Client>,
) -> StopReason {
    match name {
        CMD_CONTEXT => {
            let status = session.context_status();
            let reply = match (status.used_tokens, status.context_window) {
                (Some(used), Some(window)) => {
                    let pct = status.ratio.map_or(0.0, |r| r * 100.0);
                    format!(
                        "Context: {} / {} tokens ({:.0}%) used.",
                        fmt_tokens(used),
                        fmt_tokens(window),
                        pct
                    )
                }
                (Some(used), None) => format!(
                    "Context: {} tokens used (model context window unknown).",
                    fmt_tokens(used)
                ),
                (None, _) => "Context: no usage recorded yet.".to_string(),
            };
            send_command_reply(session_id, reply, cx);
        }
        CMD_COMPACT => {
            let reply = match session.compact_now().await {
                Ok(Some(report)) => format!(
                    "Compacted context: {} → {} tokens.",
                    fmt_tokens(report.tokens_before),
                    fmt_tokens(report.tokens_after)
                ),
                Ok(None) => "Nothing to compact yet — the conversation is too short to summarize."
                    .to_string(),
                Err(TurnError::TurnInProgress) => {
                    "Cannot compact while a turn is running. Cancel it or wait, then try again."
                        .to_string()
                }
                Err(err) => format!("Compaction failed: {err}"),
            };
            send_command_reply(session_id, reply, cx);
        }
        _ => unreachable!("run_slash_command called with an unrecognized command"),
    }
    StopReason::EndTurn
}

#[cfg(test)]
mod tests;
