//! [`Session`] / [`AgentCore`] 的 v0 默认实现。
//!
//! 装配关系：
//!
//! ```text
//! DefaultAgentCore
//!   ├── Arc<dyn LlmProvider>          (装配时传入，所有 session 共享)
//!   ├── Arc<dyn ToolRegistry>         (进程级内置工具)
//!   ├── TurnConfig                    (默认配置)
//!   └── DashMap<SessionId, Arc<dyn Session>>
//!
//! DefaultSession
//!   ├── id: SessionId
//!   ├── cwd: PathBuf
//!   ├── history: Box<dyn History>
//!   ├── tools:   Arc<dyn ToolRegistry>   (CompositeRegistry: per-session + process)
//!   ├── provider: Arc<dyn LlmProvider>
//!   ├── events:   Arc<EventEmitter>
//!   ├── permissions: Arc<PermissionGate>
//!   ├── turn_lock: tokio::sync::Mutex<TurnSlot>
//!   └── config: RwLock<TurnConfig>
//! ```
//!
//! turn 互斥用 `Mutex<TurnSlot>`：`run_turn` 在最外层 `try_lock`，失败即返回
//! `TurnError::TurnInProgress`。`TurnSlot` 内部存当前 turn 的
//! [`CancellationToken`]，`cancel_turn` 取出后 `cancel()`。

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use agent_client_protocol::schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use dashmap::DashMap;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::error::BoxError;
use crate::event::PermissionResolution;
use crate::fs::FsBackend;
use crate::hooks::{HookCtx, HookEngine, HookEvent, NoopHookEngine, SessionSource};
use crate::http::{HttpClient, NoopHttpClient};
use crate::llm::{
    HostedCapabilities, LlmProvider, ModelInfo, ProviderError, ProviderErrorKind, ProviderInfo,
};
use crate::policy::{AskWritesPolicy, SandboxPolicy};
use crate::session::capabilities::{ResolvedSessionCapabilities, SessionCapabilitiesConfig};
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::prompt::resolve_system_prompt;
use crate::session::tool_registry::{CompositeRegistry, StaticToolRegistry};
use crate::session::turn::{TurnConfig, TurnRunner};
use crate::session::{
    AgentCore, AgentError, EventStream, History, Session, SessionCreateInfo, SessionLoader,
    SessionObserver, SessionToolFactory, ToolRegistry, TurnError, VecHistory,
};
use crate::shell::ShellBackend;

/// 默认 [`AgentCore`]。
pub struct DefaultAgentCore {
    provider: Arc<dyn LlmProvider>,
    process_tools: Arc<dyn ToolRegistry>,
    policy: Arc<dyn SandboxPolicy>,
    config: RwLock<TurnConfig>,
    capabilities: SessionCapabilitiesConfig,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    /// 进程级 HTTP fetch 后端。所有 session 共享一份——HTTP 没有 per-client
    /// capability 协商，多 session 间也无须隔离连接池。CLI 入口按
    /// [`HttpClientConfig`] 构造一次后注入；测试 / `echo` provider 走
    /// [`NoopHttpClient`]。
    ///
    /// [`HttpClientConfig`]: defect_config::HttpClientConfig
    http: Arc<dyn HttpClient>,
    /// 进程级 hook 引擎。所有 session 共享——hook 配置走全局 + per-session
    /// matcher（`docs/internal/hooks.md` §5）。CLI 入口装配；不显式注入时走
    /// [`NoopHookEngine`]，等价"未配置 hook = 主循环不变"。
    hook_engine: Arc<dyn HookEngine>,
    sessions: DashMap<SessionId, Arc<dyn Session>>,
}

impl DefaultAgentCore {
    pub fn builder() -> DefaultAgentCoreBuilder {
        DefaultAgentCoreBuilder::default()
    }
}

#[derive(Default)]
pub struct DefaultAgentCoreBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    process_tools: Option<Arc<dyn ToolRegistry>>,
    policy: Option<Arc<dyn SandboxPolicy>>,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    http: Option<Arc<dyn HttpClient>>,
    hook_engine: Option<Arc<dyn HookEngine>>,
    config: TurnConfig,
    capabilities: SessionCapabilitiesConfig,
}

impl DefaultAgentCoreBuilder {
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn process_tools(mut self, tools: Arc<dyn ToolRegistry>) -> Self {
        self.process_tools = Some(tools);
        self
    }

    pub fn policy(mut self, policy: Arc<dyn SandboxPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn session_loader(mut self, loader: Arc<dyn SessionLoader>) -> Self {
        self.loader = Some(loader);
        self
    }

    pub fn session_tool_factory(mut self, factory: Arc<dyn SessionToolFactory>) -> Self {
        self.session_tools = Some(factory);
        self
    }

    pub fn observe_session(mut self, observer: Arc<dyn SessionObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    pub fn config(mut self, config: TurnConfig) -> Self {
        self.config = config;
        self
    }

    pub fn capabilities(mut self, capabilities: SessionCapabilitiesConfig) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// 设置进程级 HTTP fetch 后端。未设置时退化为 [`NoopHttpClient`]——
    /// 任何 `fetch` 调用都会以 [`crate::http::HttpClientError::Transport`]
    /// 失败，便于不需要网络的测试 / `echo` 装配跳过真实 HTTP 栈构造。
    pub fn http(mut self, http: Arc<dyn HttpClient>) -> Self {
        self.http = Some(http);
        self
    }

    /// 设置进程级 hook 引擎。未设置时退化为 [`NoopHookEngine`]——所有 hook
    /// 调用直接返回 `Pass`，主循环行为与未引入 hook 系统时一致。
    pub fn hook_engine(mut self, hook_engine: Arc<dyn HookEngine>) -> Self {
        self.hook_engine = Some(hook_engine);
        self
    }

    /// # Panics
    /// 如果 `provider` 没有设置。
    pub fn build(self) -> DefaultAgentCore {
        DefaultAgentCore {
            provider: self.provider.expect("DefaultAgentCore requires a provider"),
            process_tools: self
                .process_tools
                .unwrap_or_else(|| Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>),
            policy: self
                .policy
                .unwrap_or_else(|| Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>),
            loader: self.loader,
            session_tools: self.session_tools,
            observers: self.observers,
            http: self
                .http
                .unwrap_or_else(|| Arc::new(NoopHttpClient) as Arc<dyn HttpClient>),
            hook_engine: self
                .hook_engine
                .unwrap_or_else(|| Arc::new(NoopHookEngine) as Arc<dyn HookEngine>),
            config: RwLock::new(self.config),
            capabilities: self.capabilities,
            sessions: DashMap::new(),
        }
    }
}

impl AgentCore for DefaultAgentCore {
    fn create_session(
        &self,
        id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>> {
        Box::pin(async move {
            if !cwd.is_absolute() || !cwd.exists() {
                return Err(AgentError::InvalidCwd(cwd));
            }
            let session_cwd = cwd.clone();
            if self.sessions.contains_key(&id) {
                return Err(AgentError::DuplicateSessionId(id));
            }

            let resolved = ResolvedSessionCapabilities::resolve(
                self.capabilities,
                self.provider.hosted_capabilities(),
                &self.provider.info().vendor,
            )?;

            let session_tools = match &self.session_tools {
                Some(factory) => factory
                    .build_registry(cwd.clone(), mcp_servers.clone())
                    .await
                    .map_err(|source| AgentError::McpStartup {
                        server: "session_tools".to_string(),
                        source,
                    })?,
                None => Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>,
            };
            let composite: Arc<dyn ToolRegistry> = Arc::new(CompositeRegistry::new(
                session_tools,
                self.process_tools.clone(),
            ));

            // SessionStart hook（Sync 拦截，但 block 字段被引擎丢弃；
            // 仅吸收 outcome.append 作为系统 prompt 后缀候选）。
            let session_start_append = {
                let cancel = CancellationToken::new();
                let ctx = HookCtx::new(&id, &cwd, cancel);
                let event = HookEvent::SessionStart {
                    source: SessionSource::New,
                    cwd: &cwd,
                };
                let outcome = self.hook_engine.fire(event, ctx).await;
                outcome.append
            };

            let session = Arc::new(DefaultSession {
                id: id.clone(),
                cwd,
                history: Box::new(VecHistory::new()) as Box<dyn History>,
                tools: composite,
                provider: self.provider.clone(),
                policy: self.policy.clone(),
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                config: RwLock::new(
                    self.config
                        .read()
                        .expect("DefaultAgentCore config rwlock poisoned")
                        .clone(),
                ),
                hosted_capabilities: resolved.hosted,
                fs,
                shell,
                http: self.http.clone(),
                hook_engine: self.hook_engine.clone(),
                session_start_append,
            }) as Arc<dyn Session>;

            let session_info = SessionCreateInfo {
                id: id.clone(),
                cwd: session_cwd,
                mcp_servers,
            };
            for observer in &self.observers {
                observer
                    .on_session_created(session.clone(), session_info.clone())
                    .map_err(AgentError::Observer)?;
            }

            self.sessions.insert(id, session.clone());
            Ok(session)
        })
    }

    fn load_session(
        &self,
        id: SessionId,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>> {
        Box::pin(async move {
            if let Some(existing) = self.sessions.get(&id) {
                return Ok(existing.value().clone());
            }
            let Some(loader) = &self.loader else {
                return Err(AgentError::Restore(BoxError::new(io::Error::other(
                    "session loader not configured",
                ))));
            };
            let loaded = loader
                .load_session(id.clone())
                .await
                .map_err(AgentError::Restore)?;
            let resolved = ResolvedSessionCapabilities::resolve(
                self.capabilities,
                self.provider.hosted_capabilities(),
                &self.provider.info().vendor,
            )?;
            let session_tools = match &self.session_tools {
                Some(factory) => factory
                    .build_registry(loaded.info.cwd.clone(), loaded.info.mcp_servers.clone())
                    .await
                    .map_err(AgentError::Restore)?,
                None => Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>,
            };

            // SessionStart hook（resume 路径）。同 create_session：
            // block 被引擎丢弃；仅取 append。
            let session_start_append = {
                let cancel = CancellationToken::new();
                let ctx = HookCtx::new(&loaded.info.id, &loaded.info.cwd, cancel);
                let event = HookEvent::SessionStart {
                    source: SessionSource::Resume {
                        session_id: &loaded.info.id,
                    },
                    cwd: &loaded.info.cwd,
                };
                let outcome = self.hook_engine.fire(event, ctx).await;
                outcome.append
            };

            let session = Arc::new(DefaultSession {
                id: loaded.info.id.clone(),
                cwd: loaded.info.cwd.clone(),
                history: Box::new(VecHistory::from_messages(loaded.history)),
                tools: Arc::new(CompositeRegistry::new(
                    session_tools,
                    self.process_tools.clone(),
                )),
                provider: self.provider.clone(),
                policy: self.policy.clone(),
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                config: RwLock::new(
                    self.config
                        .read()
                        .expect("DefaultAgentCore config rwlock poisoned")
                        .clone(),
                ),
                hosted_capabilities: resolved.hosted,
                fs,
                shell,
                http: self.http.clone(),
                hook_engine: self.hook_engine.clone(),
                session_start_append,
            }) as Arc<dyn Session>;

            self.sessions.insert(id, session.clone());
            Ok(session)
        })
    }

    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.get(id).map(|r| r.value().clone())
    }
}

pub struct DefaultSession {
    id: SessionId,
    cwd: PathBuf,
    history: Box<dyn History>,
    tools: Arc<dyn ToolRegistry>,
    provider: Arc<dyn LlmProvider>,
    policy: Arc<dyn SandboxPolicy>,
    events: Arc<EventEmitter>,
    permissions: Arc<PermissionGate>,
    /// 单 turn 互斥 + cancel 通道。`Some(token)` 表示有 turn 在跑；
    /// `None` 表示空闲。`std::sync::Mutex` 仅短暂持锁、不跨 await。
    turn_state: Mutex<TurnSlot>,
    config: RwLock<TurnConfig>,
    /// session 启动期一次性裁决出的 hosted capability 集合。
    /// 每次 `run_turn` 装配 [`TurnRunner`] 时直接复用——`(provider, mode)`
    /// 在 session 生命周期内不变。
    hosted_capabilities: HostedCapabilities,
    /// session 级 fs 后端。由 [`AgentCore::create_session`] 注入；
    /// `TurnRunner` 把 `&dyn FsBackend` 借到 [`crate::tool::ToolContext`] 传给工具。
    fs: Arc<dyn FsBackend>,
    /// session 级 shell 后端。与 `fs` 同款由 [`AgentCore::create_session`] 注入；
    /// `bash` 工具通过 [`crate::tool::ToolContext`] 拿它。
    shell: Arc<dyn ShellBackend>,
    /// 进程级 HTTP fetch 后端（多 session 共享，由 [`DefaultAgentCore`]
    /// 一份持有 / clone）。`fetch` 工具通过 [`crate::tool::ToolContext`] 拿它。
    http: Arc<dyn HttpClient>,
    /// 进程级 hook 引擎（多 session 共享）。`run_turn` 装配
    /// [`TurnRunner`] 时把 `&dyn HookEngine` 借给主循环。
    hook_engine: Arc<dyn HookEngine>,
    /// session 启动期 [`HookEvent::SessionStart`] hook 返回的 append 内容。
    /// 由 [`AgentCore::create_session`] / `load_session` 在 SessionStart hook
    /// 跑完之后填进来；目前**保留但暂不消费**——system_prompt 在 turn 装配
    /// 时由 [`crate::session::prompt::resolve_system_prompt`] 计算，
    /// SessionStart preload 的落地等系统 prompt 动态拼接落地后接入。
    /// 详见 `docs/internal/hooks.md` §3.2 / §9.1。
    #[allow(dead_code)]
    session_start_append: Vec<agent_client_protocol::schema::ContentBlock>,
}

impl DefaultSession {}

#[derive(Default)]
struct TurnSlot {
    cancel: Option<CancellationToken>,
}

/// `run_turn` 的"占位 / 释放" guard：构造时占用 turn slot、drop 时释放。
struct TurnGuard<'a> {
    state: &'a Mutex<TurnSlot>,
}

impl<'a> Drop for TurnGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.state.lock() {
            slot.cancel = None;
        }
    }
}

impl Session for DefaultSession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn provider_info(&self) -> ProviderInfo {
        self.provider.info()
    }

    fn current_model(&self) -> String {
        self.config
            .read()
            .expect("DefaultSession config rwlock poisoned")
            .model
            .clone()
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async move {
            let allowed_models = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .allowed_models
                .clone();
            match self.provider.list_models().await {
                Ok(available_models) => Ok(filter_allowed_models(
                    available_models,
                    allowed_models.as_deref(),
                )),
                Err(err) => match allowed_models {
                    Some(allowed_models) => {
                        tracing::warn!(
                            provider = %self.provider.info().vendor,
                            error = %err,
                            "provider list_models failed; falling back to configured allowed models"
                        );
                        Ok(fallback_allowed_models(
                            self.provider.as_ref(),
                            &allowed_models,
                        ))
                    }
                    None => Err(err),
                },
            }
        })
    }

    fn set_model(&self, model_id: String) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            let allowed_models = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .allowed_models
                .clone();
            if let Some(allowed_models) = allowed_models.as_ref()
                && !allowed_models.iter().any(|allowed| allowed == &model_id)
            {
                return Err(ProviderError::new(ProviderErrorKind::ModelNotFound {
                    model: model_id,
                }));
            }

            if self.provider.model_info(&model_id).is_some() {
                let mut config = self
                    .config
                    .write()
                    .expect("DefaultSession config rwlock poisoned");
                config.model = model_id;
                return Ok(());
            }

            let available_models = self.provider.list_models().await?;
            let available_models =
                filter_allowed_models(available_models, allowed_models.as_deref());
            let known_model = available_models.iter().any(|model| model.id == model_id);
            if !known_model {
                return Err(ProviderError::new(ProviderErrorKind::ModelNotFound {
                    model: model_id,
                }));
            }

            let mut config = self
                .config
                .write()
                .expect("DefaultSession config rwlock poisoned");
            config.model = model_id;
            Ok(())
        })
    }

    fn subscribe(&self) -> EventStream {
        self.events.subscribe()
    }

    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>> {
        // 整个 turn 包在一个 span 里——LLM 调用 / 工具调用 / 权限请求 都
        // 自动挂成子 span，排障时一棵 trace 走到底。session_id 截短，避免
        // 输出整段 uuid 噪音。详见 docs/outbound/tracing.md §2.2。
        let span = tracing::info_span!(
            "turn",
            session_id = %short_id(self.id.0.as_ref()),
            model = %self.current_model(),
        );
        Box::pin(
            async move {
                let cancel = {
                    let mut slot = self
                        .turn_state
                        .lock()
                        .expect("DefaultSession turn_state mutex poisoned");
                    if slot.cancel.is_some() {
                        return Err(TurnError::TurnInProgress);
                    }
                    let cancel = CancellationToken::new();
                    slot.cancel = Some(cancel.clone());
                    cancel
                };

                // RAII：函数任意路径退出（含 await 内 panic）都释放 slot
                let _guard = TurnGuard {
                    state: &self.turn_state,
                };

                let config = self
                    .config
                    .read()
                    .expect("DefaultSession config rwlock poisoned")
                    .clone();
                let system_prompt = resolve_system_prompt(
                    &self.cwd,
                    &self.provider.info().vendor,
                    &config.model,
                    &config.base_prompt,
                    &config.prompt,
                    config.system_prompt.as_deref(),
                )
                .map_err(|err| TurnError::Internal(BoxError::new(err)))?;
                let runner = TurnRunner {
                    history: self.history.as_ref(),
                    tools: self.tools.as_ref(),
                    provider: self.provider.as_ref(),
                    policy: self.policy.as_ref(),
                    events: self.events.clone(),
                    permissions: self.permissions.as_ref(),
                    cancel,
                    config: &config,
                    system_prompt,
                    cwd: &self.cwd,
                    fs: self.fs.clone(),
                    shell: self.shell.clone(),
                    http: self.http.clone(),
                    hosted_capabilities: self.hosted_capabilities,
                    hooks: self.hook_engine.as_ref(),
                    session_id: &self.id,
                };

                runner.run(prompt).await
            }
            .instrument(span),
        )
    }

    fn cancel_turn(&self) {
        let token = {
            let slot = self
                .turn_state
                .lock()
                .expect("DefaultSession turn_state mutex poisoned");
            slot.cancel.clone()
        };
        if let Some(token) = token {
            token.cancel();
        }
        // 没 turn 在跑 → no-op（幂等）
    }

    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution) {
        self.permissions.resolve(&id, outcome);
    }
}

fn filter_allowed_models(
    available_models: Vec<ModelInfo>,
    allowed_models: Option<&[String]>,
) -> Vec<ModelInfo> {
    let Some(allowed_models) = allowed_models else {
        return available_models;
    };

    available_models
        .into_iter()
        .filter(|model| allowed_models.iter().any(|allowed| allowed == &model.id))
        .collect()
}

fn fallback_allowed_models(
    provider: &dyn LlmProvider,
    allowed_models: &[String],
) -> Vec<ModelInfo> {
    allowed_models
        .iter()
        .map(|model_id| {
            provider.model_info(model_id).unwrap_or_else(|| ModelInfo {
                id: model_id.clone(),
                display_name: Some(model_id.clone()),
                context_window: None,
                max_output_tokens: None,
                deprecated: false,
                capabilities_overrides: Default::default(),
            })
        })
        .collect()
}

/// v0 的 session id 生成：进程内单调递增 + 时间戳。引入 uuid crate 时再换。
///
/// `defect-acp` 的 `session/new` handler 在调用
/// [`AgentCore::create_session`] 之前需要 `SessionId`（用于构造
/// `AcpFsBackend`，详见 `docs/inbound/acp-fs.md` §3.2）；这个函数对外公开，
/// 让 acp / 测试都能拿到一致格式的 id。
pub fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("session-{ts:x}-{n:x}")
}

/// 给 tracing span 用的 session id 短形：按字符取前 12 个。仅诊断用。
fn short_id(s: &str) -> &str {
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
