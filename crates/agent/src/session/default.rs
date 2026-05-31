//! [`Session`] / [`AgentCore`] 的 v0 默认实现。
//!
//! 装配关系：
//!
//! ```text
//! DefaultAgentCore
//!   ├── Arc<dyn LlmProvider>          (装配时传入，本 core 的所有 session 共享)
//!   ├── Arc<dyn ToolRegistry>         (内置工具，本 core 的所有 session 共享一份)
//!   ├── TurnConfig                    (默认配置)
//!   └── DashMap<SessionId, Arc<dyn Session>>
//!
//! 注：这些"共享"都以 **`AgentCore` 实例**为界，不是进程全局——把 defect 当库
//! 引用时一个进程可装配多个 `AgentCore`，各持自己的 provider / 工具集 / 配置。
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

use agent_client_protocol_schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use dashmap::DashMap;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::error::BoxError;
use crate::event::PermissionResolution;
use crate::fs::FsBackend;
use crate::hooks::{HookCtx, HookEngine, NoopHookEngine};
use crate::http::{HttpClient, NoopHttpClient};
use crate::llm::{
    HostedCapabilities, LlmProvider, Message, ModelCandidate, ModelInfo, ProviderError,
    ProviderErrorKind, ProviderInfo, ProviderRegistry,
};
use crate::policy::{AskWritesPolicy, SandboxPolicy};
use crate::session::capabilities::{ResolvedSessionCapabilities, SessionCapabilitiesConfig};
use crate::session::context::{Frontend, RunningContext};
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::prompt::resolve_system_prompt;
use crate::session::tool_registry::{CompositeRegistry, StaticToolRegistry};
use crate::session::turn::{RequestAuditTracker, TurnConfig, TurnRunner};
use crate::session::{
    AgentCore, AgentError, EventStream, History, Session, SessionCreateInfo, SessionLoader,
    SessionObserver, SessionToolFactory, ToolRegistry, TurnError, VecHistory,
};
use crate::shell::ShellBackend;

/// 默认 [`AgentCore`]。
pub struct DefaultAgentCore {
    /// 装配期落地的 provider 目录。session 持有同一份 `Arc`，按当前选中
    /// 的 model id 解析对应的真实 [`LlmProvider`]——本类不再"持有单一
    /// provider"。详见 `docs/internal/llm-trait.md` §2 与
    /// `docs/internal/session.md`。
    registry: Arc<ProviderRegistry>,
    process_tools: Arc<dyn ToolRegistry>,
    policy: Arc<dyn SandboxPolicy>,
    config: RwLock<TurnConfig>,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    /// HTTP fetch 后端。本 core 的所有 session 共享一份——HTTP 没有 per-client
    /// capability 协商，多 session 间也无须隔离连接池。CLI 入口按
    /// [`HttpClientConfig`] 构造一次后注入；测试 / `echo` provider 走
    /// [`NoopHttpClient`]。
    ///
    /// [`HttpClientConfig`]: defect_config::HttpClientConfig
    http: Arc<dyn HttpClient>,
    /// hook 引擎。本 core 的所有 session 共享——hook 配置走全局 + per-session
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
    registry: Option<Arc<ProviderRegistry>>,
    /// 单 provider 便捷入口：[`Self::provider`] 写到这里，`build()` 时
    /// 与 `config.model` 一起合成一份单 entry 的 [`ProviderRegistry`]。
    /// `registry` 已显式注入时此字段被忽略。
    single_provider: Option<Arc<dyn LlmProvider>>,
    /// 单 provider 入口下的 session capabilities。`registry` 显式注入时
    /// 由 entry 自带，本字段被忽略。
    single_capabilities: SessionCapabilitiesConfig,
    process_tools: Option<Arc<dyn ToolRegistry>>,
    policy: Option<Arc<dyn SandboxPolicy>>,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    http: Option<Arc<dyn HttpClient>>,
    hook_engine: Option<Arc<dyn HookEngine>>,
    config: TurnConfig,
}

impl DefaultAgentCoreBuilder {
    /// 装配期注入 provider 目录。CLI / 真实启动路径走这条；测试与单
    /// provider 场景用 [`Self::provider`] 简便。
    pub fn registry(mut self, registry: Arc<ProviderRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// 单 provider 便捷入口。`build()` 时把它包成单 entry 的
    /// [`ProviderRegistry`]，default model = [`TurnConfig::model`]。
    /// 与 [`Self::registry`] 互斥；同时设置时以 `registry` 为准。
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.single_provider = Some(provider);
        self
    }

    /// 单 provider 便捷入口下的 session capabilities 配置——会被合到
    /// `build()` 自动构造的单 entry registry 上。多 provider 路径下应直接
    /// 把 capabilities 写到 [`ProviderEntry`] 里，本字段会被忽略。
    pub fn capabilities(mut self, capabilities: SessionCapabilitiesConfig) -> Self {
        self.single_capabilities = capabilities;
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

    /// 设置本 core 的 HTTP fetch 后端。未设置时退化为 [`NoopHttpClient`]——
    /// 任何 `fetch` 调用都会以 [`crate::http::HttpClientError::Transport`]
    /// 失败，便于不需要网络的测试 / `echo` 装配跳过真实 HTTP 栈构造。
    pub fn http(mut self, http: Arc<dyn HttpClient>) -> Self {
        self.http = Some(http);
        self
    }

    /// 设置本 core 的 hook 引擎。未设置时退化为 [`NoopHookEngine`]——所有 hook
    /// 调用直接返回 `Pass`，主循环行为与未引入 hook 系统时一致。
    pub fn hook_engine(mut self, hook_engine: Arc<dyn HookEngine>) -> Self {
        self.hook_engine = Some(hook_engine);
        self
    }

    /// # Panics
    /// `registry` 与 `provider` 都未设置；或单 provider 路径下 `config.model`
    /// 是空字符串（registry 至少要有一个 default model）。
    pub fn build(self) -> DefaultAgentCore {
        let registry = self.registry.unwrap_or_else(|| {
            let provider = self
                .single_provider
                .expect("DefaultAgentCore requires a provider or a registry");
            let model_id = self.config.model.clone();
            assert!(
                !model_id.is_empty(),
                "DefaultAgentCoreBuilder::provider() requires TurnConfig::model to be set; \
                 use registry() for multi-provider setups"
            );
            // 单 provider 路径下，把 `TurnConfig::allowed_models` 当作模型
            // 候选清单——这与 CLI 的多 provider 装配保持对称：用户用
            // `[providers.<p>.models]` 声明候选，agent 不向 adapter 发
            // `list_models` 网络请求。`allowed_models` 缺省时退回到只暴露
            // 默认模型。
            let model_ids = match self.config.allowed_models.as_ref() {
                Some(ids) if !ids.is_empty() => ids.clone(),
                _ => vec![model_id.clone()],
            };
            let mut model_infos: Vec<ModelInfo> = model_ids
                .into_iter()
                .map(|id| {
                    provider.model_info(&id).unwrap_or(ModelInfo {
                        id,
                        display_name: None,
                        context_window: None,
                        max_output_tokens: None,
                        deprecated: false,
                        capabilities_overrides: Default::default(),
                    })
                })
                .collect();
            if !model_infos.iter().any(|m| m.id == model_id) {
                model_infos.insert(
                    0,
                    ModelInfo {
                        id: model_id.clone(),
                        display_name: None,
                        context_window: None,
                        max_output_tokens: None,
                        deprecated: false,
                        capabilities_overrides: Default::default(),
                    },
                );
            }
            Arc::new(
                crate::llm::ProviderRegistry::new(
                    vec![crate::llm::ProviderEntry::new(
                        provider,
                        model_infos,
                        self.single_capabilities,
                    )],
                    &model_id,
                )
                .expect("single-entry registry must satisfy invariants"),
            )
        });

        DefaultAgentCore {
            registry,
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
        frontend: Frontend,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>> {
        Box::pin(async move {
            if !cwd.is_absolute() || !cwd.exists() {
                return Err(AgentError::InvalidCwd(cwd));
            }
            let session_cwd = cwd.clone();
            if self.sessions.contains_key(&id) {
                return Err(AgentError::DuplicateSessionId(id));
            }

            let initial = self.resolve_initial_provider()?;

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

            // after session enter hook：吸收注入的 additional_context 作为系统 prompt 后缀候选。
            let session_start_append = {
                let cancel = CancellationToken::new();
                let ctx = HookCtx::new(&id, &cwd, cancel);
                let mut step = crate::hooks::step::AfterSessionEnter {
                    cwd: cwd.to_string_lossy().into_owned(),
                    source: crate::hooks::step::SessionSource::New,
                    additional_context: Vec::new(),
                };
                let _ = self.hook_engine.dispatch(&mut step, ctx).await;
                step.additional_context
            };

            // session 级取消令牌：driver loop 退出信号 + 后台任务取消令牌来源（同一个）。
            let session_cancel = CancellationToken::new();
            let concrete = Arc::new(DefaultSession {
                id: id.clone(),
                cwd,
                history: Box::new(VecHistory::new()) as Box<dyn History>,
                tools: composite,
                registry: self.registry.clone(),
                provider_state: RwLock::new(initial),
                policy: self.policy.clone(),
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                background: crate::session::BackgroundTasks::new(session_cancel.clone()),
                turn_freed: Arc::new(tokio::sync::Notify::new()),
                session_cancel,
                config: RwLock::new(
                    self.config
                        .read()
                        .expect("DefaultAgentCore config rwlock poisoned")
                        .clone(),
                ),
                fs,
                shell,
                frontend,
                http: self.http.clone(),
                hook_engine: self.hook_engine.clone(),
                session_start_append,
                request_audit: RequestAuditTracker::new(),
            });
            // 起 session driver（主动续转）。driver 持 Weak 自引——session 外部
            // 强引用清零时它 upgrade 失败而退出，不让 session 永生。
            tokio::spawn(DefaultSession::drive(Arc::downgrade(&concrete)));
            let session = concrete as Arc<dyn Session>;

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
        frontend: Frontend,
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
            let initial = self.resolve_initial_provider()?;
            let session_tools = match &self.session_tools {
                Some(factory) => factory
                    .build_registry(loaded.info.cwd.clone(), loaded.info.mcp_servers.clone())
                    .await
                    .map_err(AgentError::Restore)?,
                None => Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>,
            };

            // after session enter hook（resume 路径）。同 create_session：取注入的 context。
            let session_start_append = {
                let cancel = CancellationToken::new();
                let ctx = HookCtx::new(&loaded.info.id, &loaded.info.cwd, cancel);
                let mut step = crate::hooks::step::AfterSessionEnter {
                    cwd: loaded.info.cwd.to_string_lossy().into_owned(),
                    source: crate::hooks::step::SessionSource::Resume,
                    additional_context: Vec::new(),
                };
                let _ = self.hook_engine.dispatch(&mut step, ctx).await;
                step.additional_context
            };

            let session_cancel = CancellationToken::new();
            let concrete = Arc::new(DefaultSession {
                id: loaded.info.id.clone(),
                cwd: loaded.info.cwd.clone(),
                history: Box::new(VecHistory::from_messages(loaded.history)),
                tools: Arc::new(CompositeRegistry::new(
                    session_tools,
                    self.process_tools.clone(),
                )),
                registry: self.registry.clone(),
                provider_state: RwLock::new(initial),
                policy: self.policy.clone(),
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                background: crate::session::BackgroundTasks::new(session_cancel.clone()),
                turn_freed: Arc::new(tokio::sync::Notify::new()),
                session_cancel,
                config: RwLock::new(
                    self.config
                        .read()
                        .expect("DefaultAgentCore config rwlock poisoned")
                        .clone(),
                ),
                fs,
                shell,
                frontend,
                http: self.http.clone(),
                hook_engine: self.hook_engine.clone(),
                session_start_append,
                request_audit: RequestAuditTracker::new(),
            });
            tokio::spawn(DefaultSession::drive(Arc::downgrade(&concrete)));
            let session = concrete as Arc<dyn Session>;

            let session_info = loaded.info;
            for observer in &self.observers {
                observer
                    .on_session_created(session.clone(), session_info.clone())
                    .map_err(AgentError::Observer)?;
            }

            self.sessions.insert(id, session.clone());
            Ok(session)
        })
    }

    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.get(id).map(|r| r.value().clone())
    }
}

impl DefaultAgentCore {
    /// 按当前 [`TurnConfig::model`] 在 registry 上查 entry，并裁决出
    /// `(provider, hosted_capabilities)`。供 `create_session` /
    /// `load_session` 复用。
    ///
    /// 配置里的 model 必须能在 registry 中找到 entry——CLI 装配期
    /// [`ProviderRegistry::new`] 已经校验过 default model，能落到这里报错
    /// 的只剩 builder 误用（registry 与 turn config 不一致）。
    fn resolve_initial_provider(&self) -> Result<SessionProviderState, AgentError> {
        let model = self
            .config
            .read()
            .expect("DefaultAgentCore config rwlock poisoned")
            .model
            .clone();
        let entry = self.registry.entry_for_model(&model).ok_or_else(|| {
            AgentError::Other(BoxError::new(io::Error::other(format!(
                "default model `{model}` is not declared by any provider entry in the registry"
            ))))
        })?;
        let provider = entry.provider().clone();
        let resolved = ResolvedSessionCapabilities::resolve(
            entry.capabilities(),
            provider.hosted_capabilities(),
            &provider.info().vendor,
        )?;
        Ok(SessionProviderState {
            provider,
            hosted_capabilities: resolved.hosted,
        })
    }
}

/// session 当前选中的真实 provider + 该 provider 的 hosted capability 解析结果。
///
/// `set_model` 跨 provider 切换时被原子替换。
struct SessionProviderState {
    provider: Arc<dyn LlmProvider>,
    hosted_capabilities: HostedCapabilities,
}

pub struct DefaultSession {
    id: SessionId,
    cwd: PathBuf,
    history: Box<dyn History>,
    tools: Arc<dyn ToolRegistry>,
    /// 全局 provider 目录。session 共享 [`DefaultAgentCore`] 持有的同一份
    /// `Arc<ProviderRegistry>`——list_models / set_model 的 candidate 与
    /// owner provider 全靠它解析。
    registry: Arc<ProviderRegistry>,
    /// 当前选中的 (provider, hosted_capabilities) 状态。`set_model` 跨
    /// provider 时整体替换，保证 `(provider, hosted_capabilities)` 总是
    /// 自洽——不存在"provider 换了但 capabilities 没换"的中间态。
    provider_state: RwLock<SessionProviderState>,
    policy: Arc<dyn SandboxPolicy>,
    events: Arc<EventEmitter>,
    permissions: Arc<PermissionGate>,
    /// 单 turn 互斥 + cancel 通道。`Some(token)` 表示有 turn 在跑；
    /// `None` 表示空闲。`std::sync::Mutex` 仅短暂持锁、不跨 await。
    turn_state: Mutex<TurnSlot>,
    /// session 级后台任务表（`run_in_background` 落点）。持有任务 `JoinHandle`
    /// 使其活过发起它的 turn；内部 cancel token 独立于 turn 子 token。`run_turn`
    /// 把 clone 经 `TurnRunner` → `ToolContext` 注入给工具。详见
    /// `docs/proposals/task-arrange.md` §3.1。
    background: crate::session::BackgroundTasks,
    /// turn slot 释放通知。`TurnGuard::drop` 时 `notify_one`——session driver 在
    /// 撞上 `TurnInProgress` 后等它，待当前 turn 结束再起自主续转 turn（主动续转的
    /// 活性保证）。详见 `docs/proposals/task-arrange.md` §3.2。
    turn_freed: Arc<tokio::sync::Notify>,
    /// session 级取消令牌——session 终结时 cancel，driver loop 据此退出。也是
    /// `background` 内任务取消令牌的来源（同一个 token）。
    session_cancel: CancellationToken,
    config: RwLock<TurnConfig>,
    /// session 级 fs 后端。由 [`AgentCore::create_session`] 注入；
    /// `TurnRunner` 把 `&dyn FsBackend` 借到 [`crate::tool::ToolContext`] 传给工具。
    fs: Arc<dyn FsBackend>,
    /// session 级 shell 后端。与 `fs` 同款由 [`AgentCore::create_session`] 注入；
    /// `bash` 工具通过 [`crate::tool::ToolContext`] 拿它。
    shell: Arc<dyn ShellBackend>,
    /// agent 接入方式。由 [`AgentCore::create_session`] / `load_session` 注入，
    /// turn 装配时组进 [`RunningContext`]，渲染进 system prompt 的 `# Environment` 段。
    frontend: Frontend,
    /// HTTP fetch 后端（本 core 的多 session 共享，由 [`DefaultAgentCore`]
    /// 一份持有 / clone）。`fetch` 工具通过 [`crate::tool::ToolContext`] 拿它。
    http: Arc<dyn HttpClient>,
    /// hook 引擎（本 core 的多 session 共享）。`run_turn` 装配
    /// [`TurnRunner`] 时把 `&dyn HookEngine` 借给主循环。
    hook_engine: Arc<dyn HookEngine>,
    /// session 启动期 [`HookEvent::SessionStart`] hook 返回的 append 内容。
    /// 由 [`AgentCore::create_session`] / `load_session` 在 SessionStart hook
    /// 跑完之后填进来；目前**保留但暂不消费**——system_prompt 在 turn 装配
    /// 时由 [`crate::session::prompt::resolve_system_prompt`] 计算，
    /// SessionStart preload 的落地等系统 prompt 动态拼接落地后接入。
    /// 详见 `docs/internal/hooks.md` §3.2 / §9.1。
    #[allow(dead_code)]
    session_start_append: Vec<agent_client_protocol_schema::ContentBlock>,
    /// 相邻请求稳定性诊断器。每次实际发给 provider 的请求都会产一条
    /// tracing 记录，帮助定位 cache miss 来源。
    request_audit: RequestAuditTracker,
}

impl DefaultSession {
    fn current_provider(&self) -> Arc<dyn LlmProvider> {
        self.provider_state
            .read()
            .expect("DefaultSession provider_state rwlock poisoned")
            .provider
            .clone()
    }

    fn current_hosted(&self) -> HostedCapabilities {
        self.provider_state
            .read()
            .expect("DefaultSession provider_state rwlock poisoned")
            .hosted_capabilities
    }

    /// 一次 turn 的执行核心——用户 turn 与自主续转 turn 共用。
    ///
    /// `prompt` 是外部输入（用户 turn）或空（自主续转 turn）。两种情况都会把已完成的
    /// 后台结果作为 prompt **前缀块**带入；空 prompt + 无后台结果时不起 turn（返回
    /// `EndTurn`，避免空转）。turn slot 互斥仍由本函数顶部把守。
    async fn run_turn_core(
        &self,
        prompt: Vec<ContentBlock>,
        ingest_source: crate::hooks::step::IngestSource,
    ) -> Result<StopReason, TurnError> {
        let span = tracing::info_span!(
            "turn",
            session_id = %short_id(self.id.0.as_ref()),
            model = %self.current_model(),
        );
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

            // RAII：函数任意路径退出（含 await 内 panic）都释放 slot + 唤醒 driver。
            let _guard = TurnGuard {
                state: &self.turn_state,
                freed: &self.turn_freed,
            };

            // 把已完成的后台任务结果作为本轮 prompt 的**前缀块**带入。
            // 详见 docs/proposals/task-arrange.md §3.1 / §5.1。
            let prompt = {
                let outcomes = self.background.drain_completed();
                if outcomes.is_empty() {
                    prompt
                } else {
                    let mut blocks: Vec<ContentBlock> = outcomes
                        .iter()
                        .map(|o| {
                            ContentBlock::from(
                                crate::session::format_background_outcome(o).as_str(),
                            )
                        })
                        .collect();
                    blocks.extend(prompt);
                    blocks
                }
            };

            // 空 prompt（自主 turn 却无任何后台结果可消化）——不起 turn，避免空转。
            if prompt.is_empty() {
                return Ok(StopReason::EndTurn);
            }

            let config = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .clone();
            // turn 在启动时拍一次 (provider, hosted) 快照——同一 turn 内即使有并发的
            // set_model 请求，本 turn 仍走选定的 provider；下一 turn 才生效。
            let provider = self.current_provider();
            let hosted = self.current_hosted();
            let running_ctx = RunningContext::new(self.frontend, &self.cwd);
            let system_prompt = resolve_system_prompt(
                &running_ctx,
                &provider.info().vendor,
                &config.model,
                &config.base_prompt,
                &config.prompt,
                config.system_prompt.as_deref(),
            )
            .map_err(|err| TurnError::Internal(BoxError::new(err)))?;
            let runner = TurnRunner {
                history: self.history.as_ref(),
                tools: self.tools.as_ref(),
                provider: provider.as_ref(),
                policy: self.policy.as_ref(),
                events: self.events.clone(),
                permissions: self.permissions.as_ref(),
                cancel,
                config: &config,
                system_prompt: system_prompt.map(Arc::from),
                cwd: &self.cwd,
                fs: self.fs.clone(),
                shell: self.shell.clone(),
                http: self.http.clone(),
                hosted_capabilities: hosted,
                hooks: self.hook_engine.as_ref(),
                session_id: &self.id,
                request_audit: &self.request_audit,
                // 顶层 turn 注入 session 级后台任务句柄——工具的 run_in_background
                // 能力由此开启。嵌套子 agent turn 不注入（见 spawn_agent）。
                background: Some(self.background.clone()),
                ingest_source,
            };

            runner.run(prompt).await
        }
        .instrument(span)
        .await
    }

    /// Session driver loop（主动续转）：常驻 task，在后台任务完成时起一个自主 turn
    /// 消化结果。`create_session` / `load_session` 时 spawn。
    ///
    /// 持 `Weak<Self>` 而非 `Arc`：driver 不能让 session 永生。每轮先 `upgrade`——
    /// 外部强引用（`AgentCore.sessions` DashMap）全没了时 upgrade 失败、driver 退出。
    /// `session_cancel` 是显式退出信号（process shutdown / 未来的 session evict）。
    ///
    /// 形态见 `docs/proposals/task-arrange.md` §3.2。两条等待腿：
    /// - `background.wait_for_completion()`：有任务完成 → 准备起自主 turn；
    /// - `session_cancel.cancelled()`：session 终结 → 退出 loop。
    ///
    /// 起 turn 前若撞上 `TurnInProgress`（用户 turn 正在跑），等 `turn_freed`
    /// 再重试——这正是"用户输入与后台结果竞争同一个 turn slot"的落点：用户 turn
    /// 先到就先跑，后台结果搭它的车（run_turn_core 的 drain）或等它结束再单独起。
    async fn drive(weak: std::sync::Weak<Self>) {
        loop {
            let Some(this) = weak.upgrade() else { break };
            if this.session_cancel.is_cancelled() {
                break;
            }
            // 先拿 notified() future 再检查队列——避免漏掉两步之间到达的完成通知。
            let completion = this.background.wait_for_completion();
            if this.background.has_completed() {
                this.run_autonomous_turn_with_retry().await;
                continue;
            }
            tokio::select! {
                () = completion => {
                    this.run_autonomous_turn_with_retry().await;
                }
                () = this.session_cancel.cancelled() => break,
            }
        }
    }

    /// 起一个自主续转 turn；若 turn slot 被占（用户 turn 在跑），等它释放再试，
    /// 最多重试到结果被消化。`session_cancel` 触发时放弃。
    async fn run_autonomous_turn_with_retry(self: &Arc<Self>) {
        loop {
            if self.session_cancel.is_cancelled() {
                return;
            }
            match self
                .run_turn_core(Vec::new(), crate::hooks::step::IngestSource::Background)
                .await
            {
                Err(TurnError::TurnInProgress) => {
                    // 用户 turn 正在跑。等它结束——它的 run_turn_core 会 drain 掉我们的
                    // 后台结果（搭车），那样这里 has_completed 就空了、自然退出。
                    tokio::select! {
                        () = self.turn_freed.notified() => {}
                        () = self.session_cancel.cancelled() => return,
                    }
                    if !self.background.has_completed() {
                        // 被在跑的用户 turn 搭车消化了——无需再起自主 turn。
                        return;
                    }
                }
                _ => return,
            }
        }
    }
}

impl Drop for DefaultSession {
    fn drop(&mut self) {
        // session 析构 → 取消 session_cancel：掐掉所有在途后台任务，并让 driver loop
        // 的 `session_cancel.cancelled()` 腿醒来退出（driver 持 Weak，此时 upgrade 也已失败）。
        self.session_cancel.cancel();
    }
}

#[derive(Default)]
struct TurnSlot {
    cancel: Option<CancellationToken>,
}

/// `run_turn` 的"占位 / 释放" guard：构造时占用 turn slot、drop 时释放。
struct TurnGuard<'a> {
    state: &'a Mutex<TurnSlot>,
    /// turn 释放时唤醒 session driver（主动续转的活性保证）。
    freed: &'a tokio::sync::Notify,
}

impl<'a> Drop for TurnGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.state.lock() {
            slot.cancel = None;
        }
        // turn slot 已空——唤醒可能等着起自主续转 turn 的 driver。
        self.freed.notify_one();
    }
}

impl Session for DefaultSession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn provider_info(&self) -> ProviderInfo {
        self.current_provider().info()
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
            // 多 provider 装配下：candidate 集来自 registry——每个 entry
            // 自带 model 列表（CLI 装配时已经塞进去）。再按 session 的
            // `allowed_models` 白名单过滤。registry 不发网络请求，这条路径
            // 永远走得通。
            let allowed_models = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .allowed_models
                .clone();
            let candidates = self.registry.list_candidates();
            let mut models: Vec<ModelInfo> = candidates
                .into_iter()
                .map(|candidate| {
                    decorate_with_provider_display(candidate.model, &candidate.provider)
                })
                .collect();
            models = filter_allowed_models(models, allowed_models.as_deref());
            Ok(models)
        })
    }

    fn list_candidates(&self) -> BoxFuture<'_, Result<Vec<ModelCandidate>, ProviderError>> {
        Box::pin(async move {
            let allowed_models = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .allowed_models
                .clone();
            let candidates = self.registry.list_candidates();
            let candidates: Vec<ModelCandidate> = match allowed_models {
                Some(allowed) => candidates
                    .into_iter()
                    .filter(|c| allowed.iter().any(|id| id == &c.model.id))
                    .collect(),
                None => candidates,
            };
            Ok(candidates)
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

            let Some(entry) = self.registry.entry_for_model(&model_id) else {
                return Err(ProviderError::new(ProviderErrorKind::ModelNotFound {
                    model: model_id,
                }));
            };

            // 跨 provider 切换时重新 resolve hosted capabilities：每个 entry
            // 自带它的 [`SessionCapabilitiesConfig`]，与 provider 的
            // hosted_capabilities 交叉裁决。Delegate 但 provider 不支持时
            // 返回 ProviderError——保持 set_model 的失败语义稳定。
            let new_provider = entry.provider().clone();
            let resolved = ResolvedSessionCapabilities::resolve(
                entry.capabilities(),
                new_provider.hosted_capabilities(),
                &new_provider.info().vendor,
            )
            .map_err(|err| {
                ProviderError::new(ProviderErrorKind::Other(BoxError::new(io::Error::other(
                    err.to_string(),
                ))))
            })?;

            // 锁顺序：provider_state 先于 config，与 run_turn 的快照路径一致。
            // 同时持有两把写锁的窗口很短（仅几条赋值），不会阻塞主循环。
            {
                let mut state = self
                    .provider_state
                    .write()
                    .expect("DefaultSession provider_state rwlock poisoned");
                state.provider = new_provider;
                state.hosted_capabilities = resolved.hosted;
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

    fn history_snapshot(&self) -> Vec<Message> {
        self.history.snapshot()
    }

    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>> {
        // 用户驱动的 turn：把已完成的后台结果作为 prompt 前缀**搭车**带入
        // （被动回流，与主动续转互补——主动续转管空闲态，搭车管"恰好用户也开口了"）。
        Box::pin(self.run_turn_core(prompt, crate::hooks::step::IngestSource::User))
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

/// 给 model 的 display_name 拼上 provider 前缀，便于 ACP 客户端区分同一
/// model id 在不同 provider 下的来源（虽然 [`ProviderRegistry::new`] 不允许
/// 重复 id，但显示层仍然需要"OpenAI: gpt-4o"这样的人读名）。
fn decorate_with_provider_display(mut model: ModelInfo, provider: &ProviderInfo) -> ModelInfo {
    let name = model
        .display_name
        .clone()
        .unwrap_or_else(|| model.id.clone());
    model.display_name = Some(format!("{}: {name}", provider.display_name));
    model
}

/// session id 生成：随机 UUID v4。
///
/// `defect-acp` 的 `session/new` handler 在调用
/// [`AgentCore::create_session`] 之前需要 `SessionId`（用于构造
/// `AcpFsBackend`，详见 `docs/inbound/acp-fs.md` §3.2）；这个函数对外公开，
/// 让 acp / 测试都能拿到一致格式的 id。
///
/// 用全局唯一的 UUID 而非进程内计数 + 时间戳：跨进程重启、并发实例都不撞，
/// 也让下游（storage 落盘目录、可观测性 trace 关联）能拿它当稳定主键。
pub fn new_session_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// 给 tracing span 用的 session id 短形：按字符取前 12 个。仅诊断用。
fn short_id(s: &str) -> &str {
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
