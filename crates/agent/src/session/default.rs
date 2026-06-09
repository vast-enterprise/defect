//! Default implementation of [`Session`] / [`AgentCore`].
//!
//! Assembly structure:
//!
//! ```text
//! DefaultAgentCore
//!   ├── Arc<dyn LlmProvider>          (injected at assembly, shared by all sessions in this core)
//!   ├── Arc<dyn ToolRegistry>         (built-in tools, shared by all sessions in this core)
//!   ├── TurnConfig                    (default configuration)
//!   └── DashMap<SessionId, Arc<dyn Session>>
//!
//! Note: "shared" here is scoped to the **`AgentCore` instance**, not process-global.
//! When using defect as a library, a single process can assemble multiple `AgentCore`
//! instances, each with its own provider / tool set / configuration.
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
//! Turn mutual exclusion uses `Mutex<TurnSlot>`: `run_turn` calls `try_lock` at the
//! outermost level, returning `TurnError::TurnInProgress` on failure. `TurnSlot`
//! internally stores the current turn's [`CancellationToken`]; `cancel_turn` extracts
//! it and calls `cancel()`.

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use agent_client_protocol_schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use dashmap::DashMap;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::error::BoxError;
use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::hooks::{HookCtx, HookEngine, NoopHookEngine};
use crate::http::{HttpClient, NoopHttpClient};
use crate::llm::{
    HostedCapabilities, LlmProvider, Message, ModelCandidate, ModelInfo, ProviderError,
    ProviderErrorKind, ProviderInfo, ProviderRegistry, ReasoningEffort,
};
use crate::policy::{AskWritesPolicy, ModeCatalog, SandboxPolicy};
use crate::session::capabilities::{ResolvedSessionCapabilities, SessionCapabilitiesConfig};
use crate::session::context::{Frontend, RunningContext};
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::prompt::resolve_system_prompt;
use crate::session::tool_registry::{CompositeRegistry, StaticToolRegistry};
use crate::session::turn::{
    CompactionCtx, RequestAuditTracker, TurnConfig, TurnRunner, run_sync_compaction,
};
use crate::session::{
    AgentCore, AgentError, CompactionReport, ContextStatus, EventStream, History, ModelSelection,
    Session, SessionCreateInfo, SessionLoader, SessionObserver, SessionToolFactory, ToolRegistry,
    TurnError, VecHistory,
};
use crate::shell::ShellBackend;

/// Default [`AgentCore`].
pub struct DefaultAgentCore {
    /// Provider registry wired at assembly time. Sessions share the same `Arc`; the
    /// active model ID is used to resolve the corresponding [`LlmProvider`] — this type
    /// no longer "holds a single provider".
    registry: Arc<ProviderRegistry>,
    process_tools: Arc<dyn ToolRegistry>,
    /// Optional tool allowlist for the top-level `--profile` mode. When `Some`, each
    /// session's tool pool (built-in **plus** connected MCP tools) is restricted to this
    /// set after assembly. Applied at session-creation time — not at CLI assembly — so
    /// that `mcp__*` tools (connected per-session and absent until then) can be allowed.
    /// `None` ⇒ no restriction (the default and non-profile path).
    tool_allow: Option<Vec<String>>,
    /// Default policy — used as the session's active policy **only when no mode catalog
    /// is present (`modes` is `None`)**. When a catalog is present, it is overridden by
    /// the catalog's current mode.
    policy: Arc<dyn SandboxPolicy>,
    /// Permission mode catalog template. When `Some`, each session holds a cloned copy
    /// and can switch independently via `session/set_mode`; when `None`, falls back to
    /// the single non-switchable `policy`.
    modes: Option<ModeCatalog>,
    config: RwLock<TurnConfig>,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    /// HTTP fetch backend. Shared across all sessions in this core — HTTP has no
    /// per-client capability negotiation, and there is no need to isolate connection
    /// pools between sessions. Constructed once at the CLI entry point from
    /// `HttpClientConfig` and injected; tests and the `echo` provider use
    /// [`NoopHttpClient`].
    http: Arc<dyn HttpClient>,
    /// Hook engine shared by all sessions in this core — hook configuration uses global +
    /// per-session matchers. Assembled at CLI entry; without explicit injection, uses
    /// [`NoopHookEngine`], equivalent to "no hooks configured = main loop unchanged".
    hook_engine: Arc<dyn HookEngine>,
    /// Background progress view configuration. Shared across all sessions (at the same
    /// level as process-wide tool configuration).
    /// Passed to each session when constructing `BackgroundTasks`.
    background_progress: crate::session::BackgroundProgressConfig,
    /// Shared state for the `--goal` goal-driven loop. When `Some`, sessions in this core
    /// run in goal mode; injected by CLI assembly based on `--goal`. All sessions share
    /// the same instance (a `--goal` process typically runs only one session). `None` =
    /// non-goal mode (default).
    goal: Option<Arc<crate::session::GoalState>>,
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
    /// Convenience entry for a single provider: set [`Self::provider`] here, and at
    /// `build()` time it is combined with `config.model` to produce a single-entry
    /// [`ProviderRegistry`]. This field is ignored when `registry` has been explicitly
    /// injected.
    single_provider: Option<Arc<dyn LlmProvider>>,
    /// Session capabilities for the single-provider path. Ignored when `registry` is
    /// explicitly injected, as the entry provides its own.
    single_capabilities: SessionCapabilitiesConfig,
    process_tools: Option<Arc<dyn ToolRegistry>>,
    tool_allow: Option<Vec<String>>,
    policy: Option<Arc<dyn SandboxPolicy>>,
    modes: Option<ModeCatalog>,
    loader: Option<Arc<dyn SessionLoader>>,
    session_tools: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
    http: Option<Arc<dyn HttpClient>>,
    hook_engine: Option<Arc<dyn HookEngine>>,
    config: TurnConfig,
    /// Background task progress view configuration (ring capacity / body limit). Each
    /// session's `BackgroundTasks` builds its progress ring from this. When unset, falls
    /// back to
    /// [`BackgroundProgressConfig::default`](crate::session::BackgroundProgressConfig)
    /// (bird's-eye view, no body content).
    background_progress: crate::session::BackgroundProgressConfig,
    /// Shared state for the `--goal` goal-driven loop. Injected by the CLI during
    /// assembly based on `--goal`; if unset, goal mode is disabled.
    goal: Option<Arc<crate::session::GoalState>>,
}

impl DefaultAgentCoreBuilder {
    /// Injects the provider registry during assembly. Used by the CLI and real startup
    /// paths; tests and single-provider scenarios should use [`Self::provider`] for
    /// convenience.
    pub fn registry(mut self, registry: Arc<ProviderRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Convenience entry point for a single provider. Wraps it into a single-entry
    /// [`ProviderRegistry`] at `build()` time; default model = [`TurnConfig::model`].
    /// Mutually exclusive with [`Self::registry`]; if both are set, `registry` takes
    /// precedence.
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.single_provider = Some(provider);
        self
    }

    /// Session capabilities for the single-provider convenience path — these are merged
    /// into the single-entry registry automatically constructed by `build()`. For the
    /// multi-provider path, write capabilities directly into
    /// [`ProviderEntry`](crate::llm::ProviderEntry) instead; this field is ignored.
    pub fn capabilities(mut self, capabilities: SessionCapabilitiesConfig) -> Self {
        self.single_capabilities = capabilities;
        self
    }

    /// Restrict every session's tool pool to this allowlist (top-level `--profile`). The
    /// filter is applied at session-creation time, after built-in and MCP tools are both
    /// in the pool, so `mcp__*` names resolve. Names absent from the assembled pool are a
    /// hard error at session creation (fail-loud). `spawn_agent` is handled by the depth
    /// gate and ignored here.
    pub fn tool_allow(mut self, allow: Vec<String>) -> Self {
        self.tool_allow = Some(allow);
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

    /// Inject a permission-mode catalog. When `Some`, each session exposes an ACP
    /// `SessionModeState` and supports `session/set_mode`; the catalog's current mode
    /// overrides [`Self::policy`] as the session's initial active policy. If not called,
    /// the session has no mode switching and is fixed to [`Self::policy`] (or the default
    /// [`AskWritesPolicy`]).
    pub fn modes(mut self, modes: ModeCatalog) -> Self {
        self.modes = Some(modes);
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

    /// Inject shared state for the `--goal` goal-driven loop. When `Some`, the session
    /// runs in goal mode: the top-level turn injects it into the `goal_done` tool via
    /// [`crate::tool::ToolContext::goal`], and the `goal-gate` hook uses it to drive
    /// multi-turn autonomous loops. If not called, the session runs in non-goal mode (the
    /// default).
    pub fn goal(mut self, goal: Arc<crate::session::GoalState>) -> Self {
        self.goal = Some(goal);
        self
    }

    /// Sets the background task progress view configuration (progress ring capacity /
    /// per-block body character limit). When not called, defaults to ring size 64 and
    /// body limit 0, meaning only summary/metadata is shown and sub-turn bodies are not
    /// populated. During CLI assembly, this is injected from the `[tools.background]`
    /// projection.
    pub fn background_progress(mut self, config: crate::session::BackgroundProgressConfig) -> Self {
        self.background_progress = config;
        self
    }

    /// Sets the HTTP fetch backend for this core. When unset, defaults to
    /// [`NoopHttpClient`]—any `fetch` call will fail with
    /// [`crate::http::HttpClientError::Transport`], allowing tests or `echo` assemblies
    /// that don't need networking to skip constructing a real HTTP stack.
    pub fn http(mut self, http: Arc<dyn HttpClient>) -> Self {
        self.http = Some(http);
        self
    }

    /// Sets the hook engine for this core. When unset, falls back to [`NoopHookEngine`] —
    /// all hook calls return `Pass` directly, and the main loop behaves as if the hook
    /// system were not introduced.
    pub fn hook_engine(mut self, hook_engine: Arc<dyn HookEngine>) -> Self {
        self.hook_engine = Some(hook_engine);
        self
    }

    /// # Panics
    /// Neither `registry` nor `provider` is set; or, in the single-provider path,
    /// `config.model` is an empty string (the registry must have at least one default
    /// model).
    pub fn build(mut self) -> DefaultAgentCore {
        let registry = self.registry.take().unwrap_or_else(|| {
            let provider = self
                .single_provider
                .take()
                .expect("DefaultAgentCore requires a provider or a registry");
            let vendor = provider.info().vendor;
            // Under the single-provider path, config usually lacks a selected vendor —
            // fill in the provider's own vendor so that `resolve_initial_provider` can
            // find the entry by the (vendor, model) pair.
            if self.config.provider.is_empty() {
                self.config.provider = vendor.clone();
            }
            let model_id = self.config.model.clone();
            assert!(
                !model_id.is_empty(),
                "DefaultAgentCoreBuilder::provider() requires TurnConfig::model to be set; \
                 use registry() for multi-provider setups"
            );
            // In the single-provider path, treat `TurnConfig::allowed_models` as the
            // model candidate list — this mirrors the multi-provider assembly in the CLI:
            // users declare candidates via `[providers.<p>.models]`, and the agent does
            // not send a `list_models` network request to the adapter. When
            // `allowed_models` is absent, fall back to exposing only the default model.
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
                    &vendor,
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
            tool_allow: self.tool_allow,
            policy: self
                .policy
                .unwrap_or_else(|| Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>),
            modes: self.modes,
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
            background_progress: self.background_progress,
            goal: self.goal,
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
            // Top-level `--profile` allowlist: applied here (not at CLI assembly) so the
            // pool already contains connected MCP tools, letting `mcp__*` names resolve.
            // `spawn_agent` is governed by the depth gate, so it is skipped by the filter.
            let composite = self.apply_tool_allow(composite)?;

            // After the session-enter hook, absorb any injected `additional_context` as
            // candidate system-prompt suffixes.
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

            // Session-level cancellation token: both the driver loop exit signal and the
            // source of background task cancellation tokens (same token).
            let session_cancel = CancellationToken::new();
            let (policy, modes) = self.session_policy_state();
            let concrete = Arc::new(DefaultSession {
                id: id.clone(),
                cwd,
                history: Arc::new(VecHistory::new()) as Arc<dyn History>,
                tools: composite,
                registry: self.registry.clone(),
                provider_state: RwLock::new(initial),
                policy,
                modes,
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                background: crate::session::BackgroundTasks::new(
                    session_cancel.clone(),
                    self.background_progress,
                ),
                goal: self.goal.clone(),
                compaction_slot: crate::session::CompactionSlot::new(),
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
            // Spawn the session driver (active keep-alive). The driver holds a `Weak`
            // self-reference so that when all external strong references to the session
            // are dropped, the driver's `upgrade` fails and it exits, preventing the
            // session from living forever.
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

            // After session enter hook (resume path). Same as `create_session`: retrieve
            // the injected context.
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
            let (policy, modes) = self.session_policy_state();
            let concrete = Arc::new(DefaultSession {
                id: loaded.info.id.clone(),
                cwd: loaded.info.cwd.clone(),
                history: Arc::new(VecHistory::from_messages(loaded.history)) as Arc<dyn History>,
                tools: self.apply_tool_allow(Arc::new(CompositeRegistry::new(
                    session_tools,
                    self.process_tools.clone(),
                )))?,
                registry: self.registry.clone(),
                provider_state: RwLock::new(initial),
                policy,
                modes,
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                background: crate::session::BackgroundTasks::new(
                    session_cancel.clone(),
                    self.background_progress,
                ),
                goal: self.goal.clone(),
                compaction_slot: crate::session::CompactionSlot::new(),
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
    /// Apply the top-level `--profile` tool allowlist (if configured) to a session's fully
    /// assembled tool pool. No-op when `tool_allow` is `None`. `spawn_agent` is skipped
    /// (governed by the depth gate, not the allowlist). An allowlisted name absent from
    /// the pool is a hard error (fail-loud).
    fn apply_tool_allow(
        &self,
        pool: Arc<dyn ToolRegistry>,
    ) -> Result<Arc<dyn ToolRegistry>, AgentError> {
        let Some(allow) = &self.tool_allow else {
            return Ok(pool);
        };
        crate::session::filter_registry_by_allowlist(
            &pool,
            allow,
            crate::tool::SPAWN_AGENT_TOOL_NAME,
        )
        .map_err(|name| {
            AgentError::Other(BoxError::new(io::Error::other(format!(
                "profile allows unknown tool `{name}` (not in built-in or MCP tool pool)"
            ))))
        })
    }

    /// Look up the entry in the registry for the current [`TurnConfig::model`] and
    /// resolve it into `(provider, hosted_capabilities)`. Shared by `create_session` /
    /// `load_session`.
    ///
    /// The configured model must have an entry in the registry —
    /// [`ProviderRegistry::new`] already validated the default model during CLI assembly,
    /// so the only way to reach this error is builder misuse (registry and turn config
    /// are inconsistent).
    fn resolve_initial_provider(&self) -> Result<SessionProviderState, AgentError> {
        let (vendor, model) = {
            let cfg = self
                .config
                .read()
                .expect("DefaultAgentCore config rwlock poisoned");
            (cfg.provider.clone(), cfg.model.clone())
        };
        let entry = self.registry.entry_for(&vendor, &model).ok_or_else(|| {
            AgentError::Other(BoxError::new(io::Error::other(format!(
                "default model `{model}` is not declared by provider `{vendor}` in the registry"
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

    /// Derive the initial `(active policy, mode catalog)` for a new session.
    ///
    /// When a [`ModeCatalog`] is configured: each session holds its own clone of the
    /// catalog (so `current` can be switched independently), and the active policy is the
    /// policy of the catalog's current mode. When not configured: the active policy is
    /// the process-level `policy`, and there is no catalog (mode switching is
    /// unavailable).
    fn session_policy_state(&self) -> (RwLock<Arc<dyn SandboxPolicy>>, Option<Mutex<ModeCatalog>>) {
        match &self.modes {
            Some(catalog) => {
                let catalog = catalog.clone();
                let active = catalog.current_policy();
                (RwLock::new(active), Some(Mutex::new(catalog)))
            }
            None => (RwLock::new(self.policy.clone()), None),
        }
    }
}

/// The currently selected real provider for the session, together with the parsed hosted
/// capabilities of that provider.
///
/// Atomically replaced by `set_model` when switching providers.
struct SessionProviderState {
    provider: Arc<dyn LlmProvider>,
    hosted_capabilities: HostedCapabilities,
}

pub struct DefaultSession {
    id: SessionId,
    cwd: PathBuf,
    /// Use `Arc` instead of `Box` because the background compaction task
    /// ([`CompactionSlot`](crate::session::CompactionSlot)) needs to hold it with a
    /// `'static` lifetime across turns, requiring shared reference counting.
    history: Arc<dyn History>,
    tools: Arc<dyn ToolRegistry>,
    /// Global provider directory. The session shares the same `Arc<ProviderRegistry>`
    /// held by [`DefaultAgentCore`] — it is used to resolve candidates and owner
    /// providers for `list_models` / `set_model`.
    registry: Arc<ProviderRegistry>,
    /// Current selected (provider, hosted_capabilities) state. `set_model` replaces the
    /// entire pair when switching providers, ensuring `(provider, hosted_capabilities)`
    /// is always consistent — there is no intermediate state where the provider has
    /// changed but capabilities have not.
    provider_state: RwLock<SessionProviderState>,
    /// The currently active decision policy. Atomically replaced on `set_mode` (lock
    /// order: after `modes`).
    /// Uses `RwLock<Arc<_>>` rather than a bare `Arc` because the per-session permission
    /// mode must be switchable at runtime; `run_turn` snapshots the policy via
    /// `.read().clone()` at the start of each turn, so in-flight turns are unaffected by
    /// subsequent switches (same semantics as `set_model`).
    policy: RwLock<Arc<dyn SandboxPolicy>>,
    /// Permission mode catalog. When `Some`, enables `session/set_mode` and ACP
    /// `SessionModeState`; when `None`, `policy` is fixed and cannot be switched. Uses
    /// `std::sync::Mutex` held only briefly, never across an await.
    modes: Option<Mutex<ModeCatalog>>,
    events: Arc<EventEmitter>,
    permissions: Arc<PermissionGate>,
    /// Single-turn mutex + cancel channel. `Some(token)` means a turn is running; `None`
    /// means idle. The `std::sync::Mutex` is held briefly and never across an await.
    turn_state: Mutex<TurnSlot>,
    /// Session-level background task table (landing point for `run_in_background`). Holds
    /// the task's `JoinHandle` to keep it alive past the originating turn; its internal
    /// cancel token is independent of the turn's child token. `run_turn` clones it
    /// through `TurnRunner` → `ToolContext` for injection into tools.
    background: crate::session::BackgroundTasks,
    /// Shared state for the `--goal` goal-driven loop. When `Some`, this session runs in
    /// goal mode; the top-level turn injects it into tools via
    /// [`crate::tool::ToolContext::goal`], and the `goal-gate` hook uses it to keep the
    /// turn alive or let it proceed. Cloned from [`DefaultAgentCore::goal`]. `None` =
    /// non-goal mode.
    goal: Option<Arc<crate::session::GoalState>>,
    /// Session-level single-flight compaction slot. When the soft watermark is exceeded,
    /// the turn main loop asynchronously triggers one summary compaction without blocking
    /// the current turn. See `session/turn/compaction_slot.rs`.
    compaction_slot: crate::session::CompactionSlot,
    /// Notifies when a turn slot is released. `TurnGuard::drop` calls `notify_one` — the
    /// session driver waits on this after hitting `TurnInProgress`, so it can start an
    /// autonomous turn continuation once the current turn ends (liveness guarantee for
    /// autonomous continuation).
    turn_freed: Arc<tokio::sync::Notify>,
    /// Session-level cancellation token; cancelled when the session terminates, causing
    /// the driver loop to exit. Also the source (same token) for cancellation tokens of
    /// tasks inside `background`.
    session_cancel: CancellationToken,
    config: RwLock<TurnConfig>,
    /// Session-level filesystem backend. Injected by [`AgentCore::create_session`];
    /// `TurnRunner` borrows a `&dyn FsBackend` into [`crate::tool::ToolContext`] for
    /// tools.
    fs: Arc<dyn FsBackend>,
    /// Session-level shell backend. Injected by [`AgentCore::create_session`] alongside
    /// `fs`; the `bash` tool accesses it via [`crate::tool::ToolContext`].
    shell: Arc<dyn ShellBackend>,
    /// How the agent is accessed. Injected by [`AgentCore::create_session`] /
    /// `load_session`, assembled into [`RunningContext`] during turn setup, and rendered
    /// into the `# Environment` section of the system prompt.
    frontend: Frontend,
    /// HTTP fetch backend, shared across sessions in this core and held/cloned by
    /// [`DefaultAgentCore`]. The `fetch` tool accesses it via
    /// [`crate::tool::ToolContext`].
    http: Arc<dyn HttpClient>,
    /// Hook engine, shared across sessions in this core. When `run_turn` assembles
    /// [`TurnRunner`], it borrows `&dyn HookEngine` to the main loop.
    hook_engine: Arc<dyn HookEngine>,
    /// Content appended by the `after_session_enter` hook during session startup (e.g.,
    /// skill L1 manifest / always-on skill body). Populated by
    /// [`AgentCore::create_session`] / `load_session` after the hook runs; on each turn,
    /// when assembling the system prompt, [`merge_session_overlay`] merges it with the
    /// explicit `config.system_prompt`, and
    /// [`crate::session::prompt::resolve_system_prompt`] places it into the "Session
    /// Instructions" section via hooks.
    session_start_append: Vec<agent_client_protocol_schema::ContentBlock>,
    /// Adjacent-request stability diagnostic. Emits a tracing record for every request
    /// actually sent to the provider, helping to identify the source of cache misses.
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

    /// Core execution of a turn, shared by user-initiated turns and automatic
    /// continuation turns.
    ///
    /// `prompt` is either external input (user turn) or empty (automatic continuation
    /// turn). In both cases, any completed background results are prepended as **prefix
    /// blocks** to the prompt. An empty prompt with no background results does not start
    /// a turn (returns `EndTurn` to avoid a no-op turn). Turn slot mutual exclusion is
    /// still enforced at the top of this function.
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

            // RAII guard: releases the slot and wakes the driver on any exit path
            // (including panic inside an await).
            let _guard = TurnGuard {
                state: &self.turn_state,
                freed: &self.turn_freed,
            };

            // Prepend completed background-task results as prefix blocks for the current
            // prompt.
            // Background task reflow.
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

            // Empty prompt (autonomous turn with no background results to consume) — skip
            // the turn to avoid spinning.
            if prompt.is_empty() {
                return Ok(StopReason::EndTurn);
            }

            let config = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .clone();
            // Snapshot (provider, hosted) once at turn start — concurrent set_model
            // requests within the same turn still use the chosen provider; changes take
            // effect on the next turn.
            let provider = self.current_provider();
            let hosted = self.current_hosted();
            let running_ctx = RunningContext::new(self.frontend, &self.cwd);
            // Merge session-scoped injection (the `additional_context` from the
            // `after_session_enter` hook, e.g. skill L1 manifest / always-on skill body)
            // with the explicit `system_prompt` into a single "Session Instructions"
            // overlay — both originate from the same source and target the same location,
            // so no extra parameter is needed.
            let session_overlay =
                merge_session_overlay(config.system_prompt.as_deref(), &self.session_start_append);
            let system_prompt = resolve_system_prompt(
                &running_ctx,
                &provider.info().vendor,
                &config.model,
                &config.base_prompt,
                &config.prompt,
                session_overlay.as_deref(),
            )
            .map_err(|err| TurnError::Internal(BoxError::new(err)))?;
            // Snapshot the active policy for this turn: an in-progress turn uses a fixed
            // policy, so
            // `session/set_mode` changes only affect subsequent turns (same semantics as
            // `set_model`).
            // Use an owned `Arc` rather than a borrow — it flows with `ToolContext` into
            // `spawn_agent`,
            // ensuring child agents capture the parent's actual policy at this moment,
            // not a stale process-level default.
            let policy = self
                .policy
                .read()
                .expect("DefaultSession policy rwlock poisoned")
                .clone();
            let runner = TurnRunner {
                history: self.history.as_ref(),
                tools: self.tools.as_ref(),
                // Owned clone of the composite (built-in + MCP) for injection into
                // ToolContext → spawn_agent, so subagent profiles can allow `mcp__*`.
                session_tools: Some(self.tools.clone()),
                provider: provider.as_ref(),
                policy,
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
                // Inject the session-level background task handle into the top-level
                // turn, enabling the tool's `run_in_background` capability. Nested
                // sub-agent turns do not receive this injection (see `spawn_agent`).
                background: Some(self.background.clone()),
                // Top-level turn injects the goal-loop state (`Some` under `--goal`
                // mode); the `goal_done` tool and `goal-gate` hook use it to drive
                // multi-turn autonomous loops. `None` in non-goal mode.
                goal: self.goal.clone(),
                // Inject the compaction slot, history Arc, and provider Arc into the
                // top-level turn so that summary compaction can be triggered
                // asynchronously when the soft watermark is exceeded. Sub-agent turns
                // pass `None` for all of these (see `spawn_agent`).
                compaction_slot: Some(self.compaction_slot.clone()),
                history_arc: Some(self.history.clone()),
                provider_arc: Some(provider.clone()),
                session_cancel: Some(self.session_cancel.clone()),
                ingest_source,
            };

            runner.run(prompt).await
        }
        .instrument(span)
        .await
    }

    /// Session driver loop (autonomous turn continuation): a long-lived task that starts
    /// an autonomous turn when a background task completes, consuming its results.
    /// Spawned during `create_session` / `load_session`.
    ///
    /// Holds `Weak<Self>` instead of `Arc`: the driver must not keep the session alive
    /// indefinitely. Each iteration first calls `upgrade` — when all external strong
    /// references (the `AgentCore.sessions` DashMap) are gone, `upgrade` fails and the
    /// driver exits. `session_cancel` is the explicit exit signal (process shutdown /
    /// future session eviction).
    ///
    /// Two waiting paths:
    /// - `background.wait_for_completion()`: a task completed → prepare to start an
    ///   autonomous turn;
    /// - `session_cancel.cancelled()`: session terminated → exit the loop.
    ///
    /// If a `TurnInProgress` is encountered before starting a turn (a user turn is
    /// running), wait for `turn_freed` and retry — this is exactly where user input and
    /// background results contend for the same turn slot: if the user turn arrives first,
    /// it runs, and the background result either hitches a ride (via `run_turn_core`'s
    /// drain) or waits for it to finish before starting its own turn.
    async fn drive(weak: std::sync::Weak<Self>) {
        loop {
            let Some(this) = weak.upgrade() else { break };
            if this.session_cancel.is_cancelled() {
                break;
            }
            // First take the notified() future, then check the queue — avoid missing
            // completion notifications that arrive between the two steps.
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

    /// Run an autonomous turn; if the turn slot is occupied (a user turn is running),
    /// wait for it to be released and retry, up to the point where the result is
    /// consumed. Abort when `session_cancel` fires.
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
                    // A user turn is in progress. Wait for it to finish — its
                    // `run_turn_core` will drain our background results (piggybacking),
                    // so `has_completed` will be empty here and we exit naturally.
                    tokio::select! {
                        () = self.turn_freed.notified() => {}
                        () = self.session_cancel.cancelled() => return,
                    }
                    if !self.background.has_completed() {
                        // Consumed by an in-flight user turn (piggybacking) — no need to
                        // start an autonomous turn.
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
        // On session drop, cancel `session_cancel`: this kills all in-flight background
        // tasks and wakes the driver loop's `session_cancel.cancelled()` branch so it
        // exits (the driver holds a `Weak`, which will now fail to upgrade).
        self.session_cancel.cancel();
    }
}

#[derive(Default)]
struct TurnSlot {
    cancel: Option<CancellationToken>,
}

/// A guard that occupies a turn slot on construction and releases it on drop.
struct TurnGuard<'a> {
    state: &'a Mutex<TurnSlot>,
    /// Notifies the session driver when the turn is released (liveness guarantee for
    /// proactive turn renewal).
    freed: &'a tokio::sync::Notify,
}

impl<'a> Drop for TurnGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.state.lock() {
            slot.cancel = None;
        }
        // Turn slot is now empty; wake the driver that may be waiting to start its own
        // turn.
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
            // Under multi-provider assembly, the candidate set comes from the registry —
            // each entry already carries its own model list (populated during CLI
            // assembly). It is then filtered against the session's `allowed_models`
            // allowlist. The registry makes no network requests, so this path always
            // succeeds.
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

    fn set_model(&self, selection: ModelSelection) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            let ModelSelection { provider, model } = selection;
            let allowed_models = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .allowed_models
                .clone();
            // Note: `allowed_models` is a flat list of model IDs without a vendor
            // dimension — models with the same name are allowed or denied uniformly
            // across all providers. Pairing is deferred to future work.
            if let Some(allowed_models) = allowed_models.as_ref()
                && !allowed_models.iter().any(|allowed| allowed == &model)
            {
                return Err(ProviderError::new(ProviderErrorKind::ModelNotFound {
                    model,
                }));
            }

            let Some(entry) = self.registry.entry_for(&provider, &model) else {
                return Err(ProviderError::new(ProviderErrorKind::ModelNotFound {
                    model,
                }));
            };

            // When switching providers, re-resolve hosted capabilities: each entry
            // carries its own [`SessionCapabilitiesConfig`], which is cross-referenced
            // with the provider's hosted_capabilities. If the delegate is not supported
            // by the provider, a `ProviderError` is returned — preserving the stable
            // failure semantics of `set_model`.
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

            // Lock order: `provider_state` before `config`, matching the snapshot path in
            // `run_turn`.
            // The window where both write locks are held is very short (just a few
            // assignments) and will not block the main loop.
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
            config.provider = provider;
            config.model = model;
            Ok(())
        })
    }

    fn current_mode(&self) -> Option<String> {
        self.modes.as_ref().map(|m| {
            m.lock()
                .expect("DefaultSession modes mutex poisoned")
                .current_id()
                .to_string()
        })
    }

    fn available_modes(&self) -> Vec<crate::session::ModeDescriptor> {
        let Some(modes) = self.modes.as_ref() else {
            return Vec::new();
        };
        modes
            .lock()
            .expect("DefaultSession modes mutex poisoned")
            .modes()
            .iter()
            .map(|m| crate::session::ModeDescriptor {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
            })
            .collect()
    }

    fn set_mode(&self, mode_id: String) -> Result<(), AgentError> {
        let Some(modes) = self.modes.as_ref() else {
            return Err(AgentError::ModeNotFound(mode_id));
        };
        // Lock order: `modes` before `policy` (no overlap with `run_turn`'s read path —
        // `run_turn` only reads `policy`). Both locks are held briefly and never across
        // an `.await`.
        let mut catalog = modes.lock().expect("DefaultSession modes mutex poisoned");
        if !catalog.set_current(&mode_id) {
            return Err(AgentError::ModeNotFound(mode_id));
        }
        let active = catalog.current_policy();
        *self
            .policy
            .write()
            .expect("DefaultSession policy rwlock poisoned") = active;
        Ok(())
    }

    fn current_reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.config
            .read()
            .expect("DefaultSession config rwlock poisoned")
            .sampling
            .reasoning_effort
    }

    fn set_reasoning_effort(&self, effort: Option<ReasoningEffort>) {
        self.config
            .write()
            .expect("DefaultSession config rwlock poisoned")
            .sampling
            .reasoning_effort = effort;
    }

    fn subscribe(&self) -> EventStream {
        self.events.subscribe()
    }

    fn history_snapshot(&self) -> Vec<Message> {
        self.history.snapshot()
    }

    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>> {
        // User-driven turn: piggybacks completed background results as a prefix to the
        // prompt.
        // (Passive backflow, complementary to active continuation — active continuation
        // handles idle state, piggybacking handles "user happened to speak up".)
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
        // No turn running → no-op (idempotent)
    }

    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution) {
        self.permissions.resolve(&id, outcome);
    }

    fn context_status(&self) -> ContextStatus {
        let used_tokens = self.history.token_estimate();
        let context_window = {
            let model = self
                .config
                .read()
                .expect("DefaultSession config rwlock poisoned")
                .model
                .clone();
            self.current_provider()
                .model_info(&model)
                .and_then(|m| m.context_window)
        };
        let ratio = match (used_tokens, context_window) {
            (Some(used), Some(window)) if window > 0 => Some(used as f64 / window as f64),
            _ => None,
        };
        ContextStatus {
            used_tokens,
            context_window,
            ratio,
        }
    }

    fn compact_now(&self) -> BoxFuture<'_, Result<Option<CompactionReport>, TurnError>> {
        Box::pin(async move {
            // A turn rewrites history concurrently with compaction; refuse rather than
            // race. The caller should `/cancel` or wait. (Held briefly, never across await.)
            {
                let slot = self
                    .turn_state
                    .lock()
                    .expect("DefaultSession turn_state mutex poisoned");
                if slot.cancel.is_some() {
                    return Err(TurnError::TurnInProgress);
                }
            }

            let (model, sampling) = {
                let config = self
                    .config
                    .read()
                    .expect("DefaultSession config rwlock poisoned");
                (config.model.clone(), config.sampling.clone())
            };
            let ctx = CompactionCtx {
                provider: self.current_provider(),
                model,
                sampling,
                tools: self.tools.schemas(),
                cancel: self.session_cancel.clone(),
            };

            // Force a compaction regardless of watermark: use the current estimate as the
            // threshold so boundary selection keeps a sensible tail. If history is empty /
            // has no estimate, there is nothing to compact.
            let Some(threshold) = self.history.token_estimate() else {
                return Ok(None);
            };
            let report = run_sync_compaction(self.history.as_ref(), &ctx, threshold).await;
            if let Some(report) = report {
                self.events
                    .emit(AgentEvent::ContextCompressed {
                        tokens_before: report.tokens_before,
                        tokens_after: report.tokens_after,
                    })
                    .await;
            }
            Ok(report)
        })
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

/// Prepend the provider name to the model's `display_name` so that ACP clients can
/// distinguish the same model ID served by different providers — the only reliable
/// disambiguation when multiple gateways expose the same model (e.g. "OpenAI: gpt-4o" vs
/// "gw-b: gpt-4o").
fn decorate_with_provider_display(mut model: ModelInfo, provider: &ProviderInfo) -> ModelInfo {
    let name = model
        .display_name
        .clone()
        .unwrap_or_else(|| model.id.clone());
    model.display_name = Some(format!("{}: {name}", provider.display_name));
    model
}

/// Generates a session ID as a random UUID v4.
///
/// The `defect-acp` `session/new` handler needs a [`SessionId`] (to construct an
/// `AcpFsBackend`) before calling [`AgentCore::create_session`]; this function is
/// public so that both acp and tests can produce IDs in a consistent format.
///
/// Using a globally unique UUID instead of an in-process counter plus timestamp
/// avoids collisions across process restarts and concurrent instances, and allows
/// downstream consumers (storage on-disk directories, observability trace
/// correlation) to use it as a stable primary key.
pub fn new_session_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Short session ID for tracing spans: takes the first 12 characters. Diagnostic use
/// only.
fn short_id(s: &str) -> &str {
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Merge the explicit `config.system_prompt` with the text-only content blocks from the
/// session-start hook's `append` into a single overlay string for the `session_overlay`
/// parameter of [`resolve_system_prompt`]. Returns `None` when both are empty (no empty
/// segment injected); when both are present they are separated by `\n\n`.
fn merge_session_overlay(system_prompt: Option<&str>, append: &[ContentBlock]) -> Option<String> {
    let appended: String = append
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    match (system_prompt, appended.is_empty()) {
        (Some(sp), true) => Some(sp.to_owned()),
        (Some(sp), false) => Some(format!("{sp}\n\n{appended}")),
        (None, false) => Some(appended),
        (None, true) => None,
    }
}
