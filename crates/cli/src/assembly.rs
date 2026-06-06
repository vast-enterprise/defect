//! CLI 默认 AgentCore 装配器。
//!
//! 这一层把 `src/bin/cli.rs` 里原本散落的 provider / tool / hook / storage /
//! MCP / observability 拼装逻辑收束成一个可扩展 builder。底层
//! [`DefaultAgentCoreBuilder`] 仍然保持最小 agent 抽象；这里表达的是
//! defect CLI 的“默认 feature set”。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use defect_agent::hooks::HookEngine;
use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::llm::{ProviderEntry, ProviderRegistry};
use defect_agent::policy::{ModeCatalog, NonInteractivePolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, SessionObserver, SessionToolFactory, StaticToolRegistry,
    ToolRegistry, TurnConfig,
};
use defect_agent::tool::{SkillEntry, Tool};
use defect_config::{HooksConfig, LoadConfigOptions, LoadedConfig, ProfileSpec, SandboxMode};
use defect_mcp::McpToolFactory;
use defect_storage::StorageObserver;

use crate::hooks::{self, HookEngineCtx};
use crate::http_stack::build_http_stack_config;
use crate::mcp_servers::build_default_mcp_servers;
use crate::observability;
use crate::paths::{default_sessions_root, local_sessions_root};
use crate::policy::{build_mode_catalog, build_policy};
use crate::providers::{build_provider_entries, build_registry};
use crate::tools::{
    build_process_tools, build_process_tools_with_subagents, filter_tools_by_allowlist,
    project_skills,
};

const SKILL_MANIFEST_HOOK_NAME: &str = "skill-manifest";
const SKILL_TRIGGERS_HOOK_NAME: &str = "skill-triggers";

/// CLI 默认装配中的可裁剪功能。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultFeature {
    /// 默认进程级工具：bash / fs / fetch / search。
    ProcessTools,
    /// profile 驱动的 `spawn_agent` 与后台任务控制工具。
    Subagents,
    /// skill 工具与自动 skill hook。
    Skills,
    /// 用户配置中的 hooks。
    Hooks,
    /// session 持久化与 `--resume`。
    Storage,
    /// 默认 MCP server factory。
    Mcp,
    /// langfuse 等旁路观测器。
    Observability,
    /// HTTP fetch 后端。
    Http,
    /// ACP 暴露的权限模式目录。
    Modes,
}

/// CLI 默认 feature set。
#[derive(Debug, Clone)]
pub struct DefaultFeatureSet {
    process_tools: bool,
    subagents: bool,
    skills: bool,
    hooks: bool,
    storage: bool,
    mcp: bool,
    observability: bool,
    http: bool,
    modes: bool,
}

impl Default for DefaultFeatureSet {
    fn default() -> Self {
        Self {
            process_tools: true,
            subagents: true,
            skills: true,
            hooks: true,
            storage: true,
            mcp: true,
            observability: true,
            http: true,
            modes: true,
        }
    }
}

impl DefaultFeatureSet {
    /// 一份全空 feature set。适合宿主只想复用配置解析，再逐项显式添加能力。
    pub fn empty() -> Self {
        Self {
            process_tools: false,
            subagents: false,
            skills: false,
            hooks: false,
            storage: false,
            mcp: false,
            observability: false,
            http: false,
            modes: false,
        }
    }

    /// 禁用某个默认功能。
    pub fn without(mut self, feature: DefaultFeature) -> Self {
        self.set(feature, false);
        self
    }

    /// 启用某个默认功能。
    pub fn with(mut self, feature: DefaultFeature) -> Self {
        self.set(feature, true);
        self
    }

    fn set(&mut self, feature: DefaultFeature, enabled: bool) {
        match feature {
            DefaultFeature::ProcessTools => self.process_tools = enabled,
            DefaultFeature::Subagents => self.subagents = enabled,
            DefaultFeature::Skills => self.skills = enabled,
            DefaultFeature::Hooks => self.hooks = enabled,
            DefaultFeature::Storage => self.storage = enabled,
            DefaultFeature::Mcp => self.mcp = enabled,
            DefaultFeature::Observability => self.observability = enabled,
            DefaultFeature::Http => self.http = enabled,
            DefaultFeature::Modes => self.modes = enabled,
        }
    }
}

/// REPL 装配语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplMode {
    /// 走 ACP server 路径。
    Disabled,
    /// 走内置 REPL；默认 sandbox 会切到 open，避免无 UI 权限确认导致卡死。
    Enabled,
}

/// 运行默认 CLI 所需的 AgentCore 与伴随信息。
pub struct BuiltCliAgent {
    pub agent: Arc<dyn AgentCore>,
    pub resume_session_id: Option<SessionId>,
    pub sandbox_mode: SandboxMode,
    pub turn_config: TurnConfig,
    /// `--goal` 模式的共享状态句柄（非 goal 模式为 `None`）。oneshot runner 在 turn
    /// 结束后读 [`GoalState::is_reached`]——续命耗尽但目标未达成时用非 0 退出码，
    /// 避免 CI 把"跑满轮数仍未达成"误判成成功。
    ///
    /// [`GoalState`]: defect_agent::session::GoalState
    pub goal: Option<Arc<defect_agent::session::GoalState>>,
}

/// defect CLI 默认 AgentCore 装配 builder。
pub struct CliAgentBuilder {
    cwd: PathBuf,
    load_options: LoadConfigOptions,
    config: LoadedConfig,
    features: DefaultFeatureSet,
    repl: ReplMode,
    local_sessions: bool,
    profile: Option<String>,
    resume: Option<Option<String>>,
    registry_override: Option<Arc<ProviderRegistry>>,
    extra_provider_entries: Vec<ProviderEntry>,
    process_tools_override: Option<Arc<dyn ToolRegistry>>,
    extra_process_tools: Vec<Arc<dyn Tool>>,
    extra_process_registries: Vec<Arc<dyn ToolRegistry>>,
    policy_override: Option<Arc<dyn SandboxPolicy>>,
    non_interactive: bool,
    goal: Option<Arc<defect_agent::session::GoalState>>,
    /// `--max-turns`：goal 模式下 before_turn_end 续命上限，映射到
    /// `TurnConfig::max_hook_continues`。`None` = 用配置/默认值。
    max_turns: Option<u32>,
    modes_override: Option<ModeCatalog>,
    hook_engine_override: Option<Arc<dyn HookEngine>>,
    builtin_registry: BuiltinRegistry,
    session_tool_factory_override: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
}

impl CliAgentBuilder {
    /// 创建一份使用完整默认 feature set 的 CLI agent builder。
    pub fn new(cwd: PathBuf, load_options: LoadConfigOptions, config: LoadedConfig) -> Self {
        Self {
            cwd,
            load_options,
            config,
            features: DefaultFeatureSet::default(),
            repl: ReplMode::Disabled,
            local_sessions: false,
            profile: None,
            resume: None,
            registry_override: None,
            extra_provider_entries: Vec::new(),
            process_tools_override: None,
            extra_process_tools: Vec::new(),
            extra_process_registries: Vec::new(),
            policy_override: None,
            non_interactive: false,
            goal: None,
            max_turns: None,
            modes_override: None,
            hook_engine_override: None,
            builtin_registry: BuiltinRegistry::defaults(),
            session_tool_factory_override: None,
            observers: Vec::new(),
        }
    }

    /// 替换默认 feature set。
    pub fn features(mut self, features: DefaultFeatureSet) -> Self {
        self.features = features;
        self
    }

    /// 启用 REPL 装配语义。
    pub fn repl(mut self, repl: ReplMode) -> Self {
        self.repl = repl;
        self
    }

    /// 使用项目本地 session 目录。
    pub fn local_sessions(mut self) -> Self {
        self.local_sessions = true;
        self
    }

    /// 指定顶层 profile。
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// 指定 resume 参数。`None` 表示裸 `--resume`，按 cwd 查最近 session。
    pub fn resume(mut self, session_id: Option<String>) -> Self {
        self.resume = Some(session_id);
        self
    }

    /// 直接覆盖 provider registry。
    pub fn provider_registry(mut self, registry: Arc<ProviderRegistry>) -> Self {
        self.registry_override = Some(registry);
        self
    }

    /// 在默认 provider entries 之后追加一个 provider entry。
    pub fn add_provider_entry(mut self, entry: ProviderEntry) -> Self {
        self.extra_provider_entries.push(entry);
        self
    }

    /// 直接覆盖进程级工具注册表。
    pub fn process_tools(mut self, tools: Arc<dyn ToolRegistry>) -> Self {
        self.process_tools_override = Some(tools);
        self
    }

    /// 给默认进程级工具叠加一个工具。重名时叠加工具优先生效。
    pub fn add_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.extra_process_tools.push(tool);
        self
    }

    /// 给默认进程级工具叠加一份 registry。越晚添加的 registry 优先级越高。
    pub fn add_tool_registry(mut self, registry: Arc<dyn ToolRegistry>) -> Self {
        self.extra_process_registries.push(registry);
        self
    }

    /// 覆盖 sandbox policy。
    pub fn policy(mut self, policy: Arc<dyn SandboxPolicy>) -> Self {
        self.policy_override = Some(policy);
        self
    }

    /// 用 [`NonInteractivePolicy`] 包裹最终 policy：内层返回 `Ask` 时降级为
    /// `Deny`，避免无 TTY 环境（`--message` 单轮模式）挂死在权限确认上。
    /// `Allow` / `Deny` 原样透传。
    pub fn non_interactive(mut self) -> Self {
        self.non_interactive = true;
        self
    }

    /// 启用 `--goal` 目标驱动循环：注册 `goal_done` 工具 + 挂 `goal-gate` hook
    /// （`before_turn_end`），并把 [`GoalState`] 接进 session。agent 多轮自主跑
    /// 直到调用 `goal_done`（达成）或撞 `max_hook_continues` 上限（`--max-turns`）。
    ///
    /// [`GoalState`]: defect_agent::session::GoalState
    pub fn goal(mut self, objective: impl Into<String>) -> Self {
        self.goal = Some(Arc::new(defect_agent::session::GoalState::new(
            objective.into(),
        )));
        self
    }

    /// `--max-turns`：goal 模式下 before_turn_end 续命上限（映射到
    /// `TurnConfig::max_hook_continues`）。撞上限后强制放停 + Exhausted 退出。
    pub fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// 覆盖权限模式目录。
    pub fn modes(mut self, modes: ModeCatalog) -> Self {
        self.modes_override = Some(modes);
        self
    }

    /// 覆盖 hook engine。
    pub fn hook_engine(mut self, hook_engine: Arc<dyn HookEngine>) -> Self {
        self.hook_engine_override = Some(hook_engine);
        self
    }

    /// 注册一份 builtin hook 工厂。
    pub fn builtin_registry(mut self, registry: BuiltinRegistry) -> Self {
        self.builtin_registry = registry;
        self
    }

    /// 覆盖 session 级工具 factory，例如自定义 MCP 接入。
    pub fn session_tool_factory(mut self, factory: Arc<dyn SessionToolFactory>) -> Self {
        self.session_tool_factory_override = Some(factory);
        self
    }

    /// 添加 session 观察器。观察器可在 session 创建后订阅事件流并推送到外部系统。
    pub fn observe_session(mut self, observer: Arc<dyn SessionObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    /// 构建 [`AgentCore`] 与 CLI 伴随信息。
    ///
    /// # Errors
    ///
    /// 配置派生失败、provider / hook / subagent 工具装配失败、持久化目录解析失败、
    /// 或显式 `resume` 但找不到目标 session 时返回错误。
    pub async fn build(mut self) -> anyhow::Result<BuiltCliAgent> {
        let profiles = defect_config::discover_profiles(&self.load_options)
            .map_err(|e| anyhow::anyhow!("profile discovery failed: {e}"))?;
        let skill_specs = if self.features.skills {
            defect_config::discover_skills(&self.load_options)
                .map_err(|e| anyhow::anyhow!("skill discovery failed: {e}"))?
        } else {
            BTreeMap::new()
        };
        let skills = project_skills(&skill_specs);
        let (registry, mut turn_config) = self.build_registry().await?;
        apply_profile_to_turn_config(&mut turn_config, self.profile.as_deref(), &profiles)?;
        // `--max-turns`：goal 模式的续命上限。映射到 before_turn_end 续命硬上限。
        if let Some(max_turns) = self.max_turns {
            turn_config.max_hook_continues = max_turns;
        }

        let sandbox_mode = self.resolve_sandbox_mode();
        let mut policy = self
            .policy_override
            .clone()
            .unwrap_or_else(|| build_policy(sandbox_mode));
        // 非交互（`--message`）：用 NonInteractivePolicy 包裹，且**不**装配 ModeCatalog。
        // 关键：DefaultSession 装配了 catalog 时，active policy 取自 catalog 当前模式
        // （`session_policy_state`），会绕过这里的 `policy`——包装就失效，Ask 不降级、
        // 在无 TTY 下永久挂死。oneshot 没有 `set_mode` 客户端，catalog 本就无意义，
        // 故直接置空，让 session 回退到这份包装过的 policy。
        let modes = if self.non_interactive {
            policy = Arc::new(NonInteractivePolicy::new(policy));
            None
        } else {
            self.modes_override.clone().or_else(|| {
                self.features
                    .modes
                    .then(|| build_mode_catalog(sandbox_mode))
            })
        };

        let skills_arc = Arc::new(skills.clone());
        if self.features.skills {
            register_skill_builtins(&mut self.builtin_registry, &skills_arc);
        }
        let builtin_registry = &self.builtin_registry;
        let hook_rt = HookEngineCtx {
            registry: &registry,
            default_model: turn_config.model.as_str(),
        };

        let mut process_tools = self.build_process_tools(
            &profiles,
            &skills,
            &registry,
            &policy,
            builtin_registry,
            &hook_rt,
        )?;
        // `--goal` 模式：叠加 goal_done 工具，让模型能声明目标达成。
        if self.goal.is_some() {
            process_tools = overlay_process_tools(
                process_tools,
                &[Arc::new(defect_agent::tool::GoalDoneTool::new()) as Arc<dyn Tool>],
                &[],
            );
        }
        let hook_engine = self.build_hook_engine(builtin_registry, &hook_rt, &skills_arc)?;
        let storage = self.build_storage()?;
        let resume_session_id = self.resolve_resume(storage.as_ref())?;
        let langfuse = self.build_langfuse()?;
        let http_client = self.build_http()?;

        let mut core = DefaultAgentCore::builder()
            .registry(registry)
            .process_tools(process_tools)
            .policy(policy)
            .config(turn_config.clone())
            .background_progress(self.config.effective.tools.background)
            .hook_engine(hook_engine);
        if let Some(modes) = modes {
            core = core.modes(modes);
        }
        if let Some(goal) = &self.goal {
            core = core.goal(goal.clone());
        }
        if let Some(storage) = storage {
            core = core
                .observe_session(storage.clone())
                .session_loader(storage as Arc<dyn defect_agent::session::SessionLoader>);
        }
        if let Some(factory) = self.build_session_tool_factory() {
            core = core.session_tool_factory(factory);
        }
        if let Some(http_client) = http_client {
            core = core.http(http_client);
        }
        if let Some(langfuse) = langfuse {
            core = core.observe_session(langfuse);
        }
        for observer in self.observers {
            core = core.observe_session(observer);
        }

        Ok(BuiltCliAgent {
            agent: Arc::new(core.build()) as Arc<dyn AgentCore>,
            resume_session_id,
            sandbox_mode,
            turn_config,
            goal: self.goal,
        })
    }

    async fn build_registry(&self) -> anyhow::Result<(Arc<ProviderRegistry>, TurnConfig)> {
        let turn_config = self.config.effective.turn.clone();
        if let Some(registry) = &self.registry_override {
            return Ok((registry.clone(), turn_config));
        }
        if self.extra_provider_entries.is_empty() {
            return build_registry(&self.config).await;
        }
        let http_config = build_http_stack_config(&self.config.effective.http)?;
        let mut entries = build_provider_entries(&self.config, http_config).await?;
        entries.extend(self.extra_provider_entries.clone());
        let registry = ProviderRegistry::new(entries, &turn_config.model)
            .map_err(|e| anyhow::anyhow!("provider registry init failed: {e}"))?;
        Ok((Arc::new(registry), turn_config))
    }

    fn resolve_sandbox_mode(&self) -> SandboxMode {
        match self.repl {
            ReplMode::Disabled => self.config.effective.sandbox.mode,
            ReplMode::Enabled => SandboxMode::Open,
        }
    }

    fn build_process_tools(
        &self,
        profiles: &BTreeMap<String, ProfileSpec>,
        skills: &BTreeMap<String, SkillEntry>,
        registry: &Arc<ProviderRegistry>,
        policy: &Arc<dyn SandboxPolicy>,
        builtin_registry: &BuiltinRegistry,
        hook_rt: &HookEngineCtx<'_>,
    ) -> anyhow::Result<Arc<dyn ToolRegistry>> {
        let base = match &self.process_tools_override {
            Some(tools) => tools.clone(),
            None if self.features.process_tools => self.build_default_process_tools(
                profiles,
                skills,
                registry,
                policy,
                builtin_registry,
                hook_rt,
            )?,
            None => Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>,
        };
        Ok(overlay_process_tools(
            base,
            &self.extra_process_tools,
            &self.extra_process_registries,
        ))
    }

    fn build_default_process_tools(
        &self,
        profiles: &BTreeMap<String, ProfileSpec>,
        skills: &BTreeMap<String, SkillEntry>,
        registry: &Arc<ProviderRegistry>,
        policy: &Arc<dyn SandboxPolicy>,
        builtin_registry: &BuiltinRegistry,
        hook_rt: &HookEngineCtx<'_>,
    ) -> anyhow::Result<Arc<dyn ToolRegistry>> {
        let Some(profile_name) = self.profile.as_deref() else {
            if self.features.subagents || self.features.skills {
                let base_prompt_text = resolve_base_prompt_text(&self.config)?;
                let empty_profiles = BTreeMap::new();
                let empty_skills = BTreeMap::new();
                let enabled_profiles = if self.features.subagents {
                    profiles
                } else {
                    &empty_profiles
                };
                let enabled_skills = if self.features.skills {
                    skills
                } else {
                    &empty_skills
                };
                return build_process_tools_with_subagents(
                    &self.config,
                    enabled_profiles,
                    enabled_skills,
                    registry,
                    policy,
                    base_prompt_text,
                    builtin_registry,
                    hook_rt,
                )
                .map_err(|e| anyhow::anyhow!("subagent hook engine build failed: {e}"));
            }
            return Ok(build_process_tools(&self.config));
        };

        let spec = profiles
            .get(profile_name)
            .ok_or_else(|| unknown_profile_error(profile_name, profiles))?;
        let base = build_process_tools(&self.config);
        filter_tools_by_allowlist(&base, &spec.tool_allow).map_err(|name| {
            anyhow::anyhow!("profile `{profile_name}` allows unknown tool `{name}`")
        })
    }

    fn build_hook_engine(
        &self,
        builtin_registry: &BuiltinRegistry,
        hook_rt: &HookEngineCtx<'_>,
        skills: &Arc<BTreeMap<String, SkillEntry>>,
    ) -> anyhow::Result<Arc<dyn HookEngine>> {
        if let Some(hook_engine) = &self.hook_engine_override {
            return Ok(hook_engine.clone());
        }
        // `--goal` 也需要挂 goal-gate hook，即便用户既没配 [hooks] 也没 skill。
        if self.features.hooks || self.features.skills || self.goal.is_some() {
            let empty_hooks = HooksConfig::default();
            let hooks_config = if self.features.hooks {
                &self.config.effective.hooks
            } else {
                &empty_hooks
            };
            return hooks::build_main_session_engine(
                hooks_config,
                builtin_registry,
                hook_rt,
                skills,
                self.goal.as_ref(),
            )
            .map_err(|e| anyhow::anyhow!("hook engine build failed: {e}"));
        }
        Ok(Arc::new(defect_agent::hooks::NoopHookEngine) as Arc<dyn HookEngine>)
    }

    fn build_storage(&self) -> anyhow::Result<Option<Arc<StorageObserver>>> {
        if !self.features.storage {
            return Ok(None);
        }
        let sessions_root = if self.local_sessions {
            local_sessions_root(&self.cwd)
        } else {
            default_sessions_root()?
        };
        Ok(Some(Arc::new(StorageObserver::new(sessions_root))))
    }

    fn resolve_resume(
        &self,
        storage: Option<&Arc<StorageObserver>>,
    ) -> anyhow::Result<Option<SessionId>> {
        match &self.resume {
            None => Ok(None),
            Some(Some(id)) => Ok(Some(SessionId::new(id.clone()))),
            Some(None) => {
                let Some(storage) = storage else {
                    return Err(anyhow::anyhow!(
                        "--resume requires the default storage feature or a session loader"
                    ));
                };
                let id = storage
                    .latest_session_id_for_cwd(&self.cwd)
                    .map_err(|e| anyhow::anyhow!("failed to scan sessions for resume: {e}"))?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "no previous session found for {} to --resume",
                            self.cwd.display()
                        )
                    })?;
                Ok(Some(id))
            }
        }
    }

    fn build_langfuse(&self) -> anyhow::Result<Option<Arc<dyn SessionObserver>>> {
        if !self.features.observability {
            return Ok(None);
        }
        let observer = observability::build_langfuse_observer(
            self.config.effective.tracing.langfuse.as_ref(),
            build_http_stack_config(&self.config.effective.http)?,
        )?
        .map(|observer| Arc::new(observer) as Arc<dyn SessionObserver>);
        Ok(observer)
    }

    fn build_http(&self) -> anyhow::Result<Option<Arc<dyn defect_agent::http::HttpClient>>> {
        if !self.features.http {
            return Ok(None);
        }
        let http = defect_http::build_fetch_client_arc(&build_http_stack_config(
            &self.config.effective.http,
        )?)
        .map_err(|e| anyhow::anyhow!("fetch http client init failed: {e}"))?;
        Ok(Some(http))
    }

    fn build_session_tool_factory(&self) -> Option<Arc<dyn SessionToolFactory>> {
        if let Some(factory) = &self.session_tool_factory_override {
            return Some(factory.clone());
        }
        self.features.mcp.then(|| {
            Arc::new(McpToolFactory::with_default_servers(
                build_default_mcp_servers(&self.config),
            )) as Arc<dyn SessionToolFactory>
        })
    }
}

fn apply_profile_to_turn_config(
    turn_config: &mut TurnConfig,
    profile_name: Option<&str>,
    profiles: &BTreeMap<String, ProfileSpec>,
) -> anyhow::Result<()> {
    let Some(profile_name) = profile_name else {
        return Ok(());
    };
    let spec = profiles
        .get(profile_name)
        .ok_or_else(|| unknown_profile_error(profile_name, profiles))?;
    if let Some(model) = &spec.model {
        turn_config.model = model.clone();
    }
    turn_config.system_prompt = Some(spec.system_prompt_text.clone());
    Ok(())
}

fn unknown_profile_error(
    profile_name: &str,
    profiles: &BTreeMap<String, ProfileSpec>,
) -> anyhow::Error {
    anyhow::anyhow!(
        "unknown --profile `{profile_name}`; available: {}",
        profiles.keys().cloned().collect::<Vec<_>>().join(", ")
    )
}

fn register_skill_builtins(
    builtin_registry: &mut BuiltinRegistry,
    skills: &Arc<BTreeMap<String, SkillEntry>>,
) {
    let skills_for_hook = skills.clone();
    builtin_registry.register_step(SKILL_MANIFEST_HOOK_NAME, move || {
        Arc::new(defect_agent::hooks::builtin::SkillManifestHook::new(
            skills_for_hook.clone(),
        ))
    });
    let skills_for_trig = skills.clone();
    builtin_registry.register_step(SKILL_TRIGGERS_HOOK_NAME, move || {
        Arc::new(defect_agent::hooks::builtin::SkillTriggersHook::new(
            skills_for_trig.clone(),
        ))
    });
}

fn overlay_process_tools(
    base: Arc<dyn ToolRegistry>,
    tools: &[Arc<dyn Tool>],
    registries: &[Arc<dyn ToolRegistry>],
) -> Arc<dyn ToolRegistry> {
    let mut current = base;
    if !tools.is_empty() {
        let mut builder = StaticToolRegistry::builder();
        for tool in tools {
            builder = builder.insert(tool.clone());
        }
        let overlay = Arc::new(builder.build()) as Arc<dyn ToolRegistry>;
        current = Arc::new(defect_agent::session::CompositeRegistry::new(
            overlay, current,
        ));
    }
    for registry in registries {
        current = Arc::new(defect_agent::session::CompositeRegistry::new(
            registry.clone(),
            current,
        ));
    }
    current
}

fn resolve_base_prompt_text(config: &LoadedConfig) -> anyhow::Result<Option<String>> {
    let base_prompt = &config.effective.base_prompt;
    let mut sections = Vec::new();
    if let Some(file) = base_prompt.file.as_deref() {
        let text = std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("base_prompt file {} read failed: {e}", file.display()))?;
        sections.push(text);
    }
    if let Some(text) = base_prompt.text.as_deref() {
        sections.push(text.to_owned());
    }
    Ok((!sections.is_empty()).then(|| sections.join("\n\n")))
}
