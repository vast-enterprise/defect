//! CLI default AgentCore assembler.
//!
//! This layer consolidates the provider / tool / hook / storage / MCP / observability
//! wiring that was previously scattered across `src/bin/cli.rs` into an extensible
//! builder. The underlying
//! [`DefaultAgentCoreBuilder`](defect_agent::session::DefaultAgentCoreBuilder) remains a
//! minimal agent abstraction; this module expresses the defect CLI's "default feature
//! set".

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

/// Clippable features in the default CLI assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultFeature {
    /// Default process-level tools: bash / fs / fetch / search.
    ProcessTools,
    /// Profile-driven `spawn_agent` and background task control tools.
    Subagents,
    /// Skill tools and automatic skill hooks.
    Skills,
    /// Hooks from the user configuration.
    Hooks,
    /// Session persistence and `--resume`.
    Storage,
    /// Default MCP server factory.
    Mcp,
    /// Bypass observers such as langfuse.
    Observability,
    /// HTTP fetch backend.
    Http,
    /// ACP-exposed permission mode directory.
    Modes,
}

/// Default feature set for the CLI.
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
    /// An empty feature set. Useful when the host only wants to reuse configuration
    /// parsing and then explicitly enable features one by one.
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

    /// Disables a default feature.
    pub fn without(mut self, feature: DefaultFeature) -> Self {
        self.set(feature, false);
        self
    }

    /// Enables a default feature.
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

/// REPL assembly semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplMode {
    /// Uses the ACP server path.
    Disabled,
    /// Uses the built-in REPL; the default sandbox switches to `open` to avoid hanging
    /// due to missing UI permission confirmation.
    Enabled,
}

/// AgentCore and associated metadata required to run the default CLI.
pub struct BuiltCliAgent {
    pub agent: Arc<dyn AgentCore>,
    pub resume_session_id: Option<SessionId>,
    pub sandbox_mode: SandboxMode,
    pub turn_config: TurnConfig,
    /// Shared state handle for `--goal` mode (`None` otherwise). The oneshot runner reads
    /// [`GoalState::is_reached`](defect_agent::session::GoalState::is_reached) after each
    /// turn — if retries are exhausted without reaching the goal, it exits with a
    /// non-zero code so CI does not mistake "ran all turns without success" for a pass.
    ///
    /// [`GoalState`]: defect_agent::session::GoalState
    pub goal: Option<Arc<defect_agent::session::GoalState>>,
}

/// Default AgentCore assembly builder for the defect CLI.
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
    /// `--max-turns`: maximum number of before_turn_end continuations in goal mode,
    /// mapped to
    /// `TurnConfig::max_hook_continues`. `None` = use config/default.
    max_turns: Option<u32>,
    modes_override: Option<ModeCatalog>,
    hook_engine_override: Option<Arc<dyn HookEngine>>,
    builtin_registry: BuiltinRegistry,
    session_tool_factory_override: Option<Arc<dyn SessionToolFactory>>,
    observers: Vec<Arc<dyn SessionObserver>>,
}

impl CliAgentBuilder {
    /// Creates a CLI agent builder with the full default feature set.
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

    /// Replace the default feature set.
    pub fn features(mut self, features: DefaultFeatureSet) -> Self {
        self.features = features;
        self
    }

    /// Enable REPL assembly semantics.
    pub fn repl(mut self, repl: ReplMode) -> Self {
        self.repl = repl;
        self
    }

    /// Use the project-local session directory.
    pub fn local_sessions(mut self) -> Self {
        self.local_sessions = true;
        self
    }

    /// Set the top-level profile.
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Set the resume parameter. `None` means bare `--resume`, which looks up the most
    /// recent session by cwd.
    pub fn resume(mut self, session_id: Option<String>) -> Self {
        self.resume = Some(session_id);
        self
    }

    /// Overrides the provider registry directly.
    pub fn provider_registry(mut self, registry: Arc<ProviderRegistry>) -> Self {
        self.registry_override = Some(registry);
        self
    }

    /// Appends a provider entry after the default provider entries.
    pub fn add_provider_entry(mut self, entry: ProviderEntry) -> Self {
        self.extra_provider_entries.push(entry);
        self
    }

    /// Overrides the process-level tool registry directly.
    pub fn process_tools(mut self, tools: Arc<dyn ToolRegistry>) -> Self {
        self.process_tools_override = Some(tools);
        self
    }

    /// Adds a tool on top of the default process-level tools. When names conflict, the
    /// added tool takes precedence.
    pub fn add_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.extra_process_tools.push(tool);
        self
    }

    /// Overlay a registry onto the default process-level tools. Registries added later
    /// take higher priority.
    pub fn add_tool_registry(mut self, registry: Arc<dyn ToolRegistry>) -> Self {
        self.extra_process_registries.push(registry);
        self
    }

    /// Overrides the sandbox policy.
    pub fn policy(mut self, policy: Arc<dyn SandboxPolicy>) -> Self {
        self.policy_override = Some(policy);
        self
    }

    /// Wraps the final policy with [`NonInteractivePolicy`]: downgrades inner `Ask` to
    /// `Deny` to prevent hanging on permission prompts in non‑TTY environments
    /// (`--message` single‑turn mode). `Allow` / `Deny` pass through unchanged.
    pub fn non_interactive(mut self) -> Self {
        self.non_interactive = true;
        self
    }

    /// Enable the `--goal` goal-driven loop: registers the `goal_done` tool, installs a
    /// `goal-gate` hook (`before_turn_end`), and attaches [`GoalState`] to the session.
    /// The agent runs autonomously for multiple turns until `goal_done` is called
    /// (success) or the `max_hook_continues` limit (`--max-turns`) is reached.
    ///
    /// [`GoalState`]: defect_agent::session::GoalState
    pub fn goal(mut self, objective: impl Into<String>) -> Self {
        self.goal = Some(Arc::new(defect_agent::session::GoalState::new(
            objective.into(),
        )));
        self
    }

    /// `--max-turns`: the maximum number of times `before_turn_end` can extend the
    /// session in goal mode (mapped to
    /// `TurnConfig::max_hook_continues`). When the limit is reached, the session is
    /// forcibly stopped and exits with `Exhausted`.
    pub fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Override the permission mode catalog.
    pub fn modes(mut self, modes: ModeCatalog) -> Self {
        self.modes_override = Some(modes);
        self
    }

    /// Override the hook engine.
    pub fn hook_engine(mut self, hook_engine: Arc<dyn HookEngine>) -> Self {
        self.hook_engine_override = Some(hook_engine);
        self
    }

    /// Registers a builtin hook factory.
    pub fn builtin_registry(mut self, registry: BuiltinRegistry) -> Self {
        self.builtin_registry = registry;
        self
    }

    /// Override the session-level tool factory, e.g. for custom MCP integration.
    pub fn session_tool_factory(mut self, factory: Arc<dyn SessionToolFactory>) -> Self {
        self.session_tool_factory_override = Some(factory);
        self
    }

    /// Adds a session observer. The observer can subscribe to the event stream after
    /// session creation and push events to an external system.
    pub fn observe_session(mut self, observer: Arc<dyn SessionObserver>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Builds an [`AgentCore`] along with CLI companion information.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration derivation fails, provider/hook/subagent tool
    /// assembly fails, the persistence directory cannot be resolved, or an explicit
    /// `resume` is requested but the target session is not found.
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
        // `--max-turns`: the maximum number of continues in goal mode. Maps to the hard
        // cap on continues in `before_turn_end`.
        if let Some(max_turns) = self.max_turns {
            turn_config.max_hook_continues = max_turns;
        }

        let sandbox_mode = self.resolve_sandbox_mode();
        let mut policy = self
            .policy_override
            .clone()
            .unwrap_or_else(|| build_policy(sandbox_mode));
        // Non-interactive mode (`--message`): wrap with `NonInteractivePolicy` and **do
        // not** attach a `ModeCatalog`.
        // Key: when `DefaultSession` has a catalog attached, the active policy comes from
        // the catalog's current mode
        // (`session_policy_state`), bypassing the `policy` set here — the wrapper becomes
        // ineffective, `Ask` is not downgraded,
        // and the process hangs forever without a TTY. `oneshot` has no `set_mode`
        // client, so the catalog is meaningless anyway;
        // set it to `None` so the session falls back to this wrapped policy.
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
        // `--goal` mode: overlay the `goal_done` tool so the model can declare a goal
        // achieved.
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
        let registry = ProviderRegistry::new(entries, &turn_config.provider, &turn_config.model)
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
        // The `--goal` flag also needs the goal-gate hook attached, even if the user has
        // configured neither `[hooks]` nor any skill.
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
