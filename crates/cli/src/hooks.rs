//! Translates `defect-config` hook configuration into the agent's [`DefaultHookEngine`].
//!
//! Hook assembly — the agent crate does not depend on the config crate, so translation
//! happens during CLI assembly; this is also where we fail-fast with "unknown builtin
//! name".
//!
//! All three handler variants are wired up:
//! - `Builtin { name }` → looks up [`BuiltinRegistry`] by name; unknown name triggers
//!   [`HookEngineBuildError::UnknownBuiltin`] fail-fast
//! - `Command(_)` → [`CommandHandler::new`] (either direct argv spawn or explicit shell)
//! - `Prompt(_)` → [`PromptHandler::new`]; during CLI assembly the current default
//!   provider/model is injected (when `HookPromptSpec.model = None`, falls back to the
//!   session default model)

use std::sync::Arc;
use std::time::Duration;

use defect_agent::hooks::builtin::BuiltinRegistry;
use defect_agent::hooks::command::{CommandHandler, CommandSpec, ShellKind as AgentShellKind};
use defect_agent::hooks::prompt::{PromptHandler, PromptRender as AgentPromptRender, PromptSpec};
use defect_agent::hooks::{
    DefaultHookEngine, HandlerTable, HookMatcher as AgentHookMatcher, StepHandler, StepHandlerEntry,
};
use defect_agent::llm::{LlmProvider, ProviderRegistry};
use defect_config::{
    HookCommandSpec, HookHandlerSpec, HookMatcher as ConfigHookMatcher, HookPromptRender,
    HookPromptSpec, HookShellKind, HooksConfig,
};

/// Build errors.
///
/// `Configuration` is a fallback for invalid combinations not caught by the configuration
/// layer (in theory the config loader has already fail-fast once).
#[derive(Debug, thiserror::Error)]
pub enum HookEngineBuildError {
    #[error("unknown builtin hook handler `{name}` (available: {available})")]
    UnknownBuiltin { name: String, available: String },

    #[error("hook configuration invalid: {0}")]
    Configuration(String),
}

/// Runtime context needed when assembling the hook engine.
///
/// `Prompt` handlers need an LLM provider; `registry` provides model-id-based provider
/// selection and a fallback model when the hook does not specify one.
pub struct HookEngineCtx<'a> {
    pub registry: &'a Arc<ProviderRegistry>,
    pub default_model: &'a str,
}

/// Constructs a [`HandlerTable`] from the `[hooks]` section (excluding auto-mounted
/// builtins).
fn build_handler_table(
    hooks: &HooksConfig,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
) -> Result<HandlerTable, HookEngineBuildError> {
    let mut table = HandlerTable::empty();

    // The event bucket keys in config are the step's `event_name` (1:1, already validated
    // by the config layer).
    // `event_name` must be `&'static str` (the bucket key of `HandlerTable`) — taken from
    // `step::ALL_EVENT_NAMES` as a static string.
    for (event_name, entries) in &hooks.buckets {
        let Some(static_name) = static_event_name(event_name) else {
            // The config layer already fail-fasts on unknown event names; skip here as a
            // safety net.
            continue;
        };
        for entry in entries {
            let matcher = translate_matcher(&entry.matcher);
            let (handler, timeout) = build_handler(&entry.handler, builtins, rt)?;
            let mut hook = StepHandlerEntry::new(matcher, handler).with_name(entry.name.clone());
            if let Some(t) = timeout {
                hook = hook.with_timeout(t);
            }
            table.push_step(static_name, hook);
        }
    }
    Ok(table)
}

/// Build a [`DefaultHookEngine`] from the `[hooks]` section and the builtin registry.
pub fn build_hook_engine(
    hooks: &HooksConfig,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
) -> Result<DefaultHookEngine, HookEngineBuildError> {
    let table = build_handler_table(hooks, builtins, rt)?;
    let engine = DefaultHookEngine::new();
    engine.reload(table);
    Ok(engine)
}

/// Converts the config's event name (owned `String`) to the `&'static str` used by the
/// step model.
fn static_event_name(name: &str) -> Option<&'static str> {
    defect_agent::hooks::step::ALL_EVENT_NAMES
        .iter()
        .copied()
        .find(|&n| n == name)
}

fn build_handler(
    spec: &HookHandlerSpec,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
) -> Result<(Arc<dyn StepHandler>, Option<Duration>), HookEngineBuildError> {
    match spec {
        HookHandlerSpec::Builtin { name } => {
            let handler = builtins.lookup_step(name).ok_or_else(|| {
                let available = builtins.names().collect::<Vec<_>>().join(", ");
                HookEngineBuildError::UnknownBuiltin {
                    name: name.clone(),
                    available,
                }
            })?;
            Ok((handler, None))
        }
        HookHandlerSpec::Command(cmd) => {
            let agent_spec = translate_command(cmd);
            let handler = CommandHandler::new(agent_spec);
            let timeout = handler.timeout();
            Ok((Arc::new(handler) as Arc<dyn StepHandler>, timeout))
        }
        HookHandlerSpec::Prompt(prompt) => {
            let provider = resolve_prompt_provider(prompt, rt)?;
            let agent_spec = translate_prompt(prompt, provider, rt.default_model.to_string());
            let handler = PromptHandler::new(agent_spec);
            let timeout = handler.timeout();
            Ok((Arc::new(handler) as Arc<dyn StepHandler>, timeout))
        }
        // `HookHandlerSpec` is non_exhaustive — new variants force the CLI to add an
        // explicit branch, preventing silent no-ops.
        other => Err(HookEngineBuildError::Configuration(format!(
            "unrecognized hook handler form: {other:?}"
        ))),
    }
}

fn resolve_prompt_provider(
    spec: &HookPromptSpec,
    rt: &HookEngineCtx<'_>,
) -> Result<Arc<dyn LlmProvider>, HookEngineBuildError> {
    let model_id = spec.model.as_deref().unwrap_or(rt.default_model);
    // The `model` field of a prompt hook has no provider dimension — take the first entry
    // that declares it by bare id.
    let entry = rt.registry.first_entry_for_model(model_id).ok_or_else(|| {
        HookEngineBuildError::Configuration(format!(
            "prompt hook references unknown model `{model_id}` (no provider registered for it)"
        ))
    })?;
    Ok(Arc::clone(entry.provider()))
}

fn translate_matcher(m: &ConfigHookMatcher) -> AgentHookMatcher {
    let mut out = AgentHookMatcher::default();
    out.tool = m.tool.clone();
    out.tool_glob = m.tool_glob.clone();
    out.safety = m.safety.clone();
    out
}

fn translate_command(spec: &HookCommandSpec) -> CommandSpec {
    match spec {
        HookCommandSpec::Argv {
            argv,
            argv_windows,
            cwd,
            env,
            timeout_sec,
        } => CommandSpec::Argv {
            argv: argv.clone(),
            argv_windows: argv_windows.clone(),
            cwd: cwd.clone(),
            env: env.clone(),
            timeout_sec: *timeout_sec,
        },
        HookCommandSpec::Shell {
            shell,
            command,
            cwd,
            env,
            timeout_sec,
        } => CommandSpec::Shell {
            shell: translate_shell(shell),
            command: command.clone(),
            cwd: cwd.clone(),
            env: env.clone(),
            timeout_sec: *timeout_sec,
        },
        // Fallback for `non_exhaustive` – conservatively produce an empty argv on unknown
        // variants, letting the agent layer report the error.
        other => {
            let _ = other;
            CommandSpec::Argv {
                argv: Vec::new(),
                argv_windows: None,
                cwd: None,
                env: Default::default(),
                timeout_sec: None,
            }
        }
    }
}

fn translate_shell(shell: &HookShellKind) -> AgentShellKind {
    match shell {
        HookShellKind::Sh => AgentShellKind::Sh,
        HookShellKind::Bash => AgentShellKind::Bash,
        HookShellKind::Pwsh => AgentShellKind::Pwsh,
        HookShellKind::Cmd => AgentShellKind::Cmd,
        HookShellKind::Custom { program, args } => AgentShellKind::Custom {
            program: program.clone(),
            args: args.clone(),
        },
        // Fallback for non_exhaustive variant
        other => {
            let _ = other;
            AgentShellKind::Sh
        }
    }
}

fn translate_prompt(
    spec: &HookPromptSpec,
    provider: Arc<dyn LlmProvider>,
    fallback_model: String,
) -> PromptSpec {
    PromptSpec {
        provider,
        model: spec.model.clone(),
        fallback_model,
        system: spec.system.clone(),
        render: match &spec.render {
            HookPromptRender::Json => AgentPromptRender::Json,
            HookPromptRender::Template { template } => AgentPromptRender::Template {
                template: template.clone(),
            },
            other => {
                // Fallback for non_exhaustive — default to Json.
                let _ = other;
                AgentPromptRender::Json
            }
        },
        timeout_sec: spec.timeout_sec,
    }
}

/// Wraps a hook engine in an [`Arc`] so that the session/turn main loop can uniformly
/// hold an `Arc<dyn HookEngine>`. When `HooksConfig::is_empty`, uses
/// [`defect_agent::hooks::NoopHookEngine`] for a zero-overhead path.
pub fn build_engine_arc(
    hooks: &HooksConfig,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
) -> Result<Arc<dyn defect_agent::hooks::HookEngine>, HookEngineBuildError> {
    if hooks.is_empty() {
        return Ok(Arc::new(defect_agent::hooks::NoopHookEngine));
    }
    let engine = build_hook_engine(hooks, builtins, rt)?;
    Ok(Arc::new(engine))
}

/// Hook engine for the main session: automatically mounts two skill builtins on top of
/// the user's `[hooks]` configuration (when any skill is discovered) —
/// - `skill-manifest` → `after_session_enter`: injects the L1 manifest + always-on body;
/// - `skill-triggers` → `before_ingest`: auto-activates relevant skills based on the
///   prompt.
///
/// This makes "auto-activation" work out of the box without requiring users to write
/// `[[hooks.*]]` manually. Both hooks have empty matchers (they match all triggers under
/// that event). When the skill index is empty, nothing is mounted (keeping zero
/// overhead), and when the user also has no `[hooks]` configured, it falls through to
/// [`NoopHookEngine`](defect_agent::hooks::NoopHookEngine). Sub-agent profiles do not
/// take this path (they still use [`build_engine_arc`]), so skill hooks do not leak into
/// sub-agents.
pub fn build_main_session_engine(
    hooks: &HooksConfig,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
    skills: &Arc<std::collections::BTreeMap<String, defect_agent::tool::SkillEntry>>,
    goal: Option<&Arc<defect_agent::session::GoalState>>,
) -> Result<Arc<dyn defect_agent::hooks::HookEngine>, HookEngineBuildError> {
    let mount_skills = !skills.is_empty();
    if hooks.is_empty() && !mount_skills && goal.is_none() {
        return Ok(Arc::new(defect_agent::hooks::NoopHookEngine));
    }

    let mut table = build_handler_table(hooks, builtins, rt)?;
    if mount_skills {
        use defect_agent::hooks::builtin::{SkillManifestHook, SkillTriggersHook};
        table.push_step(
            "after_session_enter",
            StepHandlerEntry::new(
                AgentHookMatcher::default(),
                Arc::new(SkillManifestHook::new(skills.clone())),
            )
            .with_name(Some("skill-manifest".to_string())),
        );
        table.push_step(
            "before_ingest",
            StepHandlerEntry::new(
                AgentHookMatcher::default(),
                Arc::new(SkillTriggersHook::new(skills.clone())),
            )
            .with_name(Some("skill-triggers".to_string())),
        );
    }
    // `--goal` mode: attach a `GoalGate` to two events — `after_session_enter` injects
    // the goal description and the `goal_done` contract (active from turn 1, so the model
    // knows from startup that it must call `goal_done` upon completion), and
    // `before_turn_end` drives the "exit only when achieved" loop. Both mount points
    // share the same `GoalState`.
    if let Some(goal) = goal {
        use defect_agent::hooks::builtin::GoalGate;
        table.push_step(
            "after_session_enter",
            StepHandlerEntry::new(
                AgentHookMatcher::default(),
                Arc::new(GoalGate::new(goal.clone())),
            )
            .with_name(Some("goal-gate".to_string())),
        );
        table.push_step(
            "before_turn_end",
            StepHandlerEntry::new(
                AgentHookMatcher::default(),
                Arc::new(GoalGate::new(goal.clone())),
            )
            .with_name(Some("goal-gate".to_string())),
        );
    }

    let engine = DefaultHookEngine::new();
    engine.reload(table);
    Ok(Arc::new(engine))
}

#[cfg(test)]
mod tests;
