//! 把 `defect-config` 的 hook 配置翻译成 agent 的 [`DefaultHookEngine`]。
//!
//! Hook assembly — the agent crate does not depend on the config crate,
//! 翻译动作放 CLI 装配期；这里也是 fail-fast 报"未知 builtin 名"的位置。
//!
//! 三种 handler 形态在 v0 全部接通：
//! - `Builtin { name }` → 按名查 [`BuiltinRegistry`]，未知 name 走
//!   [`HookEngineBuildError::UnknownBuiltin`] fail-fast
//! - `Command(_)` → [`CommandHandler::new`]（argv 直 spawn / 显式 shell 二选一）
//! - `Prompt(_)` → [`PromptHandler::new`]，CLI 装配期把当前 default
//!   provider/model 喂进去（`HookPromptSpec.model = None` 时回退到 session
//!   默认 model）

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

/// 装配错误。
///
/// `Configuration` 兜底仅出现在配置层未捕获的非法组合（理论上 config
/// loader 已经 fail-fast 过一轮）。
#[derive(Debug, thiserror::Error)]
pub enum HookEngineBuildError {
    #[error("unknown builtin hook handler `{name}` (available: {available})")]
    UnknownBuiltin { name: String, available: String },

    #[error("hook configuration invalid: {0}")]
    Configuration(String),
}

/// 装配 hook engine 时还需要的运行期上下文。
///
/// `Prompt` handler 要 LLM provider；`registry` 提供"按 model id 选 provider"
/// 与"如果 hook 没指定 model 用哪个 fallback"。
pub struct HookEngineCtx<'a> {
    pub registry: &'a Arc<ProviderRegistry>,
    pub default_model: &'a str,
}

/// 从 `[hooks]` 段构造 [`HandlerTable`]（不含自动挂载的 builtin）。
fn build_handler_table(
    hooks: &HooksConfig,
    builtins: &BuiltinRegistry,
    rt: &HookEngineCtx<'_>,
) -> Result<HandlerTable, HookEngineBuildError> {
    let mut table = HandlerTable::empty();

    // config 的事件桶键就是 step 的 `event_name`（1:1，config 层已校验过合法性）。
    // `event_name` 须为 `&'static str`（HandlerTable 的桶键）——从 step::ALL_EVENT_NAMES 取静态串。
    for (event_name, entries) in &hooks.buckets {
        let Some(static_name) = static_event_name(event_name) else {
            // config 层已 fail-fast 掉未知事件名；这里兜底跳过。
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

/// 用 `[hooks]` 段 + builtin 注册表构造一个 [`DefaultHookEngine`]。
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

/// 把 config 的事件名（owned String）换成 step 模型用的 `&'static str`。
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
        // `HookHandlerSpec` 是 non_exhaustive 的——出现新形态时强制 CLI 显式
        // 加一条分支，避免默默 noop。
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
    // prompt hook 的 `model` 字段没有 provider 维度——按裸 id 取首个声明它的 entry。
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
        // non_exhaustive 兜底——遇到新形态保守翻成空 argv 让 agent 层报错。
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
        // non_exhaustive 兜底
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
                // non_exhaustive 兜底——按 Json 兜底。
                let _ = other;
                AgentPromptRender::Json
            }
        },
        timeout_sec: spec.timeout_sec,
    }
}

/// 在 [`Arc`] 里封装一份 hook engine——session/turn 主循环统一拿
/// `Arc<dyn HookEngine>`。`HooksConfig::is_empty` 时用
/// [`defect_agent::hooks::NoopHookEngine`] 走零开销路径。
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

/// 主 session 的 hook 引擎：在用户 `[hooks]` 配置之上**自动挂载**两个 skill
/// builtin（发现到任意 skill 时）——
/// - `skill-manifest` → `after_session_enter`：注入 L1 清单 + always-on body；
/// - `skill-triggers` → `before_ingest`：按 prompt 自动激活相关 skill。
///
/// 这让"自动激活"开箱即用、不要求用户手写 `[[hooks.*]]`。两个 hook 的 matcher
/// 全空（命中该事件下所有触发）。skill 索引为空时不挂（保持零开销），且当用户
/// 也没配 `[hooks]` 时直接走 [`NoopHookEngine`](defect_agent::hooks::NoopHookEngine)。子 agent profile 不走这条路径
/// （仍用 [`build_engine_arc`]），故 skill hook 不会渗进子 agent。
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
    // `--goal` 模式：挂 goal-gate 到两个事件——after_session_enter 注入目标说明 +
    // goal_done 契约（turn 1 起生效，模型一开机就知道完成后要调 goal_done），
    // before_turn_end 驱动"达成才退出"的循环。两个挂载点共用同一份 GoalState。
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
mod tests {
    use super::*;
    use defect_agent::llm::{
        Capabilities, FeatureSupport, LlmProvider, ModelInfo, ProtocolId, ProviderEntry,
        ProviderInfo, ProviderRegistry, ProviderStream, ThinkingEcho,
    };
    use defect_agent::session::SessionCapabilitiesConfig;
    use defect_agent::tool::SafetyClass;
    use defect_config::{ConfigSource, HookEntry};
    use futures::future::BoxFuture;
    use std::collections::BTreeMap;
    use tokio_util::sync::CancellationToken;

    struct StubProvider;
    impl LlmProvider for StubProvider {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                vendor: "stub".into(),
                protocol: ProtocolId::OpenAiChat,
                display_name: "stub".into(),
            }
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                tool_calls: FeatureSupport::Unsupported,
                parallel_tool_calls: FeatureSupport::Unsupported,
                thinking: FeatureSupport::Unsupported,
                vision: FeatureSupport::Unsupported,
                prompt_cache: FeatureSupport::Unsupported,
                thinking_echo: ThinkingEcho::Forbidden,
            }
        }
        fn list_models(
            &self,
        ) -> BoxFuture<'_, Result<Vec<ModelInfo>, defect_agent::llm::ProviderError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn model_info(&self, _id: &str) -> Option<ModelInfo> {
            None
        }
        fn complete(
            &self,
            _req: defect_agent::llm::CompletionRequest,
            _cancel: CancellationToken,
        ) -> BoxFuture<'_, Result<ProviderStream, defect_agent::llm::ProviderError>> {
            unreachable!()
        }
    }

    fn stub_registry() -> Arc<ProviderRegistry> {
        let model = ModelInfo {
            id: "stub-1".into(),
            display_name: None,
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: Default::default(),
        };
        Arc::new(
            ProviderRegistry::new(
                vec![ProviderEntry::new(
                    Arc::new(StubProvider),
                    vec![model],
                    SessionCapabilitiesConfig::default(),
                )],
                "stub",
                "stub-1",
            )
            .expect("registry"),
        )
    }

    fn ctx<'a>(reg: &'a Arc<ProviderRegistry>) -> HookEngineCtx<'a> {
        HookEngineCtx {
            registry: reg,
            default_model: "stub-1",
        }
    }

    #[test]
    fn empty_config_yields_noop_engine() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let arc = build_engine_arc(&HooksConfig::default(), &builtins, &ctx(&reg)).expect("ok");
        let session_id = agent_client_protocol_schema::SessionId::new("s");
        let cwd = std::path::Path::new("/");
        let hctx = defect_agent::hooks::HookCtx::new(&session_id, cwd, CancellationToken::new());
        // 空配置 → NoopHookEngine：dispatch 返回 Proceed，step 不被改动。
        let mut step = defect_agent::hooks::step::AfterSessionEnter {
            cwd: "/".to_string(),
            source: defect_agent::hooks::step::SessionSource::New,
            additional_context: Vec::new(),
        };
        let control = futures::executor::block_on(arc.dispatch(&mut step, hctx));
        assert_eq!(control, defect_agent::hooks::step::HookControl::Proceed);
        assert!(step.additional_context.is_empty());
    }

    #[test]
    fn unknown_builtin_fails_fast() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "after_session_enter",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher::default(),
                handler: HookHandlerSpec::Builtin {
                    name: "does-not-exist".into(),
                },
                source: ConfigSource::User,
            },
        );
        let err = match build_engine_arc(&hooks, &builtins, &ctx(&reg)) {
            Ok(_) => panic!("should fail"),
            Err(e) => e,
        };
        assert!(matches!(
            err,
            HookEngineBuildError::UnknownBuiltin { ref name, .. } if name == "does-not-exist"
        ));
    }

    #[test]
    fn known_builtin_loads() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "before_tool_apply",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher {
                    tool: Some("login".into()),
                    ..Default::default()
                },
                handler: HookHandlerSpec::Builtin {
                    name: "redact-secrets".into(),
                },
                source: ConfigSource::Project,
            },
        );
        let _arc = build_engine_arc(&hooks, &builtins, &ctx(&reg)).expect("ok");
    }

    #[test]
    fn command_handler_argv_loads() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "before_tool_apply",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher::default(),
                handler: HookHandlerSpec::Command(HookCommandSpec::Argv {
                    argv: vec!["true".into()],
                    argv_windows: None,
                    cwd: None,
                    env: BTreeMap::new(),
                    timeout_sec: Some(5),
                }),
                source: ConfigSource::User,
            },
        );
        let _arc = build_engine_arc(&hooks, &builtins, &ctx(&reg)).expect("ok");
    }

    #[test]
    fn command_handler_shell_loads() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "before_tool_apply",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher::default(),
                handler: HookHandlerSpec::Command(HookCommandSpec::Shell {
                    shell: HookShellKind::Bash,
                    command: "echo hi".into(),
                    cwd: None,
                    env: BTreeMap::new(),
                    timeout_sec: Some(5),
                }),
                source: ConfigSource::User,
            },
        );
        let _arc = build_engine_arc(&hooks, &builtins, &ctx(&reg)).expect("ok");
    }

    #[test]
    fn prompt_handler_loads_with_default_model() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "after_session_enter",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher::default(),
                handler: HookHandlerSpec::Prompt(HookPromptSpec::new(
                    None,
                    "summarize".into(),
                    HookPromptRender::Json,
                    Some(5),
                )),
                source: ConfigSource::User,
            },
        );
        let _arc = build_engine_arc(&hooks, &builtins, &ctx(&reg)).expect("ok");
    }

    #[test]
    fn prompt_handler_unknown_model_fails() {
        let builtins = BuiltinRegistry::defaults();
        let reg = stub_registry();
        let mut hooks = HooksConfig::default();
        hooks.push(
            "after_session_enter",
            HookEntry {
                name: None,
                matcher: ConfigHookMatcher::default(),
                handler: HookHandlerSpec::Prompt(HookPromptSpec::new(
                    Some("not-registered".into()),
                    "x".into(),
                    HookPromptRender::Json,
                    None,
                )),
                source: ConfigSource::User,
            },
        );
        let err = match build_engine_arc(&hooks, &builtins, &ctx(&reg)) {
            Ok(_) => panic!("should fail"),
            Err(e) => e,
        };
        assert!(matches!(err, HookEngineBuildError::Configuration(_)));
    }

    #[test]
    fn matcher_translation_preserves_fields() {
        let cm = ConfigHookMatcher {
            tool: Some("bash".into()),
            tool_glob: Some("mcp.*".into()),
            safety: vec![SafetyClass::Destructive, SafetyClass::Network],
        };
        let am = translate_matcher(&cm);
        assert_eq!(am.tool.as_deref(), Some("bash"));
        assert_eq!(am.tool_glob.as_deref(), Some("mcp.*"));
        assert_eq!(am.safety.len(), 2);
    }
}
