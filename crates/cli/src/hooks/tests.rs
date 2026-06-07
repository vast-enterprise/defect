use super::*;
use defect_agent::llm::{
    Capabilities, FeatureSupport, LlmProvider, ModelInfo, ProtocolId, ProviderEntry, ProviderInfo,
    ProviderRegistry, ProviderStream, ThinkingEcho,
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
    // Empty config → NoopHookEngine: dispatch returns Proceed, step is unchanged.
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
