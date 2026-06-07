use super::*;
use agent_client_protocol_schema::StopReason as AcpStopReason;

fn ctx<'a>(session_id: &'a SessionId, cwd: &'a Path) -> HookCtx<'a> {
    HookCtx::new(session_id, cwd, CancellationToken::new())
}

#[test]
fn glob_basic() {
    // Tool name matching semantics after migrating to globset (`.` is not a path
    // separator; `*`/`?` behave normally).
    assert!(tool_name_matches("*.rs", "main.rs"));
    assert!(tool_name_matches("*", ""));
    assert!(tool_name_matches("a*c", "abc"));
    assert!(tool_name_matches("a*c", "ac"));
    assert!(!tool_name_matches("a*c", "abd"));
    assert!(tool_name_matches("???", "abc"));
    assert!(!tool_name_matches("???", "abcd"));
    assert!(tool_name_matches("mcp.*", "mcp.fs.read"));
    // Invalid patterns do not panic; they are treated as non-matching.
    assert!(!tool_name_matches("[bad", "anything"));
}

// ----- step model dispatch (migrate slice 1) -----

/// A step handler that returns a fixed verdict.
struct StubStepHandler {
    verdict: Value,
}

impl StepHandler for StubStepHandler {
    fn handle_step<'a>(
        &'a self,
        _envelope: &'a Value,
        _ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        let v = self.verdict.clone();
        Box::pin(async move { Ok(Some(v)) })
    }
}

#[tokio::test]
async fn dispatch_routes_to_step_handler_by_event_name() {
    let engine = DefaultHookEngine::new();
    let mut table = HandlerTable::empty();
    table.push_step(
        "before_turn_end",
        StepHandlerEntry::new(
            HookMatcher::default(),
            Arc::new(StubStepHandler {
                verdict: serde_json::json!({
                    "control": "continue",
                    "additional_context": ["keep going"],
                }),
            }),
        ),
    );
    engine.reload(table);

    let session_id = SessionId::new("s1");
    let cwd = Path::new("/");
    let mut step = step::BeforeTurnEnd {
        stop_reason: AcpStopReason::EndTurn,
        continues_so_far: 0,
        voluntary: true,
        feedback: Vec::new(),
    };
    let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
    assert_eq!(control, step::HookControl::Continue);
    // The verdict injection landed on the step.
    assert_eq!(step.feedback.len(), 1);
}

#[tokio::test]
async fn dispatch_no_handler_returns_proceed() {
    let engine = DefaultHookEngine::new();
    let session_id = SessionId::new("s1");
    let cwd = Path::new("/");
    let mut step = step::BeforeToolApply {
        tool_name: "bash".to_string(),
        safety: crate::tool::SafetyClass::ReadOnly,
        args: serde_json::json!({}),
        result: None,
    };
    let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
    assert_eq!(control, step::HookControl::Proceed);
}

#[tokio::test]
async fn dispatch_matcher_filters_by_tool() {
    let engine = DefaultHookEngine::new();
    let mut table = HandlerTable::empty();
    // Only matches handlers where tool=="edit"; the step's tool is "bash" → no match.
    table.push_step(
        "before_tool_apply",
        StepHandlerEntry::new(
            HookMatcher {
                tool: Some("edit".to_string()),
                ..Default::default()
            },
            Arc::new(StubStepHandler {
                verdict: serde_json::json!({"control": "break"}),
            }),
        ),
    );
    engine.reload(table);

    let session_id = SessionId::new("s1");
    let cwd = Path::new("/");
    let mut step = step::BeforeToolApply {
        tool_name: "bash".to_string(),
        safety: crate::tool::SafetyClass::ReadOnly,
        args: serde_json::json!({}),
        result: None,
    };
    let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
    // No match → Proceed.
    assert_eq!(control, step::HookControl::Proceed);
}

#[tokio::test]
async fn dispatch_matcher_filters_by_safety() {
    let engine = DefaultHookEngine::new();
    let mut table = HandlerTable::empty();
    // Only match handlers with `Destructive` safety; the step's safety is `ReadOnly`,
    // so it does not match.
    table.push_step(
        "before_tool_apply",
        StepHandlerEntry::new(
            HookMatcher {
                safety: vec![SafetyClass::Destructive],
                ..Default::default()
            },
            Arc::new(StubStepHandler {
                verdict: serde_json::json!({"control": "break"}),
            }),
        ),
    );
    engine.reload(table);

    let session_id = SessionId::new("s1");
    let cwd = Path::new("/");
    let mut step = step::BeforeToolApply {
        tool_name: "bash".to_string(),
        safety: SafetyClass::ReadOnly,
        args: serde_json::json!({}),
        result: None,
    };
    let control = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
    assert_eq!(control, step::HookControl::Proceed);

    // Safety hit (Destructive) → handler runs, returns break.
    let mut step2 = step::BeforeToolApply {
        tool_name: "bash".to_string(),
        safety: SafetyClass::Destructive,
        args: serde_json::json!({}),
        result: None,
    };
    let control2 = engine.dispatch(&mut step2, ctx(&session_id, cwd)).await;
    assert!(matches!(control2, step::HookControl::Break { .. }));
}

#[tokio::test]
async fn dispatch_merges_common_header() {
    let engine = DefaultHookEngine::new();
    // Use an echo handler to verify that the common header is merged.
    struct EchoHandler {
        seen: std::sync::Arc<std::sync::Mutex<Option<Value>>>,
    }
    impl StepHandler for EchoHandler {
        fn handle_step<'a>(
            &'a self,
            envelope: &'a Value,
            _ctx: HookCtx<'a>,
        ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
            *self.seen.lock().unwrap() = Some(envelope.clone());
            Box::pin(async { Ok(None) })
        }
    }
    let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut table = HandlerTable::empty();
    table.push_step(
        "after_session_enter",
        StepHandlerEntry::new(
            HookMatcher::default(),
            Arc::new(EchoHandler { seen: seen.clone() }),
        ),
    );
    engine.reload(table);

    let session_id = SessionId::new("sess-9");
    let cwd = Path::new("/repo");
    let mut step = step::AfterSessionEnter {
        cwd: "/repo".to_string(),
        source: step::SessionSource::New,
        additional_context: Vec::new(),
    };
    let _ = engine.dispatch(&mut step, ctx(&session_id, cwd)).await;
    let env = seen.lock().unwrap().clone().expect("handler saw envelope");
    assert_eq!(env["session_id"], "sess-9");
    assert_eq!(env["cwd"], "/repo");
    assert_eq!(env["hook_event"], "after_session_enter");
}
