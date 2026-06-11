//! End-to-end turn test: runs a full turn with a mock provider and a mock tool.
//!
//! Verifies:
//! - The user prompt is appended to the history
//! - The LLM stream is consumed, and `TextDelta` / `ToolUse` are correctly translated
//!   into `AgentEvent`
//! - `tool_use` → tool dispatch → tool result is written back to the history
//! - On the second LLM `EndTurn`, `run_turn` returns `Ok(EndTurn)`

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_client_protocol_schema::SessionId;
use agent_client_protocol_schema::{
    ContentBlock, PermissionOptionId, StopReason, TextContent, ToolCallUpdateFields,
};
use defect_agent::event::{AgentEvent, PermissionResolution};
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderInfo, ProviderStream, StopReason as LlmStopReason,
    ThinkingEcho, Usage,
};
use defect_agent::policy::{AskWritesPolicy, OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, Frontend, LoadedSession, Session, SessionCreateInfo,
    SessionLoader, SessionObserver, StaticToolRegistry, ToolRegistry, TurnConfig, new_session_id,
};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn unsupported_caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

/// Returns different chunk sequences depending on the call count:
/// - 1st call: emit text + a tool_use, then Stop=ToolUse
/// - 2nd call: emit text, then Stop=EndTurn
struct ScriptedProvider {
    calls: Mutex<u32>,
}

impl ScriptedProvider {
    fn new() -> Self {
        Self {
            calls: Mutex::new(0),
        }
    }
}

impl LlmProvider for ScriptedProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "scripted".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Scripted Test Provider".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn model_info(&self, _model_id: &str) -> Option<ModelInfo> {
        None
    }

    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let mut calls = self.calls.lock().expect("calls poisoned");
        *calls += 1;
        let nth = *calls;
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = match nth {
                1 => vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "msg-1".to_string(),
                        model: "scripted-001".to_string(),
                    }),
                    Ok(ProviderChunk::Usage(Usage {
                        input_tokens: Some(11),
                        output_tokens: None,
                        cache_read_input_tokens: Some(7),
                        cache_creation_input_tokens: Some(19),
                    })),
                    Ok(ProviderChunk::TextDelta {
                        text: "calling tool".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseStart {
                        id: "tu-1".to_string(),
                        name: "echo".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseArgsDelta {
                        id: "tu-1".to_string(),
                        fragment: r#"{"msg":"hi"}"#.to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseEnd {
                        id: "tu-1".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::ToolUse,
                    }),
                ],
                _ => vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "msg-2".to_string(),
                        model: "scripted-001".to_string(),
                    }),
                    Ok(ProviderChunk::Usage(Usage {
                        input_tokens: Some(5),
                        output_tokens: Some(3),
                        cache_read_input_tokens: Some(2),
                        cache_creation_input_tokens: None,
                    })),
                    Ok(ProviderChunk::TextDelta {
                        text: "done".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ],
            };
            let s: Pin<
                Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
            > = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

/// Echo tool: returns `args.msg` as-is.
struct EchoTool {
    schema: ToolSchema,
}

struct StubLoader {
    loaded: LoadedSession,
}

struct CountingObserver {
    count: Arc<AtomicUsize>,
}

impl SessionLoader for StubLoader {
    fn load_session(
        &self,
        _id: SessionId,
    ) -> BoxFuture<'_, Result<LoadedSession, defect_agent::error::BoxError>> {
        let loaded = self.loaded.clone();
        Box::pin(async move { Ok(loaded) })
    }
}

impl SessionObserver for CountingObserver {
    fn on_session_created(
        &self,
        _session: Arc<dyn Session>,
        _info: SessionCreateInfo,
    ) -> Result<(), defect_agent::error::BoxError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl EchoTool {
    fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "echo".to_string(),
                description: "echo the msg field".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "msg": {"type": "string"}
                    },
                    "required": ["msg"]
                }),
            },
        }
    }
}

impl Tool for EchoTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }

    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async {
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some("echo".to_string());
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        let text = args
            .get("msg")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mut completed = ToolCallUpdateFields::default();
        completed.content = Some(vec![
            agent_client_protocol_schema::ToolCallContent::Content(
                agent_client_protocol_schema::Content::new(text),
            ),
        ]);
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(stream::iter(vec![ToolEvent::Completed(completed)]));
        s
    }
}

#[tokio::test]
async fn full_turn_with_one_tool_call() {
    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(EchoTool::new()))
            .build(),
    );

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");

    // Subscribe to the event stream before `run_turn` starts, otherwise events may arrive
    // before any subscriber is ready.
    let mut events = session.subscribe();

    let prompt = vec![ContentBlock::Text(TextContent::new("hello"))];
    let stop = session.run_turn(prompt).await.expect("turn");

    assert!(matches!(stop, StopReason::EndTurn));

    // Consume emitted events until TurnEnded
    let mut got_user_prompt_committed = false;
    let mut got_text = false;
    let mut got_tool_call_started = false;
    let mut got_tool_call_finished = false;
    let mut got_turn_ended = false;
    let mut turn_usage = None;
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::UserPromptCommitted { .. } => got_user_prompt_committed = true,
            AgentEvent::AssistantText { .. } => got_text = true,
            AgentEvent::ToolCallStarted { fields, .. } => {
                got_tool_call_started = true;
                // raw_input must carry the original arguments from the LLM (filled by the
                // outer main loop).
                assert_eq!(
                    fields.raw_input,
                    Some(serde_json::json!({ "msg": "hi" })),
                    "ToolCallStarted should carry the tool args as raw_input"
                );
            }
            AgentEvent::ToolCallFinished { .. } => got_tool_call_finished = true,
            AgentEvent::TurnEnded { usage, .. } => {
                got_turn_ended = true;
                turn_usage = Some(usage);
                break;
            }
            _ => {}
        }
    }

    assert!(got_user_prompt_committed, "should see UserPromptCommitted");
    assert!(got_text, "should see at least one AssistantText");
    assert!(got_tool_call_started, "should see ToolCallStarted");
    assert!(got_tool_call_finished, "should see ToolCallFinished");
    assert!(got_turn_ended, "should see TurnEnded");
    assert_eq!(
        turn_usage,
        Some(Usage {
            input_tokens: Some(16),
            output_tokens: Some(3),
            cache_read_input_tokens: Some(9),
            cache_creation_input_tokens: Some(19),
        })
    );
}

#[tokio::test]
async fn second_run_turn_while_first_in_flight_returns_in_progress() {
    use defect_agent::session::TurnError;

    // A provider whose stream hangs forever (never stops), keeping the turn running
    // indefinitely.
    struct HangingProvider;
    impl LlmProvider for HangingProvider {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                vendor: "hang".to_string(),
                protocol: ProtocolId::AnthropicMessages,
                display_name: "Hanging Test Provider".to_string(),
            }
        }
        fn capabilities(&self) -> Capabilities {
            unsupported_caps()
        }
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn model_info(&self, _: &str) -> Option<ModelInfo> {
            None
        }
        fn complete(
            &self,
            _: CompletionRequest,
            cancel: CancellationToken,
        ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
            Box::pin(async move {
                let s: Pin<
                    Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
                > = Box::pin(futures::stream::unfold(cancel, |cancel| async move {
                    cancel.cancelled().await;
                    None
                }));
                Ok(s)
            })
        }
    }

    let provider = Arc::new(HangingProvider) as Arc<dyn LlmProvider>;
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .config(TurnConfig {
            model: "hang".to_string(),
            ..TurnConfig::default()
        })
        .build();
    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("session");

    let s1 = session.clone();
    let h1 = tokio::spawn(async move {
        s1.run_turn(vec![ContentBlock::Text(TextContent::new("a"))])
            .await
    });

    // Give h1 a moment to enter the turn
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let res2 = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("b"))])
        .await;
    assert!(matches!(res2, Err(TurnError::TurnInProgress)));

    // Cleanup: cancel h1
    session.cancel_turn();
    let r1 = h1.await.expect("join h1");
    assert!(matches!(
        r1,
        Ok(StopReason::Cancelled) | Ok(StopReason::EndTurn)
    ));
}

#[tokio::test]
async fn load_session_restores_history_for_next_turn() {
    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let loaded = LoadedSession {
        info: SessionCreateInfo {
            id: SessionId::new(new_session_id()),
            cwd: std::env::current_dir().expect("cwd"),
            mcp_servers: Vec::new(),
        },
        history: vec![defect_agent::llm::Message {
            role: defect_agent::llm::Role::User,
            content: vec![defect_agent::llm::MessageContent::Text {
                text: "restored".to_string(),
            }]
            .into(),
        }],
    };

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .session_loader(Arc::new(StubLoader {
            loaded: loaded.clone(),
        }))
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let session = core
        .load_session(
            loaded.info.id.clone(),
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("load session");

    let stop = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("hello"))])
        .await
        .expect("turn");

    assert!(matches!(stop, StopReason::EndTurn));
}

#[tokio::test]
async fn load_session_triggers_observers() {
    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let loaded = LoadedSession {
        info: SessionCreateInfo {
            id: SessionId::new(new_session_id()),
            cwd: std::env::current_dir().expect("cwd"),
            mcp_servers: Vec::new(),
        },
        history: Vec::new(),
    };
    let count = Arc::new(AtomicUsize::new(0));

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .session_loader(Arc::new(StubLoader {
            loaded: loaded.clone(),
        }))
        .observe_session(Arc::new(CountingObserver {
            count: count.clone(),
        }))
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let _session = core
        .load_session(
            loaded.info.id.clone(),
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("load session");

    assert_eq!(count.load(Ordering::SeqCst), 1);
}

/// Uses `AskWritesPolicy` (default) with a mutating tool to exercise the Ask path.
/// After the client replies with `allow_once`, the tool should execute and the turn
/// should complete.
#[tokio::test]
async fn ask_writes_policy_runs_after_allow_once() {
    /// A Mutating tool that triggers the Ask path.
    struct WriteEcho {
        schema: ToolSchema,
    }
    impl Tool for WriteEcho {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
            SafetyClass::Mutating
        }
        fn describe<'a>(
            &'a self,
            _args: &'a serde_json::Value,
            _ctx: ToolContext<'a>,
        ) -> BoxFuture<'a, ToolCallDescription> {
            Box::pin(async {
                let mut fields = ToolCallUpdateFields::default();
                fields.title = Some("write".to_string());
                ToolCallDescription { fields }
            })
        }
        fn execute(&self, args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
            let text = args
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let mut completed = ToolCallUpdateFields::default();
            completed.content = Some(vec![
                agent_client_protocol_schema::ToolCallContent::Content(
                    agent_client_protocol_schema::Content::new(text),
                ),
            ]);
            let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
                Box::pin(stream::iter(vec![ToolEvent::Completed(completed)]));
            s
        }
    }

    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(WriteEcho {
                schema: ToolSchema {
                    name: "echo".to_string(),
                    description: "write echo".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .build(),
    );

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");
    let mut events = session.subscribe();

    let s_clone: Arc<dyn Session> = session.clone();
    let turn = tokio::spawn(async move {
        s_clone
            .run_turn(vec![ContentBlock::Text(TextContent::new("hello"))])
            .await
    });

    // Wait for the `PolicyDecision::Ask` event before resolving
    let mut resolved = false;
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::PolicyDecision { id, decision } => {
                use defect_agent::policy::PolicyDecision;
                if matches!(decision, PolicyDecision::Ask(_)) {
                    session.resolve_permission(
                        id,
                        PermissionResolution::Selected {
                            option_id: PermissionOptionId::new("allow_once"),
                        },
                    );
                    resolved = true;
                }
            }
            AgentEvent::TurnEnded { .. } => break,
            _ => {}
        }
    }
    assert!(resolved, "expected a PolicyDecision::Ask event");

    let stop = turn.await.expect("join").expect("turn ok");
    assert!(matches!(stop, StopReason::EndTurn));
}

/// `AskWritesPolicy` + user cancels turn → `Cancelled`, not an internal error.
#[tokio::test]
async fn ask_writes_policy_cancel_during_ask_returns_cancelled() {
    struct WriteEcho {
        schema: ToolSchema,
    }
    impl Tool for WriteEcho {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
            SafetyClass::Destructive
        }
        fn describe<'a>(
            &'a self,
            _args: &'a serde_json::Value,
            _ctx: ToolContext<'a>,
        ) -> BoxFuture<'a, ToolCallDescription> {
            Box::pin(async {
                ToolCallDescription {
                    fields: ToolCallUpdateFields::default(),
                }
            })
        }
        fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
            let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
                Box::pin(stream::iter(Vec::<ToolEvent>::new()));
            s
        }
    }

    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(WriteEcho {
                schema: ToolSchema {
                    name: "echo".to_string(),
                    description: "write".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .build(),
    );
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("session");
    let mut events = session.subscribe();

    let s_clone: Arc<dyn Session> = session.clone();
    let turn = tokio::spawn(async move {
        s_clone
            .run_turn(vec![ContentBlock::Text(TextContent::new("hello"))])
            .await
    });

    // Wait for the Ask event before cancelling
    while let Some(ev) = events.next().await {
        if let AgentEvent::PolicyDecision { decision, .. } = &ev {
            use defect_agent::policy::PolicyDecision;
            if matches!(decision, PolicyDecision::Ask(_)) {
                session.cancel_turn();
                break;
            }
        }
    }

    let stop = turn.await.expect("join").expect("turn ok");
    assert!(
        matches!(stop, StopReason::Cancelled),
        "expected Cancelled, got {stop:?}"
    );
}

/// When the user denies, the main loop feeds `tool_result(is_error=true)` back to the LLM
/// and issues another request; if the provider returns `EndTurn` on the second round, the
/// overall turn should be `Ok(EndTurn)` rather than `TurnError::Internal` (the ACP bridge
/// layer would project that as a wire `Internal error`).
#[tokio::test]
async fn deny_during_ask_completes_cleanly() {
    struct DestructiveTool {
        schema: ToolSchema,
    }
    impl Tool for DestructiveTool {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
            SafetyClass::Destructive
        }
        fn describe<'a>(
            &'a self,
            _args: &'a serde_json::Value,
            _ctx: ToolContext<'a>,
        ) -> BoxFuture<'a, ToolCallDescription> {
            Box::pin(async {
                let mut fields = ToolCallUpdateFields::default();
                fields.title = Some("$ ls".to_string());
                ToolCallDescription { fields }
            })
        }
        fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
            // Should not be executed after rejection; if called, the decision path is
            // incorrect.
            let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
                Box::pin(stream::iter(Vec::<ToolEvent>::new()));
            s
        }
    }

    let provider = Arc::new(ScriptedProvider::new()) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(DestructiveTool {
                schema: ToolSchema {
                    name: "echo".to_string(),
                    description: "destructive echo".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .build(),
    );
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "scripted-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("session");
    let mut events = session.subscribe();

    let s_clone: Arc<dyn Session> = session.clone();
    let turn = tokio::spawn(async move {
        s_clone
            .run_turn(vec![ContentBlock::Text(TextContent::new("hello"))])
            .await
    });

    // Wait for an Ask event, then resolve with reject_once
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::PolicyDecision { id, decision } => {
                use defect_agent::policy::PolicyDecision;
                if matches!(decision, PolicyDecision::Ask(_)) {
                    session.resolve_permission(
                        id,
                        PermissionResolution::Selected {
                            option_id: PermissionOptionId::new("reject_once"),
                        },
                    );
                }
            }
            AgentEvent::TurnEnded { .. } => break,
            _ => {}
        }
    }

    let stop = turn.await.expect("join").expect("turn ok");
    assert!(matches!(stop, StopReason::EndTurn), "got {stop:?}");
}

/// `max_concurrent_tools` rate‑limiting: when N concurrent `tool_use` calls are issued in
/// a single turn, the number of tools running simultaneously must not exceed the
/// configured limit. This is the concurrency gate for fanout (multiple `spawn_agent`
/// calls in the same turn).
#[tokio::test]
async fn max_concurrent_tools_caps_fanout() {
    /// Provider: on the first call, emits 4 tool_use requests then Stop=ToolUse;
    /// afterwards, EndTurn.
    struct FanoutProvider {
        calls: Mutex<u32>,
    }
    impl LlmProvider for FanoutProvider {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                vendor: "fanout".to_string(),
                protocol: ProtocolId::AnthropicMessages,
                display_name: "Fanout Test Provider".to_string(),
            }
        }
        fn capabilities(&self) -> Capabilities {
            unsupported_caps()
        }
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn model_info(&self, _: &str) -> Option<ModelInfo> {
            None
        }
        fn complete(
            &self,
            _: CompletionRequest,
            _: CancellationToken,
        ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
            let mut calls = self.calls.lock().expect("calls poisoned");
            *calls += 1;
            let nth = *calls;
            Box::pin(async move {
                let chunks: Vec<Result<ProviderChunk, ProviderError>> = if nth == 1 {
                    let mut c = vec![Ok(ProviderChunk::MessageStart {
                        id: "msg-1".to_string(),
                        model: "fanout-001".to_string(),
                    })];
                    // 4 concurrent tool_use calls, all invoking the same slow tool
                    for i in 0..4 {
                        c.push(Ok(ProviderChunk::ToolUseStart {
                            id: format!("tu-{i}"),
                            name: "slow".to_string(),
                        }));
                        c.push(Ok(ProviderChunk::ToolUseArgsDelta {
                            id: format!("tu-{i}"),
                            fragment: "{}".to_string(),
                        }));
                        c.push(Ok(ProviderChunk::ToolUseEnd {
                            id: format!("tu-{i}"),
                        }));
                    }
                    c.push(Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::ToolUse,
                    }));
                    c
                } else {
                    vec![
                        Ok(ProviderChunk::MessageStart {
                            id: "msg-2".to_string(),
                            model: "fanout-001".to_string(),
                        }),
                        Ok(ProviderChunk::Stop {
                            reason: LlmStopReason::EndTurn,
                        }),
                    ]
                };
                let s: Pin<
                    Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
                > = Box::pin(stream::iter(chunks));
                Ok(s)
            })
        }
    }

    /// Tool: increments a shared counter on entry, updates the peak, sleeps, then
    /// decrements on exit. Uses the shared counters to observe real concurrency.
    struct SlowTool {
        schema: ToolSchema,
        live: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }
    impl Tool for SlowTool {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
            SafetyClass::ReadOnly
        }
        fn describe<'a>(
            &'a self,
            _args: &'a serde_json::Value,
            _ctx: ToolContext<'a>,
        ) -> BoxFuture<'a, ToolCallDescription> {
            Box::pin(async {
                ToolCallDescription {
                    fields: ToolCallUpdateFields::default(),
                }
            })
        }
        fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
            let live = self.live.clone();
            let peak = self.peak.clone();
            let fut = async move {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                live.fetch_sub(1, Ordering::SeqCst);
                ToolEvent::Completed(ToolCallUpdateFields::default())
            };
            let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
                Box::pin(stream::once(fut));
            s
        }
    }

    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(FanoutProvider {
        calls: Mutex::new(0),
    }) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(SlowTool {
                schema: ToolSchema {
                    name: "slow".to_string(),
                    description: "slow tool".to_string(),
                    input_schema: json!({"type":"object"}),
                },
                live: live.clone(),
                peak: peak.clone(),
            }))
            .build(),
    );

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(OpenPolicy) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "fanout-001".to_string(),
            max_concurrent_tools: 2,
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("session");

    let stop = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("go"))])
        .await
        .expect("turn");

    assert!(matches!(stop, StopReason::EndTurn));
    let observed = peak.load(Ordering::SeqCst);
    assert!(
        observed >= 2,
        "expected fanout to actually run concurrently (peak >= 2), got {observed}"
    );
    assert!(
        observed <= 2,
        "max_concurrent_tools=2 should cap concurrency, but peak was {observed}"
    );
}

// Ensure the compiler sees that `OpenPolicy` is used (to prevent future dead-code
// warnings).
#[allow(dead_code)]
fn _types_in_use() -> Arc<dyn SandboxPolicy> {
    Arc::new(OpenPolicy)
}

// ---------------------------------------------------------------------------
// End-to-end: before_turn_end hook keeps the turn alive (your original goal)
// ---------------------------------------------------------------------------

/// A provider that always immediately calls EndTurn (without invoking any tool), ensuring
/// every turn reaches the `before_turn_end` check.
struct AlwaysEndTurnProvider {
    calls: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl LlmProvider for AlwaysEndTurnProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "always-end".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Always EndTurn".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _id: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
            Ok(ProviderChunk::TextDelta {
                text: "done".to_string(),
            }),
            Ok(ProviderChunk::Stop {
                reason: LlmStopReason::EndTurn,
            }),
        ];
        Box::pin(async move { Ok(Box::pin(stream::iter(chunks)) as ProviderStream) })
    }
}

/// A hook engine that returns `continue` (injecting one feedback sentence) for the first
/// N calls to `before_turn_end`, then stops. Simulates the effect of "command hook exit 2
/// → continue" without relying on a real subprocess.
struct ContinueNTimesEngine {
    remaining: std::sync::Mutex<u32>,
}

impl defect_agent::hooks::HookEngine for ContinueNTimesEngine {
    fn dispatch<'a>(
        &'a self,
        step: &'a mut dyn defect_agent::hooks::step::HookStep,
        _ctx: defect_agent::hooks::HookCtx<'a>,
    ) -> BoxFuture<'a, defect_agent::hooks::step::HookControl> {
        use defect_agent::hooks::step::HookControl;
        Box::pin(async move {
            if step.event_name() != "before_turn_end" {
                return HookControl::Proceed;
            }
            let mut rem = self.remaining.lock().expect("mutex");
            if *rem == 0 {
                return HookControl::Proceed; // let it pass through
            }
            *rem -= 1;
            // Inject a keep-alive feedback (via `apply_verdict`'s `additional_context` →
            // `step.feedback`).
            let _ = step.apply_verdict(&json!({
                "control": "continue",
                "additional_context": ["keep going: condition not yet met"],
            }));
            HookControl::Continue
        })
    }
}

#[tokio::test]
async fn turn_end_hook_continue_makes_turn_loop() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let provider = Arc::new(AlwaysEndTurnProvider {
        calls: calls.clone(),
    }) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(StaticToolRegistry::builder().build());

    // Hook: allow 2 turn extensions, then stop → expect 3 LLM calls total (1 initial + 2
    // extensions).
    let engine = Arc::new(ContinueNTimesEngine {
        remaining: std::sync::Mutex::new(2),
    }) as Arc<dyn defect_agent::hooks::HookEngine>;

    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .hook_engine(engine)
        .config(TurnConfig {
            model: "always-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");

    let prompt = vec![ContentBlock::Text(TextContent::new("hello"))];
    let stop = session.run_turn(prompt).await.expect("turn");

    assert!(matches!(stop, StopReason::EndTurn));
    // 1 initial + 2 renewals = 3 LLM calls, confirming the turn-end hook caused the loop
    // to run two extra rounds.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "turn should loop 2 extra rounds"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: run_in_background background task + passive backflow (phase 1)
// ---------------------------------------------------------------------------

/// On the first call, emits a `tool_use` (calls `bg_tool`) then returns `Stop=ToolUse`;
/// every subsequent call returns `EndTurn`.
struct BgScriptedProvider {
    calls: Mutex<u32>,
}

impl LlmProvider for BgScriptedProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "bg".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Background Test Provider".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _: CompletionRequest,
        _: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let mut calls = self.calls.lock().expect("calls poisoned");
        *calls += 1;
        let nth = *calls;
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = if nth == 1 {
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "m1".to_string(),
                        model: "bg-001".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseStart {
                        id: "tu-1".to_string(),
                        name: "bg_tool".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseArgsDelta {
                        id: "tu-1".to_string(),
                        fragment: "{}".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseEnd {
                        id: "tu-1".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::ToolUse,
                    }),
                ]
            } else {
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "m2".to_string(),
                        model: "bg-001".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ]
            };
            let s: Pin<
                Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
            > = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

/// A fire-and-forget tool: if `ctx` has a background handle, spawns a background task
/// (which completes immediately) and returns "started" right away. This tool does not
/// block waiting for the task.
struct BgSpawnTool {
    schema: ToolSchema,
}

impl Tool for BgSpawnTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async {
            ToolCallDescription {
                fields: ToolCallUpdateFields::default(),
            }
        })
    }
    fn execute(&self, _args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let bg = ctx.background.clone();
        let fut = async move {
            let mut fields = ToolCallUpdateFields::default();
            let text = match bg {
                Some(bg) => {
                    let id = bg.spawn("worker".to_string(), |_cancel, _progress| async move {
                        defect_agent::session::BackgroundResult::Completed(
                            "THE-BACKGROUND-ANSWER".to_string(),
                        )
                    });
                    format!("started {id}")
                }
                None => "no background available".to_string(),
            };
            fields.content = Some(vec![
                agent_client_protocol_schema::ToolCallContent::Content(
                    agent_client_protocol_schema::Content::new(text),
                ),
            ]);
            ToolEvent::Completed(fields)
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

/// End-to-end (active reflow): turn 1 calls `bg_tool` to spawn a background task → turn 1
/// does not block and completes normally; after the background task finishes, the
/// **session driver spontaneously** starts an autonomous turn to consume the result,
/// without requiring a second user input. Asserts that the autonomous turn's
/// `UserPromptCommitted` carries the background answer text.
#[tokio::test]
async fn run_in_background_result_actively_reflows() {
    let provider = Arc::new(BgScriptedProvider {
        calls: Mutex::new(0),
    }) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(BgSpawnTool {
                schema: ToolSchema {
                    name: "bg_tool".to_string(),
                    description: "spawn a background task".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .build(),
    );
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(OpenPolicy) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "bg-001".to_string(),
            ..TurnConfig::default()
        })
        .build();
    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("session");

    // Subscribe to events before the background task completes, so we can receive the
    // driver's automatic continuation turn.
    let mut events = session.subscribe();

    // Turn 1: call `bg_tool`, spawn a background task, non-blocking.
    let stop1 = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("kick off"))])
        .await
        .expect("turn 1");
    assert!(matches!(stop1, StopReason::EndTurn));

    // Active reflow: the driver spontaneously starts an autonomous turn to consume the
    // result when the background task completes — no second user input is needed. Assert
    // that the autonomous turn's `UserPromptCommitted` carries the background answer. Use
    // a timeout as a fallback to avoid hanging if the driver does not start the turn.
    let saw_active_reflow = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while let Some(ev) = events.next().await {
            if let AgentEvent::UserPromptCommitted { content } = &ev {
                let joined: String = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                // This `UserPromptCommitted` comes from a driver-initiated continuation
                // turn (not user input), and its content is the background answer —
                // confirming that the proactive re-invoke succeeded.
                if joined.contains("THE-BACKGROUND-ANSWER") {
                    return true;
                }
            }
        }
        false
    })
    .await
    .unwrap_or(false);

    assert!(
        saw_active_reflow,
        "driver should autonomously start a turn carrying the background result (active re-invoke)"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: hitting the per-turn request cap still consults before_turn_end,
// and a continuing hook (goal mode) resets the budget and keeps working.
// ---------------------------------------------------------------------------

/// Provider that emits one `noop` tool_use on every call, so the turn loop always reaches
/// the request-cap check (after the tool batch) rather than the voluntary EndTurn path.
struct AlwaysToolProvider {
    calls: Arc<std::sync::atomic::AtomicU32>,
}

impl LlmProvider for AlwaysToolProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "always-tool".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Always Tool".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _: CompletionRequest,
        _: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
                Ok(ProviderChunk::MessageStart {
                    id: format!("m-{n}"),
                    model: "cap-001".to_string(),
                }),
                Ok(ProviderChunk::ToolUseStart {
                    id: format!("tu-{n}"),
                    name: "noop".to_string(),
                }),
                Ok(ProviderChunk::ToolUseArgsDelta {
                    id: format!("tu-{n}"),
                    fragment: "{}".to_string(),
                }),
                Ok(ProviderChunk::ToolUseEnd {
                    id: format!("tu-{n}"),
                }),
                Ok(ProviderChunk::Stop {
                    reason: LlmStopReason::ToolUse,
                }),
            ];
            Ok(Box::pin(stream::iter(chunks)) as ProviderStream)
        })
    }
}

/// A tool that does nothing and immediately completes. Counts as "progress" (an executed
/// tool), which under Adaptive would expand the cap — here we use Fixed to keep the cap
/// flat and force the cap-hit path.
struct NoopTool {
    schema: ToolSchema,
}
impl Tool for NoopTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async {
            ToolCallDescription {
                fields: ToolCallUpdateFields::default(),
            }
        })
    }
    fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        Box::pin(stream::once(async {
            ToolEvent::Completed(ToolCallUpdateFields::default())
        }))
    }
}

#[tokio::test]
async fn request_cap_hit_consults_turn_end_hook_and_resets_budget() {
    use defect_agent::session::TurnRequestLimit;

    let calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let provider = Arc::new(AlwaysToolProvider {
        calls: calls.clone(),
    }) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(NoopTool {
                schema: ToolSchema {
                    name: "noop".to_string(),
                    description: "noop".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .build(),
    );

    // Goal-mode-like gate: continue on every before_turn_end (including the involuntary
    // MaxTurnRequests stop). Bounded by max_hook_continues below.
    let engine = Arc::new(ContinueNTimesEngine {
        remaining: std::sync::Mutex::new(2),
    }) as Arc<dyn defect_agent::hooks::HookEngine>;

    // Fixed cap of 1 LLM call per logical turn: each round makes one call, runs the tool,
    // hits the cap, and must consult the hook. The gate continues twice (budget reset each
    // time) then stops → 3 LLM calls total.
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .hook_engine(engine)
        .config(TurnConfig {
            model: "cap-001".to_string(),
            request_limit: TurnRequestLimit::Fixed(1),
            max_hook_continues: 2,
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");

    let prompt = vec![ContentBlock::Text(TextContent::new("go"))];
    let stop = session.run_turn(prompt).await.expect("turn");

    // Without the fix: cap is hit on round 1, turn returns MaxTurnRequests immediately,
    // only 1 LLM call. With the fix: the hook continues twice (budget reset each round),
    // so 3 calls happen, then the hard cap forces the final stop with MaxTurnRequests.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "cap-hit should consult the hook and reset the budget for 2 extra rounds"
    );
    assert!(
        matches!(stop, StopReason::MaxTurnRequests),
        "final stop after exhausting hook continues should be MaxTurnRequests, got {stop:?}"
    );
}

// ---------------------------------------------------------------------------
// Goal mode force-keeps `goal_done` past a restrictive `--profile` allowlist:
// it is the only way for the agent to signal completion, so the allowlist must
// not be able to strip it.
// ---------------------------------------------------------------------------

/// Provider that records the tool names offered in each `CompletionRequest`, then ends the
/// turn without calling anything.
struct ToolRecordingProvider {
    seen_tools: Arc<Mutex<Vec<String>>>,
}

impl LlmProvider for ToolRecordingProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "rec".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Tool Recording".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        req: CompletionRequest,
        _: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        *self.seen_tools.lock().unwrap() = req.tools.iter().map(|t| t.name.clone()).collect();
        Box::pin(async {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
                Ok(ProviderChunk::MessageStart {
                    id: "m-0".to_string(),
                    model: "rec-001".to_string(),
                }),
                Ok(ProviderChunk::TextDelta {
                    text: "ok".to_string(),
                }),
                Ok(ProviderChunk::Stop {
                    reason: LlmStopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(chunks)) as ProviderStream)
        })
    }
}

#[tokio::test]
async fn goal_mode_force_keeps_goal_done_past_restrictive_allowlist() {
    use defect_agent::session::GoalState;
    use defect_agent::tool::{GOAL_DONE_TOOL_NAME, GoalDoneTool};

    let seen_tools = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ToolRecordingProvider {
        seen_tools: seen_tools.clone(),
    }) as Arc<dyn LlmProvider>;

    // Process pool mirrors `--goal` assembly: a normal tool plus the overlaid goal_done.
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(NoopTool {
                schema: ToolSchema {
                    name: "noop".to_string(),
                    description: "noop".to_string(),
                    input_schema: json!({"type":"object"}),
                },
            }))
            .insert(Arc::new(GoalDoneTool::new()))
            .build(),
    );

    // Profile allowlist deliberately omits goal_done (only allows `noop`).
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .tool_allow(vec!["noop".to_string()])
        .goal(Arc::new(GoalState::new("do the thing".to_string())))
        .config(TurnConfig {
            model: "rec-001".to_string(),
            ..TurnConfig::default()
        })
        .build();

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");

    let prompt = vec![ContentBlock::Text(TextContent::new("go"))];
    session.run_turn(prompt).await.expect("turn");

    let offered = seen_tools.lock().unwrap().clone();
    assert!(
        offered.iter().any(|n| n == GOAL_DONE_TOOL_NAME),
        "goal mode must offer goal_done to the LLM even when the profile allowlist omits it; \
         got {offered:?}"
    );
    assert!(
        offered.iter().any(|n| n == "noop"),
        "the allowlisted tool should still be offered; got {offered:?}"
    );
}
