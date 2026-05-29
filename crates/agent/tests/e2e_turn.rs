//! 端到端 turn 测试：mock provider + mock tool 跑一次完整 turn。
//!
//! 验证：
//! - 用户 prompt 被 append 进 history
//! - LLM 流被消费、TextDelta / ToolUse 被正确翻译为 AgentEvent
//! - tool_use → tool 调度 → tool 结果回写 history
//! - 第二轮 LLM EndTurn → run_turn 返回 Ok(EndTurn)

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_client_protocol::schema::SessionId;
use agent_client_protocol::schema::{
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
    AgentCore, DefaultAgentCore, LoadedSession, Session, SessionCreateInfo, SessionLoader,
    SessionObserver, StaticToolRegistry, ToolRegistry, TurnConfig, new_session_id,
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

/// 按调用次数返回不同的 chunk 序列：
/// - 第 1 次：emit 一段文本 + 一个 tool_use 然后 Stop=ToolUse
/// - 第 2 次：emit 一段文本然后 Stop=EndTurn
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

/// echo 工具：把 args.msg 原样返回。
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
            agent_client_protocol::schema::ToolCallContent::Content(
                agent_client_protocol::schema::Content::new(text),
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
        )
        .await
        .expect("create session");

    // 订阅事件流——必须在 run_turn 开始前订阅，否则事件先到没人接
    let mut events = session.subscribe();

    let prompt = vec![ContentBlock::Text(TextContent::new("hello"))];
    let stop = session.run_turn(prompt).await.expect("turn");

    assert!(matches!(stop, StopReason::EndTurn));

    // 收 emit 事件直到 TurnEnded
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
            AgentEvent::ToolCallStarted { .. } => got_tool_call_started = true,
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

    // provider 总是无限挂起的 stream（不 Stop），让 turn 一直跑
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
        )
        .await
        .expect("session");

    let s1 = session.clone();
    let h1 = tokio::spawn(async move {
        s1.run_turn(vec![ContentBlock::Text(TextContent::new("a"))])
            .await
    });

    // 给 h1 一点时间进入 turn
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let res2 = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("b"))])
        .await;
    assert!(matches!(res2, Err(TurnError::TurnInProgress)));

    // 收尾：取消 h1
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
            }],
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
        )
        .await
        .expect("load session");

    assert_eq!(count.load(Ordering::SeqCst), 1);
}

/// 用 `AskWritesPolicy`（默认）+ Mutating 工具走 Ask 路径。
/// 客户端"应答" `allow_once` 后工具应被执行、turn 完成。
#[tokio::test]
async fn ask_writes_policy_runs_after_allow_once() {
    /// Mutating 工具：触发 Ask 分支。
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
                agent_client_protocol::schema::ToolCallContent::Content(
                    agent_client_protocol::schema::Content::new(text),
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

    // 等到 PolicyDecision::Ask 事件出来再 resolve
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

/// `AskWritesPolicy` + 用户取消 turn → `Cancelled`，不是 internal error。
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

    // 等到 Ask 事件出现再 cancel
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

/// 用户拒绝 → 主循环把 tool_result(is_error=true) 喂回 LLM、再发一轮请求；
/// 若 provider 在第二轮返回 EndTurn，整体 turn 应当 Ok(EndTurn) 而非
/// `TurnError::Internal`（acp 桥接层会把它投影成 wire `Internal error`）。
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
            // 拒绝后不应被 execute——若被调说明决策路径出错。
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

    // 等到 Ask 事件出现 → resolve 为 reject_once
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

// 让编译期看到我们用到了 OpenPolicy（避免之后引用断裂）
#[allow(dead_code)]
fn _types_in_use() -> Arc<dyn SandboxPolicy> {
    Arc::new(OpenPolicy)
}
