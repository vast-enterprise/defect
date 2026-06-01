use super::*;

use std::path::Path;

use agent_client_protocol_schema::ToolCallUpdateFields;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use tokio_util::sync::CancellationToken;

use crate::fs::{FsBackend, NoopFsBackend};
use crate::http::NoopHttpClient;
use crate::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderInfo, ProviderStream, StopReason, ThinkingEcho,
};
use crate::policy::AskWritesPolicy;
use crate::shell::ShellBackend;
use crate::tool::{Tool, ToolContext};

// --- 测试用 provider ----------------------------------------------------

fn fake_caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Unsupported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

fn fake_info() -> ProviderInfo {
    ProviderInfo {
        vendor: "fake".to_string(),
        protocol: ProtocolId::OpenAiChat,
        display_name: "Fake".to_string(),
    }
}

fn model(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        display_name: None,
        context_window: None,
        max_output_tokens: None,
        deprecated: false,
        capabilities_overrides: Default::default(),
    }
}

/// 固定回一条文本 + EndTurn。
struct TextProvider {
    text: String,
}

impl LlmProvider for TextProvider {
    fn info(&self) -> ProviderInfo {
        fake_info()
    }
    fn capabilities(&self) -> Capabilities {
        fake_caps()
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
        let chunks = vec![
            Ok(ProviderChunk::MessageStart {
                id: "m".into(),
                model: "fake-1".into(),
            }),
            Ok(ProviderChunk::TextDelta {
                text: self.text.clone(),
            }),
            Ok(ProviderChunk::Stop {
                reason: StopReason::EndTurn,
            }),
        ];
        let s: ProviderStream = Box::pin(stream::iter(chunks));
        Box::pin(async move { Ok(s) })
    }
}

/// 第一轮请求工具调用，之后（历史里出现 tool_result）回文本结束。
/// 用来验证：子 agent 命中写工具时，NonInteractivePolicy 把 Ask 降级为 Deny，
/// 子 turn 不挂在 PermissionGate 上、能正常结束。
struct ToolThenTextProvider {
    tool_name: String,
}

impl LlmProvider for ToolThenTextProvider {
    fn info(&self) -> ProviderInfo {
        fake_info()
    }
    fn capabilities(&self) -> Capabilities {
        fake_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _model_id: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        // 历史里已有 tool_result ⇒ 第二轮，结束。
        let has_tool_result = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, MessageContent::ToolResult { .. }))
        });
        let chunks = if has_tool_result {
            vec![
                Ok(ProviderChunk::MessageStart {
                    id: "m2".into(),
                    model: "fake-1".into(),
                }),
                Ok(ProviderChunk::TextDelta {
                    text: "done after denial".into(),
                }),
                Ok(ProviderChunk::Stop {
                    reason: StopReason::EndTurn,
                }),
            ]
        } else {
            vec![
                Ok(ProviderChunk::MessageStart {
                    id: "m1".into(),
                    model: "fake-1".into(),
                }),
                Ok(ProviderChunk::ToolUseStart {
                    id: "tu1".into(),
                    name: self.tool_name.clone(),
                }),
                Ok(ProviderChunk::ToolUseArgsDelta {
                    id: "tu1".into(),
                    fragment: "{}".into(),
                }),
                Ok(ProviderChunk::ToolUseEnd { id: "tu1".into() }),
                Ok(ProviderChunk::Stop {
                    reason: StopReason::ToolUse,
                }),
            ]
        };
        let s: ProviderStream = Box::pin(stream::iter(chunks));
        Box::pin(async move { Ok(s) })
    }
}

// --- 测试用写工具（Mutating）-------------------------------------------

struct DummyWriteTool {
    schema: ToolSchema,
}

impl DummyWriteTool {
    fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "write_file".to_string(),
                description: "dummy".to_string(),
                input_schema: json!({"type": "object"}),
            },
        }
    }
}

impl Tool for DummyWriteTool {
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
            ToolCallDescription {
                fields: ToolCallUpdateFields::default(),
            }
        })
    }
    fn execute(&self, _args: serde_json::Value, _ctx: ToolContext<'_>) -> ToolStream {
        // 不该被调用（policy 应 Deny）；真被调用也无副作用。
        let ev = ToolEvent::Completed(ToolCallUpdateFields::default());
        Box::pin(stream::once(async move { ev }))
    }
}

// --- helpers ------------------------------------------------------------

fn profiles_with(profile: SubagentProfile) -> Arc<BTreeMap<String, SubagentProfile>> {
    let mut m = BTreeMap::new();
    m.insert("reviewer".to_string(), profile);
    Arc::new(m)
}

fn registry_with(provider: Arc<dyn LlmProvider>) -> Arc<ProviderRegistry> {
    ProviderRegistry::single(provider, model("fake-1"))
}

fn process_tools_with(tools: Vec<Arc<dyn Tool>>) -> Arc<dyn ToolRegistry> {
    let mut b = StaticToolRegistry::builder();
    for t in tools {
        b = b.insert(t);
    }
    Arc::new(b.build())
}

fn run_tool(tool: &SpawnAgentTool, args: serde_json::Value, cwd: &Path) -> Vec<ToolEvent> {
    let fs: Arc<dyn FsBackend> = Arc::new(NoopFsBackend);
    let shell: Arc<dyn ShellBackend> = Arc::new(crate::shell::NoopShellBackend);
    let http = Arc::new(NoopHttpClient);
    let ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let mut stream = tool.execute(args, ctx);
        let mut out = Vec::new();
        while let Some(ev) = stream.next().await {
            out.push(ev);
        }
        out
    })
}

/// 跑工具，但注入一个指向 `parent_events` 的 subagent 桥，并在跑之前订阅父
/// emitter；返回 (工具事件, 父 emitter 收到的事件)。用于验证子 turn 事件被包成
/// `AgentEvent::Subagent` 桥接回父。
fn run_tool_with_bridge(
    tool: &SpawnAgentTool,
    args: serde_json::Value,
    cwd: &Path,
    parent_tool_call_id: &str,
) -> (Vec<ToolEvent>, Vec<AgentEvent>) {
    let fs: Arc<dyn FsBackend> = Arc::new(NoopFsBackend);
    let shell: Arc<dyn ShellBackend> = Arc::new(crate::shell::NoopShellBackend);
    let http = Arc::new(NoopHttpClient);
    let parent_events = Arc::new(EventEmitter::new());
    let bridge = crate::tool::SubagentBridge {
        parent_events: parent_events.clone(),
        parent_tool_call_id: agent_client_protocol_schema::ToolCallId::new(parent_tool_call_id),
    };
    let ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1")
        .with_subagent_bridge(bridge);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let mut parent_sub = parent_events.subscribe();
        let mut tool_events = Vec::new();
        let mut stream = tool.execute(args, ctx);
        while let Some(ev) = stream.next().await {
            tool_events.push(ev);
        }
        // 工具返回前已 await 桥接 task（drop events + task.await），父 emitter 的
        // 事件此刻都已 send 完毕——逐条排空（不会阻塞，缓冲里就这些）。
        drop(parent_events);
        let mut bridged = Vec::new();
        while let Some(ev) = parent_sub.next().await {
            bridged.push(ev);
        }
        (tool_events, bridged)
    })
}

fn completed_text(events: &[ToolEvent]) -> Option<String> {
    events.iter().find_map(|ev| match ev {
        ToolEvent::Completed(fields) => fields.content.as_ref().and_then(|c| {
            c.iter().find_map(|cc| match cc {
                ToolCallContent::Content(content) => match &content.content {
                    ContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                },
                _ => None,
            })
        }),
        _ => None,
    })
}

// --- tests --------------------------------------------------------------

#[test]
fn schema_has_profile_enum_and_catalog() {
    let profile = SubagentProfile {
        description: "review diffs for races".to_string(),
        model: None,
        system_prompt: "you are reviewer".to_string(),
        tool_allow: vec!["read_file".to_string()],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(TextProvider { text: "hi".into() })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let schema = tool.schema();
    assert_eq!(schema.name, "spawn_agent");
    // catalog 进 description。
    assert!(schema.description.contains("review diffs for races"));
    assert!(schema.description.contains("- reviewer:"));
    // profile enum 含发现到的名字。
    let enum_vals = schema.input_schema["properties"]["profile"]["enum"]
        .as_array()
        .expect("enum array");
    assert_eq!(enum_vals.len(), 1);
    assert_eq!(enum_vals[0], "reviewer");
}

#[test]
fn returns_subagent_final_text() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec![],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(TextProvider {
            text: "the answer".into(),
        })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let events = run_tool(
        &tool,
        json!({"profile": "reviewer", "task": "do it"}),
        tmp.path(),
    );
    assert_eq!(completed_text(&events).as_deref(), Some("the answer"));
}

#[test]
fn subagent_events_bridged_to_parent() {
    // 子 turn 跑一个回文本的 provider。注入 subagent 桥后，父 emitter 应收到
    // 一串 AgentEvent::Subagent（带正确的 parent_tool_call_id / agent_type），
    // 内含子 turn 的 TurnStarted / LlmCall / AssistantText / TurnEnded。
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec![],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(TextProvider {
            text: "the answer".into(),
        })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let (tool_events, bridged) = run_tool_with_bridge(
        &tool,
        json!({"profile": "reviewer", "task": "do it"}),
        tmp.path(),
        "parent-call-1",
    );
    // 工具本身仍正常返回最终文本。
    assert_eq!(completed_text(&tool_events).as_deref(), Some("the answer"));

    // 父 emitter 收到的全是 Subagent 包裹，且 parent_tool_call_id / agent_type 正确。
    assert!(!bridged.is_empty(), "expected bridged subagent events");
    for ev in &bridged {
        match ev {
            AgentEvent::Subagent {
                parent_tool_call_id,
                agent_type,
                ..
            } => {
                assert_eq!(parent_tool_call_id.0.as_ref(), "parent-call-1");
                assert_eq!(agent_type, "reviewer");
            }
            other => panic!("expected Subagent wrapper, got {other:?}"),
        }
    }
    // 至少应包含子 turn 的边界与一段助手文本。
    let has_turn_started = bridged.iter().any(|ev| {
        matches!(ev, AgentEvent::Subagent { inner, .. } if matches!(**inner, AgentEvent::TurnStarted))
    });
    let has_assistant_text = bridged.iter().any(|ev| {
        matches!(ev, AgentEvent::Subagent { inner, .. } if matches!(**inner, AgentEvent::AssistantText { .. }))
    });
    assert!(has_turn_started, "subagent TurnStarted should be bridged");
    assert!(
        has_assistant_text,
        "subagent AssistantText should be bridged"
    );
}

#[test]
fn unknown_profile_fails() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec![],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(TextProvider { text: "x".into() })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let events = run_tool(&tool, json!({"profile": "nope", "task": "t"}), tmp.path());
    assert!(matches!(
        events.last(),
        Some(ToolEvent::Failed(ToolError::InvalidArgs(_)))
    ));
}

#[test]
fn unknown_allowed_tool_fails_loud() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec!["does_not_exist".to_string()],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(TextProvider { text: "x".into() })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let events = run_tool(
        &tool,
        json!({"profile": "reviewer", "task": "t"}),
        tmp.path(),
    );
    assert!(matches!(
        events.last(),
        Some(ToolEvent::Failed(ToolError::InvalidArgs(_)))
    ));
}

#[test]
fn deadlock_guard_mutating_tool_is_denied_and_turn_completes() {
    // 子 agent 第一轮请求调 write_file（Mutating）。父 policy 是 AskWrites，
    // 会对 Mutating 返回 Ask；NonInteractivePolicy 必须把它降级为 Deny，
    // 子 turn 不挂在 PermissionGate 上，第二轮回文本正常结束。
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec!["write_file".to_string()],
        sampling: None,
        hooks: None,
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        registry_with(Arc::new(ToolThenTextProvider {
            tool_name: "write_file".into(),
        })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![Arc::new(DummyWriteTool::new())]),
        None,
    );

    // 整个 run_tool 必须在合理时间内返回（不死锁）。current_thread runtime
    // 上若 PermissionGate 永久 await，这个调用永不返回——测试会挂死而非通过，
    // 即等价于 fail。
    let events = run_tool(
        &tool,
        json!({"profile": "reviewer", "task": "t"}),
        tmp.path(),
    );
    assert_eq!(
        completed_text(&events).as_deref(),
        Some("done after denial")
    );
}

/// 一个最小 HookEngine：在 `before_generate` 上 short-circuit，填合成 assistant
/// 文本，从而完全跳过真实 LLM 调用。用来证明 profile 自带的 hook 引擎确实在子
/// agent 的 turn 里被调度——否则子 agent 会回 provider 的文本而非这条。
struct ShortCircuitHooks {
    text: String,
}

impl crate::hooks::HookEngine for ShortCircuitHooks {
    fn dispatch<'a>(
        &'a self,
        step: &'a mut dyn crate::hooks::step::HookStep,
        _ctx: crate::hooks::HookCtx<'a>,
    ) -> BoxFuture<'a, crate::hooks::step::HookControl> {
        let text = self.text.clone();
        Box::pin(async move {
            if step.event_name() == "before_generate" {
                // apply_verdict 走 BeforeGenerate 的 `assistant` 字段 → assistant_text。
                let _ = step.apply_verdict(&json!({ "assistant": text }));
            }
            crate::hooks::step::HookControl::Proceed
        })
    }
}

#[test]
fn profile_hook_engine_runs_in_subagent_turn() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec![],
        sampling: None,
        hooks: Some(Arc::new(ShortCircuitHooks {
            text: "from hook".into(),
        })),
    };
    let tool = SpawnAgentTool::new(
        profiles_with(profile),
        // provider 会回 "from provider"——若 hook 没跑，结果就是它。
        registry_with(Arc::new(TextProvider {
            text: "from provider".into(),
        })),
        Arc::new(AskWritesPolicy::new()),
        process_tools_with(vec![]),
        None,
    );
    let events = run_tool(
        &tool,
        json!({"profile": "reviewer", "task": "do it"}),
        tmp.path(),
    );
    // hook short-circuit 生效 ⇒ 最终文本来自 hook，而非 provider。
    assert_eq!(completed_text(&events).as_deref(), Some("from hook"));
}
