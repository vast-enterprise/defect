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

// --- test provider ----------------------------------------------------

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

/// Always returns a single text message followed by EndTurn.
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

/// First round requests a tool call, then returns text after a `tool_result` appears in
/// history.
/// Used to verify that when a sub-agent hits a write tool, `NonInteractivePolicy`
/// downgrades `Ask` to `Deny`,
/// and the sub-turn does not hang on `PermissionGate` and finishes normally.
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
        // Tool result already present in history → second round, done.
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

// --- Test write tool (Mutating) -------------------------------------------

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
        // Should not be called (policy should Deny); calling it has no side effects.
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
    // Non-zero depth to avoid the new depth gate failing loudly before dispatch.
    let ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1")
        .with_subagent_depth(4);
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

/// Run the tool, but inject a subagent bridge pointing to `parent_events` and subscribe
/// to the parent emitter before running; returns (tool events, events received by the
/// parent emitter). Used to verify that child turn events are wrapped as
/// `AgentEvent::Subagent` and bridged back to the parent.
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
    // Use a non-zero depth, otherwise the newly added depth gate will fail loud before
    // dispatch.
    let ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1")
        .with_subagent_bridge(bridge)
        .with_subagent_depth(4);
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
        // By the time the tool returns, the bridging task has already been awaited (drop
        // events + task.await), so all events from the parent emitter have been sent.
        // Drain them one by one (non-blocking; only the buffered events remain).
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

// --- tests ---

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
    // The catalog is included in the description.
    assert!(schema.description.contains("review diffs for races"));
    assert!(schema.description.contains("- reviewer:"));
    // profile enum contains the discovered names.
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
    // The child turn runs a text-returning provider. After injecting the subagent bridge,
    // the parent emitter should receive a sequence of `AgentEvent::Subagent` (with
    // correct `parent_tool_call_id` / `agent_type`), containing the child turn's
    // `TurnStarted`, `LlmCall`, `AssistantText`, and `TurnEnded`.
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
    // The tool itself still returns the final text normally.
    assert_eq!(completed_text(&tool_events).as_deref(), Some("the answer"));

    // The parent emitter receives only `Subagent` wrappers. In the single-layer case,
    // `ancestor_path` is exactly `[this call's id]`, and `agent_type` is correct.
    assert!(!bridged.is_empty(), "expected bridged subagent events");
    for ev in &bridged {
        match ev {
            AgentEvent::Subagent {
                ancestor_path,
                agent_type,
                ..
            } => {
                assert_eq!(
                    ancestor_path
                        .iter()
                        .map(|id| id.0.as_ref())
                        .collect::<Vec<_>>(),
                    vec!["parent-call-1"]
                );
                assert_eq!(agent_type, "reviewer");
            }
            other => panic!("expected Subagent wrapper, got {other:?}"),
        }
    }
    // Must contain at least one sub-turn boundary and one assistant text segment.
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
    // The sub-agent requests `write_file` (Mutating) in its first turn. The parent policy
    // is `AskWrites`, which returns `Ask` for Mutating; `NonInteractivePolicy` must
    // downgrade it to `Deny`. The sub-turn does not hang on the `PermissionGate`, and the
    // second turn completes normally with text.
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

    // `run_tool` must return within a reasonable time (no deadlock). On a
    // `current_thread` runtime, if `PermissionGate` awaits forever, this call never
    // returns — the test hangs instead of passing, which is equivalent to a failure.
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

/// A minimal `HookEngine` that short-circuits on `before_generate`, synthesizing
/// assistant text to completely skip the real LLM call. Used to prove that the profile's
/// built-in hook engine is indeed dispatched during the sub-agent's turn — otherwise the
/// sub-agent would return the provider's text instead.
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
                // apply_verdict uses the `assistant` field of BeforeGenerate to set
                // assistant_text.
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
        // The provider returns "from provider" — this is the result if the hook does not
        // run.
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
    // The hook short-circuit takes effect, so the final text comes from the hook, not the
    // provider.
    assert_eq!(completed_text(&events).as_deref(), Some("from hook"));
}
