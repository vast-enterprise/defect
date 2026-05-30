use super::*;

use std::path::Path;

use agent_client_protocol::schema::ToolCallUpdateFields;
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
fn unknown_profile_fails() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let profile = SubagentProfile {
        description: "d".to_string(),
        model: None,
        system_prompt: "sys".to_string(),
        tool_allow: vec![],
        sampling: None,
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
        json!({"profile": "nope", "task": "t"}),
        tmp.path(),
    );
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
    assert_eq!(completed_text(&events).as_deref(), Some("done after denial"));
}
