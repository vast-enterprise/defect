//! Common scaffolding for e2e integration tests: mock server startup, echo tool, and
//! agent core assembly.
//!
//! Imported by `anthropic_e2e.rs` / `openai_e2e.rs` via `mod common;`.

use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, SessionId, TextContent, ToolCallContent, ToolCallUpdateFields,
};
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::LlmProvider;
use defect_agent::policy::{OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, Frontend, Session, StaticToolRegistry, ToolRegistry, TurnConfig,
    new_session_id,
};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};
use futures::future::BoxFuture;
use futures::stream;
use serde_json::json;
use wiremock::MockServer;

/// Starts a local `wiremock` server.
///
/// The server is automatically shut down when dropped; the caller holds it for the entire
/// test.
pub async fn start_mock_server() -> MockServer {
    MockServer::start().await
}

/// Concatenates a sequence of SSE events into the wire-format byte stream (`event:` +
/// `data:` + blank line), suitable for feeding directly to
/// [`wiremock::ResponseTemplate::set_body_raw`].
pub fn encode_sse_events(events: &[(&str, &str)]) -> Vec<u8> {
    let mut out = String::new();
    for (name, data) in events {
        if !name.is_empty() {
            out.push_str("event: ");
            out.push_str(name);
            out.push('\n');
        }
        // data may contain newlines (raw JSON does not, but leave room — per SSE spec
        // each line must be prefixed with `data:`)
        for line in data.split('\n') {
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }
    out.into_bytes()
}

/// Echo tool: outputs the `args.msg` field, used to exercise the full
/// "tool_use → tool execution → tool_result written back to history" round-trip within a
/// turn.
pub struct EchoTool {
    schema: ToolSchema,
}

impl EchoTool {
    pub fn new() -> Self {
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

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
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
        completed.content = Some(vec![ToolCallContent::Content(Content::new(text))]);
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(stream::iter(vec![ToolEvent::Completed(completed)]));
        s
    }
}

/// Assembles a [`DefaultAgentCore`] with the given provider and creates a session.
///
/// Uses `OpenPolicy` to skip permission interactions, making it easier to exercise the
/// fast path for ReadOnly tools.
pub async fn build_session(provider: Arc<dyn LlmProvider>, model: &str) -> Arc<dyn Session> {
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(EchoTool::new()))
            .build(),
    );
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(OpenPolicy) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: model.to_string(),
            ..TurnConfig::default()
        })
        .build();
    let cwd = std::env::current_dir().expect("cwd");
    core.create_session(
        SessionId::new(new_session_id()),
        cwd,
        vec![],
        Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
        Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
        Frontend::Headless,
    )
    .await
    .expect("create session")
}

pub fn user_prompt(text: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text(TextContent::new(text))]
}
