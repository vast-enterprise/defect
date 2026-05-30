//! e2e 集测的公共脚手架：mock server 启动、回显工具、agent core 装配。
//!
//! 由 `anthropic_e2e.rs` / `openai_e2e.rs` 通过 `mod common;` 引入。

use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, SessionId, TextContent, ToolCallContent, ToolCallUpdateFields,
};
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::LlmProvider;
use defect_agent::policy::{OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, Session, StaticToolRegistry, ToolRegistry, TurnConfig,
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

/// 启动一个本地 wiremock server。
///
/// `wiremock` 的 server 在 drop 时自动停掉，调用方持有它的整个测试期。
pub async fn start_mock_server() -> MockServer {
    MockServer::start().await
}

/// 把 SSE 事件序列拼成 wire 格式（`event:` + `data:` + 空行）的字节流，
/// 直接喂给 [`wiremock::ResponseTemplate::set_body_raw`]。
pub fn encode_sse_events(events: &[(&str, &str)]) -> Vec<u8> {
    let mut out = String::new();
    for (name, data) in events {
        if !name.is_empty() {
            out.push_str("event: ");
            out.push_str(name);
            out.push('\n');
        }
        // data 可能含换行（裸 JSON 不会，但留余地——按 SSE 规范每行 data:）
        for line in data.split('\n') {
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }
    out.into_bytes()
}

/// 回显工具：把 `args.msg` 字段当作输出，用来在 turn 里走完整的
/// "tool_use → tool 执行 → tool_result 回写历史" 闭环。
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

/// 用给定 provider 装配一个 [`DefaultAgentCore`]，并创建一个 session。
///
/// 用 `OpenPolicy` 跳过权限交互，便于跑通 ReadOnly 工具的快路径。
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
    )
    .await
    .expect("create session")
}

pub fn user_prompt(text: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text(TextContent::new(text))]
}
