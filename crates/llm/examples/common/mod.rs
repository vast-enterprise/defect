//! 冒烟脚本的公用骨架。**仅供 examples/ 用**，不进 crate 公共 API。
//!
//! 设计取舍：
//! - 凭证从 env 读，命令行只挑 scenario / 覆盖 model
//! - tracing 默认 INFO，能用 `RUST_LOG=defect_llm=debug` 拨亮
//! - 每个 scenario 走完打一行 `=== PASS / FAIL ===` 摘要，便于肉眼扫
//!
//! 见 `docs/outbound/llm-anthropic.md` §10 / `docs/outbound/llm-openai.md` §9
//! 末尾的"真端点 smoke"清单。

use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol::schema::{
    Content, ContentBlock, StopReason as AcpStopReason, TextContent, ToolCallContent,
    ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::llm::{LlmProvider, SamplingParams, ThinkingConfig};
use defect_agent::policy::{OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, Session, StaticToolRegistry, ToolRegistry, TurnConfig,
};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};
use futures::{StreamExt, stream};
use serde_json::json;

/// 进程退出码：0 全部 PASS（含 SKIP）/ 1 至少一个 FAIL。
/// SKIP 不视作失败——thinking 在不支持的模型上自动 skip 不该让 CI 红。
pub const EXIT_OK: i32 = 0;
pub const EXIT_FAIL: i32 = 1;

/// 装好 tracing：默认 `info,toac=warn`，环境变量 `RUST_LOG` 整体覆盖。
///
/// `toac=warn` 默认 silence——toac wire crate 的 INFO request 事件含
/// authorization header 明文（详见 `docs/outbound/tracing.md` §5.2）。
/// 调试 wire 时显式 `RUST_LOG=...,toac=debug`。
///
/// # Panics
///
/// 重复初始化会 panic——examples 只在 main 调一次。
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,toac=warn"));
    fmt().with_env_filter(filter).with_target(true).init();
}

/// 从 env 读字符串；为空 / 不存在时返回 None。
pub fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// 简易回显工具：args.msg → text。tool-use scenario 用它走 tool_use →
/// 工具执行 → tool_result 闭环。
pub struct EchoTool {
    schema: ToolSchema,
}

impl EchoTool {
    pub fn new() -> Self {
        Self {
            schema: ToolSchema {
                name: "echo".to_string(),
                description: "Echoes the `msg` field back. Call this tool to confirm tool wiring."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "msg": {"type": "string", "description": "Text to echo back"}
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

    fn describe(&self, _args: &serde_json::Value) -> ToolCallDescription {
        let mut fields = ToolCallUpdateFields::default();
        fields.title = Some("echo".to_string());
        ToolCallDescription { fields }
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

/// 装配 agent core + session：注入回显工具，权限走 [`OpenPolicy`]
/// 直放（冒烟不测权限交互）。
pub async fn build_session(
    provider: Arc<dyn LlmProvider>,
    model: &str,
    sampling: SamplingParams,
) -> Arc<dyn Session> {
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
            sampling,
            ..TurnConfig::default()
        })
        .build();
    let cwd = std::env::current_dir().expect("cwd");
    core.create_session(cwd, vec![])
        .await
        .expect("create session")
}

/// 跑一个 turn，把 emit 的事件实时打到 stdout，最后回 (stop_reason, 文本拼接)。
pub async fn run_turn_and_print(
    session: Arc<dyn Session>,
    prompt: &str,
) -> Result<(AcpStopReason, String, ToolHits), Box<dyn std::error::Error>> {
    let mut events = session.subscribe();
    let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt))];

    let session_for_turn = session.clone();
    let prompt_owned = prompt_blocks.clone();
    let turn = tokio::spawn(async move { session_for_turn.run_turn(prompt_owned).await });

    let mut text_buf = String::new();
    let mut thought_buf = String::new();
    let mut hits = ToolHits::default();

    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::AssistantText {
                content: ContentBlock::Text(t),
            } => {
                print!("{}", t.text);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                text_buf.push_str(&t.text);
            }
            AgentEvent::AssistantThought {
                content: ContentBlock::Text(t),
            } => {
                eprint!("\x1b[90m{}\x1b[0m", t.text); // gray
                let _ = std::io::Write::flush(&mut std::io::stderr());
                thought_buf.push_str(&t.text);
            }
            AgentEvent::ToolCallStarted { fields, .. } => {
                hits.started += 1;
                let title = fields
                    .title
                    .clone()
                    .unwrap_or_else(|| "<no-title>".to_string());
                println!("\n[tool started] {title}");
            }
            AgentEvent::ToolCallFinished { fields, .. } => {
                hits.finished += 1;
                let summary = first_text_content(&fields).unwrap_or_default();
                println!("[tool finished] {summary}");
            }
            AgentEvent::TurnEnded { reason, usage } => {
                println!("\n[turn ended] reason={reason:?} usage={usage:?}");
                break;
            }
            _ => {}
        }
    }

    let stop = turn.await??;
    hits.thought_text = thought_buf;
    Ok((stop, text_buf, hits))
}

#[derive(Debug, Default)]
pub struct ToolHits {
    pub started: u32,
    pub finished: u32,
    pub thought_text: String,
}

fn first_text_content(fields: &ToolCallUpdateFields) -> Option<String> {
    let content = fields.content.as_ref()?;
    for c in content {
        if let ToolCallContent::Content(inner) = c
            && let agent_client_protocol::schema::ContentBlock::Text(t) = &inner.content
        {
            return Some(t.text.clone());
        }
    }
    None
}

/// 把 thinking enabled 的 sampling params 装好。Anthropic 接受 budget；
/// OpenAI o-系列只看 reasoning_effort。
pub fn sampling_with_thinking(budget_tokens: Option<u32>) -> SamplingParams {
    SamplingParams {
        thinking: ThinkingConfig::Enabled { budget_tokens },
        ..SamplingParams::default()
    }
}

pub fn print_pass(label: &str) {
    println!("\n=== PASS: {label} ===");
}

pub fn print_fail(label: &str, err: &dyn std::fmt::Display) {
    eprintln!("\n=== FAIL: {label}: {err} ===");
}

pub fn print_skip(label: &str, reason: &str) {
    println!("\n=== SKIP: {label}: {reason} ===");
}

/// 命令行第一个位置参数：scenario 名（默认 `all`）。
pub fn scenario_from_args() -> String {
    std::env::args().nth(1).unwrap_or_else(|| "all".to_string())
}
