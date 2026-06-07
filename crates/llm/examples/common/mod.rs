//! Common skeleton for smoke-test scripts. **For `examples/` only** — not part of the
//! crate's public API.
//!
//! Design decisions:
//! - Credentials are read from the environment; the command line only selects a scenario
//!   or overrides the model.
//! - `tracing` defaults to `INFO`; use `RUST_LOG=defect_llm=debug` to increase verbosity.
//! - Each scenario prints a one-line `=== PASS / FAIL ===` summary for easy visual
//!   scanning.
//!
//! See LLM provider integration tests for the list of "real-endpoint smoke" tests.

use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, SessionId, StopReason as AcpStopReason, TextContent, ToolCallContent,
    ToolCallUpdateFields,
};
use defect_agent::event::AgentEvent;
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::{LlmProvider, SamplingParams, ThinkingConfig, Usage};
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
use futures::{StreamExt, stream};
use serde_json::json;

/// Process exit codes: 0 for all PASS (including SKIP), 1 for at least one FAIL.
/// SKIP is not considered a failure — auto-skipping on unsupported models should not turn
/// CI red.
pub const EXIT_OK: i32 = 0;
pub const EXIT_FAIL: i32 = 1;

/// Set up tracing: default `info,toac=warn`, overridden entirely by the `RUST_LOG`
/// environment variable.
///
/// `toac=warn` silences the `toac` wire crate's INFO-level request events by default.
/// Use `RUST_LOG=...,toac=debug` to enable debug logging for the wire crate.
///
/// # Panics
///
/// Panics if called more than once — examples call this only once in `main`.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,toac=warn"));
    fmt().with_env_filter(filter).with_target(true).init();
}

/// Reads a string from the environment; returns `None` if the variable is unset or empty.
pub fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// A simple echo tool: `args.msg` → text. In a tool-use scenario, it exercises the full
/// tool_use → tool execution → tool_result round trip.
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

/// Assemble agent core + session: inject an echo tool, with permissions governed by
/// [`OpenPolicy`]; directly expose it (smoke tests skip permission checks).
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

/// Runs a single turn, printing emitted events to stdout in real time, and returns
/// `(stop_reason, concatenated text)`.
pub async fn run_turn_and_print(
    session: Arc<dyn Session>,
    prompt: &str,
) -> Result<(AcpStopReason, String, ToolHits), Box<dyn std::error::Error>> {
    let (stop, text, hits, _usage) = run_turn_and_print_with_usage(session, prompt).await?;
    Ok((stop, text, hits))
}

pub async fn run_turn_and_print_with_usage(
    session: Arc<dyn Session>,
    prompt: &str,
) -> Result<(AcpStopReason, String, ToolHits, Usage), Box<dyn std::error::Error>> {
    let mut events = session.subscribe();
    let prompt_blocks = vec![ContentBlock::Text(TextContent::new(prompt))];

    let session_for_turn = session.clone();
    let prompt_owned = prompt_blocks.clone();
    let turn = tokio::spawn(async move { session_for_turn.run_turn(prompt_owned).await });

    let mut text_buf = String::new();
    let mut thought_buf = String::new();
    let mut hits = ToolHits::default();
    let mut usage = Usage::default();

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
            AgentEvent::TurnEnded {
                reason,
                usage: turn_usage,
            } => {
                println!("\n[turn ended] reason={reason:?} usage={turn_usage:?}");
                usage = turn_usage;
                break;
            }
            _ => {}
        }
    }

    let stop = turn.await??;
    hits.thought_text = thought_buf;
    Ok((stop, text_buf, hits, usage))
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
            && let agent_client_protocol_schema::ContentBlock::Text(t) = &inner.content
        {
            return Some(t.text.clone());
        }
    }
    None
}

/// Build sampling params with thinking enabled. Anthropic accepts a budget; OpenAI
/// o-series only uses reasoning_effort.
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

/// The first positional CLI argument: scenario name (default `all`).
pub fn scenario_from_args() -> String {
    std::env::args().nth(1).unwrap_or_else(|| "all".to_string())
}
