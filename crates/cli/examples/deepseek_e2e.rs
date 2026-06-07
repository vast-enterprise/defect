//! End-to-end smoke test: uses [`Channel::duplex`] to connect an ACP client and server
//! in-process. The server is assembled the same way as our CLI (`--provider deepseek`),
//! so the prompt actually reaches DeepSeek `/v1/chat/completions` and streams
//! `agent_message_chunk` back to the client.
//!
//! Usage:
//!
//! ```bash
//! # Credentials are read from .env (repo root); or export DEEPSEEK_API_KEY=...
//! cargo run -p defect-cli --example deepseek_e2e
//! ```

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol::{Channel, Client, SessionMessage};
use agent_client_protocol_schema::{
    ContentBlock, InitializeRequest, ProtocolVersion, SessionNotification, SessionUpdate,
};
use defect_acp::serve_on;
use defect_agent::llm::LlmProvider;
use defect_agent::policy::{OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, StaticToolRegistry, ToolRegistry, TurnConfig,
};
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
use defect_tools::BashTool;

const DEFAULT_PROMPT: &str = "Say hello in one short sentence, then stop.";
const MODEL: &str = "deepseek-v4-flash";

fn prompt_text() -> String {
    std::env::var("DEEPSEEK_E2E_PROMPT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::load_env_file(Path::new(".env"));
    common::init_tracing();

    let provider = DeepSeekProvider::new(DeepSeekConfig::from_env())
        .map_err(|e| anyhow::anyhow!("deepseek provider init failed: {e}"))?;
    let provider: Arc<dyn LlmProvider> = Arc::new(provider);

    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(BashTool::new()))
            .build(),
    );
    // OpenPolicy allows bash to pass through directly in e2e — this is a smoke test, not
    // testing permission interactions.
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(OpenPolicy) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: MODEL.to_string(),
            ..TurnConfig::default()
        })
        .build();
    let agent: Arc<dyn AgentCore> = Arc::new(core);

    // The server uses `channel_b` (agent's perspective), and the client uses `channel_a`
    // (client's perspective).
    let (channel_a, channel_b) = Channel::duplex();

    // Spawn the server task; let it exit naturally after the client driver finishes.
    let server_handle = tokio::spawn(serve_on(agent, channel_b));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();

    let stop_reason = Client
        .builder()
        .name("deepseek-e2e-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                // Stream chunks to stdout so you can visually verify streaming is
                // working.
                match &notif.update {
                    SessionUpdate::AgentMessageChunk(chunk) => {
                        if let ContentBlock::Text(t) = &chunk.content {
                            print!("{}", t.text);
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                        }
                    }
                    SessionUpdate::ToolCall(tc) => {
                        let title = tc.title.clone();
                        eprintln!("\n[tool start] {title}");
                    }
                    SessionUpdate::ToolCallUpdate(upd) => {
                        if matches!(
                            upd.fields.status,
                            Some(agent_client_protocol_schema::ToolCallStatus::Completed)
                                | Some(agent_client_protocol_schema::ToolCallStatus::Failed)
                        ) {
                            eprintln!("[tool end]   status={:?}", upd.fields.status);
                        }
                    }
                    _ => {}
                }
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(channel_a, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let cwd = std::env::current_dir()
                .map_err(|e| agent_client_protocol::Error::internal_error().data(e.to_string()))?;
            let mut session = cx.build_session(cwd).block_task().start_session().await?;

            session.send_prompt(prompt_text())?;
            loop {
                match session.read_update().await? {
                    SessionMessage::SessionMessage(_) => {}
                    SessionMessage::StopReason(stop_reason) => break Ok(stop_reason),
                    _ => {}
                }
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("client connection failed: {e}"))?;

    server_handle.abort();
    let _ = server_handle.await;

    println!();
    println!("\n=== stop_reason = {stop_reason:?} ===");
    let updates = updates.lock().expect("updates mutex");
    let assistant_text: String = updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(chunk) => Some(&chunk.content),
            _ => None,
        })
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    if assistant_text.trim().is_empty() {
        anyhow::bail!("expected at least one AgentMessageChunk; got none");
    }
    Ok(())
}
