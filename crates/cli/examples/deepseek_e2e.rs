//! 端到端冒烟：用 [`Channel::duplex`] 在进程内对接一个 ACP 客户端 / 服务端，
//! 服务端走我们 CLI 同款的装配（`--provider deepseek`），让 prompt 真正走到
//! DeepSeek `/v1/chat/completions` 并把流式 `agent_message_chunk` 打回客户端。
//!
//! 用法：
//!
//! ```bash
//! # 凭证从 .env 读（仓库根目录）；或 export DEEPSEEK_API_KEY=...
//! cargo run -p defect-cli --example deepseek_e2e
//! ```

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    ContentBlock, InitializeRequest, ProtocolVersion, SessionNotification, SessionUpdate,
};
use agent_client_protocol::{Channel, Client, SessionMessage};
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
    // OpenPolicy 让 bash 在 e2e 里直放——这是 smoke 脚本，不测权限交互。
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

    // server 用 channel_b（agent 视角），client 用 channel_a（client 视角）。
    let (channel_a, channel_b) = Channel::duplex();

    // 把 server task spawn 出去；client driver 跑完后让它自然退出。
    let server_handle = tokio::spawn(serve_on(agent, channel_b));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();

    let stop_reason = Client
        .builder()
        .name("deepseek-e2e-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                // 实时把 chunk 打到 stdout，方便肉眼看流式有没有真的在出。
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
                            Some(agent_client_protocol::schema::ToolCallStatus::Completed)
                                | Some(agent_client_protocol::schema::ToolCallStatus::Failed)
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
