//! `defect` 二进制入口。
//!
//! v0：装配 [`EchoProvider`] + 空工具注册表 + [`DefaultAgentCore`]，
//! 以 stdio 启动 ACP server。真实 provider / 工具集后续在 `defect-llm`
//! 与 `defect-tools` 接入后替换装配。

use std::sync::Arc;

use defect_acp::EchoProvider;
use defect_agent::session::{AgentCore, DefaultAgentCore, TurnConfig};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 默认到 stderr——stdio ACP 占用 stdout，日志走 stderr 才不会污染线协议。
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_writer(std::io::stderr)
        .init();

    let provider = Arc::new(EchoProvider::new());
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent = DefaultAgentCore::builder()
        .provider(provider)
        .config(config)
        .build();
    let agent: Arc<dyn AgentCore> = Arc::new(agent);

    defect_acp::serve(agent).await?;
    Ok(())
}
