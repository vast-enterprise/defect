//! `defect` 二进制入口。
//!
//! v0：根据显式 provider 配置装配 LLM provider，组装 [`DefaultAgentCore`]，
//! 以 stdio 启动 ACP server。

#![warn(clippy::indexing_slicing, clippy::unwrap_used)]
//!
//! Provider 选择：
//! 1. `--provider <name>` 命令行参数
//! 2. `DEFECT_PROVIDER` 环境变量
//! 3. 配置文件
//! 4. 默认 `echo`（无外部依赖，便于无凭证环境冒烟）
//!
//! 取值：`echo` | `anthropic` | `openai` | `deepseek`。
//! 凭证仍由各 provider 从 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
//! `DEEPSEEK_API_KEY` 读取。
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::{
    EnvVariable, HttpHeader, McpServer, McpServerHttp, McpServerSse, McpServerStdio,
};
use clap::{Parser, ValueEnum};
use defect_acp::EchoProvider;
use defect_agent::llm::LlmProvider;
use defect_agent::policy::{
    AskWritesPolicy, DenyAllPolicy, OpenPolicy, ReadOnlyPolicy, SandboxPolicy,
};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, StaticToolRegistry, ToolRegistry, TurnConfig,
};
use defect_config::{
    CliOverrides, HttpClientConfig, HttpProxyMode, HttpProxySettings, LoadConfigOptions,
    LoadedConfig, McpServerConfig as ConfigMcpServerConfig, ProviderKind as ConfigProviderKind,
    SandboxMode, load_dotenv_compat, parse_cli_override,
};
use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
use defect_mcp::McpToolFactory;
use defect_storage::StorageObserver;
use defect_tools::{BashTool, EditFileTool, ReadFileTool, WriteFileTool};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cwd = env::current_dir()?;
    load_dotenv_compat(&cwd).map_err(|e| anyhow::anyhow!("dotenv load failed: {e}"))?;

    let cli = CliArgs::parse();
    let config = defect_config::load_config(LoadConfigOptions {
        cwd,
        cli: cli.to_overrides()?,
        ..LoadConfigOptions::default()
    })
    .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;
    init_tracing(config.effective.tracing.filter.as_deref())?;

    for warning in &config.warnings {
        tracing::warn!("{warning:?}");
    }

    let (provider, turn_config) = build_provider(&config)?;

    tracing::info!(
        provider = %provider.info().vendor,
        model = %turn_config.model,
        "starting defect ACP server on stdio"
    );

    let tools = build_process_tools(&config);
    let storage = Arc::new(StorageObserver::new(default_sessions_root()?));
    let agent = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(build_policy(config.effective.sandbox.mode))
        .observe_session(storage.clone())
        .session_loader(storage)
        .session_tool_factory(Arc::new(McpToolFactory::with_default_servers(
            build_default_mcp_servers(&config),
        )))
        .config(turn_config)
        .build();
    let agent: Arc<dyn AgentCore> = Arc::new(agent);

    defect_acp::serve(agent).await?;
    Ok(())
}

/// Headless agent over ACP/stdio.
#[derive(Debug, Parser)]
#[command(
    name = "defect",
    about = "Headless agent over ACP/stdio",
    long_about = "defect — headless agent over ACP/stdio.\n\n\
                  Auth env: ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY.\n\
                  Logging: RUST_LOG controls tracing-subscriber EnvFilter (default: info)."
)]
struct CliArgs {
    /// LLM provider to use. CLI flag wins over DEFECT_PROVIDER env and config.
    #[arg(long, value_enum, env = "DEFECT_PROVIDER")]
    provider: Option<ProviderKind>,

    /// Override the default model id. CLI flag wins over DEFECT_MODEL env.
    #[arg(long, env = "DEFECT_MODEL")]
    model: Option<String>,

    /// Additional dotted-path config overrides. May be repeated.
    #[arg(long = "config", value_name = "KEY=VALUE")]
    config_override: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderKind {
    Echo,
    Anthropic,
    Openai,
    Deepseek,
}

impl CliArgs {
    fn to_overrides(&self) -> anyhow::Result<CliOverrides> {
        let config_overrides = self
            .config_override
            .iter()
            .map(|spec| parse_cli_override(spec).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(CliOverrides {
            provider: self.provider.map(Into::into),
            model: self.model.clone(),
            config_overrides,
        })
    }
}

fn build_provider(config: &LoadedConfig) -> anyhow::Result<(Arc<dyn LlmProvider>, TurnConfig)> {
    let http_config = build_http_stack_config(&config.effective.http)?;
    let provider: Arc<dyn LlmProvider> = match config.effective.cli.provider {
        ConfigProviderKind::Echo => Arc::new(EchoProvider::new()) as Arc<dyn LlmProvider>,
        ConfigProviderKind::Anthropic => Arc::new(
            AnthropicProvider::new(AnthropicConfig {
                api_key: None,
                base_url: config.effective.providers.anthropic.base_url.clone(),
                http: http_config.clone(),
            })
            .map_err(|e| anyhow::anyhow!("anthropic provider init failed: {e}"))?,
        ) as Arc<dyn LlmProvider>,
        ConfigProviderKind::Openai => Arc::new(
            OpenAiProvider::new(OpenAiConfig {
                api_key: None,
                base_url: config.effective.providers.openai.base_url.clone(),
                organization: config.effective.providers.openai.organization.clone(),
                project: config.effective.providers.openai.project.clone(),
                capabilities_override: None,
                http: http_config.clone(),
            })
            .map_err(|e| anyhow::anyhow!("openai provider init failed: {e}"))?,
        ) as Arc<dyn LlmProvider>,
        ConfigProviderKind::Deepseek => Arc::new(
            DeepSeekProvider::new(DeepSeekConfig {
                api_key: None,
                base_url: config.effective.providers.deepseek.base_url.clone(),
                http: http_config,
            })
            .map_err(|e| anyhow::anyhow!("deepseek provider init failed: {e}"))?,
        ) as Arc<dyn LlmProvider>,
    };

    Ok((provider, config.effective.turn.clone()))
}

/// 把 `defect-config` 的 typed 配置翻译成 `defect_http::HttpStackConfig`。
///
/// `defect-config` 不直接依赖 `defect-http` 是为了保持 crate 单向依赖
/// （详见 `defect_config::HttpClientConfig` 注释），翻译动作放在 CLI 装配
/// 期最自然——同一份 stack config 三家 provider 共用，proxy URI 解析失败
/// 在这里集中报错。
fn build_http_stack_config(
    config: &HttpClientConfig,
) -> anyhow::Result<defect_http::HttpStackConfig> {
    use std::time::Duration;

    let mut stack = defect_http::HttpStackConfig::default();
    if let Some(ms) = config.total_timeout_ms {
        stack.total_timeout = if ms == 0 {
            None
        } else {
            Some(Duration::from_millis(ms))
        };
    }
    if let Some(retries) = config.transport_retries {
        stack.transport_retries = retries;
    }
    if let Some(ms) = config.initial_backoff_ms {
        stack.initial_backoff = Duration::from_millis(ms);
    }
    if let Some(ua) = &config.user_agent {
        stack.user_agent = Some(ua.clone());
    }
    stack.proxy = match config.proxy.mode {
        HttpProxyMode::FromEnv => defect_http::ProxyConfig::FromEnv,
        HttpProxyMode::Disabled => defect_http::ProxyConfig::Disabled,
        HttpProxyMode::Explicit => {
            defect_http::ProxyConfig::Explicit(parse_proxy_settings(&config.proxy.explicit)?)
        }
    };
    Ok(stack)
}

fn parse_proxy_settings(
    settings: &HttpProxySettings,
) -> anyhow::Result<defect_http::ProxySettings> {
    let parse_uri = |raw: &str, field: &str| -> anyhow::Result<http::Uri> {
        raw.parse::<http::Uri>()
            .map_err(|e| anyhow::anyhow!("invalid http.proxy.{field} `{raw}`: {e}"))
    };
    Ok(defect_http::ProxySettings {
        http_proxy: settings
            .http_proxy
            .as_deref()
            .map(|raw| parse_uri(raw, "http_proxy"))
            .transpose()?,
        https_proxy: settings
            .https_proxy
            .as_deref()
            .map(|raw| parse_uri(raw, "https_proxy"))
            .transpose()?,
        no_proxy: settings.no_proxy.clone(),
    })
}

fn build_process_tools(config: &LoadedConfig) -> Arc<dyn ToolRegistry> {
    Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(BashTool::from_config(
                &config.effective.tools.bash,
            )))
            .insert(Arc::new(ReadFileTool::from_config(
                &config.effective.tools.fs,
            )))
            .insert(Arc::new(WriteFileTool::new()))
            .insert(Arc::new(EditFileTool::new()))
            .build(),
    )
}

fn build_default_mcp_servers(config: &LoadedConfig) -> Vec<McpServer> {
    config
        .effective
        .mcp
        .enabled_servers
        .iter()
        .filter_map(|name| {
            let server = config.effective.mcp.servers.get(name)?;
            Some(match server {
                ConfigMcpServerConfig::Stdio(server) => McpServer::Stdio(
                    McpServerStdio::new(name, PathBuf::from(&server.command))
                        .args(server.args.clone())
                        .env(
                            server
                                .env
                                .iter()
                                .map(|(name, value)| EnvVariable::new(name, value))
                                .collect(),
                        ),
                ),
                ConfigMcpServerConfig::Http(server) => McpServer::Http(
                    McpServerHttp::new(name, &server.url).headers(
                        server
                            .headers
                            .iter()
                            .map(|(name, value)| HttpHeader::new(name, value))
                            .collect(),
                    ),
                ),
                ConfigMcpServerConfig::Sse(server) => McpServer::Sse(
                    McpServerSse::new(name, &server.url).headers(
                        server
                            .headers
                            .iter()
                            .map(|(name, value)| HttpHeader::new(name, value))
                            .collect(),
                    ),
                ),
            })
        })
        .collect()
}

fn build_policy(mode: SandboxMode) -> Arc<dyn SandboxPolicy> {
    match mode {
        SandboxMode::ReadOnly => Arc::new(ReadOnlyPolicy),
        SandboxMode::AskWrites => Arc::new(AskWritesPolicy::new()),
        SandboxMode::Open => Arc::new(OpenPolicy),
        SandboxMode::DenyAll => Arc::new(DenyAllPolicy),
    }
}

fn default_sessions_root() -> anyhow::Result<std::path::PathBuf> {
    if let Ok(xdg_state_home) = env::var("XDG_STATE_HOME") {
        return Ok(std::path::PathBuf::from(xdg_state_home).join("defect/sessions"));
    }
    if let Ok(home) = env::var("HOME") {
        return Ok(std::path::PathBuf::from(home).join(".local/state/defect/sessions"));
    }
    Err(anyhow::anyhow!(
        "cannot resolve session storage root: neither XDG_STATE_HOME nor HOME is set"
    ))
}

impl From<ProviderKind> for ConfigProviderKind {
    fn from(value: ProviderKind) -> Self {
        match value {
            ProviderKind::Echo => Self::Echo,
            ProviderKind::Anthropic => Self::Anthropic,
            ProviderKind::Openai => Self::Openai,
            ProviderKind::Deepseek => Self::Deepseek,
        }
    }
}

fn init_tracing(filter: Option<&str>) -> anyhow::Result<()> {
    let default_filter = filter.unwrap_or("info,toac=warn");
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))
}

#[cfg(test)]
mod test;
