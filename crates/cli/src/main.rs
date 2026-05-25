//! `defect` 二进制入口。
//!
//! v0：根据显式 provider 配置装配 LLM provider，组装 [`DefaultAgentCore`]，
//! 以 stdio 启动 ACP server。
//!
//! Provider 选择（**显式**，不嗅探 API_KEY env）：
//! 1. `--provider <name>` 命令行参数
//! 2. `DEFECT_PROVIDER` 环境变量
//! 3. 默认 `echo`（无外部依赖，便于无凭证环境冒烟）
//!
//! 取值：`echo` | `anthropic` | `openai` | `deepseek`。
//! 凭证仍由各 provider 从 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
//! `DEEPSEEK_API_KEY` 读取，但只用于鉴权，不参与"选哪家"。

use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use defect_acp::EchoProvider;
use defect_agent::llm::LlmProvider;
use defect_agent::session::{AgentCore, DefaultAgentCore, TurnConfig};
use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
use tracing_subscriber::EnvFilter;

const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
const DEFAULT_DEEPSEEK_MODEL: &str = "deepseek-chat";
const DEFAULT_ECHO_MODEL: &str = "echo";

const PROVIDER_ENV: &str = "DEFECT_PROVIDER";
const MODEL_ENV: &str = "DEFECT_MODEL";
const DEFAULT_PROVIDER: &str = "echo";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // .env 必须在 tracing_subscriber init 前加载——否则 RUST_LOG 这类 env 不生效。
    // 已在进程 env 里的同名变量优先（user-set wins），避免文件覆盖 shell 显式覆盖。
    load_env_file(Path::new(".env"));

    // 默认到 stderr——stdio ACP 占用 stdout，日志走 stderr 才不会污染线协议。
    // `toac=warn` 默认 silence——toac wire crate 的 INFO 级 request 事件含
    // authorization header 明文（详见 docs/outbound/tracing.md §5.2）。
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,toac=warn")),
        )
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();

    let cli = CliArgs::parse(env::args().skip(1))?;
    let (provider, model) = build_provider(&cli)?;

    tracing::info!(
        provider = %provider.info().vendor,
        model = %model,
        "starting defect ACP server on stdio"
    );

    let config = TurnConfig {
        model,
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

#[derive(Debug, Default)]
struct CliArgs {
    provider: Option<String>,
    model: Option<String>,
}

impl CliArgs {
    fn parse<I: IntoIterator<Item = String>>(args: I) -> anyhow::Result<Self> {
        let mut out = CliArgs::default();
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--provider" => {
                    out.provider = Some(iter.next().ok_or_else(|| {
                        anyhow::anyhow!("--provider requires a value (echo|anthropic|openai|deepseek)")
                    })?);
                }
                s if s.starts_with("--provider=") => {
                    out.provider = Some(s["--provider=".len()..].to_string());
                }
                "--model" => {
                    out.model = Some(
                        iter.next()
                            .ok_or_else(|| anyhow::anyhow!("--model requires a value"))?,
                    );
                }
                s if s.starts_with("--model=") => {
                    out.model = Some(s["--model=".len()..].to_string());
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    anyhow::bail!("unknown argument: {other}");
                }
            }
        }
        Ok(out)
    }
}

fn print_help() {
    eprintln!(
        "defect — headless agent over ACP/stdio\n\n\
         Usage: defect [OPTIONS]\n\n\
         Options:\n\
           --provider <name>   echo | anthropic | openai | deepseek (default: echo)\n\
                               Also configurable via DEFECT_PROVIDER env var.\n\
           --model <id>        Override the default model id\n\
                               Also configurable via DEFECT_MODEL env var.\n\
           -h, --help          Show this help\n\n\
         Environment:\n\
           DEFECT_PROVIDER     Same as --provider; CLI flag wins\n\
           DEFECT_MODEL        Same as --model; CLI flag wins\n\
           ANTHROPIC_API_KEY   Auth for the anthropic provider\n\
           OPENAI_API_KEY      Auth for the openai provider\n\
           DEEPSEEK_API_KEY    Auth for the deepseek provider\n\
           RUST_LOG            tracing-subscriber EnvFilter (default: info)"
    );
}

fn build_provider(cli: &CliArgs) -> anyhow::Result<(Arc<dyn LlmProvider>, String)> {
    let kind = cli
        .provider
        .clone()
        .or_else(|| env_nonempty(PROVIDER_ENV))
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());

    let model_override = cli.model.clone().or_else(|| env_nonempty(MODEL_ENV));

    match kind.as_str() {
        "echo" => Ok((
            Arc::new(EchoProvider::new()) as Arc<dyn LlmProvider>,
            model_override.unwrap_or_else(|| DEFAULT_ECHO_MODEL.to_string()),
        )),
        "anthropic" => {
            let provider = AnthropicProvider::new(AnthropicConfig::from_env())
                .map_err(|e| anyhow::anyhow!("anthropic provider init failed: {e}"))?;
            Ok((
                Arc::new(provider) as Arc<dyn LlmProvider>,
                model_override.unwrap_or_else(|| DEFAULT_ANTHROPIC_MODEL.to_string()),
            ))
        }
        "openai" => {
            let provider = OpenAiProvider::new(OpenAiConfig::from_env())
                .map_err(|e| anyhow::anyhow!("openai provider init failed: {e}"))?;
            Ok((
                Arc::new(provider) as Arc<dyn LlmProvider>,
                model_override.unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string()),
            ))
        }
        "deepseek" => {
            let provider = DeepSeekProvider::new(DeepSeekConfig::from_env())
                .map_err(|e| anyhow::anyhow!("deepseek provider init failed: {e}"))?;
            Ok((
                Arc::new(provider) as Arc<dyn LlmProvider>,
                model_override.unwrap_or_else(|| DEFAULT_DEEPSEEK_MODEL.to_string()),
            ))
        }
        other => anyhow::bail!(
            "unknown provider {other:?}; expected one of echo|anthropic|openai|deepseek"
        ),
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.is_empty())
}

/// 极简 `.env` 加载器：`KEY=VALUE` 一行一条，`#` 开头注释、空行跳过；
/// 支持外层 `"..."` / `'...'` 包裹去除。**已在进程 env 里的变量保留原值**，
/// 避免 .env 覆盖 shell 显式 export。读不到文件 / 解析失败仅 warn。
fn load_env_file(path: &Path) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            eprintln!("warning: failed to read {}: {err}", path.display());
            return;
        }
    };

    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            eprintln!(
                "warning: {}:{} skipped: missing '=' in {line:?}",
                path.display(),
                lineno + 1
            );
            continue;
        };
        let key = key.trim();
        let value = strip_quotes(value.trim());
        if key.is_empty() {
            continue;
        }
        // 已显式 set 的不动；空字符串视作 unset。
        if env::var_os(key).is_some() {
            continue;
        }
        // SAFETY: 进入 main 前未起任何 spawn / signal handler，单线程读 env 安全。
        unsafe {
            env::set_var(key, value);
        }
    }
}

fn strip_quotes(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}
