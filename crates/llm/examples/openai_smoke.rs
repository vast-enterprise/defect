//! OpenAI 兼容 provider 真端点冒烟。
//!
//! 用法：
//!
//! ```bash
//! OPENAI_API_KEY=sk-... \
//!   cargo run -p defect-llm --example openai_smoke -- [scenario]
//! ```
//!
//! `[scenario]` ∈ `list-models | text | tool | thinking | all`，默认 `all`。
//!
//! 可选 env：
//! - `OPENAI_BASE_URL`：覆盖默认 `https://api.openai.com/v1`（也可指 DeepSeek 等兼容端点）
//! - `OPENAI_MODEL`：覆盖默认模型 `gpt-4o-mini`
//! - `OPENAI_ORG` / `OPENAI_PROJECT`：可选组织 / 项目标识
//! - `RUST_LOG=defect_llm=debug` 打开协议层调试日志
//!
//! `thinking` 场景仅在模型 ID 看起来支持 reasoning（`o1*` / `o3*` / `o4*` /
//! `*reasoner*` / `*reasoning*`）时跑，否则自动 skip。

mod common;

use std::sync::Arc;

use agent_client_protocol_schema::StopReason as AcpStopReason;
use defect_agent::llm::{LlmProvider, SamplingParams};
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};

use common::{
    EXIT_FAIL, EXIT_OK, build_session, env_string, init_tracing, print_fail, print_pass,
    print_skip, run_turn_and_print, sampling_with_thinking, scenario_from_args,
};

const DEFAULT_MODEL: &str = "gpt-4o-mini";
const THINKING_BUDGET_TOKENS: u32 = 2048;

const TEXT_PROMPT: &str = "Say hello in one short sentence.";
const TOOL_PROMPT: &str = "Please call the `echo` tool with msg=\"hello from smoke\", \
     then briefly summarize what the tool returned.";
const THINKING_PROMPT: &str = "Think step by step: a farmer has 17 sheep and all but 9 die. How many are left? \
     Show your reasoning briefly, then give the final number.";

#[tokio::main]
async fn main() {
    init_tracing();

    let api_key = match env_string("OPENAI_API_KEY") {
        Some(k) => k,
        None => {
            eprintln!("OPENAI_API_KEY is required for openai_smoke");
            std::process::exit(EXIT_FAIL);
        }
    };
    let base_url = env_string("OPENAI_BASE_URL");
    let organization = env_string("OPENAI_ORG");
    let project = env_string("OPENAI_PROJECT");
    let model = env_string("OPENAI_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let provider: Arc<dyn LlmProvider> = match OpenAiProvider::new(OpenAiConfig {
        api_key: Some(api_key),
        api_key_env: None,
        base_url,
        organization,
        project,
        vendor: "openai".to_string(),
        display_name: "OpenAI Chat Completions".to_string(),
        headers: std::collections::HashMap::new(),
        capabilities_override: None,
        reasoning_effort: None,
        chat_dialect: defect_llm::protocol::openai_chat::ChatDialect::OpenAi,
        http: defect_http::HttpStackConfig::default(),
    }) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("provider init failed: {e}");
            std::process::exit(EXIT_FAIL);
        }
    };

    let scenario = scenario_from_args();
    println!("=== openai smoke: scenario={scenario} model={model} ===");

    let mut failed = 0u32;
    let mut ran = 0u32;

    for label in scenarios_for(&scenario) {
        ran += 1;
        let outcome = run_scenario(label, provider.clone(), &model).await;
        match outcome {
            ScenarioOutcome::Pass => print_pass(label),
            ScenarioOutcome::Skip(reason) => print_skip(label, &reason),
            ScenarioOutcome::Fail(err) => {
                failed += 1;
                print_fail(label, &err);
            }
        }
    }

    println!("\n=== openai smoke done: ran={ran} failed={failed} ===");
    if failed > 0 {
        std::process::exit(EXIT_FAIL);
    } else {
        std::process::exit(EXIT_OK);
    }
}

fn scenarios_for(name: &str) -> Vec<&'static str> {
    match name {
        "list-models" => vec!["list-models"],
        "text" => vec!["text"],
        "tool" => vec!["tool"],
        "thinking" => vec!["thinking"],
        _ => vec!["list-models", "text", "tool", "thinking"],
    }
}

/// OpenAI 系：仅 reasoning-capable 模型才走 thinking 路径。
/// gpt-4o / gpt-4 / gpt-3.5 等纯 chat 模型不支持，跑了只会 400。
fn model_supports_thinking(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.contains("reasoner")
        || m.contains("reasoning")
}

enum ScenarioOutcome {
    Pass,
    Skip(String),
    Fail(String),
}

async fn run_scenario(label: &str, provider: Arc<dyn LlmProvider>, model: &str) -> ScenarioOutcome {
    println!("\n--- running: {label} ---");
    let res = match label {
        "list-models" => scenario_list_models(provider).await,
        "text" => scenario_text(provider, model).await,
        "tool" => scenario_tool(provider, model).await,
        "thinking" => scenario_thinking(provider, model).await,
        other => Err(format!("unknown scenario {other}")),
    };
    match res {
        Ok(None) => ScenarioOutcome::Pass,
        Ok(Some(reason)) => ScenarioOutcome::Skip(reason),
        Err(e) => ScenarioOutcome::Fail(e),
    }
}

async fn scenario_list_models(provider: Arc<dyn LlmProvider>) -> Result<Option<String>, String> {
    let models = provider.list_models().await.map_err(|e| e.to_string())?;
    if models.is_empty() {
        return Err("list_models returned empty".to_string());
    }
    println!("got {} models, first 5:", models.len());
    for m in models.iter().take(5) {
        println!(
            "  - {} ({})",
            m.id,
            m.display_name.as_deref().unwrap_or("-")
        );
    }
    Ok(None)
}

async fn scenario_text(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    let session = build_session(provider, model, SamplingParams::default()).await;
    let (stop, text, _hits) = run_turn_and_print(session, TEXT_PROMPT)
        .await
        .map_err(|e| e.to_string())?;
    if !matches!(stop, AcpStopReason::EndTurn) {
        return Err(format!("unexpected stop reason: {stop:?}"));
    }
    if text.trim().is_empty() {
        return Err("empty assistant text".to_string());
    }
    Ok(None)
}

async fn scenario_tool(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    let session = build_session(provider, model, SamplingParams::default()).await;
    let (stop, _text, hits) = run_turn_and_print(session, TOOL_PROMPT)
        .await
        .map_err(|e| e.to_string())?;
    if !matches!(stop, AcpStopReason::EndTurn) {
        return Err(format!("unexpected stop reason: {stop:?}"));
    }
    if hits.started == 0 || hits.finished == 0 {
        return Err(format!(
            "expected at least one tool call (started={}, finished={})",
            hits.started, hits.finished
        ));
    }
    Ok(None)
}

async fn scenario_thinking(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    if !model_supports_thinking(model) {
        return Ok(Some(format!(
            "model {model} does not look like a reasoning model; \
             set OPENAI_MODEL=o3-mini or similar to exercise this path"
        )));
    }
    let sampling = sampling_with_thinking(Some(THINKING_BUDGET_TOKENS));
    let session = build_session(provider, model, sampling).await;
    let (stop, text, _hits) = run_turn_and_print(session, THINKING_PROMPT)
        .await
        .map_err(|e| e.to_string())?;
    if !matches!(stop, AcpStopReason::EndTurn) {
        return Err(format!("unexpected stop reason: {stop:?}"));
    }
    if text.trim().is_empty() {
        return Err("empty assistant text".to_string());
    }
    // OpenAI o-系列默认不外露 reasoning content，所以不强制要求 thought_text。
    Ok(None)
}
