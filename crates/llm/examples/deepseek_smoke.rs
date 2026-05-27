//! DeepSeek provider 真端点冒烟。
//!
//! 用法：
//!
//! ```bash
//! DEEPSEEK_API_KEY=sk-... \
//!   cargo run -p defect-llm --example deepseek_smoke -- [scenario]
//! ```
//!
//! `[scenario]` ∈ `list-models | text | tool | thinking | thinking-tool | cache-smoke | all`，
//! 默认 `all`。
//!
//! 可选 env：
//! - `DEEPSEEK_BASE_URL`：覆盖默认 `https://api.deepseek.com`
//! - `DEEPSEEK_MODEL`：覆盖默认模型 `deepseek-v4-flash`
//! - `RUST_LOG=defect_llm=debug` 打开协议层调试日志
//!
//! `thinking-tool` 场景验证 thinking + tool_use 多轮 round-trip：
//! v4 系列在 thinking 模式下要求把上一轮 `reasoning_content` 回放回去，
//! 否则第二轮（送 tool_result 时）400
//! "reasoning_content must be passed back to the API"。本场景跑一个会
//! 触发工具调用的 prompt，agent core 在一个 turn 内自动闭环：第一轮
//! LLM 出 thinking + tool_use → 工具执行 → 第二轮把 thinking + tool_result
//! 一起送回。失败说明 [`MessageContent::Thinking`] 的 echo 路径没拼对。
//! 模型不在 `list_models` 返回里时 SKIP，不 FAIL。

mod common;

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::StopReason as AcpStopReason;
use defect_agent::llm::{LlmProvider, SamplingParams};
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};

use common::{
    EXIT_FAIL, EXIT_OK, build_session, env_string, init_tracing, print_fail, print_pass,
    print_skip, run_turn_and_print, run_turn_and_print_with_usage, sampling_with_thinking,
    scenario_from_args,
};

const DEFAULT_MODEL: &str = "deepseek-v4-flash";
const THINKING_BUDGET_TOKENS: u32 = 2048;
const CACHE_SMOKE_PROBE_ATTEMPTS: usize = 4;
const CACHE_SMOKE_SETTLE_DELAY: Duration = Duration::from_secs(3);

const TEXT_PROMPT: &str = "Say hello in one short sentence.";
const TOOL_PROMPT: &str = "Please call the `echo` tool with msg=\"hello from smoke\", \
     then briefly summarize what the tool returned.";
const THINKING_PROMPT: &str = "Think step by step: a farmer has 17 sheep and all but 9 die. How many are left? \
     Show your reasoning briefly, then give the final number.";
// 让模型必须先 thinking、必须用工具、再给文本——这样第二轮请求里
// 的 assistant message 一定带 reasoning_content（否则服务端拒）。
const THINKING_TOOL_PROMPT: &str = "Think briefly about which message to echo, then call \
     the `echo` tool with msg=\"hello from thinking-tool\", and after it returns \
     summarize what came back in one sentence.";
const CACHE_SMOKE_PROMPT: &str = concat!(
    "You are validating prompt-cache reuse for a deterministic smoke test. ",
    "Read the stable instructions below carefully and then answer with exactly one short sentence. ",
    "Stable block A: defect is a headless ACP agent. Stable block B: this request intentionally repeats a long prefix. ",
    "Stable block C: do not call tools, do not ask questions, do not add bullet points. ",
    "Stable block D: summarize the phrase 'cache smoke baseline' in plain English. ",
    "Stable block E: keep the response under twelve words. ",
    "Stable block F: this paragraph exists only to make the prompt prefix long enough for cache validation. ",
    "Stable block G: repeat the key idea mentally but not verbatim; the important property is identical prompt bytes across runs. ",
    "Stable block H: avoid markdown, code fences, and JSON. ",
    "Stable block I: answer once and stop."
);

#[tokio::main]
async fn main() {
    init_tracing();

    let api_key = match env_string("DEEPSEEK_API_KEY").or_else(|| env_string("OPENAI_API_KEY")) {
        Some(k) => k,
        None => {
            eprintln!("DEEPSEEK_API_KEY (or OPENAI_API_KEY) is required for deepseek_smoke");
            std::process::exit(EXIT_FAIL);
        }
    };
    let base_url = env_string("DEEPSEEK_BASE_URL");
    let model = env_string("DEEPSEEK_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let provider: Arc<dyn LlmProvider> = match DeepSeekProvider::new(DeepSeekConfig {
        api_key: Some(api_key),
        base_url,
        http: defect_http::HttpStackConfig::default(),
    }) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("provider init failed: {e}");
            std::process::exit(EXIT_FAIL);
        }
    };

    let scenario = scenario_from_args();
    println!("=== deepseek smoke: scenario={scenario} model={model} ===");

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

    println!("\n=== deepseek smoke done: ran={ran} failed={failed} ===");
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
        "thinking-tool" => vec!["thinking-tool"],
        "cache-smoke" => vec!["cache-smoke"],
        _ => vec![
            "list-models",
            "text",
            "tool",
            "thinking",
            "thinking-tool",
            "cache-smoke",
        ],
    }
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
        "thinking-tool" => scenario_thinking_tool_multi_turn(provider, model).await,
        "cache-smoke" => scenario_cache_smoke(provider, model).await,
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

async fn scenario_thinking_tool_multi_turn(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    let sampling = sampling_with_thinking(Some(THINKING_BUDGET_TOKENS));
    let session = build_session(provider, model, sampling).await;
    let (stop, text, hits) = match run_turn_and_print(session, THINKING_TOOL_PROMPT).await {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            // 模型未上线时 SKIP 而非 FAIL（避免 CI 因上游清单变动而红）。
            if msg.contains("ModelNotFound") || msg.contains("model_not_found") {
                return Ok(Some(format!(
                    "model {model} not available on DeepSeek API; \
                     override with DEEPSEEK_MODEL"
                )));
            }
            return Err(msg);
        }
    };
    if !matches!(stop, AcpStopReason::EndTurn) {
        return Err(format!("unexpected stop reason: {stop:?}"));
    }
    if hits.started == 0 || hits.finished == 0 {
        return Err(format!(
            "expected at least one tool call (started={}, finished={})",
            hits.started, hits.finished
        ));
    }
    if hits.thought_text.trim().is_empty() {
        return Err("no reasoning_content emitted by thinking-tool model; \
             cannot verify echo path"
            .to_string());
    }
    if text.trim().is_empty() {
        return Err("empty assistant text after tool turn".to_string());
    }
    // 走到这里说明：第一轮带 thinking + tool_use → tool_result 注入 →
    // 第二轮请求里 assistant message 必带 reasoning_content（否则
    // v4 系列会在第二轮直接 400，run_turn 拿不到 EndTurn）。
    Ok(None)
}

async fn scenario_thinking(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    let sampling = sampling_with_thinking(Some(THINKING_BUDGET_TOKENS));
    let session = build_session(provider, model, sampling).await;
    let (stop, text, hits) = run_turn_and_print(session, THINKING_PROMPT)
        .await
        .map_err(|e| e.to_string())?;
    if !matches!(stop, AcpStopReason::EndTurn) {
        return Err(format!("unexpected stop reason: {stop:?}"));
    }
    if text.trim().is_empty() {
        return Err("empty assistant text".to_string());
    }
    if hits.thought_text.trim().is_empty() {
        return Ok(Some(format!(
            "no reasoning_content emitted by {model}; \
             check upstream changed shape"
        )));
    }
    Ok(None)
}

async fn scenario_cache_smoke(
    provider: Arc<dyn LlmProvider>,
    model: &str,
) -> Result<Option<String>, String> {
    let total_attempts = CACHE_SMOKE_PROBE_ATTEMPTS.saturating_add(1);
    let mut observed_cache_reads = Vec::with_capacity(total_attempts);
    let mut any_usage_reported = false;

    for attempt_index in 0..total_attempts {
        let session = build_session(provider.clone(), model, SamplingParams::default()).await;
        let (stop, text, _hits, usage) = run_turn_and_print_with_usage(session, CACHE_SMOKE_PROMPT)
            .await
            .map_err(|e| e.to_string())?;
        if !matches!(stop, AcpStopReason::EndTurn) {
            return Err(format!(
                "unexpected cache-smoke stop reason on attempt {}: {stop:?}",
                attempt_index + 1
            ));
        }
        if text.trim().is_empty() {
            return Err(format!(
                "empty assistant text on cache-smoke attempt {}",
                attempt_index + 1
            ));
        }

        any_usage_reported |= usage_reported(&usage);
        let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
        observed_cache_reads.push(cache_read);

        let phase = if attempt_index == 0 {
            "warmup"
        } else {
            "probe"
        };
        println!(
            "[cache-smoke] attempt={} phase={phase} cache_read={cache_read}",
            attempt_index + 1
        );

        if attempt_index > 0 && cache_read > 0 {
            println!(
                "[cache-smoke] cache hit observed after warmup: {:?}",
                observed_cache_reads
            );
            return Ok(None);
        }

        if attempt_index + 1 < total_attempts {
            tokio::time::sleep(CACHE_SMOKE_SETTLE_DELAY).await;
        }
    }

    if !any_usage_reported {
        return Ok(Some(
            "DeepSeek stream did not report usage/prompt_cache_hit_tokens; \
             cannot verify cache hits from streaming usage on this endpoint"
                .to_string(),
        ));
    }

    Err(format!(
        "no post-warmup cache hit observed across {} probes: {:?}",
        CACHE_SMOKE_PROBE_ATTEMPTS, observed_cache_reads
    ))
}

fn usage_reported(usage: &defect_agent::llm::Usage) -> bool {
    usage.input_tokens.is_some()
        || usage.output_tokens.is_some()
        || usage.cache_read_input_tokens.is_some()
        || usage.cache_creation_input_tokens.is_some()
}
