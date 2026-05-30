//! DeepSeek provider 集测：复用 OpenAI-compatible transport，但把
//! DeepSeek 特有的 `/models` 兼容差异收敛在 wrapper 自己。

use std::sync::Arc;

use agent_client_protocol_schema::StopReason as AcpStopReason;
use defect_agent::event::AgentEvent;
use defect_agent::llm::LlmProvider;
use defect_llm::provider::deepseek::{DeepSeekConfig, DeepSeekProvider};
use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, ResponseTemplate};

mod common;
use common::{build_session, encode_sse_events, start_mock_server, user_prompt};

const TEST_API_KEY: &str = "test-deepseek-key";
const TEST_AUTH_HEADER: &str = "Bearer test-deepseek-key";

fn provider_for(server_uri: &str) -> Arc<dyn LlmProvider> {
    let cfg = DeepSeekConfig {
        api_key: Some(TEST_API_KEY.to_string()),
        api_key_env: None,
        base_url: Some(server_uri.to_string()),
        reasoning_effort: None,
        http: defect_http::HttpStackConfig::default(),
    };
    Arc::new(DeepSeekProvider::new(cfg).expect("provider")) as Arc<dyn LlmProvider>
}

const MODEL_ID: &str = "deepseek-v4-flash";
const DONE: &str = "[DONE]";

fn deepseek_sse_body(chunks: &[&str], include_done: bool) -> Vec<u8> {
    let mut events: Vec<(&str, &str)> = chunks.iter().map(|chunk| ("", *chunk)).collect();
    if include_done {
        events.push(("", DONE));
    }
    encode_sse_events(&events)
}

#[tokio::test]
async fn list_models_falls_back_to_builtin_deepseek_models_when_decode_fails() {
    let server = start_mock_server().await;

    let incompatible_body = json!({
        "object": "list",
        "data": [
            {"id": "deepseek-v4-pro"},
            {"id": "deepseek-v4-flash"}
        ]
    });
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", TEST_AUTH_HEADER))
        .respond_with(ResponseTemplate::new(200).set_body_json(incompatible_body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri());
    let models = provider.list_models().await.expect("list models");

    assert!(
        models.iter().any(|model| model.id == "deepseek-v4-pro"),
        "expected built-in deepseek-v4-pro fallback"
    );
    assert!(
        models.iter().any(|model| model.id == "deepseek-v4-flash"),
        "expected built-in deepseek-v4-flash fallback"
    );
}

#[tokio::test]
async fn stream_usage_reads_deepseek_prompt_cache_hit_tokens() {
    let server = start_mock_server().await;

    let sse_body = deepseek_sse_body(
        &[
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-flash","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_cache_hit_tokens":6,"prompt_cache_miss_tokens":4}}"#,
        ],
        true,
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", TEST_AUTH_HEADER))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let session = build_session(provider_for(&server.uri()), MODEL_ID).await;
    let mut events_stream = session.subscribe();

    let stop = session
        .run_turn(user_prompt("hello"))
        .await
        .expect("turn ok");
    assert!(matches!(stop, AcpStopReason::EndTurn));

    let mut turn_usage = None;
    while let Some(ev) = events_stream.next().await {
        if let AgentEvent::TurnEnded { usage, .. } = ev {
            turn_usage = Some(usage);
            break;
        }
    }

    let usage = turn_usage.expect("turn usage");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.cache_read_input_tokens, Some(6));
    assert_eq!(usage.cache_creation_input_tokens, None);
}
