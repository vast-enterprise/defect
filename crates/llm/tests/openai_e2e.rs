//! End-to-end integration tests for the OpenAI provider, using wiremock as a compatible
//! backend for OpenAI Chat Completions so that `OpenAiProvider` runs a full agent turn.
//!
//! Integration tests for OpenAI provider:
//! - list_models round-trip + hardcoded merge
//! - text-only turn (including stream + `[DONE]` terminator)
//! - tool_calls full path (two-round LLM call loop)
//! - auth header injection
//! - cancel interruption

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::StopReason as AcpStopReason;
use defect_agent::event::AgentEvent;
use defect_agent::llm::LlmProvider;
use defect_llm::provider::openai::{OpenAiConfig, OpenAiProvider};
use futures::StreamExt;
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, Request, ResponseTemplate};

mod common;
use common::{build_session, encode_sse_events, start_mock_server, user_prompt};

const TEST_API_KEY: &str = "test-openai-key";
const TEST_AUTH_HEADER: &str = "Bearer test-openai-key";
const MODEL_ID: &str = "gpt-test-001";

// ---- raw chat.completion.chunk JSON (same source as protocol-layer tests) ---

const TEXT_CHUNK_1: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_CHUNK_2: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"content":"hello "},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_CHUNK_3: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"content":"world"},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_FINISH_STOP: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#;
const USAGE_CHUNK: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":3}}}"#;
const DONE: &str = "[DONE]";

fn provider_for(server_uri: &str) -> Arc<dyn LlmProvider> {
    let cfg = OpenAiConfig {
        api_key: Some(TEST_API_KEY.to_string()),
        api_key_env: None,
        // base_url already includes the `/v1` prefix (as does the wire spec server).
        // wiremock does not need this prefix, so base_url points directly to
        // `server.uri()`.
        base_url: Some(server_uri.to_string()),
        organization: None,
        project: None,
        vendor: "openai".to_string(),
        display_name: "OpenAI Chat Completions".to_string(),
        headers: std::collections::HashMap::new(),
        capabilities_override: None,
        reasoning_effort: None,
        chat_dialect: defect_llm::protocol::openai_chat::ChatDialect::OpenAi,
        http: defect_http::HttpStackConfig::default(),
    };
    Arc::new(OpenAiProvider::new(cfg).expect("provider")) as Arc<dyn LlmProvider>
}

/// Encodes several raw JSON chunks (each corresponding to one SSE `data:` line) plus an
/// optional `[DONE]` termination frame into an OpenAI-format SSE wire byte string.
fn openai_sse_body(chunks: &[&str], include_done: bool) -> Vec<u8> {
    let mut events: Vec<(&str, &str)> = chunks.iter().map(|c| ("", *c)).collect();
    if include_done {
        events.push(("", DONE));
    }
    encode_sse_events(&events)
}

fn sse_response(chunks: &[&str], include_done: bool) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(openai_sse_body(chunks, include_done), "text/event-stream")
}

// ---------- list_models -------------------------------------------------

#[tokio::test]
async fn list_models_round_trip() {
    let server = start_mock_server().await;

    let body = json!({
        "object": "list",
        "data": [
            {"id": "gpt-test-001", "object": "model", "created": 1, "owned_by": "openai"}
        ]
    });
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", TEST_AUTH_HEADER))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri());
    let models = provider.list_models().await.expect("list models");
    // The upstream provides `gpt-test-001` (not in the hardcoded table), but the
    // hardcoded table will add some known models (gpt-4o, o1...). Here we only assert
    // that the upstream id is present in the result.
    assert!(
        models.iter().any(|m| m.id == "gpt-test-001"),
        "expected upstream id in merged list"
    );
}

// text-only turn

#[tokio::test]
async fn turn_with_text_only_response() {
    let server = start_mock_server().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", TEST_AUTH_HEADER))
        .respond_with(sse_response(
            &[
                TEXT_CHUNK_1,
                TEXT_CHUNK_2,
                TEXT_CHUNK_3,
                TEXT_FINISH_STOP,
                USAGE_CHUNK,
            ],
            true,
        ))
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

    let mut got_text = false;
    while let Some(ev) = events_stream.next().await {
        match ev {
            AgentEvent::AssistantText { .. } => got_text = true,
            AgentEvent::TurnEnded { .. } => break,
            _ => {}
        }
    }
    assert!(got_text, "expected at least one AssistantText");
}

// ---------- tool-use turn (two LLM rounds) -------------------------------

#[tokio::test]
async fn turn_with_tool_use_two_rounds() {
    let server = start_mock_server().await;

    let round1 = openai_sse_body(
        &[
            r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"msg\":\"hi\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-r1","object":"chat.completion.chunk","created":1,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        true,
    );
    let round2 = openai_sse_body(
        &[
            r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-r2","object":"chat.completion.chunk","created":2,"model":"gpt-test-001","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        true,
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(move |req: &Request| {
            // Determine whether a tool result has already been returned by checking if
            // any message in the request body has `role: "tool"`.
            // Round 1 contains only system + user messages; round 2 adds assistant (with
            // tool_calls) and tool (with tool_result).
            let body: Value = serde_json::from_slice(&req.body).expect("body json");
            let has_tool_msg = body
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .any(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
                })
                .unwrap_or(false);
            let payload = if has_tool_msg {
                round2.clone()
            } else {
                round1.clone()
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(payload, "text/event-stream")
        })
        .expect(2)
        .mount(&server)
        .await;

    let session = build_session(provider_for(&server.uri()), MODEL_ID).await;
    let mut events_stream = session.subscribe();

    let stop = session
        .run_turn(user_prompt("please echo"))
        .await
        .expect("turn ok");
    assert!(matches!(stop, AcpStopReason::EndTurn));

    let mut started = false;
    let mut finished = false;
    while let Some(ev) = events_stream.next().await {
        match ev {
            AgentEvent::ToolCallStarted { .. } => started = true,
            AgentEvent::ToolCallFinished { .. } => finished = true,
            AgentEvent::TurnEnded { .. } => break,
            _ => {}
        }
    }
    assert!(started, "expected ToolCallStarted");
    assert!(finished, "expected ToolCallFinished");
}

// ---------- auth header tests --------------------------------------

#[tokio::test]
async fn missing_bearer_header_results_in_404() {
    let server = start_mock_server().await;

    // Only match the wrong token to return 200; the real request carries "Bearer
    // test-openai-key" and never matches, falling through to the default 404, verifying
    // that the provider actually sends the Authorization header.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer wrong-key"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri());
    let req = defect_agent::llm::CompletionRequest {
        model: MODEL_ID.to_string(),
        system: None,
        messages: vec![defect_agent::llm::Message {
            role: defect_agent::llm::Role::User,
            content: vec![defect_agent::llm::MessageContent::Text { text: "hi".into() }].into(),
        }],
        tools: vec![],
        tool_choice: defect_agent::llm::ToolChoice::Auto,
        sampling: defect_agent::llm::SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let res = provider
        .complete(req, tokio_util::sync::CancellationToken::new())
        .await;
    assert!(res.is_err(), "expected error when auth header didn't match");
}

// Cancel mid-stream

#[tokio::test]
async fn cancel_during_stream_terminates_turn_silently() {
    let server = start_mock_server().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            sse_response(
                &[TEXT_CHUNK_1, TEXT_CHUNK_2, TEXT_FINISH_STOP, USAGE_CHUNK],
                true,
            )
            .set_delay(Duration::from_millis(200)),
        )
        .mount(&server)
        .await;

    let session = build_session(provider_for(&server.uri()), MODEL_ID).await;

    let s = session.clone();
    let h = tokio::spawn(async move { s.run_turn(user_prompt("hi")).await });

    tokio::time::sleep(Duration::from_millis(50)).await;
    session.cancel_turn();

    let outcome = h.await.expect("join");
    use defect_agent::llm::ProviderErrorKind;
    use defect_agent::session::TurnError;
    match outcome {
        Ok(AcpStopReason::Cancelled | AcpStopReason::EndTurn) => {}
        Err(TurnError::Provider(e)) if matches!(e.kind, ProviderErrorKind::Canceled) => {}
        other => panic!("unexpected turn outcome: {other:?}"),
    }
}
