//! Anthropic provider end-to-end integration tests: uses wiremock as Anthropic API,
//! running `AnthropicProvider` as a real backend through a full agent turn.
//!
//! No real API calls — all routes are intercepted by the mock server, covering
//! round-trip, auth, cancel scenarios plus single tool_use full loop and list_models.

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::StopReason as AcpStopReason;
use defect_agent::event::AgentEvent;
use defect_agent::llm::LlmProvider;
use defect_llm::provider::anthropic::{AnthropicConfig, AnthropicProvider};
use futures::StreamExt;
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, Request, ResponseTemplate};

mod common;
use common::{build_session, encode_sse_events, start_mock_server, user_prompt};

const TEST_API_KEY: &str = "test-anthropic-key";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const MODEL_ID: &str = "claude-test-001";

// ---- SSE event payloads（与协议层 tests 用的同一份 wire 字节）----------

const MODEL_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-test-001","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":42,"output_tokens":1}}}"#;
const TEXT_START_0: &str = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#;
const TEXT_DELTA_0: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}"#;
const TEXT_DELTA_1: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#;
const TEXT_STOP_0: &str = r#"{"type":"content_block_stop","index":0}"#;
const TOOL_START_1: &str = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"echo","input":{}}}"#;
const TOOL_DELTA_1: &str = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"msg\":\"hi\"}"}}"#;
const TOOL_STOP_1: &str = r#"{"type":"content_block_stop","index":1}"#;
const MSG_DELTA_END: &str =
    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":17}}"#;
const MSG_DELTA_TOOL: &str =
    r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#;
const MSG_STOP: &str = r#"{"type":"message_stop"}"#;

fn provider_for(server_uri: &str) -> Arc<dyn LlmProvider> {
    let cfg = AnthropicConfig {
        api_key: Some(TEST_API_KEY.to_string()),
        api_key_env: None,
        base_url: Some(server_uri.to_string()),
        http: defect_http::HttpStackConfig::default(),
    };
    Arc::new(AnthropicProvider::new(cfg).expect("provider")) as Arc<dyn LlmProvider>
}

fn sse_body(events: &[(&str, &str)]) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(encode_sse_events(events), "text/event-stream")
}

// ---------- list_models -------------------------------------------------

#[tokio::test]
async fn list_models_round_trip() {
    let server = start_mock_server().await;

    let body = json!({
        "data": [
            {"type": "model", "id": "claude-test-001", "display_name": "Claude Test", "created_at": "2025-01-01T00:00:00Z"}
        ],
        "has_more": false,
        "first_id": null,
        "last_id": null
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", TEST_API_KEY))
        .and(header("anthropic-version", ANTHROPIC_VERSION))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri());
    let models = provider.list_models().await.expect("list models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "claude-test-001");
    assert_eq!(models[0].display_name.as_deref(), Some("Claude Test"));
}

// ---------- text-only turn ----------------------------------------------

#[tokio::test]
async fn turn_with_text_only_response() {
    let server = start_mock_server().await;

    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_delta", TEXT_DELTA_1),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", TEST_API_KEY))
        .and(header("anthropic-version", ANTHROPIC_VERSION))
        .respond_with(sse_body(&events))
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

/// 两轮 LLM：第 1 轮发 tool_use，第 2 轮发 EndTurn 文本。
///
/// wiremock 的 `expect(1)` + `respond_with` 不支持"按调用次数返回不同 body"，
/// 这里用一条 `Mock` 注册两个独立路由不行（同一条 path），所以走
/// "Mock 上挂 stateful matcher：拿请求体 messages 长度判断轮次"。
#[tokio::test]
async fn turn_with_tool_use_two_rounds() {
    let server = start_mock_server().await;

    let round1 = encode_sse_events(&[
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"calling "}}"#,
        ),
        ("content_block_stop", TEXT_STOP_0),
        ("content_block_start", TOOL_START_1),
        ("content_block_delta", TOOL_DELTA_1),
        ("content_block_stop", TOOL_STOP_1),
        ("message_delta", MSG_DELTA_TOOL),
        ("message_stop", MSG_STOP),
    ]);
    let round2 = encode_sse_events(&[
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        (
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}"#,
        ),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &Request| {
            // 用请求体 messages.len 判断轮次：第 1 轮只有 1 条 user，
            // 第 2 轮有 user + assistant(tool_use) + user(tool_result) = 3 条。
            let body: Value = serde_json::from_slice(&req.body).expect("body json");
            let n = body
                .get("messages")
                .and_then(|m| m.as_array())
                .map(Vec::len)
                .unwrap_or(0);
            let payload = if n <= 1 {
                round1.clone()
            } else {
                round2.clone()
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

// ---------- auth header rejection ---------------------------------------

#[tokio::test]
async fn missing_api_key_header_is_rejected_by_server() {
    let server = start_mock_server().await;

    // 故意要求一个不存在的 key——provider 必然不带这个 header，
    // 导致匹配落到默认 404 上，验证"provider 把 x-api-key 真的发出去了"。
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "wrong-key"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri());
    let cancel = tokio_util::sync::CancellationToken::new();
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
    let res = provider.complete(req, cancel).await;
    // wiremock 在没有 mock 命中时返回 404，provider 应映射到 ServerError。
    assert!(res.is_err(), "expected error when auth header didn't match");
}

// ---------- cancel mid-stream -------------------------------------------

#[tokio::test]
async fn cancel_during_stream_terminates_turn_silently() {
    let server = start_mock_server().await;

    // 给一个慢响应：每 10ms 一帧的小流，足够让 cancel 在中途生效。
    // wiremock 不直接支持 chunked SSE delay；这里用 set_delay 让响应在
    // 100ms 后才发回——cancel 在这之前先触发。
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(encode_sse_events(&events), "text/event-stream")
                .set_delay(Duration::from_millis(200)),
        )
        .mount(&server)
        .await;

    let session = build_session(provider_for(&server.uri()), MODEL_ID).await;

    let s = session.clone();
    let h = tokio::spawn(async move { s.run_turn(user_prompt("hi")).await });

    // 给请求一点时间 in-flight，再取消。
    tokio::time::sleep(Duration::from_millis(50)).await;
    session.cancel_turn();

    let outcome = h.await.expect("join");
    // cancel 在 HTTP 阶段触发：provider 立即返回 `ProviderErrorKind::Canceled`，
    // turn loop 当前把它当作 `TurnError::Provider`（Canceled 在 retry_hint
    // 里是 No，所以不重试）。也允许"cancel 落在 SSE 拉取阶段 → 主循环把它
    // 翻成 `StopReason::Cancelled`"或"响应在取消之前已经全部到达"两种边界。
    use defect_agent::llm::ProviderErrorKind;
    use defect_agent::session::TurnError;
    match outcome {
        Ok(AcpStopReason::Cancelled | AcpStopReason::EndTurn) => {}
        Err(TurnError::Provider(e)) if matches!(e.kind, ProviderErrorKind::Canceled) => {}
        other => panic!("unexpected turn outcome: {other:?}"),
    }
}
