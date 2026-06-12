//! Unit tests for the Anthropic Messages protocol layer.
//!
//! Key coverage:
//! - `encode_request` field mapping (model / system / tools / tool_choice /
//!   thinking / sampling / messages with 4 content types)
//! - `decode_stream` SSE state machine:
//!   - Single tool_use full path (Start→ArgsDelta→Stop→ToolUseEnd→message_delta)
//!   - Interleaved ArgsDelta for two concurrent tool_use (different indices)
//!   - Alternating thinking + signature_delta
//!   - `event: error` terminates the stream
//!   - `event: ping` is swallowed
//!   - Single SSE JSON parse failure → `Malformed`, stream continues
//!   - Missing stop at end of stream → `ProtocolViolation`
//!   - Cancel triggers silent stream termination
//!
//! Tests feed a `Stream<Item = Result<Sse, _>>` constructed via
//! [`decode_stream_generic`] with `futures::stream::iter`, bypassing the
//! pub-crate restriction on `SseBody::new`; at runtime [`decode_stream`] uses
//! the same decoding core, so coverage is equivalent.

use defect_core::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderErrorKind,
    ReasoningEffort, Role, SamplingParams, StopReason, ThinkingConfig, ToolChoice, ToolResultBody,
    ToolResultContent,
};
use defect_core::tool::ToolSchema;
use futures::StreamExt;
use serde_json::json;
use sse_stream::Sse;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::wire::anthropic::components as wire;

// Helpers

#[derive(Debug, thiserror::Error)]
#[error("test sse never errors")]
struct NeverError;

fn make_sse_events(events: &[(&str, &str)]) -> Vec<Sse> {
    events
        .iter()
        .map(|(name, data)| Sse {
            event: Some((*name).to_owned()),
            data: Some((*data).to_owned()),
            id: None,
            retry: None,
        })
        .collect()
}

/// Feed `Vec<Sse>` directly into `process_sse` — the test avoids the hyper body and the
/// entire transport layer, focusing solely on the state machine.
fn run_state_machine(
    events: &[(&str, &str)],
) -> (DecoderState, Vec<Result<ProviderChunk, ProviderError>>) {
    let mut state = DecoderState::default();
    let mut out = Vec::new();
    for sse in make_sse_events(events) {
        let mut buf = Vec::new();
        process_sse(&mut state, sse, &mut buf);
        // process_sse pushes multiple chunks onto a stack in reverse order for poll_next
        // to pop; here we reverse them back to chronological order for testing.
        buf.reverse();
        out.extend(buf);
        if state.fatal {
            break;
        }
    }
    (state, out)
}

/// Packages several `(event, data)` pairs into a `Stream<Item = Result<Sse, NeverError>>`
/// and feeds it to [`decode_stream_generic`] for an end-to-end run through
/// [`AnthropicSseDecoder`].
async fn run_decode_stream_generic(
    events: &[(&str, &str)],
    cancel: CancellationToken,
) -> Vec<Result<ProviderChunk, ProviderError>> {
    let items: Vec<Result<Sse, NeverError>> = make_sse_events(events).into_iter().map(Ok).collect();
    let stream = futures::stream::iter(items);
    decode_stream_generic(stream, cancel)
        .collect::<Vec<_>>()
        .await
}

fn ok_chunks(results: Vec<Result<ProviderChunk, ProviderError>>) -> Vec<ProviderChunk> {
    results.into_iter().map(|r| r.expect("err chunk")).collect()
}

// ---------- encode_request ----------

#[test]
fn encode_minimal_request() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: Some("you are helpful".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text { text: "hi".into() }].into(),
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let wire_req = encode_request(&req);
    assert_eq!(wire_req.max_tokens, i64::from(DEFAULT_MAX_TOKENS));
    assert!(matches!(wire_req.stream, Some(true)));
    assert!(matches!(
        wire_req.system,
        Some(wire::SystemPrompt::SystemPromptVariant1(ref blocks))
            if matches!(
                blocks.as_slice(),
                [wire::TextBlockParam {
                    text,
                    cache_control: Some(_),
                    ..
                }] if text == "you are helpful"
            )
    ));
    assert_eq!(wire_req.messages.len(), 1);
    assert!(matches!(
        wire_req.messages[0].role,
        wire::MessageParamRole::User
    ));
    let wire::MessageParamContent::MessageParamContentVariant1(content) =
        &wire_req.messages[0].content
    else {
        panic!("expected list content");
    };
    assert!(matches!(
        content.as_slice(),
        [wire::ContentBlockParam::TextBlockParam(wire::TextBlockParam {
            text,
            cache_control: Some(_),
            ..
        })] if text == "hi"
    ));
    assert!(wire_req.tools.is_none());
    assert!(matches!(
        wire_req.tool_choice,
        Some(wire::ToolChoice::ToolChoiceAuto(_))
    ));
    assert!(wire_req.thinking.is_none());
}

#[test]
fn encode_request_carries_sampling() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text { text: "x".into() }].into(),
        }],
        tools: vec![],
        tool_choice: ToolChoice::Required,
        sampling: SamplingParams {
            max_tokens: Some(8000),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(40),
            stop_sequences: vec!["END".into()],
            thinking: ThinkingConfig::Enabled {
                budget_tokens: Some(2000),
            },
            reasoning_effort: None,
        },
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);
    assert_eq!(w.max_tokens, 8000);
    assert_eq!(w.temperature, Some(0.5));
    assert_eq!(w.top_p, Some(0.9));
    assert_eq!(w.top_k, Some(40));
    assert_eq!(w.stop_sequences.as_deref(), Some(&["END".to_string()][..]));
    assert!(matches!(
        w.tool_choice,
        Some(wire::ToolChoice::ToolChoiceAny(_))
    ));
    assert!(matches!(
        w.thinking,
        Some(wire::ThinkingConfigParam::ThinkingConfigEnabled(ref t)) if t.budget_tokens == 2000
    ));
}

fn thinking_budget(w: &wire::CreateMessageParams) -> Option<i64> {
    match &w.thinking {
        Some(wire::ThinkingConfigParam::ThinkingConfigEnabled(t)) => Some(t.budget_tokens),
        _ => None,
    }
}

fn req_with(
    effort: Option<ReasoningEffort>,
    thinking: ThinkingConfig,
    max_tokens: u32,
) -> CompletionRequest {
    CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text { text: "x".into() }].into(),
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams {
            max_tokens: Some(max_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            thinking,
            reasoning_effort: effort,
        },
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    }
}

#[test]
fn effort_maps_to_thinking_budget() {
    let w = encode_request(&req_with(
        Some(ReasoningEffort::High),
        ThinkingConfig::Disabled,
        64_000,
    ));
    assert_eq!(thinking_budget(&w), Some(16_384));
}

#[test]
fn effort_none_disables_thinking() {
    let w = encode_request(&req_with(
        Some(ReasoningEffort::None),
        ThinkingConfig::Enabled {
            budget_tokens: Some(8_000),
        },
        64_000,
    ));
    // effort override wins over the explicit thinking config and disables thinking.
    assert!(w.thinking.is_none());
}

#[test]
fn effort_takes_precedence_over_thinking_config() {
    let w = encode_request(&req_with(
        Some(ReasoningEffort::Low),
        ThinkingConfig::Enabled {
            budget_tokens: Some(30_000),
        },
        64_000,
    ));
    assert_eq!(thinking_budget(&w), Some(4_096));
}

#[test]
fn effort_budget_clamped_below_max_tokens() {
    // xhigh wants 32_768 but max_tokens is only 5_000 → clamp to max_tokens - 1.
    let w = encode_request(&req_with(
        Some(ReasoningEffort::Xhigh),
        ThinkingConfig::Disabled,
        5_000,
    ));
    assert_eq!(thinking_budget(&w), Some(4_999));
}

#[test]
fn thinking_dropped_when_max_tokens_too_small_for_minimum_budget() {
    // max_tokens - 1 = 500 < MIN_THINKING_BUDGET (1024) → thinking dropped entirely.
    let w = encode_request(&req_with(
        Some(ReasoningEffort::High),
        ThinkingConfig::Disabled,
        501,
    ));
    assert!(w.thinking.is_none());
}

#[test]
fn thinking_config_used_when_no_effort() {
    let w = encode_request(&req_with(
        None,
        ThinkingConfig::Enabled {
            budget_tokens: Some(2_000),
        },
        64_000,
    ));
    assert_eq!(thinking_budget(&w), Some(2_000));
}

#[test]
fn encode_request_tool_uses_and_results() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![
            Message {
                role: Role::Assistant,
                content: vec![MessageContent::ToolUse {
                    id: "toolu_1".into(),
                    name: "fs_read".into(),
                    args: json!({"path": "/tmp/a"}),
                }]
                .into(),
            },
            Message {
                role: Role::User,
                content: vec![MessageContent::ToolResult {
                    tool_use_id: "toolu_1".into(),
                    output: ToolResultBody::Text {
                        text: "hello".into(),
                    },
                    is_error: false,
                }]
                .into(),
            },
        ],
        tools: vec![ToolSchema {
            name: "fs_read".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }],
        tool_choice: ToolChoice::Named {
            name: "fs_read".into(),
        },
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    assert!(matches!(
        w.tool_choice,
        Some(wire::ToolChoice::ToolChoiceTool(ref t)) if t.name == "fs_read"
    ));

    let tools = w.tools.as_ref().expect("tools");
    assert_eq!(tools.len(), 1);
    let wire::ToolUnion::Tool(t) = &tools[0] else {
        panic!("expected Tool");
    };
    assert_eq!(t.name, "fs_read");
    assert_eq!(t.description.as_deref(), Some("Read a file"));
    assert_eq!(
        t.input_schema.required.as_deref(),
        Some(&["path".to_string()][..])
    );

    // Round-trip assistant tool_use
    let assistant = match &w.messages[0].content {
        wire::MessageParamContent::MessageParamContentVariant1(v) => v,
        _ => panic!("expected list content"),
    };
    let wire::ContentBlockParam::ToolUseBlockParam(tu) = &assistant[0] else {
        panic!("expected tool_use_block_param");
    };
    assert_eq!(tu.id, "toolu_1");
    assert_eq!(tu.name, "fs_read");
    assert_eq!(tu.input.get("path"), Some(&json!("/tmp/a")));
    assert!(tu.cache_control.is_some());

    // Tool result round-trip
    let user = match &w.messages[1].content {
        wire::MessageParamContent::MessageParamContentVariant1(v) => v,
        _ => panic!("expected list content"),
    };
    let wire::ContentBlockParam::ToolResultBlockParam(tr) = &user[0] else {
        panic!("expected tool_result_block_param");
    };
    assert_eq!(tr.tool_use_id, "toolu_1");
    assert_eq!(tr.is_error, Some(false));

    let wire::ToolUnion::Tool(tool) = &tools[0] else {
        panic!("expected Tool");
    };
    assert!(tool.cache_control.is_some());
}

#[test]
fn encode_multimodal_tool_result_emits_text_and_image_blocks() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::ToolResult {
                tool_use_id: "toolu_img".into(),
                output: ToolResultBody::Content {
                    blocks: vec![
                        ToolResultContent::Text {
                            text: "here is the screenshot".into(),
                        },
                        ToolResultContent::Image {
                            mime: "image/png".into(),
                            data: ImageData::Base64 {
                                encoded: "AAAA".into(),
                            },
                        },
                    ],
                },
                is_error: false,
            }]
            .into(),
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    let user = match &w.messages[0].content {
        wire::MessageParamContent::MessageParamContentVariant1(v) => v,
        _ => panic!("expected list content"),
    };
    let wire::ContentBlockParam::ToolResultBlockParam(tr) = &user[0] else {
        panic!("expected tool_result_block_param");
    };
    let Some(wire::ToolResultBlockParamContent102::ToolResultBlockParamContent102Variant1(blocks)) =
        &tr.content
    else {
        panic!("expected list tool_result content");
    };
    assert_eq!(blocks.len(), 2);
    assert!(matches!(
        &blocks[0],
        wire::ToolResultBlockParamContent::TextBlockParam(t) if t.text == "here is the screenshot"
    ));
    assert!(matches!(
        &blocks[1],
        wire::ToolResultBlockParamContent::ImageBlockParam(_)
    ));
}

// ---------- thinking round-trip (signature gating) ---------------------

/// Encodes an assistant message containing [`MessageContent::Thinking`] and returns a
/// list of content blocks for asserting the presence or absence of `ThinkingBlockParam`.
fn encode_with_thinking(text: &str, signature: Option<&str>) -> Vec<wire::ContentBlockParam> {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![Message {
            role: Role::Assistant,
            content: vec![
                MessageContent::Thinking {
                    text: text.to_owned(),
                    signature: signature.map(str::to_owned),
                },
                MessageContent::Text {
                    text: "answer".into(),
                },
            ]
            .into(),
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);
    let wire::MessageParamContent::MessageParamContentVariant1(blocks) =
        w.messages[0].content.clone()
    else {
        panic!("expected list content");
    };
    blocks
}

#[test]
fn encode_thinking_with_signature_emits_thinking_block_param() {
    let blocks = encode_with_thinking("step 1", Some("sig-abc"));
    // Expect two blocks: thinking + text.
    assert_eq!(blocks.len(), 2);
    let wire::ContentBlockParam::ThinkingBlockParam(t) = &blocks[0] else {
        panic!("expected thinking block first, got {:?}", blocks[0]);
    };
    assert_eq!(t.thinking, "step 1");
    assert_eq!(t.signature, "sig-abc");
}

#[test]
fn encode_thinking_without_signature_skips_thinking_block_param() {
    // When switching back to Anthropic from another provider: the previous turn's
    // thinking text came from OpenAI/DeepSeek and has no signature. Since `signature` is
    // required on the Anthropic wire, the entire thinking block is skipped — only the
    // text is kept.
    let blocks = encode_with_thinking("step 1", None);
    assert_eq!(blocks.len(), 1);
    assert!(matches!(
        &blocks[0],
        wire::ContentBlockParam::TextBlockParam(t) if t.text == "answer"
    ));
}

// decode_stream / state machine

const MODEL_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":42,"output_tokens":1}}}"#;

const TEXT_START_0: &str = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#;
const TEXT_DELTA_0: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}"#;
const TEXT_DELTA_1: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#;
const TEXT_STOP_0: &str = r#"{"type":"content_block_stop","index":0}"#;

const TOOL_START_1: &str = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"calc","input":{}}}"#;
const TOOL_DELTA_1A: &str = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1"}}"#;
const TOOL_DELTA_1B: &str = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"}"}}"#;
const TOOL_STOP_1: &str = r#"{"type":"content_block_stop","index":1}"#;

const MSG_DELTA_END: &str =
    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":17}}"#;
const MSG_DELTA_TOOL: &str =
    r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#;
const MSG_STOP: &str = r#"{"type":"message_stop"}"#;
const PING: &str = r#"{"type":"ping"}"#;

#[test]
fn decode_text_then_tool_use() {
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_delta", TEXT_DELTA_1),
        ("content_block_stop", TEXT_STOP_0),
        ("content_block_start", TOOL_START_1),
        ("content_block_delta", TOOL_DELTA_1A),
        ("content_block_delta", TOOL_DELTA_1B),
        ("content_block_stop", TOOL_STOP_1),
        ("message_delta", MSG_DELTA_TOOL),
        ("message_stop", MSG_STOP),
    ];
    let (state, results) = run_state_machine(&events);
    assert!(state.stopped);
    let chunks = ok_chunks(results);

    // Expected sequence: MessageStart, Usage(input=42), TextDelta x2, ToolUseStart,
    // ArgsDelta x2, ToolUseEnd, Stop(ToolUse), Usage(output=3)
    let mut iter = chunks.into_iter();
    assert!(
        matches!(iter.next().unwrap(), ProviderChunk::MessageStart { id, .. } if id == "msg_1")
    );
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Usage(u) if u.input_tokens == Some(42)
    ));
    assert!(matches!(iter.next().unwrap(), ProviderChunk::TextDelta { text } if text == "hello "));
    assert!(matches!(iter.next().unwrap(), ProviderChunk::TextDelta { text } if text == "world"));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseStart { id, name } if id == "toolu_a" && name == "calc"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseArgsDelta { id, fragment } if id == "toolu_a" && fragment.starts_with("{\"x\"")
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseArgsDelta { id, .. } if id == "toolu_a"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseEnd { id } if id == "toolu_a"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Stop {
            reason: StopReason::ToolUse
        }
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Usage(u) if u.output_tokens == Some(3)
    ));
}

#[test]
fn decode_two_concurrent_tool_uses_interleaved() {
    let tool_start_b = r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_b","name":"echo","input":{}}}"#;
    let tool_delta_a = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#;
    let tool_delta_b = r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"y\":2}"}}"#;
    let tool_stop_b = r#"{"type":"content_block_stop","index":2}"#;

    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TOOL_START_1),
        ("content_block_start", tool_start_b),
        ("content_block_delta", tool_delta_a),
        ("content_block_delta", tool_delta_b),
        ("content_block_stop", TOOL_STOP_1),
        ("content_block_stop", tool_stop_b),
        ("message_delta", MSG_DELTA_TOOL),
        ("message_stop", MSG_STOP),
    ];
    let (state, results) = run_state_machine(&events);
    assert!(state.stopped);
    let chunks = ok_chunks(results);

    let tool_use_starts: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseStart { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_use_starts, vec!["toolu_a", "toolu_b"]);

    let args_pairs: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseArgsDelta { id, fragment } => {
                Some((id.clone(), fragment.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        args_pairs,
        vec![
            ("toolu_a".into(), "{\"x\":1}".into()),
            ("toolu_b".into(), "{\"y\":2}".into()),
        ]
    );

    let tool_use_ends: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseEnd { id } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_use_ends, vec!["toolu_a", "toolu_b"]);
}

#[test]
fn decode_thinking_with_signature() {
    let think_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#;
    let think_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step 1"}}"#;
    let sig_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}"#;
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", think_start),
        ("content_block_delta", think_delta),
        ("content_block_delta", sig_delta),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    let (_state, results) = run_state_machine(&events);
    let chunks = ok_chunks(results);
    let mut saw_think = false;
    let mut saw_sig = false;
    for c in chunks {
        match c {
            ProviderChunk::ThinkingDelta { text } if text == "step 1" => saw_think = true,
            ProviderChunk::ThinkingSignature { signature } if signature == "abc" => saw_sig = true,
            _ => {}
        }
    }
    assert!(saw_think, "expected ThinkingDelta");
    assert!(saw_sig, "expected ThinkingSignature");
}

#[test]
fn decode_ping_is_swallowed() {
    let events = [
        ("message_start", MODEL_START),
        ("ping", PING),
        ("ping", PING),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    let (_state, results) = run_state_machine(&events);
    let chunks = ok_chunks(results);
    assert!(
        !chunks
            .iter()
            .any(|c| matches!(c, ProviderChunk::TextDelta { .. }))
    );
    let stops = chunks
        .iter()
        .filter(|c| matches!(c, ProviderChunk::Stop { .. }))
        .count();
    assert_eq!(stops, 1);
}

#[test]
fn decode_error_event_terminates() {
    let err = r#"{"type":"error","error":{"type":"overloaded_error","message":"too busy"}}"#;
    let events = [("message_start", MODEL_START), ("error", err)];
    let (state, results) = run_state_machine(&events);
    assert!(state.fatal);
    let last = results.last().expect("at least one chunk");
    assert!(last.is_err(), "last must be Err");
    let kind = &last.as_ref().err().unwrap().kind;
    assert!(matches!(kind, ProviderErrorKind::ServerError { .. }));
}

#[test]
fn decode_malformed_json_continues() {
    // A bad data item in the middle; the state machine should yield Malformed and then
    // continue.
    let bad = r#"{not json}"#;
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", bad),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    let (state, results) = run_state_machine(&events);
    assert!(state.stopped);
    let mut saw_malformed = false;
    let mut saw_text = false;
    for r in results {
        match r {
            Err(e) if matches!(e.kind, ProviderErrorKind::Malformed(_)) => saw_malformed = true,
            Ok(ProviderChunk::TextDelta { text }) if text == "hello " => saw_text = true,
            _ => {}
        }
    }
    assert!(saw_malformed);
    assert!(saw_text);
}

// ---------- decode_stream_generic end-to-end: via AnthropicSseDecoder ----

#[tokio::test]
async fn decode_stream_end_to_end_text_path() {
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_stop", TEXT_STOP_0),
        ("message_delta", MSG_DELTA_END),
        ("message_stop", MSG_STOP),
    ];
    let chunks = run_decode_stream_generic(&events, CancellationToken::new()).await;
    assert!(
        chunks.iter().all(|r| r.is_ok()),
        "got error chunks: {:?}",
        chunks
    );
    let last = chunks.last().unwrap().as_ref().ok().unwrap();
    assert!(matches!(last, ProviderChunk::Usage(_)));
}

#[tokio::test]
async fn decode_stream_protocol_violation_when_no_stop() {
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
        ("content_block_stop", TEXT_STOP_0),
        // no message_delta
    ];
    let chunks = run_decode_stream_generic(&events, CancellationToken::new()).await;
    let last = chunks.last().expect("chunks");
    assert!(last.is_err());
    let kind = &last.as_ref().err().unwrap().kind;
    assert!(matches!(kind, ProviderErrorKind::ProtocolViolation { .. }));
}

// ---------- cache breakpoint placement ----------

/// True if the (last) content block of `messages[idx]` carries a cache breakpoint.
fn message_has_breakpoint(w: &wire::CreateMessageParams, idx: usize) -> bool {
    let wire::MessageParamContent::MessageParamContentVariant1(blocks) = &w.messages[idx].content
    else {
        return false;
    };
    blocks.iter().any(|b| match b {
        wire::ContentBlockParam::TextBlockParam(b) => b.cache_control.is_some(),
        wire::ContentBlockParam::ToolUseBlockParam(b) => b.cache_control.is_some(),
        wire::ContentBlockParam::ToolResultBlockParam(b) => b.cache_control.is_some(),
        wire::ContentBlockParam::ImageBlockParam(b) => b.cache_control.is_some(),
        _ => false,
    })
}

fn system_has_breakpoint(w: &wire::CreateMessageParams) -> bool {
    matches!(
        &w.system,
        Some(wire::SystemPrompt::SystemPromptVariant1(blocks))
            if blocks.iter().any(|b| b.cache_control.is_some())
    )
}

fn text_msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: vec![MessageContent::Text { text: text.into() }].into(),
    }
}

/// The static-prefix breakpoint goes on `system`, and the rolling breakpoints land on the
/// **most recent** messages (end-biased ladder), never on the oldest ones.
#[test]
fn cache_breakpoints_are_end_biased() {
    let messages: Vec<Message> = (0..6)
        .map(|i| {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            text_msg(role, &format!("m{i}"))
        })
        .collect();
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: Some("sys".into()),
        messages,
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    // 1 static (system) + 3 rolling = 4 total, never exceeding MAX_CACHE_BREAKPOINTS.
    assert!(
        system_has_breakpoint(&w),
        "system must carry the static breakpoint"
    );
    // The three most recent messages (indices 3,4,5) get the rolling breakpoints.
    assert!(message_has_breakpoint(&w, 5));
    assert!(message_has_breakpoint(&w, 4));
    assert!(message_has_breakpoint(&w, 3));
    // Older messages must NOT — otherwise the budget was wasted on a short prefix.
    assert!(!message_has_breakpoint(&w, 2));
    assert!(!message_has_breakpoint(&w, 1));
    assert!(!message_has_breakpoint(&w, 0));
}

/// Without a system prompt, the static-prefix breakpoint falls on the last tool, and the
/// rolling budget grows to 3 messages.
#[test]
fn cache_breakpoint_falls_back_to_last_tool_without_system() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: None,
        messages: vec![text_msg(Role::User, "hi")],
        tools: vec![
            ToolSchema {
                name: "a".into(),
                description: "first".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            },
            ToolSchema {
                name: "b".into(),
                description: "second".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            },
        ],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_core::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    assert!(!system_has_breakpoint(&w));
    let tools = w.tools.as_ref().expect("tools");
    let breakpoint_on = |i: usize| {
        let wire::ToolUnion::Tool(t) = &tools[i] else {
            panic!("expected Tool");
        };
        t.cache_control.is_some()
    };
    // Only the LAST tool carries the static-prefix breakpoint (it caches tools[0..=last]).
    assert!(!breakpoint_on(0));
    assert!(breakpoint_on(1));
}

#[tokio::test]
async fn decode_stream_cancel_terminates_silently() {
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
    ];
    let cancel = CancellationToken::new();
    cancel.cancel(); // Cancel immediately.
    let chunks = run_decode_stream_generic(&events, cancel).await;
    // Cancel immediately → the stream should end at once, yielding no `Err(Canceled)`.
    assert!(chunks.iter().all(|r| r.is_ok()), "expected no Err chunks");
}
