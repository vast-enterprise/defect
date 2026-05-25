//! Anthropic Messages 协议层单测。
//!
//! 重点覆盖：
//! - `encode_request` 字段映射（model / system / tools / tool_choice /
//!   thinking / sampling / messages 的 4 种 content）
//! - `decode_stream` SSE 状态机：
//!   - 单 tool_use 完整路径（Start→ArgsDelta→Stop→ToolUseEnd→message_delta）
//!   - 两个 tool_use 并发（不同 index）的 ArgsDelta 交错
//!   - thinking + signature_delta 交替
//!   - `event: error` 终止流
//!   - `event: ping` 被吞
//!   - 单条 SSE JSON 解析失败 → `Malformed`，流不终止
//!   - 流末未收到 stop → `ProtocolViolation`
//!   - cancel 触发 → 流静默终结
//!
//! 测试通过 [`decode_stream_generic`] 喂 `futures::stream::iter` 构造的
//! `Stream<Item = Result<Sse, _>>`，避开 toac `SseBody::new` 的 pub-crate
//! 限制；运行期 [`decode_stream`] 走的是同一份解码核心，覆盖等价。

use defect_agent::llm::{
    CompletionRequest, Message, MessageContent, ProviderChunk, ProviderErrorKind, Role,
    SamplingParams, StopReason, ThinkingConfig, ToolChoice, ToolResultBody,
};
use defect_agent::tool::ToolSchema;
use futures::StreamExt;
use serde_json::json;
use sse_stream::Sse;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::wire::anthropic::components as wire;

// ---------- helpers ------------------------------------------------------

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

/// 直接喂 `Vec<Sse>` 给 `process_sse` —— 测试不走 hyper body，避免引入
/// 整套 transport，专门测状态机本身。
fn run_state_machine(
    events: &[(&str, &str)],
) -> (DecoderState, Vec<Result<ProviderChunk, ProviderError>>) {
    let mut state = DecoderState::default();
    let mut out = Vec::new();
    for sse in make_sse_events(events) {
        let mut buf = Vec::new();
        process_sse(&mut state, sse, &mut buf);
        // process_sse 内部把多 chunk 反序压栈给 poll_next 用 pop()——
        // 这里测试要按时间序还原，反转一次。
        buf.reverse();
        out.extend(buf);
        if state.fatal {
            break;
        }
    }
    (state, out)
}

/// 把若干 `(event, data)` 装进一个 `Stream<Item = Result<Sse, NeverError>>`，
/// 喂给 [`decode_stream_generic`] 来端到端地走 [`AnthropicSseDecoder`]。
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

// ---------- encode_request ----------------------------------------------

#[test]
fn encode_minimal_request() {
    let req = CompletionRequest {
        model: "claude-opus-4-7".into(),
        system: Some("you are helpful".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text { text: "hi".into() }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
    };
    let wire_req = encode_request(&req);
    assert_eq!(wire_req.max_tokens, i64::from(DEFAULT_MAX_TOKENS));
    assert!(matches!(wire_req.stream, Some(true)));
    assert!(matches!(
        wire_req.system,
        Some(wire::SystemPrompt::SystemPromptVariant0(ref s)) if s == "you are helpful"
    ));
    assert_eq!(wire_req.messages.len(), 1);
    assert!(matches!(
        wire_req.messages[0].role,
        wire::MessageParamRole::User
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
            content: vec![MessageContent::Text { text: "x".into() }],
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
        },
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
                }],
            },
            Message {
                role: Role::User,
                content: vec![MessageContent::ToolResult {
                    tool_use_id: "toolu_1".into(),
                    output: ToolResultBody::Text {
                        text: "hello".into(),
                    },
                    is_error: false,
                }],
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
    };
    let w = encode_request(&req);

    assert!(matches!(
        w.tool_choice,
        Some(wire::ToolChoice::ToolChoiceTool(ref t)) if t.name == "fs_read"
    ));

    let tools = w.tools.expect("tools");
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

    // assistant tool_use round-trip
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

    // user tool_result round-trip
    let user = match &w.messages[1].content {
        wire::MessageParamContent::MessageParamContentVariant1(v) => v,
        _ => panic!("expected list content"),
    };
    let wire::ContentBlockParam::ToolResultBlockParam(tr) = &user[0] else {
        panic!("expected tool_result_block_param");
    };
    assert_eq!(tr.tool_use_id, "toolu_1");
    assert_eq!(tr.is_error, Some(false));
}

// ---------- thinking round-trip (signature gating) ---------------------

/// 编码一条带 [`MessageContent::Thinking`] 的 assistant message，返回
/// content blocks 列表用于断言 ThinkingBlockParam 的存在/缺失。
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
            ],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
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
    // 期望两个 block：thinking + text。
    assert_eq!(blocks.len(), 2);
    let wire::ContentBlockParam::ThinkingBlockParam(t) = &blocks[0] else {
        panic!("expected thinking block first, got {:?}", blocks[0]);
    };
    assert_eq!(t.thinking, "step 1");
    assert_eq!(t.signature, "sig-abc");
}

#[test]
fn encode_thinking_without_signature_skips_thinking_block_param() {
    // 跨 provider 切回 Anthropic：上一轮是 OpenAI/DeepSeek 出的 thinking
    // 文本，没有 signature。Anthropic wire 上 signature 是 required，
    // 整块跳过——只保留 text。
    let blocks = encode_with_thinking("step 1", None);
    assert_eq!(blocks.len(), 1);
    assert!(matches!(
        &blocks[0],
        wire::ContentBlockParam::TextBlockParam(t) if t.text == "answer"
    ));
}

// ---------- decode_stream / state machine -------------------------------

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

    // 期望序列：MessageStart, Usage(input=42), TextDelta x2, ToolUseStart,
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
    // 一条坏 data 在中间，状态机应该 yield Malformed 而后继续。
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

// ---------- decode_stream_generic 端到端：经过 AnthropicSseDecoder ----

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
        // 没有 message_delta
    ];
    let chunks = run_decode_stream_generic(&events, CancellationToken::new()).await;
    let last = chunks.last().expect("chunks");
    assert!(last.is_err());
    let kind = &last.as_ref().err().unwrap().kind;
    assert!(matches!(kind, ProviderErrorKind::ProtocolViolation { .. }));
}

#[tokio::test]
async fn decode_stream_cancel_terminates_silently() {
    let events = [
        ("message_start", MODEL_START),
        ("content_block_start", TEXT_START_0),
        ("content_block_delta", TEXT_DELTA_0),
    ];
    let cancel = CancellationToken::new();
    cancel.cancel(); // 一上来就取消
    let chunks = run_decode_stream_generic(&events, cancel).await;
    // 立即取消 → 流应该立刻结束，不 yield 任何 Err（Canceled）。
    assert!(chunks.iter().all(|r| r.is_ok()), "expected no Err chunks");
}
