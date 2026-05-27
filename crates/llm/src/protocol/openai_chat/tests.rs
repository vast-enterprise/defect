//! OpenAI Chat Completions 协议层单测。
//!
//! 重点覆盖：
//! - `encode_request` 字段映射（system 提升到 messages[0]、tool_choice
//!   多 variant、tools.function 包装、ToolUse / ToolResult 拆分、Image
//!   url + base64、stream / stream_options 强制开启）
//! - `decode_stream` SSE 状态机：
//!   - 单 tool_call 完整路径（Start→ArgsDelta→finish_reason→ToolUseEnd→Stop）
//!   - 两个 tool_call 并发（不同 index）的 ArgsDelta 交错
//!   - DeepSeek `reasoning_content` → ThinkingDelta（裸 JSON 提取）
//!   - `data: [DONE]` 终止流
//!   - 单条 SSE JSON 解析失败 → `Malformed`，流不终止
//!   - 流末未收到 finish_reason / [DONE] → `ProtocolViolation`
//!   - cancel 触发 → 流静默终结
//!   - 末尾 usage chunk（`choices: []` + `usage: {...}`）→ Usage chunk
//!   - finish_reason 各 variant → StopReason 映射

use defect_agent::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderErrorKind, Role,
    SamplingParams, StopReason, ThinkingConfig, ThinkingEcho, ToolChoice, ToolResultBody,
};
use defect_agent::tool::ToolSchema;
use futures::StreamExt;
use serde_json::json;
use sse_stream::Sse;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::wire::openai::components as wire;

// ---------- helpers ------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("test sse never errors")]
struct NeverError;

fn make_sse_events(datas: &[&str]) -> Vec<Sse> {
    datas
        .iter()
        .map(|data| Sse {
            event: None,
            data: Some((*data).to_owned()),
            id: None,
            retry: None,
        })
        .collect()
}

/// 直接喂 `Vec<Sse>` 给 `process_sse`，跳过 hyper / transport，专测状态机。
fn run_state_machine(datas: &[&str]) -> (DecoderState, Vec<Result<ProviderChunk, ProviderError>>) {
    let mut state = DecoderState::default();
    let mut out = Vec::new();
    for sse in make_sse_events(datas) {
        let mut buf = Vec::new();
        process_sse(&mut state, sse, &mut buf, usage_from_wire);
        // process_sse 内部反序压栈给 poll_next 的 pop()——测试要按时间序，
        // 这里再反一次。
        buf.reverse();
        out.extend(buf);
        if state.fatal || state.done {
            break;
        }
    }
    (state, out)
}

async fn run_decode_stream_generic(
    datas: &[&str],
    cancel: CancellationToken,
) -> Vec<Result<ProviderChunk, ProviderError>> {
    let items: Vec<Result<Sse, NeverError>> = make_sse_events(datas).into_iter().map(Ok).collect();
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
fn encode_minimal_request_promotes_system_to_messages0() {
    let req = CompletionRequest {
        model: "gpt-4o-mini".into(),
        system: Some("you are helpful".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text { text: "hi".into() }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    // stream 强制 true + include_usage 强制 true
    assert_eq!(w.stream, Some(true));
    assert!(matches!(
        w.stream_options,
        Some(
            wire::ChatCompletionStreamOptions::ChatCompletionStreamOptionsVariant0(
                wire::ChatCompletionStreamOptionsVariant0 {
                    include_usage: Some(true),
                    ..
                }
            )
        )
    ));

    // messages[0] 是 system，messages[1] 是 user
    assert_eq!(w.messages.len(), 2);
    assert!(matches!(
        &w.messages[0],
        wire::ChatCompletionRequestMessage::ChatCompletionRequestSystemMessage(s)
            if matches!(
                &s.content,
                wire::ChatCompletionRequestSystemMessageContent::ChatCompletionRequestSystemMessageContentVariant0(t) if t == "you are helpful"
            )
    ));
    assert!(matches!(
        &w.messages[1],
        wire::ChatCompletionRequestMessage::ChatCompletionRequestUserMessage(_)
    ));

    // tools / tool_choice / reasoning_effort 默认
    assert!(w.tools.is_none());
    assert!(matches!(
        w.tool_choice,
        Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionToolChoiceOptionVariant0(
                wire::ChatCompletionToolChoiceOptionVariant0::Auto
            )
        )
    ));
    assert!(w.reasoning_effort.is_none());

    // model 用 Variant0 (free string)
    assert!(matches!(
        w.model,
        wire::ModelIdsShared::ModelIdsSharedVariant0(ref s) if s == "gpt-4o-mini"
    ));
}

#[test]
fn encode_request_carries_sampling_and_thinking() {
    let req = CompletionRequest {
        model: "o3-mini".into(),
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
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    assert_eq!(w.max_completion_tokens, Some(8000));
    assert!(matches!(
        w.temperature,
        Some(wire::CreateChatCompletionRequestTemperature::CreateChatCompletionRequestTemperatureVariant0(t)) if (t - 0.5).abs() < 1e-6
    ));
    assert!(matches!(
        w.top_p,
        Some(wire::CreateChatCompletionRequestTopP::CreateChatCompletionRequestTopPVariant0(t)) if (t - 0.9).abs() < 1e-6
    ));
    assert!(matches!(
        w.stop,
        Some(wire::StopConfiguration::StopConfigurationVariant1(ref v)) if v == &["END".to_string()]
    ));
    assert!(matches!(
        w.tool_choice,
        Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionToolChoiceOptionVariant0(
                wire::ChatCompletionToolChoiceOptionVariant0::Required
            )
        )
    ));
    assert!(matches!(
        w.reasoning_effort,
        Some(wire::ReasoningEffort::ReasoningEffortVariant0(
            wire::ReasoningEffortVariant0::Medium
        ))
    ));
    // top_k 被 OpenAI 协议层丢弃，wire 上没有该字段。
}

#[test]
fn encode_request_sets_stable_prompt_cache_key_from_prefix_shape() {
    let req = CompletionRequest {
        model: "gpt-4o-mini".into(),
        system: Some("you are helpful".into()),
        messages: vec![Message {
            role: Role::User,
            content: vec![MessageContent::Text {
                text: "turn-specific text".into(),
            }],
        }],
        tools: vec![ToolSchema {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }),
        }],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };

    let first = encode_request(&req).prompt_cache_key;
    let mut req_with_new_turn_text = req.clone();
    req_with_new_turn_text.messages = vec![Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: "different turn text".into(),
        }],
    }];
    let second = encode_request(&req_with_new_turn_text).prompt_cache_key;

    assert_eq!(
        first, second,
        "turn-local messages should not perturb cache key"
    );
    assert!(first.is_some());
}

#[test]
fn encode_request_changes_prompt_cache_key_when_prefix_changes() {
    let req = CompletionRequest {
        model: "gpt-4o-mini".into(),
        system: Some("system-a".into()),
        messages: vec![],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let base = encode_request(&req).prompt_cache_key;

    let mut changed = req.clone();
    changed.system = Some("system-b".into());
    let changed_key = encode_request(&changed).prompt_cache_key;

    assert_ne!(base, changed_key);
}

#[test]
fn encode_request_splits_tool_use_and_tool_result_into_separate_messages() {
    let req = CompletionRequest {
        model: "gpt-4o-mini".into(),
        system: None,
        messages: vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    MessageContent::Text {
                        text: "calling".into(),
                    },
                    MessageContent::ToolUse {
                        id: "call_1".into(),
                        name: "fs_read".into(),
                        args: json!({"path": "/tmp/a"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![
                    MessageContent::Text {
                        text: "see results below".into(),
                    },
                    MessageContent::ToolResult {
                        tool_use_id: "call_1".into(),
                        output: ToolResultBody::Text {
                            text: "hello".into(),
                        },
                        is_error: false,
                    },
                ],
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
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    // tool_choice = Named → ChatCompletionNamedToolChoice
    assert!(matches!(
        w.tool_choice,
        Some(wire::ChatCompletionToolChoiceOption::ChatCompletionNamedToolChoice(ref t))
            if t.function.name == "fs_read"
    ));

    // tools.0 = ChatCompletionTool { function: {...} }
    let tools = w.tools.expect("tools");
    let wire::CreateChatCompletionRequestTools::ChatCompletionTool(t) = &tools[0] else {
        panic!("expected ChatCompletionTool");
    };
    assert_eq!(t.function.name, "fs_read");
    assert_eq!(t.function.description.as_deref(), Some("Read a file"));
    assert!(
        t.function
            .parameters
            .as_ref()
            .unwrap()
            .contains_key("properties")
    );

    // 期望 wire messages：
    //   [0] assistant (text "calling" + tool_calls=[call_1])
    //   [1] user      (text "see results below")
    //   [2] tool      (tool_call_id=call_1, content="hello")
    assert_eq!(w.messages.len(), 3);

    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!("expected assistant");
    };
    assert!(matches!(
        asst.content,
        Some(wire::ChatCompletionRequestAssistantMessageContent::ChatCompletionRequestAssistantMessageContentVariant0(
            wire::ChatCompletionRequestAssistantMessageContentVariant0::ChatCompletionRequestAssistantMessageContentVariant0Variant0(ref s)
        )) if s == "calling"
    ));
    let calls = asst.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(calls.len(), 1);
    let wire::ChatCompletionMessageToolCallsItem::ChatCompletionMessageToolCall(call) = &calls[0]
    else {
        panic!("expected function tool call");
    };
    assert_eq!(call.id, "call_1");
    assert_eq!(call.function.name, "fs_read");
    // arguments 是 stringified JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&call.function.arguments).expect("valid JSON");
    assert_eq!(parsed.get("path"), Some(&json!("/tmp/a")));

    let wire::ChatCompletionRequestMessage::ChatCompletionRequestUserMessage(user) = &w.messages[1]
    else {
        panic!("expected user");
    };
    let wire::ChatCompletionRequestUserMessageContent::ChatCompletionRequestUserMessageContentVariant1(parts) = &user.content else {
        panic!("expected list user content");
    };
    assert_eq!(parts.len(), 1);
    let wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartText(t) = &parts[0] else {
        panic!("expected text part");
    };
    assert_eq!(t.text, "see results below");

    let wire::ChatCompletionRequestMessage::ChatCompletionRequestToolMessage(tool_msg) =
        &w.messages[2]
    else {
        panic!("expected tool");
    };
    assert_eq!(tool_msg.tool_call_id, "call_1");
    assert!(matches!(
        &tool_msg.content,
        wire::ChatCompletionRequestToolMessageContent::ChatCompletionRequestToolMessageContentVariant0(s) if s == "hello"
    ));
}

#[test]
fn encode_request_image_base64_and_url() {
    let req = CompletionRequest {
        model: "gpt-4o".into(),
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![
                MessageContent::Image {
                    mime: "image/png".into(),
                    data: ImageData::Base64 {
                        encoded: "AAAA".into(),
                    },
                },
                MessageContent::Image {
                    mime: "image/jpeg".into(),
                    data: ImageData::Url {
                        url: "https://example.com/x.jpg".into(),
                    },
                },
            ],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);

    let wire::ChatCompletionRequestMessage::ChatCompletionRequestUserMessage(user) = &w.messages[0]
    else {
        panic!("expected user");
    };
    let wire::ChatCompletionRequestUserMessageContent::ChatCompletionRequestUserMessageContentVariant1(parts) = &user.content else {
        panic!("expected list");
    };
    assert_eq!(parts.len(), 2);
    let wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartImage(img0) = &parts[0] else {
        panic!("expected image part");
    };
    assert_eq!(img0.image_url.url, "data:image/png;base64,AAAA");
    let wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartImage(img1) = &parts[1] else {
        panic!("expected image part");
    };
    assert_eq!(img1.image_url.url, "https://example.com/x.jpg");
}

// ---------- thinking round-trip (Required vs Forbidden) ----------------

/// 给定一条带 [`MessageContent::Thinking`] 的 assistant message，按
/// `echo_mode` 调 `encode_request_with_echo` 并返回 wire 上 assistant
/// message 的 `reasoning_content`。
fn encode_with_thinking(
    text: &str,
    signature: Option<&str>,
    echo_mode: ThinkingEcho,
) -> Option<String> {
    let req = CompletionRequest {
        model: "deepseek-v4-pro".into(),
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
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request_with_echo(&req, echo_mode);
    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!("expected assistant message");
    };
    asst.reasoning_content.clone()
}

#[test]
fn encode_thinking_required_writes_reasoning_content() {
    let rc = encode_with_thinking("step 1\nstep 2", None, ThinkingEcho::Required);
    assert_eq!(rc.as_deref(), Some("step 1\nstep 2"));
}

#[test]
fn encode_thinking_forbidden_drops_reasoning_content() {
    // Forbidden 不写——OpenAI 官方与 deepseek-reasoner/R1 都按这条走。
    let rc = encode_with_thinking("step 1", None, ThinkingEcho::Forbidden);
    assert!(rc.is_none(), "Forbidden must not emit reasoning_content");
}

#[test]
fn encode_thinking_optional_writes_reasoning_content() {
    let rc = encode_with_thinking("step 1", None, ThinkingEcho::Optional);
    assert_eq!(rc.as_deref(), Some("step 1"));
}

#[test]
fn encode_thinking_required_but_empty_text_is_none() {
    // 空 buf —— 没东西可回放，不要硬塞空字符串触发服务端 invalid_request。
    let rc = encode_with_thinking("", None, ThinkingEcho::Required);
    assert!(rc.is_none());
}

#[test]
fn encode_thinking_concatenates_multiple_thinking_blocks() {
    let req = CompletionRequest {
        model: "deepseek-v4-pro".into(),
        system: None,
        messages: vec![Message {
            role: Role::Assistant,
            content: vec![
                MessageContent::Thinking {
                    text: "a".into(),
                    signature: None,
                },
                MessageContent::Thinking {
                    text: "b".into(),
                    signature: None,
                },
                MessageContent::Text { text: "ok".into() },
            ],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request_with_echo(&req, ThinkingEcho::Required);
    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!();
    };
    assert_eq!(asst.reasoning_content.as_deref(), Some("ab"));
}

#[test]
fn encode_thinking_only_required_adds_empty_content() {
    let req = CompletionRequest {
        model: "deepseek-v4-pro".into(),
        system: None,
        messages: vec![Message {
            role: Role::Assistant,
            content: vec![MessageContent::Thinking {
                text: "step 1".into(),
                signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request_with_echo(&req, ThinkingEcho::Required);
    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!("expected assistant message");
    };
    assert_eq!(asst.reasoning_content.as_deref(), Some("step 1"));
    assert!(matches!(
        asst.content,
        Some(
            wire::ChatCompletionRequestAssistantMessageContent::ChatCompletionRequestAssistantMessageContentVariant0(
                wire::ChatCompletionRequestAssistantMessageContentVariant0::ChatCompletionRequestAssistantMessageContentVariant0Variant0(ref text)
            )
        ) if text.is_empty()
    ));
}

#[test]
fn encode_thinking_only_forbidden_keeps_content_none() {
    let req = CompletionRequest {
        model: "gpt-4o".into(),
        system: None,
        messages: vec![Message {
            role: Role::Assistant,
            content: vec![MessageContent::Thinking {
                text: "step 1".into(),
                signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request_with_echo(&req, ThinkingEcho::Forbidden);
    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!("expected assistant message");
    };
    assert!(asst.reasoning_content.is_none());
    assert!(asst.content.is_none());
}

#[test]
fn encode_request_default_forbids_thinking_echo() {
    // 默认 `encode_request` (无 echo arg) 等价于 Forbidden —— 防止
    // 通过该入口绕过 capability 矩阵把 reasoning_content 漏到
    // 不该收的厂商上。
    let req = CompletionRequest {
        model: "gpt-4o".into(),
        system: None,
        messages: vec![Message {
            role: Role::Assistant,
            content: vec![
                MessageContent::Thinking {
                    text: "leak?".into(),
                    signature: None,
                },
                MessageContent::Text { text: "ok".into() },
            ],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: SamplingParams::default(),
        hosted_capabilities: ::defect_agent::llm::HostedCapabilities::default(),
    };
    let w = encode_request(&req);
    let wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(asst) =
        &w.messages[0]
    else {
        panic!();
    };
    assert!(asst.reasoning_content.is_none());
}

// ---------- decode_stream / state machine -------------------------------

const TEXT_CHUNK_1: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_CHUNK_2: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hello "},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_CHUNK_3: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"world"},"logprobs":null,"finish_reason":null}]}"#;
const TEXT_CHUNK_FINISH_STOP: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#;
const USAGE_CHUNK: &str = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":3}}}"#;
const DONE: &str = "[DONE]";

#[test]
fn decode_text_path_emits_message_start_then_text_then_stop_then_usage() {
    let datas = [
        TEXT_CHUNK_1,
        TEXT_CHUNK_2,
        TEXT_CHUNK_3,
        TEXT_CHUNK_FINISH_STOP,
        USAGE_CHUNK,
        DONE,
    ];
    let (state, results) = run_state_machine(&datas);
    assert!(state.stopped);
    assert!(state.done);
    let chunks = ok_chunks(results);

    let mut iter = chunks.into_iter();
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::MessageStart { id, model } if id == "chatcmpl-1" && model == "gpt-4o-mini"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::TextDelta { text } if text == "hello "
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::TextDelta { text } if text == "world"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Stop {
            reason: StopReason::EndTurn
        }
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Usage(u) if u.input_tokens == Some(10)
            && u.output_tokens == Some(5)
            && u.cache_read_input_tokens == Some(3)
    ));
}

#[test]
fn decode_single_tool_call_full_path() {
    let chunks_raw = [
        r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":2,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"calc","arguments":""}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":2,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":2,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}"#,
        r#"{"id":"chatcmpl-2","object":"chat.completion.chunk","created":2,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        DONE,
    ];
    let (state, results) = run_state_machine(&chunks_raw);
    assert!(state.stopped);
    let chunks = ok_chunks(results);

    let mut iter = chunks.into_iter();
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::MessageStart { .. }
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseStart { id, name } if id == "call_a" && name == "calc"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseArgsDelta { id, fragment } if id == "call_a" && fragment.starts_with("{\"x\"")
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseArgsDelta { id, .. } if id == "call_a"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::ToolUseEnd { id } if id == "call_a"
    ));
    assert!(matches!(
        iter.next().unwrap(),
        ProviderChunk::Stop {
            reason: StopReason::ToolUse
        }
    ));
}

#[test]
fn decode_two_concurrent_tool_calls_interleaved_by_index() {
    let chunks_raw = [
        // call A start
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"alpha","arguments":""}}]},"finish_reason":null}]}"#,
        // call B start
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"beta","arguments":""}}]},"finish_reason":null}]}"#,
        // call A args
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"finish_reason":null}]}"#,
        // call B args
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":2}"}}]},"finish_reason":null}]}"#,
        // finish
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        DONE,
    ];
    let (_state, results) = run_state_machine(&chunks_raw);
    let chunks = ok_chunks(results);

    let starts: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseStart { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(starts, vec!["call_a", "call_b"]);

    let args: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseArgsDelta { id, fragment } => {
                Some((id.clone(), fragment.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        args,
        vec![
            ("call_a".into(), "{\"x\":1}".into()),
            ("call_b".into(), "{\"y\":2}".into()),
        ]
    );

    let ends: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            ProviderChunk::ToolUseEnd { id } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ends, vec!["call_a", "call_b"]);
}

#[test]
fn decode_reasoning_content_extension_emits_thinking_delta() {
    // DeepSeek 在 delta 上挂 `reasoning_content`，wire OAS 没有，从 raw 取。
    let chunks_raw = [
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking...","content":null},"finish_reason":null}]}"#,
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}"#,
        r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        DONE,
    ];
    let (_state, results) = run_state_machine(&chunks_raw);
    let chunks = ok_chunks(results);

    let mut saw_think = false;
    let mut saw_text = false;
    for c in &chunks {
        match c {
            ProviderChunk::ThinkingDelta { text } if text == "thinking..." => saw_think = true,
            ProviderChunk::TextDelta { text } if text == "answer" => saw_text = true,
            _ => {}
        }
    }
    assert!(saw_think, "expected ThinkingDelta from reasoning_content");
    assert!(saw_text, "expected TextDelta from content");
}

#[test]
fn decode_done_terminates_stream() {
    let datas = [TEXT_CHUNK_1, TEXT_CHUNK_FINISH_STOP, DONE, USAGE_CHUNK];
    let (state, _results) = run_state_machine(&datas);
    assert!(state.done, "[DONE] should set done flag");
    // run_state_machine 在 done=true 时 break，USAGE_CHUNK 不被处理——
    // 真实场景里上游也不会在 [DONE] 之后再发数据。
}

#[test]
fn decode_malformed_json_continues() {
    let bad = r#"{not json}"#;
    let datas = [
        TEXT_CHUNK_1,
        bad,
        TEXT_CHUNK_2,
        TEXT_CHUNK_FINISH_STOP,
        DONE,
    ];
    let (state, results) = run_state_machine(&datas);
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

#[test]
fn decode_finish_reason_variants_map_to_stop_reason() {
    let cases = [
        ("stop", StopReason::EndTurn),
        ("length", StopReason::MaxTokens),
        ("tool_calls", StopReason::ToolUse),
        ("function_call", StopReason::ToolUse),
        ("content_filter", StopReason::Refusal),
    ];
    for (wire_name, expected) in cases {
        let chunk1 =
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#
                .to_string();
        let final_chunk = format!(
            r#"{{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{{"index":0,"delta":{{}},"finish_reason":"{wire_name}"}}]}}"#
        );
        let datas = [chunk1.as_str(), final_chunk.as_str(), DONE];
        let (_state, results) = run_state_machine(&datas);
        let chunks = ok_chunks(results);
        let stop = chunks
            .iter()
            .find_map(|c| match c {
                ProviderChunk::Stop { reason } => Some(*reason),
                _ => None,
            })
            .expect("expected Stop");
        assert_eq!(stop, expected, "wire finish_reason={wire_name}");
    }
}

#[test]
fn decode_final_usage_chunk_has_empty_choices() {
    let datas = [TEXT_CHUNK_1, TEXT_CHUNK_FINISH_STOP, USAGE_CHUNK, DONE];
    let (_state, results) = run_state_machine(&datas);
    let chunks = ok_chunks(results);
    let usage = chunks
        .iter()
        .find_map(|c| match c {
            ProviderChunk::Usage(u) => Some(*u),
            _ => None,
        })
        .expect("expected Usage");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.cache_read_input_tokens, Some(3));
}

// ---------- decode_stream_generic 端到端：经过 OpenAiSseDecoder ---------

#[tokio::test]
async fn decode_stream_end_to_end_text_path() {
    let datas = [
        TEXT_CHUNK_1,
        TEXT_CHUNK_2,
        TEXT_CHUNK_FINISH_STOP,
        USAGE_CHUNK,
        DONE,
    ];
    let chunks = run_decode_stream_generic(&datas, CancellationToken::new()).await;
    assert!(
        chunks.iter().all(|r| r.is_ok()),
        "got error chunks: {:?}",
        chunks
    );
    let last = chunks.last().unwrap().as_ref().ok().unwrap();
    assert!(matches!(last, ProviderChunk::Usage(_)));
}

#[tokio::test]
async fn decode_stream_protocol_violation_when_no_finish_no_done() {
    let datas = [TEXT_CHUNK_1, TEXT_CHUNK_2];
    let chunks = run_decode_stream_generic(&datas, CancellationToken::new()).await;
    let last = chunks.last().expect("chunks");
    assert!(last.is_err());
    let kind = &last.as_ref().err().unwrap().kind;
    assert!(matches!(kind, ProviderErrorKind::ProtocolViolation { .. }));
}

#[tokio::test]
async fn decode_stream_cancel_terminates_silently() {
    let datas = [TEXT_CHUNK_1, TEXT_CHUNK_2];
    let cancel = CancellationToken::new();
    cancel.cancel();
    let chunks = run_decode_stream_generic(&datas, cancel).await;
    // 立即取消 → 流应该立刻结束，不 yield 任何 Err（Canceled）。
    assert!(chunks.iter().all(|r| r.is_ok()), "expected no Err chunks");
}
