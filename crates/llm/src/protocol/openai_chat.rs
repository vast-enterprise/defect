//! OpenAI Chat Completions 协议编解码。
//!
//! 把 [`defect_agent::llm::CompletionRequest`] 编为 wire
//! [`crate::wire::openai::components::CreateChatCompletionRequest`]，
//! 把 SSE [`Sse`] 流（[`CreateChatCompletionStreamResponse`]）解码为
//! [`defect_agent::llm::ProviderChunk`] 流。
//!
//! 设计映射详见 `docs/outbound/llm-openai.md` §6。
//!
//! [`Sse`]: ::sse_stream::Sse
//! [`CreateChatCompletionStreamResponse`]:
//!     crate::wire::openai::components::CreateChatCompletionStreamResponse

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use defect_agent::error::BoxError;
use defect_agent::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderError,
    ProviderErrorKind, Role, StopReason, ThinkingConfig, ThinkingEcho, ToolChoice, ToolResultBody,
    Usage,
};
use defect_agent::tool::ToolSchema;
use futures::Stream;
use sse_stream::Sse;
use toac::body::codec::sse::SseEventStream;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::wire::openai::components as wire;

// ---------- encode -------------------------------------------------------

/// 把 [`CompletionRequest`] 编为 wire 请求体。
///
/// 关键映射决策（详见 `docs/outbound/llm-openai.md` §6.1）：
///
/// - 强制 `stream = true` + `stream_options.include_usage = true`：
///   协议层只跑 SSE 分支，且**必须**让上游发末尾 usage chunk，否则
///   token 计费拿不到。
/// - `system` 提升为 `messages[0]` 的 system message —— OpenAI 没有
///   顶层 system 字段（与 Anthropic 不同）。
/// - 单条 [`Message`] 在 OpenAI 形态下可能拆成多条 wire message：
///   user 消息里若混了 [`MessageContent::ToolResult`]，需要拆出独立的
///   tool message（OpenAI 用 `role: tool` + `tool_call_id` 表达工具结果，
///   不能跟 user 文本混在同一条 message）。
/// - assistant 消息的 [`MessageContent::ToolUse`] 投到 `tool_calls`
///   字段，而不是 content blocks。`args` 经 `serde_json::to_string` 转字符串
///   （OpenAI 协议规定 `function.arguments` 为 stringified JSON）。
/// - `top_k` 在 OpenAI 协议里不存在；这里直接丢弃，由 provider 层负责
///   warn（`docs/internal/llm-trait.md` §6 能力矩阵）。
/// - `max_tokens`：OpenAI 把 `max_tokens` 标 deprecated，新字段是
///   `max_completion_tokens`。一律走新字段，且**不**像 Anthropic 那样
///   兜底默认值 —— OpenAI 不强制（不传时由模型决定）。
pub fn encode_request(req: &CompletionRequest) -> wire::CreateChatCompletionRequest {
    encode_request_with_echo(req, ThinkingEcho::Forbidden)
}

/// 与 [`encode_request`] 同形态，但显式接收 thinking 回放策略。
///
/// `echo_mode` 由 provider 层从 [`defect_agent::llm::Capabilities`] 读取
/// 并传入：`Required` 时 assistant message 上的
/// [`MessageContent::Thinking`] 文本会被写到 wire 的非标
/// `reasoning_content` 字段（详见 `docs/internal/thinking-roundtrip.md` §4.2）；
/// `Forbidden`（含未配置）一律不写。
pub fn encode_request_with_echo(
    req: &CompletionRequest,
    echo_mode: ThinkingEcho,
) -> wire::CreateChatCompletionRequest {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system.as_ref() {
        messages.push(encode_system_message(sys));
    }
    for m in &req.messages {
        encode_message_into(m, echo_mode, &mut messages);
    }

    #[allow(deprecated)]
    wire::CreateChatCompletionRequest {
        // ---- 我们使用的字段 ----
        messages,
        model: wire::ModelIdsShared::ModelIdsSharedVariant0(req.model.clone()),
        stream: Some(true),
        stream_options: Some(wire::ChatCompletionStreamOptions::ChatCompletionStreamOptionsVariant0(
            wire::ChatCompletionStreamOptionsVariant0 {
                include_usage: Some(true),
                include_obfuscation: None,
            },
        )),
        max_completion_tokens: req.sampling.max_tokens.map(i64::from),
        temperature: req.sampling.temperature.map(|t| {
            wire::CreateChatCompletionRequestTemperature::CreateChatCompletionRequestTemperatureVariant0(
                f64::from(t),
            )
        }),
        top_p: req.sampling.top_p.map(|t| {
            wire::CreateChatCompletionRequestTopP::CreateChatCompletionRequestTopPVariant0(
                f64::from(t),
            )
        }),
        stop: if req.sampling.stop_sequences.is_empty() {
            None
        } else {
            Some(wire::StopConfiguration::StopConfigurationVariant1(
                req.sampling.stop_sequences.clone(),
            ))
        },
        reasoning_effort: encode_thinking(req.sampling.thinking),
        tools: if req.tools.is_empty() {
            None
        } else {
            Some(req.tools.iter().map(encode_tool).collect())
        },
        tool_choice: encode_tool_choice(&req.tool_choice),
        // ---- 不使用的字段：显式 None，方便日后 grep ----
        metadata: None,
        top_logprobs: None,
        user: None,
        safety_identifier: None,
        prompt_cache_key: None,
        service_tier: None,
        prompt_cache_retention: None,
        modalities: None,
        verbosity: None,
        frequency_penalty: None,
        presence_penalty: None,
        web_search_options: None,
        response_format: None,
        audio: None,
        store: None,
        logit_bias: None,
        logprobs: None,
        max_tokens: None,
        n: None,
        prediction: None,
        seed: None,
        parallel_tool_calls: None,
        function_call: None,
        functions: None,
    }
}

fn encode_system_message(text: &str) -> wire::ChatCompletionRequestMessage {
    wire::ChatCompletionRequestMessage::ChatCompletionRequestSystemMessage(
        wire::ChatCompletionRequestSystemMessage {
            content: wire::ChatCompletionRequestSystemMessageContent::ChatCompletionRequestSystemMessageContentVariant0(
                text.to_owned(),
            ),
            role: wire::ChatCompletionRequestSystemMessageRole::System,
            name: None,
        },
    )
}

/// 一条 [`Message`] 可能 fan-out 成多条 wire message：
/// - user 里夹带的每个 [`MessageContent::ToolResult`] 都要拆出独立 tool
///   message；
/// - assistant 的 [`MessageContent::ToolUse`] 抽到顶层 `tool_calls` 字段，
///   而不是 content。
fn encode_message_into(
    m: &Message,
    echo_mode: ThinkingEcho,
    out: &mut Vec<wire::ChatCompletionRequestMessage>,
) {
    match m.role {
        Role::User => encode_user_message_into(m, out),
        Role::Assistant => encode_assistant_message_into(m, echo_mode, out),
    }
}

fn encode_user_message_into(m: &Message, out: &mut Vec<wire::ChatCompletionRequestMessage>) {
    let mut user_parts: Vec<wire::ChatCompletionRequestUserMessageContentPart> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_use_id, text)

    for c in &m.content {
        match c {
            MessageContent::Text { text } => {
                user_parts.push(
                    wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartText(
                        wire::ChatCompletionRequestMessageContentPartText {
                            r#type: wire::ChatCompletionRequestMessageContentPartTextType::Text,
                            text: text.clone(),
                        },
                    ),
                );
            }
            MessageContent::Image { mime, data } => {
                user_parts.push(
                    wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartImage(
                        wire::ChatCompletionRequestMessageContentPartImage {
                            r#type: wire::ChatCompletionRequestMessageContentPartImageType::ImageUrl,
                            image_url: wire::ChatCompletionRequestMessageContentPartImageImageUrl {
                                url: image_url_string(mime, data),
                                detail: None,
                            },
                        },
                    ),
                );
            }
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error: _,
            } => {
                // OpenAI 的 tool message 没有 is_error 字段；用 prefix 标记，
                // 让模型从 content 里读到错误状态。is_error 主要给 Anthropic
                // 用的；这里保留它的语义但形态不一样。
                let text = match output {
                    ToolResultBody::Text { text } => text.clone(),
                    ToolResultBody::Json { value } => value.to_string(),
                    _ => String::new(),
                };
                tool_results.push((tool_use_id.clone(), text));
            }
            // non_exhaustive 的兜底：保留位置但内容空。
            _ => {
                user_parts.push(
                    wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartText(
                        wire::ChatCompletionRequestMessageContentPartText {
                            r#type: wire::ChatCompletionRequestMessageContentPartTextType::Text,
                            text: String::new(),
                        },
                    ),
                );
            }
        }
    }

    // OpenAI 的 tool message 必须独立成条 —— 不能跟 user 文本混。
    // 时序约定：user 文本 / 图片在前，tool 结果在后（模型把 tool 结果
    // 视为系统态消息，不该影响用户当前发问）。
    if !user_parts.is_empty() {
        out.push(wire::ChatCompletionRequestMessage::ChatCompletionRequestUserMessage(
            wire::ChatCompletionRequestUserMessage {
                content: wire::ChatCompletionRequestUserMessageContent::ChatCompletionRequestUserMessageContentVariant1(
                    user_parts,
                ),
                role: wire::ChatCompletionRequestUserMessageRole::User,
                name: None,
            },
        ));
    }
    for (tool_use_id, text) in tool_results {
        out.push(wire::ChatCompletionRequestMessage::ChatCompletionRequestToolMessage(
            wire::ChatCompletionRequestToolMessage {
                role: wire::ChatCompletionRequestToolMessageRole::Tool,
                content: wire::ChatCompletionRequestToolMessageContent::ChatCompletionRequestToolMessageContentVariant0(
                    text,
                ),
                tool_call_id: tool_use_id,
            },
        ));
    }
}

fn encode_assistant_message_into(
    m: &Message,
    echo_mode: ThinkingEcho,
    out: &mut Vec<wire::ChatCompletionRequestMessage>,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<wire::ChatCompletionMessageToolCallsItem> = Vec::new();
    let mut reasoning_text = String::new();

    for c in &m.content {
        match c {
            MessageContent::Text { text } => text_parts.push(text.clone()),
            MessageContent::Thinking { text, .. } => {
                // signature 字段在 OpenAI 路径上无意义（DeepSeek 不要、
                // OpenAI 自己也不要），只取文本。
                reasoning_text.push_str(text);
            }
            MessageContent::ToolUse { id, name, args } => {
                tool_calls.push(
                    wire::ChatCompletionMessageToolCallsItem::ChatCompletionMessageToolCall(
                        wire::ChatCompletionMessageToolCall {
                            id: id.clone(),
                            r#type: wire::ChatCompletionMessageToolCallType::Function,
                            function: wire::ChatCompletionMessageToolCallFunction {
                                name: name.clone(),
                                arguments: serde_json::to_string(args).unwrap_or_default(),
                            },
                        },
                    ),
                );
            }
            // ToolResult/Image 不该出现在 assistant 角色里；non_exhaustive
            // 兜底也走到这里。忽略，不投到 wire。
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        // OpenAI 要求 assistant message 必须有 content **或** tool_calls；
        // 我们这里若没有 text 就走 None，依赖 tool_calls 撑场。
        None
    } else {
        Some(wire::ChatCompletionRequestAssistantMessageContent::ChatCompletionRequestAssistantMessageContentVariant0(
            wire::ChatCompletionRequestAssistantMessageContentVariant0::ChatCompletionRequestAssistantMessageContentVariant0Variant0(
                text_parts.join(""),
            ),
        ))
    };

    let reasoning_content = match (echo_mode, reasoning_text.is_empty()) {
        (ThinkingEcho::Required, false) => Some(reasoning_text),
        // Optional 也按 Required 处理：服务端容忍多发的场景下回放更
        // 安全（DeepSeek-v4-pro 文档把它列为 must、其它 Optional 厂商
        // 多发也不报错）。
        (ThinkingEcho::Optional, false) => Some(reasoning_text),
        _ => None,
    };

    #[allow(deprecated)]
    out.push(
        wire::ChatCompletionRequestMessage::ChatCompletionRequestAssistantMessage(
            wire::ChatCompletionRequestAssistantMessage {
                content,
                refusal: None,
                role: wire::ChatCompletionRequestAssistantMessageRole::Assistant,
                name: None,
                audio: None,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                function_call: None,
                reasoning_content,
            },
        ),
    );
}

fn image_url_string(mime: &str, data: &ImageData) -> String {
    match data {
        ImageData::Url { url } => url.clone(),
        ImageData::Base64 { encoded } => format!("data:{mime};base64,{encoded}"),
        // non_exhaustive 兜底——空 URL，明显的 placeholder。
        _ => String::new(),
    }
}

fn encode_thinking(t: ThinkingConfig) -> Option<wire::ReasoningEffort> {
    match t {
        ThinkingConfig::Disabled => None,
        // OpenAI 的 thinking 不接受 budget_tokens（与 Anthropic 不同），
        // 只有等级。budget 值送丢，统一映射到 medium。
        ThinkingConfig::Enabled { .. } => Some(wire::ReasoningEffort::ReasoningEffortVariant0(
            wire::ReasoningEffortVariant0::Medium,
        )),
        _ => None,
    }
}

fn encode_tool_choice(c: &ToolChoice) -> Option<wire::ChatCompletionToolChoiceOption> {
    match c {
        ToolChoice::Auto => Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionToolChoiceOptionVariant0(
                wire::ChatCompletionToolChoiceOptionVariant0::Auto,
            ),
        ),
        ToolChoice::Required => Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionToolChoiceOptionVariant0(
                wire::ChatCompletionToolChoiceOptionVariant0::Required,
            ),
        ),
        ToolChoice::None => Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionToolChoiceOptionVariant0(
                wire::ChatCompletionToolChoiceOptionVariant0::None,
            ),
        ),
        ToolChoice::Named { name } => Some(
            wire::ChatCompletionToolChoiceOption::ChatCompletionNamedToolChoice(
                wire::ChatCompletionNamedToolChoice {
                    r#type: wire::ChatCompletionNamedToolChoiceType::Function,
                    function: wire::ChatCompletionNamedToolChoiceFunction { name: name.clone() },
                },
            ),
        ),
        _ => None,
    }
}

fn encode_tool(t: &ToolSchema) -> wire::CreateChatCompletionRequestTools {
    wire::CreateChatCompletionRequestTools::ChatCompletionTool(wire::ChatCompletionTool {
        r#type: wire::ChatCompletionToolType::Function,
        function: wire::FunctionObject {
            name: t.name.clone(),
            description: if t.description.is_empty() {
                None
            } else {
                Some(t.description.clone())
            },
            parameters: Some(json_value_to_parameters(&t.input_schema)),
            strict: None,
        },
    })
}

fn json_value_to_parameters(v: &serde_json::Value) -> wire::FunctionParameters {
    v.as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

// ---------- decode -------------------------------------------------------

/// 解码状态机内部状态。
#[derive(Debug, Default)]
struct DecoderState {
    /// 已发出 MessageStart。
    started: bool,
    /// 已发出 Stop。Stop 之后只允许再发 Usage。
    stopped: bool,
    /// 见过 `data: [DONE]` 标记。
    done: bool,
    /// 收到致命错误（解析失败重试不能继续）。
    fatal: bool,
    /// `delta.tool_calls[].index` → `tool_call_id`。OpenAI 流式 chunk
    /// 的工具调用通过 index 关联，第一帧带 id+name，后续 args 帧只有
    /// index。我们用此表把 index 还原为 ProviderChunk 里的 id。
    tool_calls: HashMap<i64, ToolCallState>,
    /// tool_calls 收到顺序（用于在 Stop 时按出现顺序发 ToolUseEnd）。
    tool_call_order: Vec<i64>,
}

#[derive(Debug, Clone)]
struct ToolCallState {
    id: String,
    /// 是否已经发过 ToolUseEnd。
    closed: bool,
}

/// SSE 流 → ProviderChunk 流。返回值实现 [`Stream`]，drop 即取消。
///
/// `cancel` 触发后流静默终结，与
/// [`docs/internal/llm-trait.md`] §2.2 一致。
pub fn decode_stream(
    sse: SseEventStream,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    decode_stream_generic(sse, cancel)
}

/// 与 [`decode_stream`] 同形态，但对入参 `Stream` 类型泛化，方便测试
/// 直接喂 `futures::stream::iter`，不经过 toac transport。
pub fn decode_stream_generic<S, E>(
    sse: S,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send
where
    S: Stream<Item = Result<Sse, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    OpenAiSseDecoder {
        inner: sse,
        cancel,
        state: DecoderState::default(),
        pending: Vec::new(),
        finished: false,
        _err: std::marker::PhantomData::<E>,
    }
}

struct OpenAiSseDecoder<S, E> {
    inner: S,
    cancel: CancellationToken,
    state: DecoderState,
    /// 单帧可能产出多个 chunk（finish_reason 帧通常会紧跟 ToolUseEnd*N
    /// + Stop）。先存到 `pending`，poll_next 用 `pop()` 逐个吐。
    pending: Vec<Result<ProviderChunk, ProviderError>>,
    finished: bool,
    _err: std::marker::PhantomData<E>,
}

impl<S, E> Stream for OpenAiSseDecoder<S, E>
where
    S: Stream<Item = Result<Sse, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<ProviderChunk, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: standard pin-projection through a single field. We
        // never move `inner` out and `_err` is a zero-sized PhantomData.
        let this = unsafe { self.get_unchecked_mut() };
        loop {
            if let Some(item) = this.pending.pop() {
                return Poll::Ready(Some(item));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            if this.cancel.is_cancelled() {
                this.finished = true;
                return Poll::Ready(None);
            }

            // SAFETY: pin-projection through a single field.
            let inner = unsafe { Pin::new_unchecked(&mut this.inner) };
            match inner.poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.finished = true;
                    // 既无 [DONE] 又无 stop chunk —— ProtocolViolation。
                    if !this.state.done
                        && !this.state.stopped
                        && this.state.started
                        && !this.state.fatal
                    {
                        return Poll::Ready(Some(Err(ProviderError::new(
                            ProviderErrorKind::ProtocolViolation {
                                hint: "stream ended without finish_reason or [DONE]".into(),
                            },
                        ))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(ProviderError::new(
                        ProviderErrorKind::Transport(BoxError::new(e)),
                    ))));
                }
                Poll::Ready(Some(Ok(sse))) => {
                    process_sse(&mut this.state, sse, &mut this.pending);
                    if this.state.done || this.state.fatal {
                        this.finished = true;
                    }
                }
            }
        }
    }
}

fn process_sse(
    state: &mut DecoderState,
    sse: Sse,
    out: &mut Vec<Result<ProviderChunk, ProviderError>>,
) {
    let data = match sse.data {
        Some(d) => d,
        None => return,
    };
    let trimmed = data.trim();
    // OpenAI 流终结符。收到后丢掉所有后续数据帧（实际不会有）。
    if trimmed == "[DONE]" {
        state.done = true;
        return;
    }

    // 先 parse 为 raw Value，提取 DeepSeek 私货 `delta.reasoning_content`
    // —— wire OAS 没有这个字段，结构化 parse 后就丢了。
    let raw: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            out.push(Err(ProviderError::new(ProviderErrorKind::Malformed(
                BoxError::new(e),
            ))));
            return;
        }
    };

    let parsed: Result<wire::CreateChatCompletionStreamResponse, _> =
        serde_json::from_value(raw.clone());
    let evt = match parsed {
        Ok(e) => e,
        Err(e) => {
            out.push(Err(ProviderError::new(ProviderErrorKind::Malformed(
                BoxError::new(e),
            ))));
            return;
        }
    };

    handle_chunk(state, &raw, evt, out);
}

fn handle_chunk(
    state: &mut DecoderState,
    raw: &serde_json::Value,
    evt: wire::CreateChatCompletionStreamResponse,
    out: &mut Vec<Result<ProviderChunk, ProviderError>>,
) {
    // poll_next 用 `pop()` 取，按时间顺序吐就要反序压栈。
    let mut buf: Vec<Result<ProviderChunk, ProviderError>> = Vec::new();

    // 第一次见到 chunk → MessageStart。OpenAI 不像 Anthropic 那样
    // 有专门的 message_start event，每个 chunk 都带 id/model；首帧即起点。
    if !state.started {
        state.started = true;
        buf.push(Ok(ProviderChunk::MessageStart {
            id: evt.id.clone(),
            model: evt.model.clone(),
        }));
    }

    // choices 通常长度为 1（`n=1`），final usage chunk 是空数组。
    for (choice_idx, choice) in evt.choices.iter().enumerate() {
        // 提取 raw delta 用于 reasoning_content。
        let raw_delta = raw
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(choice_idx))
            .and_then(|c| c.get("delta"));

        let delta = &choice.delta;

        // DeepSeek `reasoning_content` —— wire OAS 没这字段，从 raw 拿。
        if let Some(rc) = raw_delta
            .and_then(|d| d.get("reasoning_content"))
            .and_then(|v| v.as_str())
        {
            if !rc.is_empty() {
                buf.push(Ok(ProviderChunk::ThinkingDelta {
                    text: rc.to_owned(),
                }));
            }
        }

        // 文本增量。
        if let Some(content) = &delta.content {
            if let wire::ChatCompletionStreamResponseDeltaContent::ChatCompletionStreamResponseDeltaContentVariant0(s) = content {
                if !s.is_empty() {
                    buf.push(Ok(ProviderChunk::TextDelta { text: s.clone() }));
                }
            }
        }

        // tool_calls：第一次出现某个 index 带 id+name → ToolUseStart；
        // 之后任意带 arguments 的 chunk → ToolUseArgsDelta。
        if let Some(calls) = &delta.tool_calls {
            for tc in calls {
                handle_tool_call_chunk(state, tc, &mut buf);
            }
        }

        // refusal：OpenAI 用 delta.refusal 表达安全拒绝。我们当 TextDelta
        // 处理（带可识别的前缀），最终 finish_reason=content_filter 时再
        // 通过 Stop 把语义往上传。
        if let Some(refusal) = &delta.refusal {
            if let wire::ChatCompletionStreamResponseDeltaRefusal::ChatCompletionStreamResponseDeltaRefusalVariant0(s) = refusal {
                if !s.is_empty() {
                    buf.push(Ok(ProviderChunk::TextDelta { text: s.clone() }));
                }
            }
        }

        // finish_reason 是必填字段（OAS 上无 Option）；流中绝大多数 chunk
        // 都是 `Stop`（"non-stop" reason）—— 但 OpenAI 实际 wire 形态下，
        // 非终结 chunk 的 finish_reason 是 null，由 codegen 投成什么样取决
        // 于 OAS。我们这里**保守对待**：只在收到 `tool_calls` / `length` /
        // `content_filter` / `function_call` 中任意一个 + 之后没有更多数据
        // 时认定为终结；`stop` 同样视作终结。简化策略：每个非空 choices
        // 的最后一个 chunk 一定带终结 finish_reason，所以收到就立即 emit。
        //
        // 注：当 wire schema 把 `finish_reason: null` 反序失败时，会落到
        // 上面的 Malformed 分支，状态机不会到这里。
        // finish_reason 在中间 chunk 是 null（OAS 已 patch 成 Option），
        // 终结 chunk 才有值。命中即关 tool_calls + emit Stop。
        if !state.stopped
            && let Some(fr) = choice.finish_reason
        {
            let order = state.tool_call_order.clone();
            for idx in order {
                if let Some(tc) = state.tool_calls.get_mut(&idx)
                    && !tc.closed
                {
                    tc.closed = true;
                    buf.push(Ok(ProviderChunk::ToolUseEnd { id: tc.id.clone() }));
                }
            }
            state.stopped = true;
            buf.push(Ok(ProviderChunk::Stop {
                reason: stop_reason_from_wire(fr),
            }));
        }
    }

    // 末尾 usage chunk：choices 为空、usage 有值。
    if let Some(usage) = &evt.usage {
        buf.push(Ok(ProviderChunk::Usage(usage_from_wire(usage))));
    }

    buf.reverse();
    out.extend(buf);
}

fn handle_tool_call_chunk(
    state: &mut DecoderState,
    tc: &wire::ChatCompletionMessageToolCallChunk,
    out: &mut Vec<Result<ProviderChunk, ProviderError>>,
) {
    let idx = tc.index;
    let entry_existed = state.tool_calls.contains_key(&idx);

    // 第一帧：必须带 id（OpenAI 文档明确：第一个 tool_calls chunk 携带
    // 完整 id 和 function.name，后续 chunk 只带 arguments）。
    if !entry_existed {
        let Some(id) = tc.id.clone() else {
            // 没 id 又没 prior state，无法关联 —— 当 ProtocolViolation 但
            // 不致命，因为下一帧可能就带 id。
            warn!(index = idx, "tool_calls chunk missing id on first frame");
            return;
        };
        let name = tc
            .function
            .as_ref()
            .and_then(|f| f.name.clone())
            .unwrap_or_default();
        state.tool_calls.insert(
            idx,
            ToolCallState {
                id: id.clone(),
                closed: false,
            },
        );
        state.tool_call_order.push(idx);
        out.push(Ok(ProviderChunk::ToolUseStart { id, name }));
    }

    if let Some(func) = &tc.function {
        if let Some(args) = &func.arguments {
            if !args.is_empty() {
                if let Some(tool) = state.tool_calls.get(&idx) {
                    out.push(Ok(ProviderChunk::ToolUseArgsDelta {
                        id: tool.id.clone(),
                        fragment: args.clone(),
                    }));
                }
            }
        }
    }
}

fn stop_reason_from_wire(
    r: wire::CreateChatCompletionStreamResponseChoicesFinishReason,
) -> StopReason {
    use wire::CreateChatCompletionStreamResponseChoicesFinishReason as W;
    match r {
        W::Stop => StopReason::EndTurn,
        W::Length => StopReason::MaxTokens,
        W::ToolCalls | W::FunctionCall => StopReason::ToolUse,
        W::ContentFilter => StopReason::Refusal,
    }
}

fn usage_from_wire(u: &wire::CompletionUsage) -> Usage {
    Usage {
        input_tokens: u64::try_from(u.prompt_tokens).ok(),
        output_tokens: u64::try_from(u.completion_tokens).ok(),
        cache_read_input_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .and_then(|v| u64::try_from(v).ok()),
        // OpenAI 不报告 cache creation tokens；只 cached_tokens 表
        // "本次命中缓存的输入 token 数"。
        cache_creation_input_tokens: None,
    }
}

#[cfg(test)]
mod tests;
