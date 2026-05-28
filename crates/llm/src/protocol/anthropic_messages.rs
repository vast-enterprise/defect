//! Anthropic Messages 协议编解码。
//!
//! 把 [`defect_agent::llm::CompletionRequest`] 编为 wire
//! [`crate::wire::anthropic::components::CreateMessageParams`]，
//! 把 SSE [`Sse`] 流（[`MessageStreamEvent`]）解码为
//! [`defect_agent::llm::ProviderChunk`] 流。
//!
//! 设计映射详见 `docs/outbound/llm-anthropic.md` §6。
//!
//! [`Sse`]: ::sse_stream::Sse
//! [`MessageStreamEvent`]: crate::wire::anthropic::components::MessageStreamEvent

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use defect_agent::error::BoxError;
use defect_agent::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderError,
    ProviderErrorKind, Role, StopReason, ThinkingConfig, ToolChoice, ToolResultBody, Usage,
};
use defect_agent::tool::ToolSchema;
use futures::{Stream, StreamExt};
use sse_stream::Sse;
use toac::body::codec::sse::SseEventStream;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::wire::anthropic::components as wire;

// ---------- encode -------------------------------------------------------

/// 默认 max_tokens——`CompletionRequest::sampling.max_tokens` 为 `None`
/// 且模型缓存里查不到 max_output_tokens 时使用的兜底值。
///
/// Anthropic Messages API `max_tokens` 是 **必填**，没有合理默认值，
/// 协议层不能少传。4096 是 Anthropic 文档里"old default"标注，覆盖
/// 绝大多数情况；调用方有更精确的值时应通过 sampling 显式给出。
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Anthropic prompt cache 最多接受 4 个 ephemeral cache breakpoint。
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// 把 [`CompletionRequest`] 编为 wire 请求体。
///
/// 强制 `stream = true`：协议层只跑 SSE 分支。
pub fn encode_request(req: &CompletionRequest) -> wire::CreateMessageParams {
    let max_tokens = req.sampling.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut prompt_cache = PromptCache::new();
    let system = req.system.as_deref().map(|text| {
        wire::SystemPrompt::SystemPromptVariant1(vec![wire::TextBlockParam {
            text: text.to_owned(),
            r#type: wire::TextBlockParamType::Text,
            cache_control: prompt_cache.next_breakpoint(),
            citations: None,
        }])
    });
    let messages = req
        .messages
        .iter()
        .map(|message| encode_message(message, &mut prompt_cache))
        .collect();
    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(
            req.tools
                .iter()
                .map(|tool| encode_tool(tool, &mut prompt_cache))
                .collect(),
        )
    };

    wire::CreateMessageParams {
        max_tokens: i64::from(max_tokens),
        messages,
        model: wire::Model::ModelVariant1(req.model.clone()),
        cache_control: None,
        container: None,
        inference_geo: None,
        metadata: None,
        output_config: None,
        service_tier: None,
        stop_sequences: if req.sampling.stop_sequences.is_empty() {
            None
        } else {
            Some(req.sampling.stop_sequences.clone())
        },
        stream: Some(true),
        system,
        temperature: req.sampling.temperature,
        thinking: encode_thinking(req.sampling.thinking),
        tool_choice: encode_tool_choice(&req.tool_choice),
        tools,
        top_k: req.sampling.top_k.map(i64::from),
        top_p: req.sampling.top_p,
    }
}

fn encode_message(m: &Message, prompt_cache: &mut PromptCache) -> wire::MessageParam {
    wire::MessageParam {
        role: match m.role {
            Role::User => wire::MessageParamRole::User,
            Role::Assistant => wire::MessageParamRole::Assistant,
        },
        content: wire::MessageParamContent::MessageParamContentVariant1(
            m.content
                .iter()
                .filter_map(|content| encode_content(content, prompt_cache))
                .collect(),
        ),
    }
}

fn encode_content(
    c: &MessageContent,
    prompt_cache: &mut PromptCache,
) -> Option<wire::ContentBlockParam> {
    match c {
        MessageContent::Text { text } => Some(wire::ContentBlockParam::TextBlockParam(
            wire::TextBlockParam {
                text: text.clone(),
                r#type: wire::TextBlockParamType::Text,
                cache_control: prompt_cache.next_breakpoint(),
                citations: None,
            },
        )),
        MessageContent::Thinking { text, signature } => {
            // Anthropic wire 上 `signature` 是 required —— 缺失等价于
            // 伪造，服务端拒。OpenAI / DeepSeek 兼容路径不会带 signature；
            // 跨 provider 切回 Anthropic 时 thinking 块没法回放，整块跳过。
            let signature = signature.as_ref()?;
            Some(wire::ContentBlockParam::ThinkingBlockParam(
                wire::ThinkingBlockParam {
                    signature: signature.clone(),
                    thinking: text.clone(),
                    r#type: wire::ThinkingBlockParamType::Thinking,
                },
            ))
        }
        MessageContent::ToolUse { id, name, args } => Some(
            wire::ContentBlockParam::ToolUseBlockParam(wire::ToolUseBlockParam {
                id: id.clone(),
                input: json_value_to_object(args),
                name: name.clone(),
                r#type: wire::ToolUseBlockParamType::ToolUse,
                cache_control: prompt_cache.next_breakpoint(),
                caller: None,
            }),
        ),
        MessageContent::ToolResult {
            tool_use_id,
            output,
            is_error,
        } => Some(encode_tool_result(
            tool_use_id,
            output,
            *is_error,
            prompt_cache,
        )),
        MessageContent::Image { mime, data } => Some(wire::ContentBlockParam::ImageBlockParam(
            wire::ImageBlockParam {
                source: encode_image_source(mime, data),
                r#type: wire::ImageBlockParamType::Image,
                cache_control: prompt_cache.next_breakpoint(),
            },
        )),
        // 内部枚举是 non_exhaustive；新增 variant 时落到此处的回退是
        // 一段空文本——明显的 placeholder，方便 grep 出来补真正的映射。
        _ => Some(wire::ContentBlockParam::TextBlockParam(
            wire::TextBlockParam {
                text: String::new(),
                r#type: wire::TextBlockParamType::Text,
                cache_control: prompt_cache.next_breakpoint(),
                citations: None,
            },
        )),
    }
}

fn encode_tool_result(
    tool_use_id: &str,
    output: &ToolResultBody,
    is_error: bool,
    prompt_cache: &mut PromptCache,
) -> wire::ContentBlockParam {
    let text = match output {
        ToolResultBody::Text { text } => text.clone(),
        ToolResultBody::Json { value } => value.to_string(),
        _ => String::new(),
    };
    wire::ContentBlockParam::ToolResultBlockParam(wire::ToolResultBlockParam {
        tool_use_id: tool_use_id.to_owned(),
        r#type: wire::ToolResultBlockParamType::ToolResult,
        cache_control: prompt_cache.next_breakpoint(),
        content: Some(
            wire::ToolResultBlockParamContent102::ToolResultBlockParamContent102Variant1(vec![
                wire::ToolResultBlockParamContent::TextBlockParam(wire::TextBlockParam {
                    text,
                    r#type: wire::TextBlockParamType::Text,
                    cache_control: None,
                    citations: None,
                }),
            ]),
        ),
        is_error: Some(is_error),
    })
}

fn encode_image_source(mime: &str, data: &ImageData) -> wire::ImageSource {
    match data {
        ImageData::Base64 { encoded } => {
            wire::ImageSource::Base64ImageSource(wire::Base64ImageSource {
                data: encoded.clone(),
                media_type: image_media_type(mime),
                r#type: wire::Base64ImageSourceType::Base64,
            })
        }
        ImageData::Url { url } => wire::ImageSource::UrlImageSource(wire::UrlImageSource {
            r#type: wire::UrlImageSourceType::Url,
            url: url.clone(),
        }),
        _ => wire::ImageSource::UrlImageSource(wire::UrlImageSource {
            r#type: wire::UrlImageSourceType::Url,
            url: String::new(),
        }),
    }
}

fn image_media_type(mime: &str) -> wire::Base64ImageSourceMediaType {
    match mime {
        "image/png" => wire::Base64ImageSourceMediaType::ImagePng,
        "image/gif" => wire::Base64ImageSourceMediaType::ImageGif,
        "image/webp" => wire::Base64ImageSourceMediaType::ImageWebp,
        _ => wire::Base64ImageSourceMediaType::ImageJpeg,
    }
}

fn encode_thinking(t: ThinkingConfig) -> Option<wire::ThinkingConfigParam> {
    match t {
        ThinkingConfig::Disabled => None,
        ThinkingConfig::Enabled { budget_tokens } => Some(
            wire::ThinkingConfigParam::ThinkingConfigEnabled(wire::ThinkingConfigEnabled {
                budget_tokens: i64::from(budget_tokens.unwrap_or(1024)),
                r#type: wire::ThinkingConfigEnabledType::Enabled,
                display: None,
            }),
        ),
        _ => None,
    }
}

fn encode_tool_choice(c: &ToolChoice) -> Option<wire::ToolChoice> {
    match c {
        ToolChoice::Auto => Some(wire::ToolChoice::ToolChoiceAuto(wire::ToolChoiceAuto {
            r#type: wire::ToolChoiceAutoType::Auto,
            disable_parallel_tool_use: None,
        })),
        ToolChoice::Required => Some(wire::ToolChoice::ToolChoiceAny(wire::ToolChoiceAny {
            r#type: wire::ToolChoiceAnyType::Any,
            disable_parallel_tool_use: None,
        })),
        ToolChoice::Named { name } => {
            Some(wire::ToolChoice::ToolChoiceTool(wire::ToolChoiceTool {
                name: name.clone(),
                r#type: wire::ToolChoiceToolType::Tool,
                disable_parallel_tool_use: None,
            }))
        }
        ToolChoice::None => Some(wire::ToolChoice::ToolChoiceNone(wire::ToolChoiceNone {
            r#type: wire::ToolChoiceNoneType::None,
        })),
        _ => None,
    }
}

fn encode_tool(t: &ToolSchema, prompt_cache: &mut PromptCache) -> wire::ToolUnion {
    let (properties, required) = split_input_schema(&t.input_schema);
    wire::ToolUnion::Tool(wire::Tool {
        input_schema: wire::ToolInputSchema {
            r#type: wire::ToolInputSchemaType::Object,
            properties,
            required,
        },
        name: t.name.clone(),
        allowed_callers: None,
        cache_control: prompt_cache.next_breakpoint(),
        defer_loading: None,
        description: if t.description.is_empty() {
            None
        } else {
            Some(t.description.clone())
        },
        eager_input_streaming: None,
        input_examples: None,
        strict: None,
        r#type: None,
    })
}

struct PromptCache {
    remaining_breakpoints: usize,
}

impl PromptCache {
    fn new() -> Self {
        Self {
            remaining_breakpoints: MAX_CACHE_BREAKPOINTS,
        }
    }

    fn next_breakpoint(&mut self) -> Option<wire::CacheControlEphemeral> {
        if self.remaining_breakpoints == 0 {
            return None;
        }
        self.remaining_breakpoints -= 1;
        Some(wire::CacheControlEphemeral {
            r#type: wire::CacheControlEphemeralType::Ephemeral,
            ttl: None,
        })
    }
}

/// 拆 JSON Schema 顶层为 `properties` / `required` —— `ToolInputSchema`
/// 受限只接 object schema，多余字段送丢即可（codegen 出来的 wire 不存）。
fn split_input_schema(
    schema: &serde_json::Value,
) -> (
    Option<HashMap<String, serde_json::Value>>,
    Option<Vec<String>>,
) {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return (None, None),
    };
    let properties = obj.get("properties").and_then(|v| v.as_object()).map(|m| {
        m.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<_, _>>()
    });
    let required = obj.get("required").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    });
    (properties, required)
}

fn json_value_to_object(v: &serde_json::Value) -> HashMap<String, serde_json::Value> {
    v.as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

// ---------- decode -------------------------------------------------------

/// 在解码状态机里追踪每个 content_block 的种类——只有
/// [`BlockKind::ToolUse`] 需要在 ArgsDelta / Stop 时反查 `tool_use_id`。
#[derive(Debug, Clone)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse {
        id: String,
    },
    /// 已知但暂不投影为 chunk（server tool / citation 等）。
    Other,
}

/// 内部状态。每次 `message_start` 重置，理论上一条流只 start 一次，
/// 这里仍按"宽容"原则：上游若重复发，沿用已有状态而不直接报错。
#[derive(Debug, Default)]
struct DecoderState {
    blocks: HashMap<i64, BlockKind>,
    started: bool,
    stopped: bool,
    /// 见过 `error` event 后置位，下游应停止消费。
    fatal: bool,
}

/// SSE 流 → ProviderChunk 流。返回值实现 [`Stream`]，调用方通过 `.next()`
/// 拉取，drop 该流即等同取消。
///
/// `cancel` 来自 [`defect_agent::llm::LlmProvider::complete`]，触发后流静默
/// 终结，不 yield `Err(Canceled)`（与 [`docs/internal/llm-trait.md`] §2.2 一致）。
pub fn decode_stream(
    sse: SseEventStream,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    decode_stream_generic(sse, cancel)
}

/// 与 [`decode_stream`] 同形态，但对入参的具体 `Stream` 类型泛化。
///
/// 运行期由 [`decode_stream`] 喂 [`SseEventStream`]；测试可以喂任意
/// `Stream<Item = Result<Sse, E>>`（例如 `futures::stream::iter`），
/// 无需经过 toac 的 `SseBody`（其 `new` 是 crate-private）。
pub fn decode_stream_generic<S, E>(
    sse: S,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send
where
    S: Stream<Item = Result<Sse, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let sse = sse.map(|item| {
        item.map_err(|e| ProviderError::new(ProviderErrorKind::Transport(BoxError::new(e))))
    });
    decode_stream_provider_errors(sse, cancel)
}

/// 与 [`decode_stream_generic`] 同形态，但入参错误已统一成
/// [`ProviderError`]。Bedrock 这类非 SSE transport 会在 event-stream 层
/// 产生已分类错误，直接走此入口复用 Anthropic 状态机。
pub fn decode_stream_provider_errors<S>(
    sse: S,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send
where
    S: Stream<Item = Result<Sse, ProviderError>> + Send + 'static,
{
    AnthropicSseDecoder {
        inner: sse,
        cancel,
        state: DecoderState::default(),
        pending: Vec::new(),
        finished: false,
    }
}

struct AnthropicSseDecoder<S> {
    inner: S,
    cancel: CancellationToken,
    state: DecoderState,
    /// 单个 SSE event 可能产出多个 chunk（如 `message_start` 同时给
    /// MessageStart + Usage）。先存到 `pending`，逐个吐。
    pending: Vec<Result<ProviderChunk, ProviderError>>,
    finished: bool,
}

impl<S> Stream for AnthropicSseDecoder<S>
where
    S: Stream<Item = Result<Sse, ProviderError>>,
{
    type Item = Result<ProviderChunk, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: standard pin-projection through a single field. We never
        // move `inner` out.
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
                    if !this.state.stopped && this.state.started && !this.state.fatal {
                        return Poll::Ready(Some(Err(ProviderError::new(
                            ProviderErrorKind::ProtocolViolation {
                                hint: "stream ended without message_delta stop".into(),
                            },
                        ))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(sse))) => {
                    process_sse(&mut this.state, sse, &mut this.pending);
                    if this.state.fatal {
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
    let event_name = sse.event.as_deref();

    let parsed: Result<wire::MessageStreamEvent, _> = serde_json::from_str(&data);
    let evt = match parsed {
        Ok(e) => e,
        Err(e) => {
            // 协议噪声：单条解析失败不终止整个流。把原始 payload 也打到 warn
            // 日志——不同 transport（Bedrock event-stream / Messages SSE）
            // 之间字段差异要靠原文对比定位。
            warn!(
                error = %e,
                event_name = ?event_name,
                payload = %data,
                "failed to decode anthropic messages stream event",
            );
            out.push(Err(ProviderError::new(ProviderErrorKind::Malformed(
                BoxError::new(e),
            ))));
            return;
        }
    };

    // event_name 与 evt 内 type 字段冗余但理论一致；不一致只 warn。
    if let Some(name) = event_name {
        let ty = event_type_of(&evt);
        if ty != name {
            warn!(
                event_name = %name,
                data_type = %ty,
                "anthropic sse event name disagrees with data.type",
            );
        }
    }

    handle_event(state, evt, out);
}

fn event_type_of(e: &wire::MessageStreamEvent) -> &'static str {
    match e {
        wire::MessageStreamEvent::MessageStartEvent(_) => "message_start",
        wire::MessageStreamEvent::ContentBlockStartEvent(_) => "content_block_start",
        wire::MessageStreamEvent::ContentBlockDeltaEvent(_) => "content_block_delta",
        wire::MessageStreamEvent::ContentBlockStopEvent(_) => "content_block_stop",
        wire::MessageStreamEvent::MessageDeltaEvent(_) => "message_delta",
        wire::MessageStreamEvent::MessageStopEvent(_) => "message_stop",
        wire::MessageStreamEvent::PingEvent(_) => "ping",
        wire::MessageStreamEvent::ErrorEvent(_) => "error",
    }
}

fn handle_event(
    state: &mut DecoderState,
    evt: wire::MessageStreamEvent,
    out: &mut Vec<Result<ProviderChunk, ProviderError>>,
) {
    use wire::MessageStreamEvent::*;

    // poll_next 用 `pop()` 取出，这里反序压栈让它按时间顺序吐出。
    let mut buf = Vec::new();
    match evt {
        MessageStartEvent(e) => {
            state.started = true;
            let m = e.message;
            buf.push(Ok(ProviderChunk::MessageStart {
                id: m.id,
                model: model_to_string(&m.model),
            }));
            buf.push(Ok(ProviderChunk::Usage(usage_from_wire(&m.usage))));
        }
        ContentBlockStartEvent(e) => {
            let kind = block_kind_from_wire(&e.content_block);
            if let BlockKind::ToolUse { id } = &kind
                && let Some((tool_id, tool_name)) = tool_use_meta(&e.content_block)
            {
                buf.push(Ok(ProviderChunk::ToolUseStart {
                    id: tool_id.clone(),
                    name: tool_name,
                }));
                debug_assert_eq!(id, &tool_id);
            }
            state.blocks.insert(e.index, kind);
        }
        ContentBlockDeltaEvent(e) => {
            let kind = state.blocks.get(&e.index);
            match e.delta {
                wire::ContentBlockDelta::TextDelta(d) => {
                    buf.push(Ok(ProviderChunk::TextDelta { text: d.text }));
                }
                wire::ContentBlockDelta::ThinkingDelta(d) => {
                    buf.push(Ok(ProviderChunk::ThinkingDelta { text: d.thinking }));
                }
                wire::ContentBlockDelta::SignatureDelta(d) => {
                    buf.push(Ok(ProviderChunk::ThinkingSignature {
                        signature: d.signature,
                    }));
                }
                wire::ContentBlockDelta::InputJsonDelta(d) => match kind {
                    Some(BlockKind::ToolUse { id }) => {
                        buf.push(Ok(ProviderChunk::ToolUseArgsDelta {
                            id: id.clone(),
                            fragment: d.partial_json,
                        }));
                    }
                    _ => warn!(index = e.index, "input_json_delta for non-tool_use block"),
                },
                wire::ContentBlockDelta::CitationsDelta(_) => {
                    // citations 不投影到 v0 ProviderChunk；忽略。
                }
            }
        }
        ContentBlockStopEvent(e) => {
            if let Some(BlockKind::ToolUse { id }) = state.blocks.get(&e.index) {
                buf.push(Ok(ProviderChunk::ToolUseEnd { id: id.clone() }));
            }
        }
        MessageDeltaEvent(e) => {
            state.stopped = true;
            buf.push(Ok(ProviderChunk::Stop {
                reason: stop_reason_from_wire(e.delta.stop_reason),
            }));
            buf.push(Ok(ProviderChunk::Usage(usage_from_delta(&e.usage))));
        }
        MessageStopEvent(_) | PingEvent(_) => {}
        ErrorEvent(e) => {
            state.fatal = true;
            buf.push(Err(stream_error_to_provider(&e.error)));
        }
    }

    buf.reverse();
    out.extend(buf);
}

fn model_to_string(m: &wire::Model) -> String {
    match m {
        wire::Model::ModelVariant0(v) => v.to_string(),
        wire::Model::ModelVariant1(s) => s.clone(),
    }
}

fn block_kind_from_wire(b: &wire::ContentBlock) -> BlockKind {
    use wire::ContentBlock::*;
    match b {
        TextBlock(_) => BlockKind::Text,
        ThinkingBlock(_) | RedactedThinkingBlock(_) => BlockKind::Thinking,
        ToolUseBlock(t) => BlockKind::ToolUse { id: t.id.clone() },
        _ => BlockKind::Other,
    }
}

fn tool_use_meta(b: &wire::ContentBlock) -> Option<(String, String)> {
    if let wire::ContentBlock::ToolUseBlock(t) = b {
        Some((t.id.clone(), t.name.clone()))
    } else {
        None
    }
}

fn stop_reason_from_wire(r: wire::StopReason) -> StopReason {
    match r {
        wire::StopReason::EndTurn => StopReason::EndTurn,
        wire::StopReason::MaxTokens => StopReason::MaxTokens,
        wire::StopReason::StopSequence => StopReason::StopSequence,
        wire::StopReason::ToolUse => StopReason::ToolUse,
        wire::StopReason::Refusal => StopReason::Refusal,
        wire::StopReason::PauseTurn => {
            // pause_turn 是 server tool 中途让步信号。v0 主循环没有
            // 对应语义，按 EndTurn 处理并 warn——后续要单独建 variant。
            warn!("anthropic stop_reason=pause_turn folded to EndTurn");
            StopReason::EndTurn
        }
    }
}

fn usage_from_wire(u: &wire::Usage) -> Usage {
    Usage {
        input_tokens: u64::try_from(u.input_tokens).ok(),
        output_tokens: u64::try_from(u.output_tokens).ok(),
        cache_read_input_tokens: u
            .cache_read_input_tokens
            .and_then(|v| u64::try_from(v).ok()),
        cache_creation_input_tokens: u
            .cache_creation_input_tokens
            .and_then(|v| u64::try_from(v).ok()),
    }
}

fn usage_from_delta(u: &wire::MessageDeltaUsage) -> Usage {
    Usage {
        input_tokens: u.input_tokens.and_then(|v| u64::try_from(v).ok()),
        output_tokens: u64::try_from(u.output_tokens).ok(),
        cache_read_input_tokens: u
            .cache_read_input_tokens
            .and_then(|v| u64::try_from(v).ok()),
        cache_creation_input_tokens: u
            .cache_creation_input_tokens
            .and_then(|v| u64::try_from(v).ok()),
    }
}

fn stream_error_to_provider(p: &wire::StreamErrorPayload) -> ProviderError {
    let kind = match p.r#type.as_str() {
        "overloaded_error" => ProviderErrorKind::ServerError {
            status: None,
            hint: Some(p.message.clone()),
        },
        "rate_limit_error" => ProviderErrorKind::RateLimit {
            retry_after: None,
            scope: defect_agent::llm::RateLimitScope::Unspecified,
        },
        "invalid_request_error" => ProviderErrorKind::BadRequest {
            hint: Some(p.message.clone()),
        },
        "authentication_error" => ProviderErrorKind::AuthRejected {
            hint: Some(p.message.clone()),
        },
        "permission_error" => ProviderErrorKind::AuthRejected {
            hint: Some(p.message.clone()),
        },
        "not_found_error" => ProviderErrorKind::ServerError {
            status: Some(404),
            hint: Some(p.message.clone()),
        },
        "api_error" => ProviderErrorKind::ServerError {
            status: None,
            hint: Some(p.message.clone()),
        },
        _ => ProviderErrorKind::ServerError {
            status: None,
            hint: Some(p.message.clone()),
        },
    };
    ProviderError::new(kind)
}

#[cfg(test)]
mod tests;
