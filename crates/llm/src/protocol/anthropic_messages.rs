//! Anthropic Messages protocol encoding and decoding.
//!
//! Encodes [`defect_core::llm::CompletionRequest`] into the wire format
//! [`crate::wire::anthropic::components::CreateMessageParams`],
//! and decodes an SSE [`Sse`] stream ([`MessageStreamEvent`]) into a
//! [`defect_core::llm::ProviderChunk`] stream.
//!
//! Anthropic Messages API protocol mapping.
//!
//! [`Sse`]: ::sse_stream::Sse
//! [`MessageStreamEvent`]: crate::wire::anthropic::components::MessageStreamEvent

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::task::{Context, Poll};

use defect_core::error::BoxError;
use defect_core::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderError,
    ProviderErrorKind, ReasoningEffort, Role, StopReason, ThinkingConfig, ToolChoice,
    ToolResultBody, ToolResultContent, Usage,
};
use defect_core::tool::ToolSchema;
use futures::{Stream, StreamExt};
use sse_stream::Sse;
use toac::body::codec::sse::SseEventStream;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::wire::anthropic::components as wire;

// encode

/// Fallback value used when `CompletionRequest::sampling.max_tokens` is `None` and no
/// `max_output_tokens` is found in the model cache.
///
/// The Anthropic Messages API requires `max_tokens` — it has no sensible default and must
/// always be sent. 4096 is documented as the "old default" in Anthropic's docs and covers
/// most cases; callers with a more precise value should set it explicitly via `sampling`.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Anthropic prompt cache accepts at most 4 ephemeral cache breakpoints.
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Encodes a [`CompletionRequest`] into the wire request body.
///
/// Forces `stream = true`: the protocol layer only uses the SSE path.
///
/// ## Cache breakpoint placement
///
/// Anthropic prompt cache is a **prefix** cache: a `cache_control` marker caches the entire
/// prefix from the start (canonical order `tools → system → messages`) up to and including
/// that block, and at most `MAX_CACHE_BREAKPOINTS` markers are allowed. To maximize the
/// cached prefix we place markers **end-biased**, not front-to-back:
///
/// - **1 static-prefix marker**: on the `system` block if present (which caches
///   `tools + system`, since `system` follows `tools` in the canonical order), otherwise on
///   the last tool. This static prefix survives across every turn.
/// - **up to 3 rolling markers**: on the last content block of the last few messages
///   (deepest first), forming a ladder so the long conversation prefix is cached and a
///   single new turn still hits the prior prefix.
pub fn encode_request(req: &CompletionRequest) -> wire::CreateMessageParams {
    let max_tokens = req.sampling.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let has_system = req.system.is_some();
    // Reserve one breakpoint for the static prefix (system, or the last tool when there is
    // no system); the rest roll over the most recent messages.
    let message_breakpoints = MAX_CACHE_BREAKPOINTS.saturating_sub(1);

    let system = req.system.as_deref().map(|text| {
        wire::SystemPrompt::SystemPromptVariant1(vec![wire::TextBlockParam {
            text: text.to_owned(),
            r#type: wire::TextBlockParamType::Text,
            // The single static-prefix breakpoint: caches `tools + system`.
            cache_control: Some(ephemeral()),
            citations: None,
        }])
    });

    let mut messages: Vec<wire::MessageParam> = req.messages.iter().map(encode_message).collect();
    mark_message_breakpoints(&mut messages, message_breakpoints);

    let tools = if req.tools.is_empty() {
        None
    } else {
        let last = req.tools.len() - 1;
        Some(
            req.tools
                .iter()
                .enumerate()
                // When there is no system prompt, the static-prefix breakpoint goes on the
                // last tool instead (caching the whole tool array).
                .map(|(i, tool)| encode_tool(tool, !has_system && i == last))
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
        thinking: encode_thinking(
            req.sampling.thinking,
            req.sampling.reasoning_effort,
            max_tokens,
        ),
        tool_choice: encode_tool_choice(&req.tool_choice),
        tools,
        top_k: req.sampling.top_k.map(i64::from),
        top_p: req.sampling.top_p,
    }
}

fn encode_message(m: &Message) -> wire::MessageParam {
    wire::MessageParam {
        role: match m.role {
            Role::User => wire::MessageParamRole::User,
            Role::Assistant => wire::MessageParamRole::Assistant,
        },
        content: wire::MessageParamContent::MessageParamContentVariant1(
            m.content.iter().filter_map(encode_content).collect(),
        ),
    }
}

fn encode_content(c: &MessageContent) -> Option<wire::ContentBlockParam> {
    match c {
        MessageContent::Text { text } => Some(wire::ContentBlockParam::TextBlockParam(
            wire::TextBlockParam {
                text: text.clone(),
                r#type: wire::TextBlockParamType::Text,
                cache_control: None,
                citations: None,
            },
        )),
        MessageContent::Thinking { text, signature } => {
            // On the Anthropic wire, `signature` is required — a missing value is
            // equivalent to forgery and will be rejected by the server. The OpenAI /
            // DeepSeek compatibility path never includes a signature; when switching
            // providers back to Anthropic, the thinking block cannot be replayed, so the
            // entire block is skipped.
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
                cache_control: None,
                caller: None,
            }),
        ),
        MessageContent::ToolResult {
            tool_use_id,
            output,
            is_error,
        } => Some(encode_tool_result(tool_use_id, output, *is_error)),
        MessageContent::Image { mime, data } => Some(wire::ContentBlockParam::ImageBlockParam(
            wire::ImageBlockParam {
                source: encode_image_source(mime, data),
                r#type: wire::ImageBlockParamType::Image,
                cache_control: None,
            },
        )),
        // The inner enum is `non_exhaustive`; when a new variant is added, the fallback
        // here is an empty string — an obvious placeholder so it can be grepped for a
        // real mapping.
        _ => Some(wire::ContentBlockParam::TextBlockParam(
            wire::TextBlockParam {
                text: String::new(),
                r#type: wire::TextBlockParamType::Text,
                cache_control: None,
                citations: None,
            },
        )),
    }
}

fn encode_tool_result(
    tool_use_id: &str,
    output: &ToolResultBody,
    is_error: bool,
) -> wire::ContentBlockParam {
    // Anthropic's `tool_result` block natively supports text and image sub-blocks; map
    // each block accordingly.
    let content: Vec<wire::ToolResultBlockParamContent> = match output {
        ToolResultBody::Text { text } => vec![text_result_block(text.clone())],
        ToolResultBody::Json { value } => vec![text_result_block(value.to_string())],
        ToolResultBody::Content { blocks } => blocks
            .iter()
            .map(|b| match b {
                ToolResultContent::Text { text } => text_result_block(text.clone()),
                ToolResultContent::Image { mime, data } => {
                    wire::ToolResultBlockParamContent::ImageBlockParam(wire::ImageBlockParam {
                        source: encode_image_source(mime, data),
                        r#type: wire::ImageBlockParamType::Image,
                        cache_control: None,
                    })
                }
            })
            .collect(),
    };
    wire::ContentBlockParam::ToolResultBlockParam(wire::ToolResultBlockParam {
        tool_use_id: tool_use_id.to_owned(),
        r#type: wire::ToolResultBlockParamType::ToolResult,
        cache_control: None,
        content: Some(
            wire::ToolResultBlockParamContent102::ToolResultBlockParamContent102Variant1(content),
        ),
        is_error: Some(is_error),
    })
}

fn text_result_block(text: String) -> wire::ToolResultBlockParamContent {
    wire::ToolResultBlockParamContent::TextBlockParam(wire::TextBlockParam {
        text,
        r#type: wire::TextBlockParamType::Text,
        cache_control: None,
        citations: None,
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

/// Anthropic's minimum extended-thinking budget. Budgets below this are rejected by the
/// API, so any "enable thinking" intent is floored to it.
const MIN_THINKING_BUDGET: u32 = 1024;

/// Maps the OpenAI-style `reasoning_effort` levels onto Anthropic thinking budgets.
/// Anthropic has no `reasoning_effort` wire field; extended thinking is controlled purely
/// by `budget_tokens`. `None` disables thinking; the other levels scale the budget.
fn effort_to_budget(effort: ReasoningEffort) -> Option<u32> {
    match effort {
        ReasoningEffort::None => None,
        ReasoningEffort::Minimal => Some(MIN_THINKING_BUDGET),
        ReasoningEffort::Low => Some(4_096),
        ReasoningEffort::Medium => Some(8_192),
        ReasoningEffort::High => Some(16_384),
        ReasoningEffort::Xhigh => Some(32_768),
    }
}

/// Resolves the wire `thinking` field. Precedence mirrors the OpenAI path: a per-session
/// `reasoning_effort` override (ACP thought-level) wins and is mapped to a thinking budget;
/// otherwise fall back to the explicit [`ThinkingConfig`].
///
/// Anthropic requires `budget_tokens < max_tokens`; the budget is clamped to stay strictly
/// below `max_tokens` (and never under [`MIN_THINKING_BUDGET`]) so a large effort level on a
/// small `max_tokens` request doesn't trigger a 400.
fn encode_thinking(
    t: ThinkingConfig,
    effort: Option<ReasoningEffort>,
    max_tokens: u32,
) -> Option<wire::ThinkingConfigParam> {
    let budget = match effort {
        Some(effort) => effort_to_budget(effort)?,
        None => match t {
            ThinkingConfig::Enabled { budget_tokens } => {
                budget_tokens.unwrap_or(MIN_THINKING_BUDGET)
            }
            _ => return None,
        },
    };

    // Keep the budget strictly below max_tokens, but never below the API minimum. When
    // max_tokens is too small to fit even the minimum budget, thinking is dropped rather
    // than sending an invalid request.
    let ceiling = max_tokens.saturating_sub(1);
    if ceiling < MIN_THINKING_BUDGET {
        return None;
    }
    let budget = budget.clamp(MIN_THINKING_BUDGET, ceiling);

    Some(wire::ThinkingConfigParam::ThinkingConfigEnabled(
        wire::ThinkingConfigEnabled {
            budget_tokens: i64::from(budget),
            r#type: wire::ThinkingConfigEnabledType::Enabled,
            display: None,
        },
    ))
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
    }
}

fn encode_tool(t: &ToolSchema, breakpoint: bool) -> wire::ToolUnion {
    let (properties, required) = split_input_schema(&t.input_schema);
    wire::ToolUnion::Tool(wire::Tool {
        input_schema: wire::ToolInputSchema {
            r#type: wire::ToolInputSchemaType::Object,
            properties,
            required,
        },
        name: t.name.clone(),
        allowed_callers: None,
        cache_control: breakpoint.then(ephemeral),
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

/// A fresh ephemeral `cache_control` marker (the only cache mode we emit).
fn ephemeral() -> wire::CacheControlEphemeral {
    wire::CacheControlEphemeral {
        r#type: wire::CacheControlEphemeralType::Ephemeral,
        ttl: None,
    }
}

/// Places up to `budget` rolling cache breakpoints on the **most recent** messages, one per
/// message, deepest first. The marker is attached to the last cacheable content block of
/// each chosen message so the cached prefix extends as far down the conversation as
/// possible; combined with the static-prefix marker (system / last tool) this forms an
/// end-biased ladder — see [`encode_request`].
///
/// A message contributes no marker if it has no content block that carries a
/// `cache_control` field (e.g. a lone thinking block), in which case the next-older message
/// is used, so the budget is never silently wasted.
fn mark_message_breakpoints(messages: &mut [wire::MessageParam], budget: usize) {
    let mut remaining = budget;
    for message in messages.iter_mut().rev() {
        if remaining == 0 {
            break;
        }
        if set_last_block_breakpoint(message) {
            remaining -= 1;
        }
    }
}

/// Sets an ephemeral breakpoint on the last content block of `message` that supports
/// `cache_control`. Returns whether a breakpoint was placed.
fn set_last_block_breakpoint(message: &mut wire::MessageParam) -> bool {
    let wire::MessageParamContent::MessageParamContentVariant1(blocks) = &mut message.content
    else {
        return false;
    };
    for block in blocks.iter_mut().rev() {
        let slot = match block {
            wire::ContentBlockParam::TextBlockParam(b) => &mut b.cache_control,
            wire::ContentBlockParam::ToolUseBlockParam(b) => &mut b.cache_control,
            wire::ContentBlockParam::ToolResultBlockParam(b) => &mut b.cache_control,
            wire::ContentBlockParam::ImageBlockParam(b) => &mut b.cache_control,
            // Thinking blocks have no cache_control field on the wire; skip.
            _ => continue,
        };
        *slot = Some(ephemeral());
        return true;
    }
    false
}

/// Splits the top-level JSON Schema into `properties` / `required` — `ToolInputSchema`.
/// Only accepts object schemas; extra fields are discarded (the codegen wire does not
/// store them).
fn split_input_schema(
    schema: &serde_json::Value,
) -> (
    Option<BTreeMap<String, serde_json::Value>>,
    Option<Vec<String>>,
) {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return (None, None),
    };
    let properties = obj.get("properties").and_then(|v| v.as_object()).map(|m| {
        m.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<BTreeMap<_, _>>()
    });
    let required = obj.get("required").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    });
    (properties, required)
}

fn json_value_to_object(v: &serde_json::Value) -> BTreeMap<String, serde_json::Value> {
    v.as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

// ---------- decode -------------------------------------------------------

/// Tracks the kind of each `content_block` in the decoding state machine — only
/// [`BlockKind::ToolUse`] needs to look up `tool_use_id` on `ArgsDelta` / `Stop`.
#[derive(Debug, Clone)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse {
        id: String,
    },
    /// Known but not yet projected as a chunk (server tool, citation, etc.).
    Other,
}

/// Internal state. Reset on each `message_start`. In theory a stream should only start
/// once, but we follow a lenient policy: if the upstream sends a duplicate start, we
/// reuse the existing state rather than erroring out.
#[derive(Debug, Default)]
struct DecoderState {
    blocks: HashMap<i64, BlockKind>,
    started: bool,
    stopped: bool,
    /// Set when an `error` event is seen; downstream should stop consuming.
    fatal: bool,
}

/// Converts an SSE stream into a `ProviderChunk` stream. The return value implements
/// [`Stream`]; callers pull from it via `.next()`, and dropping the stream is equivalent
/// to cancellation.
///
/// The `cancel` token originates from [`defect_core::llm::LlmProvider::complete`]. When
/// triggered, the stream silently terminates without yielding `Err(Canceled)`.
pub fn decode_stream(
    sse: SseEventStream,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    decode_stream_generic(sse, cancel)
}

/// Same shape as [`decode_stream`], but generic over the concrete `Stream` type of the
/// input.
///
/// At runtime [`decode_stream`] feeds it a [`SseEventStream`]; tests can feed any
/// `Stream<Item = Result<Sse, E>>` (e.g. `futures::stream::iter`),
/// without going through toac's `SseBody` (whose `new` is crate-private).
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

/// Same shape as [`decode_stream_generic`], but input errors are already unified to
/// [`ProviderError`]. Non-SSE transports like Bedrock produce classified errors at the
/// event-stream layer; this entry point reuses the Anthropic state machine directly.
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
    /// A single SSE event may produce multiple chunks (e.g. `message_start` yields both
    /// MessageStart and Usage). These are buffered in `pending` and yielded one by one.
    pending: Vec<Result<ProviderChunk, ProviderError>>,
    finished: bool,
}

impl<S> Stream for AnthropicSseDecoder<S>
where
    S: Stream<Item = Result<Sse, ProviderError>>,
{
    type Item = Result<ProviderChunk, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Safety: standard pin-projection through a single field. We never move `inner`
        // out.
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
            // Protocol noise: a single parse failure does not terminate the entire
            // stream. Also log the raw payload at warn level — field differences between
            // transports (Bedrock event-stream / Messages SSE) must be located by
            // comparing the original text.
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

    // event_name and the `type` field inside `evt` are redundant but should agree; log a
    // warning if they don't.
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

    // poll_next drains with `pop()`, so push in reverse order to preserve chronological
    // output.
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
                    // Citations are not projected into `ProviderChunk`; ignore them.
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
            // `pause_turn` is a server-tool yield signal. The main loop has no
            // corresponding semantics, so it is treated as `EndTurn` with a warning; a
            // dedicated variant should be added later.
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
            scope: defect_core::llm::RateLimitScope::Unspecified,
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
