//! OpenAI Chat Completions protocol encoding and decoding.
//!
//! Encodes [`defect_core::llm::CompletionRequest`] into the wire format
//! [`crate::wire::openai::components::CreateChatCompletionRequest`],
//! and decodes an SSE [`Sse`] stream of [`CreateChatCompletionStreamResponse`] into a
//! [`defect_core::llm::ProviderChunk`] stream.
//!
//! OpenAI Chat Completions API protocol mapping.
//!
//! [`Sse`]: ::sse_stream::Sse
//! [`CreateChatCompletionStreamResponse`]:
//!     crate::wire::openai::components::CreateChatCompletionStreamResponse

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use defect_core::error::BoxError;
use defect_core::llm::{
    CompletionRequest, ImageData, Message, MessageContent, ProviderChunk, ProviderError,
    ProviderErrorKind, ReasoningEffort, Role, StopReason, ThinkingConfig, ThinkingEcho, ToolChoice,
    ToolResultBody, ToolResultContent, Usage,
};
use defect_core::tool::ToolSchema;
use futures::Stream;
use sse_stream::Sse;
use toac::body::codec::sse::SseEventStream;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::wire::openai::components as wire;

// encode

const PROMPT_CACHE_KEY_PREFIX: &str = "defect:chat:v1:";
const PROMPT_CACHE_KEY_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PROMPT_CACHE_KEY_PRIME: u64 = 0x0000_0001_0000_01b3;

type UsageParser = fn(Option<&serde_json::Value>, &wire::CompletionUsage) -> Usage;

/// Encodes a [`CompletionRequest`] into a wire request body.
///
/// Key mapping decisions:
///
/// - Forces `stream = true` + `stream_options.include_usage = true`:
///   the protocol layer only runs the SSE branch, and **must** let the upstream
///   send a trailing usage chunk, otherwise token billing is unavailable.
/// - Promotes `system` to `messages[0]` as a system message — OpenAI has no
///   top-level `system` field (unlike Anthropic).
/// - A single [`Message`] may be split into multiple wire messages in OpenAI
///   format: if a user message mixes [`MessageContent::ToolResult`], it needs
///   a separate tool message (OpenAI uses `role: tool` + `tool_call_id` for
///   tool results, which cannot be mixed with user text in the same message).
/// - [`MessageContent::ToolUse`] in assistant messages maps to the `tool_calls`
///   field rather than content blocks. `args` is serialized via `serde_json::to_string`
///   (the OpenAI protocol requires `function.arguments` to be stringified JSON).
/// - `top_k` is absent in the OpenAI protocol; the provider layer handles this.
/// - `max_tokens`: the OpenAI dialect deprecates `max_tokens` in favor of
///   `max_completion_tokens`. The DeepSeek-compatible dialect still uses
///   `max_tokens` to align with its OpenAI-compatible endpoint and opencode
///   request format. Neither sets a default like Anthropic — the model decides
///   when not specified.
pub fn encode_request(req: &CompletionRequest) -> wire::CreateChatCompletionRequest {
    encode_request_with_echo(req, ThinkingEcho::Forbidden)
}

/// Same shape as [`encode_request`], but explicitly accepts a thinking-echo policy.
///
/// `echo_mode` is read by the provider layer from [`defect_core::llm::Capabilities`]
/// and passed in: when `Required`, the [`MessageContent::Thinking`] text on the
/// assistant message is written to the non-standard `reasoning_content` field on the
/// wire; when `Forbidden` (including unconfigured), it is never written.
pub fn encode_request_with_echo(
    req: &CompletionRequest,
    echo_mode: ThinkingEcho,
) -> wire::CreateChatCompletionRequest {
    encode_request_full(req, echo_mode, None)
}

/// Same shape as [`encode_request_with_echo`], but allows the provider layer to forcibly
/// override the `reasoning_effort` field. When `effort_override` is `Some(_)`, the value
/// of `SamplingParams::thinking` is ignored and the override is written directly to the
/// wire; when `None`, the old behavior (thinking enabled → medium) is preserved.
pub fn encode_request_full(
    req: &CompletionRequest,
    echo_mode: ThinkingEcho,
    effort_override: Option<ReasoningEffort>,
) -> wire::CreateChatCompletionRequest {
    encode_request_with_dialect(req, echo_mode, effort_override, ChatDialect::OpenAi)
}

/// OpenAI Chat-compatible request dialect.
///
/// Even though OpenAI and compatible providers share the same JSON schema, there are
/// still minor semantic differences in a few fields.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ChatDialect {
    #[default]
    OpenAi,
    DeepSeek,
}

/// Same shape as [`encode_request_full`], but allows the provider to specify a compatible
/// vendor dialect.
pub fn encode_request_with_dialect(
    req: &CompletionRequest,
    echo_mode: ThinkingEcho,
    effort_override: Option<ReasoningEffort>,
    dialect: ChatDialect,
) -> wire::CreateChatCompletionRequest {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system.as_ref() {
        messages.push(encode_system_message(sys));
    }
    for m in &req.messages {
        encode_message_into(m, echo_mode, dialect, &mut messages);
    }

    let max_tokens = req.sampling.max_tokens.map(i64::from);
    #[allow(deprecated)]
    wire::CreateChatCompletionRequest {
        // ---- fields we use ----
        messages,
        model: wire::ModelIdsShared::ModelIdsSharedVariant0(req.model.clone()),
        stream: Some(true),
        stream_options: Some(wire::ChatCompletionStreamOptions::ChatCompletionStreamOptionsVariant0(
            wire::ChatCompletionStreamOptionsVariant0 {
                include_usage: Some(true),
                include_obfuscation: None,
            },
        )),
        max_completion_tokens: match dialect {
            ChatDialect::OpenAi => max_tokens,
            ChatDialect::DeepSeek => None,
        },
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
        // Priority: per-session `sampling.reasoning_effort` (ACP thought-level,
        // switchable at runtime) > provider-level `effort_override` (fixed in config) >
        // derived from `thinking`. The first two directly materialize the level; the last
        // can only map to medium.
        reasoning_effort: req
            .sampling
            .reasoning_effort
            .or(effort_override)
            .map(encode_reasoning_effort)
            .or_else(|| encode_thinking(req.sampling.thinking)),
        tools: if req.tools.is_empty() {
            None
        } else {
            Some(req.tools.iter().map(encode_tool).collect())
        },
        tool_choice: encode_tool_choice(&req.tool_choice),
        // Unused fields: explicitly set to None for easy grepping later
        metadata: None,
        top_logprobs: None,
        user: None,
        safety_identifier: None,
        prompt_cache_key: match dialect {
            ChatDialect::OpenAi => Some(build_prompt_cache_key(req, echo_mode)),
            ChatDialect::DeepSeek => None,
        },
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
        max_tokens: match dialect {
            ChatDialect::OpenAi => None,
            ChatDialect::DeepSeek => max_tokens,
        },
        n: None,
        prediction: None,
        seed: None,
        parallel_tool_calls: None,
        function_call: None,
        functions: None,
    }
}

fn build_prompt_cache_key(req: &CompletionRequest, echo_mode: ThinkingEcho) -> String {
    let mut hasher = PromptCacheKeyHasher::new();
    hasher.write_str(&req.model);
    if let Some(system) = req.system.as_deref() {
        hasher.write_str(system);
    }
    hasher.write_str(prompt_cache_echo_mode(echo_mode));
    hasher.write_str(prompt_cache_tool_choice(&req.tool_choice));
    hasher.write_json(&req.tools);
    format!("{PROMPT_CACHE_KEY_PREFIX}{:016x}", hasher.finish())
}

fn prompt_cache_echo_mode(mode: ThinkingEcho) -> &'static str {
    match mode {
        ThinkingEcho::Forbidden => "forbidden",
        ThinkingEcho::Required => "required",
        ThinkingEcho::Optional => "optional",
    }
}

fn prompt_cache_tool_choice(choice: &ToolChoice) -> &str {
    match choice {
        ToolChoice::Auto => "auto",
        ToolChoice::Required => "required",
        ToolChoice::Named { name } => name.as_str(),
        ToolChoice::None => "none",
    }
}

struct PromptCacheKeyHasher {
    state: u64,
}

impl PromptCacheKeyHasher {
    fn new() -> Self {
        Self {
            state: PROMPT_CACHE_KEY_OFFSET_BASIS,
        }
    }

    fn write_json<T>(&mut self, value: &T)
    where
        T: serde::Serialize,
    {
        let Ok(encoded) = serde_json::to_vec(value) else {
            return;
        };
        self.write_bytes(&encoded);
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(PROMPT_CACHE_KEY_PRIME);
        }
        self.state ^= u64::from(b'\n');
        self.state = self.state.wrapping_mul(PROMPT_CACHE_KEY_PRIME);
    }

    fn finish(self) -> u64 {
        self.state
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

/// A single [`Message`] may fan out into multiple wire messages:
/// - Each [`MessageContent::ToolResult`] embedded in a user message becomes a separate
///   tool message.
/// - Each [`MessageContent::ToolUse`] in an assistant message is lifted to the top-level
///   `tool_calls` field instead of being part of the content.
fn encode_message_into(
    m: &Message,
    echo_mode: ThinkingEcho,
    dialect: ChatDialect,
    out: &mut Vec<wire::ChatCompletionRequestMessage>,
) {
    match m.role {
        Role::User => encode_user_message_into(m, out),
        Role::Assistant => encode_assistant_message_into(m, echo_mode, dialect, out),
    }
}

fn encode_user_message_into(m: &Message, out: &mut Vec<wire::ChatCompletionRequestMessage>) {
    let mut user_parts: Vec<wire::ChatCompletionRequestUserMessageContentPart> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_use_id, text)

    for c in m.content.iter() {
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
                user_parts.push(image_part(mime, data));
            }
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error: _,
            } => {
                // OpenAI's tool message has no `is_error` field; we use a prefix to
                // signal the error state so the model can read it from the content.
                // `is_error` is primarily for Anthropic; here we preserve its semantics
                // but in a different form.
                //
                // OpenAI's tool message only accepts text—images from multimodal results
                // cannot be placed inside a tool message. Strategy: extract image blocks
                // and push them into `user_parts` (the user message immediately following
                // the tool message), leaving only text plus a placeholder hint in the
                // tool message so the model knows the images are in the next message.
                let text = match output {
                    ToolResultBody::Text { text } => text.clone(),
                    ToolResultBody::Json { value } => value.to_string(),
                    ToolResultBody::Content { blocks } => {
                        let mut text = String::new();
                        let mut image_count = 0usize;
                        for block in blocks {
                            match block {
                                ToolResultContent::Text { text: t } => {
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                }
                                ToolResultContent::Image { mime, data } => {
                                    image_count += 1;
                                    user_parts.push(image_part(mime, data));
                                }
                            }
                        }
                        if image_count > 0 {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&format!(
                                "[{image_count} image(s) from this tool result follow in the next user message]"
                            ));
                        }
                        text
                    }
                };
                tool_results.push((tool_use_id.clone(), text));
            }
            // Fallback for `non_exhaustive`: keep the slot but leave the content empty.
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

    // OpenAI / LiteLLM require that an assistant message with tool_calls must be
    // immediately followed by the corresponding tool messages; a subsequent user message
    // cannot be inserted in between.
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
}

fn encode_assistant_message_into(
    m: &Message,
    echo_mode: ThinkingEcho,
    dialect: ChatDialect,
    out: &mut Vec<wire::ChatCompletionRequestMessage>,
) {
    const EMPTY_ASSISTANT_CONTENT: &str = "";

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<wire::ChatCompletionMessageToolCallsItem> = Vec::new();
    let mut reasoning_text = String::new();

    for c in m.content.iter() {
        match c {
            MessageContent::Text { text } => text_parts.push(text.clone()),
            MessageContent::Thinking { text, .. } => {
                // The `signature` field is irrelevant on the OpenAI path (neither
                // DeepSeek nor OpenAI uses it); only the text is taken.
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
            // ToolResult/Image should not appear in the assistant role; the
            // non_exhaustive fallback also reaches here. Ignore and do not send over the
            // wire.
            _ => {}
        }
    }

    let reasoning_content = match dialect {
        ChatDialect::DeepSeek => Some(reasoning_text),
        ChatDialect::OpenAi => match (echo_mode, reasoning_text.is_empty()) {
            (ThinkingEcho::Required, false) => Some(reasoning_text),
            // Treat `Optional` the same as `Required`: replaying is safer when the server
            // tolerates extra thinking fields (DeepSeek-v4-pro docs list it as `must`;
            // other `Optional` vendors do not error on extra fields either).
            (ThinkingEcho::Optional, false) => Some(reasoning_text),
            _ => None,
        },
    };
    let content = if text_parts.is_empty() {
        if tool_calls.is_empty() && reasoning_content.is_some() {
            // DeepSeek v4 series validates that assistant messages have at least
            // `content` or `tool_calls`; replaying a thinking-only history requires
            // adding an empty `content`.
            Some(wire::ChatCompletionRequestAssistantMessageContent::ChatCompletionRequestAssistantMessageContentVariant0(
                wire::ChatCompletionRequestAssistantMessageContentVariant0::ChatCompletionRequestAssistantMessageContentVariant0Variant0(
                    EMPTY_ASSISTANT_CONTENT.to_owned(),
                ),
            ))
        } else {
            None
        }
    } else {
        Some(wire::ChatCompletionRequestAssistantMessageContent::ChatCompletionRequestAssistantMessageContentVariant0(
            wire::ChatCompletionRequestAssistantMessageContentVariant0::ChatCompletionRequestAssistantMessageContentVariant0Variant0(
                text_parts.join(""),
            ),
        ))
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

/// Build an OpenAI user-message image part. This is shared with the image block extracted
/// from a multimodal `tool_result` via `MessageContent::Image`.
fn image_part(mime: &str, data: &ImageData) -> wire::ChatCompletionRequestUserMessageContentPart {
    wire::ChatCompletionRequestUserMessageContentPart::ChatCompletionRequestMessageContentPartImage(
        wire::ChatCompletionRequestMessageContentPartImage {
            r#type: wire::ChatCompletionRequestMessageContentPartImageType::ImageUrl,
            image_url: wire::ChatCompletionRequestMessageContentPartImageImageUrl {
                url: image_url_string(mime, data),
                detail: None,
            },
        },
    )
}

fn image_url_string(mime: &str, data: &ImageData) -> String {
    match data {
        ImageData::Url { url } => url.clone(),
        ImageData::Base64 { encoded } => format!("data:{mime};base64,{encoded}"),
    }
}

fn encode_thinking(t: ThinkingConfig) -> Option<wire::ReasoningEffort> {
    match t {
        ThinkingConfig::Disabled => None,
        // OpenAI's thinking does not accept `budget_tokens` (unlike Anthropic); it only
        // supports effort levels. The budget value is discarded and uniformly mapped to
        // `medium`.
        ThinkingConfig::Enabled { .. } => Some(wire::ReasoningEffort::ReasoningEffortVariant0(
            wire::ReasoningEffortVariant0::Medium,
        )),
    }
}

fn encode_reasoning_effort(effort: ReasoningEffort) -> wire::ReasoningEffort {
    use ReasoningEffort as E;
    use wire::ReasoningEffortVariant0 as V;
    let v = match effort {
        E::None => V::None,
        E::Minimal => V::Minimal,
        E::Low => V::Low,
        E::Medium => V::Medium,
        E::High => V::High,
        E::Xhigh => V::Xhigh,
    };
    wire::ReasoningEffort::ReasoningEffortVariant0(v)
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

/// Internal state of the decoding state machine.
#[derive(Debug, Default)]
struct DecoderState {
    /// MessageStart has been emitted.
    started: bool,
    /// `Stop` has been emitted. After `Stop`, only `Usage` is allowed.
    stopped: bool,
    /// Whether the `data: [DONE]` marker has been seen.
    done: bool,
    /// Received a fatal error (parsing failure; retry cannot continue).
    fatal: bool,
    /// Maps `delta.tool_calls[].index` → `tool_call_id`. OpenAI streaming chunks
    /// associate tool calls by index: the first frame carries `id` + `name`, while
    /// subsequent `args` frames only have the index. This table maps the index back to
    /// the `id` in `ProviderChunk`.
    tool_calls: HashMap<i64, ToolCallState>,
    /// Order in which tool_calls were received (used to emit ToolUseEnd in arrival order
    /// on Stop).
    tool_call_order: Vec<i64>,
}

#[derive(Debug, Clone)]
struct ToolCallState {
    id: String,
    /// Whether the `ToolUseEnd` has already been sent.
    closed: bool,
}

/// SSE stream → `ProviderChunk` stream. The return value implements [`Stream`]; dropping
/// it cancels the stream.
///
/// After `cancel` is triggered, the stream silently terminates, consistent with the LLM
/// trait contract.
pub fn decode_stream(
    sse: SseEventStream,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    decode_stream_with_usage_parser(sse, cancel, usage_from_wire)
}

/// Same shape as [`decode_stream`], but generic over the input `Stream` type for easier
/// testing — feed it directly with `futures::stream::iter` without going through the toac
/// transport.
pub fn decode_stream_generic<S, E>(
    sse: S,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send
where
    S: Stream<Item = Result<Sse, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    decode_stream_generic_with_usage_parser(sse, cancel, usage_from_wire)
}

/// Same shape as [`decode_stream`], but allows vendor-specific overrides of the usage
/// parsing logic.
pub(crate) fn decode_stream_with_usage_parser(
    sse: SseEventStream,
    cancel: CancellationToken,
    usage_parser: UsageParser,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    decode_stream_generic_with_usage_parser(sse, cancel, usage_parser)
}

fn decode_stream_generic_with_usage_parser<S, E>(
    sse: S,
    cancel: CancellationToken,
    usage_parser: UsageParser,
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
        usage_parser,
        _err: std::marker::PhantomData::<E>,
    }
}

struct OpenAiSseDecoder<S, E> {
    inner: S,
    cancel: CancellationToken,
    state: DecoderState,
    /// A single SSE frame may produce multiple chunks (a finish_reason frame is usually
    /// followed by ToolUseEnd*N + Stop). Store them in `pending` first, then `poll_next`
    /// pops them one by one.
    pending: Vec<Result<ProviderChunk, ProviderError>>,
    finished: bool,
    usage_parser: UsageParser,
    _err: std::marker::PhantomData<E>,
}

impl<S, E> Stream for OpenAiSseDecoder<S, E>
where
    S: Stream<Item = Result<Sse, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<ProviderChunk, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: Standard pin-projection through a single field. We never move `inner`
        // out, and `_err` is a zero-sized `PhantomData`.
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
                    // Neither [DONE] nor stop chunk — ProtocolViolation.
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
                    process_sse(&mut this.state, sse, &mut this.pending, this.usage_parser);
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
    usage_parser: UsageParser,
) {
    let data = match sse.data {
        Some(d) => d,
        None => return,
    };
    let trimmed = data.trim();
    // OpenAI stream terminator. Drop all subsequent data frames after receiving it (there
    // won't be any in practice).
    if trimmed == "[DONE]" {
        state.done = true;
        return;
    }

    // First parse as a raw `Value` to extract DeepSeek's proprietary
    // `delta.reasoning_content` — the wire OAS lacks this field, so it would be lost
    // after structured parsing.
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

    handle_chunk(state, &raw, evt, out, usage_parser);
}

fn handle_chunk(
    state: &mut DecoderState,
    raw: &serde_json::Value,
    evt: wire::CreateChatCompletionStreamResponse,
    out: &mut Vec<Result<ProviderChunk, ProviderError>>,
    usage_parser: UsageParser,
) {
    // `poll_next` uses `pop()`, so to emit in chronological order we must push in
    // reverse.
    let mut buf: Vec<Result<ProviderChunk, ProviderError>> = Vec::new();

    // The first chunk seen implies a `MessageStart`. Unlike Anthropic, OpenAI does not
    // have a dedicated `message_start` event; every chunk carries `id`/`model`, so the
    // first frame is the start.
    if !state.started {
        state.started = true;
        buf.push(Ok(ProviderChunk::MessageStart {
            id: evt.id.clone(),
            model: evt.model.clone(),
        }));
    }

    // choices are typically length 1 (`n=1`); the final usage chunk is an empty array.
    for (choice_idx, choice) in evt.choices.iter().enumerate() {
        // Extract the raw delta for `reasoning_content`.
        let raw_delta = raw
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(choice_idx))
            .and_then(|c| c.get("delta"));

        let delta = &choice.delta;

        // DeepSeek `reasoning_content` is not present in the wire OAS, so it is taken
        // from the raw delta.
        if let Some(rc) = raw_delta
            .and_then(|d| d.get("reasoning_content"))
            .and_then(|v| v.as_str())
            && !rc.is_empty()
        {
            buf.push(Ok(ProviderChunk::ThinkingDelta {
                text: rc.to_owned(),
            }));
        }

        // Text delta.
        if let Some(
            wire::ChatCompletionStreamResponseDeltaContent::ChatCompletionStreamResponseDeltaContentVariant0(
                s,
            ),
        ) = &delta.content
            && !s.is_empty()
        {
            buf.push(Ok(ProviderChunk::TextDelta { text: s.clone() }));
        }

        // For each tool call index, the first chunk containing `id` and `name` triggers a
        // `ToolUseStart`; subsequent chunks with `arguments` become `ToolUseArgsDelta`.
        if let Some(calls) = &delta.tool_calls {
            for tc in calls {
                handle_tool_call_chunk(state, tc, &mut buf);
            }
        }

        // OpenAI uses `delta.refusal` to signal a safety refusal. We treat it as a
        // `TextDelta` (with a distinguishable prefix), and later, when
        // `finish_reason=content_filter`, we propagate the semantics upward via `Stop`.
        if let Some(
            wire::ChatCompletionStreamResponseDeltaRefusal::ChatCompletionStreamResponseDeltaRefusalVariant0(
                s,
            ),
        ) = &delta.refusal
            && !s.is_empty()
        {
            buf.push(Ok(ProviderChunk::TextDelta { text: s.clone() }));
        }

        // `finish_reason` is required in the OAS (no `Option`); most chunks in the stream
        // are `Stop` (a "non-stop" reason). However, in OpenAI's actual wire format,
        // non-terminal chunks have `finish_reason: null`, and what the codegen produces
        // depends on the OAS. We take a **conservative approach**: only treat a chunk as
        // terminal when we see any of `tool_calls` / `length` / `content_filter` /
        // `function_call` and no more data follows; `stop` is also terminal. Simplified
        // strategy: the last chunk of every non-empty `choices` always carries a terminal
        // `finish_reason`, so emit immediately upon receipt.
        //
        // Note: when the wire schema fails to deserialize `finish_reason: null`, it falls
        // into the `Malformed` branch above, and the state machine never reaches here.
        // `finish_reason` is `null` on intermediate chunks (the OAS has been patched to
        // `Option`); only terminal chunks have a value. When hit, close `tool_calls` and
        // emit `Stop`.
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

    // Final usage chunk: choices are empty, usage is present.
    if let Some(usage) = &evt.usage {
        buf.push(Ok(ProviderChunk::Usage(usage_parser(
            raw.get("usage"),
            usage,
        ))));
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

    // First frame: must carry `id` (OpenAI docs specify that the first `tool_calls` chunk
    // carries the full `id` and `function.name`; subsequent chunks carry only
    // `arguments`).
    if !entry_existed {
        let Some(id) = tc.id.clone() else {
            // No id and no prior state, so the chunk cannot be associated — treat as a
            // protocol violation, but it is not fatal because the next frame may carry
            // the id.
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

    if let Some(func) = &tc.function
        && let Some(args) = &func.arguments
        && !args.is_empty()
        && let Some(tool) = state.tool_calls.get(&idx)
    {
        out.push(Ok(ProviderChunk::ToolUseArgsDelta {
            id: tool.id.clone(),
            fragment: args.clone(),
        }));
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

fn usage_from_wire(_raw_usage: Option<&serde_json::Value>, u: &wire::CompletionUsage) -> Usage {
    Usage {
        input_tokens: u64::try_from(u.prompt_tokens).ok(),
        output_tokens: u64::try_from(u.completion_tokens).ok(),
        cache_read_input_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .and_then(|v| u64::try_from(v).ok()),
        // OpenAI does not report cache creation tokens; `cached_tokens` only indicates
        // the number of input tokens that hit the cache.
        cache_creation_input_tokens: None,
    }
}

#[cfg(test)]
mod tests;
