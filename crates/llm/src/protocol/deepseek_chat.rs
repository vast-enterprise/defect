//! Compatibility layer for DeepSeek Chat Completions responses.
//!
//! The request side reuses the OpenAI Chat Completions encoding; only the response
//! `usage` field is augmented with DeepSeek-specific fields: `prompt_cache_hit_tokens` /
//! `prompt_cache_miss_tokens`.

use defect_core::llm::{ProviderChunk, ProviderError, Usage};
use futures::Stream;
use toac::body::codec::sse::SseEventStream;
use tokio_util::sync::CancellationToken;

use super::openai_chat;
use crate::wire::openai::components as wire;

/// DeepSeek SSE stream → ProviderChunk stream.
///
/// Reuses the OpenAI-compatible state machine, only changing the usage extraction logic.
pub fn decode_stream(
    sse: SseEventStream,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ProviderChunk, ProviderError>> + Send {
    openai_chat::decode_stream_with_usage_parser(sse, cancel, usage_from_deepseek_wire)
}

fn usage_from_deepseek_wire(
    raw_usage: Option<&serde_json::Value>,
    wire_usage: &wire::CompletionUsage,
) -> Usage {
    let raw_cache_hit_tokens = raw_usage
        .and_then(|usage| usage.get("prompt_cache_hit_tokens"))
        .and_then(serde_json::Value::as_u64);

    Usage {
        input_tokens: u64::try_from(wire_usage.prompt_tokens).ok(),
        output_tokens: u64::try_from(wire_usage.completion_tokens).ok(),
        cache_read_input_tokens: raw_cache_hit_tokens.or_else(|| {
            wire_usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .and_then(|value| u64::try_from(value).ok())
        }),
        // DeepSeek's `prompt_cache_miss_tokens` represents the number of cache-missed
        // prompt tokens, which is not equivalent to Anthropic's
        // `cache_creation_input_tokens` ("write-to-cache cost"); we do not conflate the
        // fields here.
        cache_creation_input_tokens: None,
    }
}
