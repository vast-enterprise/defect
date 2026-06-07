//! LLM invocation, retry, and streaming drain.
//!
//! Extracted from the turn main flow: `call_llm_with_retry` / `call_llm_attempt` /
//! `drain_provider_stream` / `handle_chunk` are implemented as methods on
//! [`super::TurnRunner`],
//! along with their dedicated accumulation types ([`DrainOutcome`] / [`LlmAttempt`] /
//! [`ToolUseAccumulated`]) and
//! pure-function helpers for usage / retry.

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{ContentBlock, TextContent};
use futures::StreamExt;
use rand::RngExt;
use serde_json::Value as JsonValue;
use tracing::Instrument;

use crate::event::{AgentEvent, LlmRequestSnapshot};
use crate::llm::{
    CompletionRequest, Message, MessageContent, ProviderChunk, ProviderStream, RetryHint, Role,
    StopReason as LlmStopReason, Usage,
};
use crate::session::TurnError;

use super::{TurnRunner, TurnState};

impl TurnRunner<'_> {
    /// Returns the stream on success, along with the attempt number (used by `run_inner`
    /// to emit `LlmCallFinished`).
    pub(super) async fn call_llm_with_retry(
        &self,
        req: &CompletionRequest,
        state: &mut TurnState,
    ) -> Result<(ProviderStream, u32), TurnError> {
        let max_attempts = self.config.max_llm_retries.saturating_add(1).max(1);
        let vendor = self.provider.info().vendor.to_string();
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            state.request_count = state.request_count.saturating_add(1);
            // One attempt = one `llm_call` span. The span wraps four steps: send request,
            // wait for response, decide whether to retry, and backoff sleep. On failure,
            // a new span is created for the next retry (attempt field +1), making it
            // easier to correlate each actual request during debugging.
            //
            // Note: use `.instrument(span).await`, **not** `span.enter()` then `await` —
            // the latter would carry the entered guard across the await point, which is
            // an anti-pattern explicitly warned about in the tracing documentation.
            let span = tracing::info_span!(
                "llm_call",
                vendor = %vendor,
                model = %req.model,
                attempt,
            );
            let step = self
                .call_llm_attempt(req, attempt, max_attempts)
                .instrument(span)
                .await;
            match step {
                LlmAttempt::Done(stream) => return Ok((stream, attempt)),
                LlmAttempt::Failed(err) => return Err(TurnError::Provider(err)),
                // Cancelled: return an empty stream; the attempt number is meaningless
                // (no `Finished` is emitted, see `run_inner`).
                LlmAttempt::Cancelled => return Ok((empty_stream(), attempt)),
                LlmAttempt::Retry => continue,
            }
        }
    }

    /// A single LLM call attempt: send the request, emit events, and decide the next
    /// step.
    /// Separated from [`Self::call_llm_with_retry`] so that `info_span!` can wrap the
    /// entire future via `.instrument(...)` without holding an entered guard across an
    /// await.
    async fn call_llm_attempt(
        &self,
        req: &CompletionRequest,
        attempt: u32,
        max_attempts: u32,
    ) -> LlmAttempt {
        self.events
            .emit(AgentEvent::LlmCallStarted {
                model: req.model.clone(),
                attempt,
                // Wrapping in `Arc` so that cloning degrades to reference counting when
                // fanning out to multiple subscribers, avoiding repeated deep copies of
                // the entire message history under long contexts.
                request: Arc::new(LlmRequestSnapshot {
                    system: req.system.clone(),
                    messages: req.messages.clone(),
                }),
            })
            .await;

        match self
            .provider
            .complete(req.clone(), self.cancel.clone())
            .await
        {
            Ok(stream) => {
                // On the success path, `LlmCallFinished` is **not** emitted here — the
                // stream has not been drained yet, so the usage for this call is not
                // available. `Finished` is emitted by `run_inner` after draining, with
                // `outcome.usage` (the actual usage for this single call).
                LlmAttempt::Done(stream)
            }
            Err(err) => {
                let hint = err.retry_hint();
                let err_text = err.to_string();
                self.events
                    .emit(AgentEvent::LlmCallFinished {
                        model: req.model.clone(),
                        attempt,
                        usage: Usage::default(),
                        error: Some(err_text),
                    })
                    .await;

                if attempt >= max_attempts || matches!(hint, RetryHint::No) {
                    tracing::warn!(error = %err, ?hint, "llm call failed permanently");
                    return LlmAttempt::Failed(err);
                }
                if let Some(delay) = retry_delay(hint, attempt) {
                    tracing::info!(
                        ?hint,
                        delay_ms = delay.as_millis() as u64,
                        "llm call failed, retrying after delay"
                    );
                    tokio::select! {
                        biased;
                        () = self.cancel.cancelled() => return LlmAttempt::Cancelled,
                        () = tokio::time::sleep(delay) => {}
                    }
                } else {
                    tracing::info!(?hint, "llm call failed, retrying immediately");
                }
                LlmAttempt::Retry
            }
        }
    }

    pub(super) async fn drain_provider_stream(
        &self,
        stream: &mut ProviderStream,
        state: &mut TurnState,
    ) -> Result<DrainOutcome, TurnError> {
        let mut outcome = DrainOutcome::default();

        loop {
            tokio::select! {
                biased;
                () = self.cancel.cancelled() => {
                    outcome.cancelled = true;
                    return Ok(outcome);
                }
                next = stream.next() => match next {
                    None => {
                        if !outcome.saw_stop {
                            outcome.stop = LlmStopReason::EndTurn;
                        }
                        return Ok(outcome);
                    }
                    Some(Err(err)) => {
                        return Err(TurnError::Provider(err));
                    }
                    Some(Ok(chunk)) => {
                        if self.handle_chunk(chunk, &mut outcome, state).await {
                            return Ok(outcome);
                        }
                    }
                }
            }
        }
    }

    /// Process a single chunk. Returns `true` if the stream has reached Stop.
    async fn handle_chunk(
        &self,
        chunk: ProviderChunk,
        outcome: &mut DrainOutcome,
        state: &mut TurnState,
    ) -> bool {
        match chunk {
            ProviderChunk::MessageStart { .. } => false,
            ProviderChunk::TextDelta { text } => {
                outcome.text_buf.push_str(&text);
                self.events
                    .emit(AgentEvent::AssistantText {
                        content: ContentBlock::Text(TextContent::new(text)),
                    })
                    .await;
                false
            }
            ProviderChunk::ThinkingDelta { text } => {
                outcome.thinking_buf.push_str(&text);
                self.events
                    .emit(AgentEvent::AssistantThought {
                        content: ContentBlock::Text(TextContent::new(text)),
                    })
                    .await;
                false
            }
            ProviderChunk::ThinkingSignature { signature } => {
                outcome.thinking_signature = Some(signature);
                false
            }
            ProviderChunk::ToolUseStart { id, name } => {
                outcome.tool_uses.push(ToolUseAccumulated {
                    id,
                    name,
                    args_buf: String::new(),
                });
                false
            }
            ProviderChunk::ToolUseArgsDelta { id, fragment } => {
                if let Some(slot) = outcome.tool_uses.iter_mut().find(|t| t.id == id) {
                    slot.args_buf.push_str(&fragment);
                }
                false
            }
            ProviderChunk::ToolUseEnd { .. } => false,
            ProviderChunk::Stop { reason } => {
                outcome.saw_stop = true;
                outcome.stop = reason;
                false
            }
            ProviderChunk::Usage(u) => {
                outcome.usage = add_usage(outcome.usage, u);
                state.usage = add_usage(state.usage, u);
                false
            }
        }
    }
}

// ----- LLM drain accumulation type -----

/// Result of a single LLM call attempt (the smallest unit wrapped by
/// `.instrument(span).await`).
enum LlmAttempt {
    Done(ProviderStream),
    Failed(crate::llm::ProviderError),
    Cancelled,
    Retry,
}

pub(super) struct DrainOutcome {
    pub(super) saw_stop: bool,
    pub(super) stop: LlmStopReason,
    pub(super) text_buf: String,
    pub(super) thinking_buf: String,
    pub(super) thinking_signature: Option<String>,
    pub(super) tool_uses: Vec<ToolUseAccumulated>,
    pub(super) usage: Usage,
    pub(super) cancelled: bool,
}

impl Default for DrainOutcome {
    fn default() -> Self {
        Self {
            saw_stop: false,
            stop: LlmStopReason::EndTurn,
            text_buf: String::new(),
            thinking_buf: String::new(),
            thinking_signature: None,
            tool_uses: Vec::new(),
            usage: Usage::default(),
            cancelled: false,
        }
    }
}

pub(super) struct ToolUseAccumulated {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) args_buf: String,
}

// ----- helpers -----

/// Assemble the content accumulated by the drain into a single assistant message.
pub(super) fn assistant_message(outcome: &DrainOutcome) -> Message {
    let mut content: Vec<MessageContent> = Vec::new();
    // Thinking must precede Text / ToolUse — the Anthropic wire protocol requires the
    // order thinking → text → tool_use; misordering causes server rejection. On the
    // OpenAI-compatible side, reasoning_content is a top-level message field and order is
    // irrelevant, but keeping a uniform shape aids readability.
    if !outcome.thinking_buf.is_empty() || outcome.thinking_signature.is_some() {
        content.push(MessageContent::Thinking {
            text: outcome.thinking_buf.clone(),
            signature: outcome.thinking_signature.clone(),
        });
    }
    if !outcome.text_buf.is_empty() {
        content.push(MessageContent::Text {
            text: outcome.text_buf.clone(),
        });
    }
    for tu in &outcome.tool_uses {
        let args = parse_args(&tu.args_buf).unwrap_or(JsonValue::Object(Default::default()));
        content.push(MessageContent::ToolUse {
            id: tu.id.clone(),
            name: tu.name.clone(),
            args,
        });
    }
    Message {
        role: Role::Assistant,
        content: content.into(),
    }
}

pub(super) fn parse_args(buf: &str) -> Result<JsonValue, String> {
    if buf.trim().is_empty() {
        return Ok(JsonValue::Object(Default::default()));
    }
    serde_json::from_str(buf).map_err(|e| e.to_string())
}

fn add_usage(a: Usage, b: Usage) -> Usage {
    Usage {
        input_tokens: add_opt(a.input_tokens, b.input_tokens),
        output_tokens: add_opt(a.output_tokens, b.output_tokens),
        cache_read_input_tokens: add_opt(a.cache_read_input_tokens, b.cache_read_input_tokens),
        cache_creation_input_tokens: add_opt(
            a.cache_creation_input_tokens,
            b.cache_creation_input_tokens,
        ),
    }
}

/// The "real input tokens" for an LLM call = `input + cache_read + cache_creation`.
/// Matches Claude Code's `getTokenCountFromUsage`: cache hits and creations are also
/// part of the model's input side and must be counted. Any field that is `None` is
/// treated as 0; if all three are `None`, returns `None` (the provider did not report
/// input tokens, so no baseline is available).
pub(super) fn real_input_tokens(usage: &Usage) -> Option<u64> {
    let input = usage.input_tokens;
    let cache_read = usage.cache_read_input_tokens;
    let cache_creation = usage.cache_creation_input_tokens;
    if input.is_none() && cache_read.is_none() && cache_creation.is_none() {
        return None;
    }
    Some(
        input
            .unwrap_or(0)
            .saturating_add(cache_read.unwrap_or(0))
            .saturating_add(cache_creation.unwrap_or(0)),
    )
}

fn add_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// `attempt` is the count of the **just-failed** attempt (1-based), used to compute the
/// backoff exponent.
fn retry_delay(hint: RetryHint, attempt: u32) -> Option<Duration> {
    match hint {
        RetryHint::No => None,
        RetryHint::Immediate => Some(Duration::from_millis(0)),
        RetryHint::After(d) => Some(d),
        // When the server provides no hint (including 529 overloaded, 5xx, or timeout),
        // use exponential backoff with jitter instead of a fixed short delay — overloads
        // occur probabilistically within time windows, so a fixed 500ms×N almost always
        // hits the same overload wave repeatedly. The formula aligns with the
        // `defect-http` transport backoff layer.
        RetryHint::Backoff => Some(backoff_delay(attempt)),
        RetryHint::AfterAction(_) => Some(Duration::from_millis(0)),
    }
}

/// `BACKOFF_INITIAL * 2^(attempt-1)` with ±25% jitter, capped at [`BACKOFF_MAX`].
/// Same formula as `defect-http`'s transport retry layer (`initial * 2^n ± 25%`).
fn backoff_delay(attempt: u32) -> Duration {
    // attempt starts at 1: first failure uses 2^0 = initial, second uses 2^1, and so on.
    let exp = attempt.saturating_sub(1).min(20);
    let base_nanos = BACKOFF_INITIAL.as_nanos().saturating_mul(1u128 << exp);
    let cap_nanos = BACKOFF_MAX.as_nanos();
    let clamped = base_nanos.min(cap_nanos);

    let mut rng = rand::rng();
    let factor: f64 = 1.0 + rng.random_range(-BACKOFF_JITTER_FRAC..BACKOFF_JITTER_FRAC);
    let nanos = (clamped as f64 * factor).round();
    let nanos = nanos.clamp(0.0, cap_nanos as f64) as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

/// Base backoff: the first retry waits approximately this long, then doubles
/// exponentially.
const BACKOFF_INITIAL: Duration = Duration::from_millis(500);
/// Backoff cap – prevents sleeping too long when `attempt` is large.
const BACKOFF_MAX: Duration = Duration::from_secs(16);
/// Jitter magnitude: ±25%, to spread out retry timing across multiple requests in the
/// same burst of load.
const BACKOFF_JITTER_FRAC: f64 = 0.25;

fn empty_stream() -> ProviderStream {
    Box::pin(futures::stream::empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_backoff_hints_unchanged() {
        assert_eq!(retry_delay(RetryHint::No, 1), None);
        assert_eq!(retry_delay(RetryHint::Immediate, 5), Some(Duration::ZERO));
        let d = Duration::from_secs(7);
        assert_eq!(retry_delay(RetryHint::After(d), 3), Some(d));
    }

    #[test]
    fn backoff_grows_exponentially_within_jitter() {
        // attempt 1 → ~500 ms ±25 % → [375, 625] ms
        for _ in 0..100 {
            let d = backoff_delay(1);
            assert!(
                d >= Duration::from_millis(374) && d <= Duration::from_millis(626),
                "attempt 1 out of jitter range: {d:?}"
            );
        }
        // attempt 3 → ~2000ms ±25% → [1500, 2500]ms
        for _ in 0..100 {
            let d = backoff_delay(3);
            assert!(
                d >= Duration::from_millis(1499) && d <= Duration::from_millis(2501),
                "attempt 3 out of jitter range: {d:?}"
            );
        }
    }

    #[test]
    fn backoff_caps() {
        for _ in 0..100 {
            assert!(backoff_delay(40) <= BACKOFF_MAX, "cap broken");
        }
    }
}
