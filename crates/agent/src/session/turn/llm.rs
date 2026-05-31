//! LLM 调用、重试与流式 drain。
//!
//! 从 turn 主流程疏散出来：`call_llm_with_retry` / `call_llm_attempt` /
//! `drain_provider_stream` / `handle_chunk` 作为 [`super::TurnRunner`] 的方法实现，
//! 加上其专属累积类型（[`DrainOutcome`] / [`LlmAttempt`] / [`ToolUseAccumulated`]）与
//! usage / 重试相关的纯函数 helper。

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{ContentBlock, TextContent};
use futures::StreamExt;
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
    /// 返回成功拿到的流 + 成功时的 attempt 号（供 run_inner 发 LlmCallFinished）。
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
            // 一次 attempt = 一个 llm_call span。span 包住"发请求 + 等响应 +
            // 决定是否重试 + 退避 sleep"四步——失败后进入下一轮重试时
            // 重新建 span（attempt 字段 +1），便于排障对齐每次实际请求。
            // 注意：用 .instrument(span).await，**不要** span.enter() 然后 await
            // ——后者会在 await 时把 entered guard 跨过 await，是 tracing
            // 文档显式警告的 anti-pattern。
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
                // Cancelled：返回空流，attempt 号无意义（不会发 Finished，见 run_inner）。
                LlmAttempt::Cancelled => return Ok((empty_stream(), attempt)),
                LlmAttempt::Retry => continue,
            }
        }
    }

    /// 一次 llm 调用 attempt：发请求、emit 事件、决定下一步。
    /// 与 [`Self::call_llm_with_retry`] 拆开是为了让 `info_span!` 通过
    /// `.instrument(...)` 包住整段 future 而不跨 await 持 entered guard。
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
                // Arc 包裹：fan-out 给多个订阅者时 clone 退化成引用计数，
                // 避免长上下文下整份 messages 历史被反复深拷贝。
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
                // 成功路径**不在这里**发 LlmCallFinished——此刻流还没 drain，
                // 本次调用的 usage 尚未到达。Finished 由 run_inner 在 drain 之后
                // 带上 outcome.usage（单次调用真 usage）发出。
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
                if let Some(delay) = retry_delay(hint) {
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

    /// 处理单个 chunk。返回 `true` 表示流已到 Stop。
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

// ----- LLM drain 累积类型 -----

/// 一次 LLM 调用 attempt 的结果（包给 `.instrument(span).await` 的最小分支）。
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

/// 把 drain 累积的内容组装成一条 assistant 消息。
pub(super) fn assistant_message(outcome: &DrainOutcome) -> Message {
    let mut content: Vec<MessageContent> = Vec::new();
    // Thinking 必须排在 Text / ToolUse 之前 —— Anthropic wire 顺序约定
    // 是 thinking → text → tool_use，错位会被服务端拒；OpenAI 兼容侧
    // reasoning_content 是 message 顶级字段不在乎顺序，统一形态便于阅读。
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

/// 一次 LLM 调用的「真实输入 token」= `input + cache_read + cache_creation`。
/// 对齐 Claude Code 的 `getTokenCountFromUsage`：缓存命中/创建的部分也都进了
/// 模型输入侧，必须计入。任一字段 `None` 视为 0；三项全 `None` 则返回 `None`
/// （provider 没报输入量，无法作为基线）。
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

fn retry_delay(hint: RetryHint) -> Option<Duration> {
    match hint {
        RetryHint::No => None,
        RetryHint::Immediate => Some(Duration::from_millis(0)),
        RetryHint::After(d) => Some(d),
        RetryHint::Backoff => Some(Duration::from_millis(500)),
        RetryHint::AfterAction(_) => Some(Duration::from_millis(0)),
    }
}

fn empty_stream() -> ProviderStream {
    Box::pin(futures::stream::empty())
}
