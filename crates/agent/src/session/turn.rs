//! Turn 主循环。
//!
//! 设计详见 `docs/internal/turn-loop.md`。本文件按 §3 的状态机落地。
//!
//! 关键依赖：
//! - [`History`]：消息历史的读写
//! - [`ToolRegistry`]：工具查找
//! - [`LlmProvider`]：LLM 调用
//! - [`EventEmitter`]：事件发布（`Arc` 共享，使工具 task 也能 emit）
//! - [`PermissionGate`]：权限请求等待

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::{
    Content as AcpContent, ContentBlock, EmbeddedResource, EmbeddedResourceResource, ImageContent,
    ResourceLink, StopReason as AcpStopReason, TextContent, TextResourceContents, ToolCallContent,
    ToolCallId, ToolCallStatus, ToolCallUpdateFields,
};
use futures::StreamExt;
use serde_json::Value as JsonValue;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::error::BoxError;
use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::llm::{
    CompletionRequest, HostedCapabilities, LlmProvider, Message, MessageContent, ProviderChunk,
    ProviderStream, RetryHint, Role, SamplingParams, StopReason as LlmStopReason, ToolChoice,
    ToolResultBody, Usage,
};
use crate::policy::{PolicyCtx, PolicyDecision, RecordedOutcome, SandboxPolicy};
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::{History, ToolRegistry, TurnError};
use crate::shell::ShellBackend;
use crate::tool::{Tool, ToolContext, ToolError, ToolEvent};

const DEFAULT_PROMPT_FILE: &str = "AGENTS.md";

/// LLM 调用次数上限策略。详见 `docs/internal/turn-loop.md` §6.1。
#[derive(Debug, Clone, Copy)]
pub enum TurnRequestLimit {
    /// 不设上限。
    Unbounded,
    /// 固定上限：到 N 即返回 [`AcpStopReason::MaxTurnRequests`]。
    Fixed(u32),
    /// 自适应：每当本轮"有 tool_use 被批准执行"视为推进，
    /// 上限自动 +1；否则按 [`Self::Fixed`] 终止。
    Adaptive {
        initial: u32,
        expand_on_progress: bool,
    },
}

impl TurnRequestLimit {
    fn initial_cap(&self) -> Option<u32> {
        match *self {
            Self::Unbounded => None,
            Self::Fixed(n) => Some(n),
            Self::Adaptive { initial, .. } => Some(initial),
        }
    }

    fn expand_on_progress(&self) -> bool {
        matches!(
            self,
            Self::Adaptive {
                expand_on_progress: true,
                ..
            }
        )
    }
}

/// turn 配置。详见 `docs/internal/turn-loop.md` §9。
#[derive(Debug, Clone)]
pub struct TurnConfig {
    pub model: String,
    pub allowed_models: Option<Vec<String>>,
    pub base_prompt: BasePromptConfig,
    pub system_prompt: Option<String>,
    pub prompt: PromptConfig,
    pub sampling: SamplingParams,
    pub request_limit: TurnRequestLimit,
    pub compact_threshold_tokens: Option<u64>,
    pub max_llm_retries: u32,
    /// `0` = 不限。v0 默认不限。
    pub max_concurrent_tools: usize,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            allowed_models: None,
            base_prompt: BasePromptConfig::default(),
            system_prompt: None,
            prompt: PromptConfig::default(),
            sampling: SamplingParams::default(),
            request_limit: TurnRequestLimit::Adaptive {
                initial: 32,
                expand_on_progress: true,
            },
            compact_threshold_tokens: None,
            max_llm_retries: 3,
            max_concurrent_tools: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BasePromptConfig {
    pub file: Option<PathBuf>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptConfig {
    pub file: String,
    pub text: Option<String>,
    pub provider_overlays: std::collections::BTreeMap<String, String>,
    pub model_overlays: std::collections::BTreeMap<String, String>,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            file: DEFAULT_PROMPT_FILE.to_owned(),
            text: None,
            provider_overlays: std::collections::BTreeMap::new(),
            model_overlays: std::collections::BTreeMap::new(),
        }
    }
}

/// turn 一次执行的全部依赖与累计状态。
///
/// 本 struct 由 [`crate::session::DefaultSession`] 在每次 `run_turn` 时构造，
/// 借用 session 的子组件、跑完即销毁。
pub struct TurnRunner<'a> {
    pub history: &'a dyn History,
    pub tools: &'a dyn ToolRegistry,
    pub provider: &'a dyn LlmProvider,
    pub policy: &'a dyn SandboxPolicy,
    pub events: Arc<EventEmitter>,
    pub permissions: &'a PermissionGate,
    pub cancel: CancellationToken,
    pub config: &'a TurnConfig,
    pub system_prompt: Option<String>,
    pub cwd: &'a std::path::Path,
    pub fs: Arc<dyn FsBackend>,
    pub shell: Arc<dyn ShellBackend>,
    /// session 启动期裁决出的 hosted capability 集合。
    /// 每轮 turn 装配请求时直接复用，不再重新查询。
    pub hosted_capabilities: HostedCapabilities,
}

impl<'a> TurnRunner<'a> {
    /// 跑完一次 turn。
    pub async fn run(&self, prompt: Vec<ContentBlock>) -> Result<AcpStopReason, TurnError> {
        self.events
            .emit(AgentEvent::UserPromptCommitted {
                content: prompt.clone(),
            })
            .await;
        self.history.append(Message {
            role: Role::User,
            content: prompt
                .into_iter()
                .map(content_block_to_message_content)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect(),
        });

        self.events.emit(AgentEvent::TurnStarted).await;

        let result = self.run_inner().await;

        if let Ok(outcome) = &result {
            self.events
                .emit(AgentEvent::TurnEnded {
                    reason: outcome.reason,
                    usage: outcome.usage,
                })
                .await;
        }
        // Err 路径不发 TurnEnded：桥接层据 future outcome 自行决定 wire 响应。

        result.map(|outcome| outcome.reason)
    }

    async fn run_inner(&self) -> Result<TurnOutcome, TurnError> {
        let mut state = TurnState::new(self.config.request_limit);

        loop {
            if self.cancel.is_cancelled() {
                return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
            }

            self.maybe_compact().await?;

            let req = self.build_request();
            let mut stream = self.call_llm_with_retry(&req, &mut state).await?;

            let outcome = self.drain_provider_stream(&mut stream, &mut state).await?;

            if outcome.cancelled {
                return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
            }

            let assistant = assistant_message(&outcome);
            if !assistant.content.is_empty() {
                self.history.append(assistant);
            }

            match outcome.stop {
                LlmStopReason::EndTurn | LlmStopReason::StopSequence => {
                    return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
                }
                LlmStopReason::Refusal => {
                    return Ok(turn_outcome(&state, AcpStopReason::Refusal));
                }
                LlmStopReason::MaxTokens => {
                    return Ok(turn_outcome(&state, AcpStopReason::MaxTokens));
                }
                LlmStopReason::ToolUse => {}
            }

            if outcome.tool_uses.is_empty() {
                return Ok(turn_outcome(&state, AcpStopReason::EndTurn));
            }

            let approved = match self.decide_permissions(&outcome.tool_uses).await? {
                DecisionFlow::Continue(list) => list,
                DecisionFlow::Cancelled => {
                    return Ok(turn_outcome(&state, AcpStopReason::Cancelled));
                }
            };

            let progressed = approved.iter().any(|a| matches!(a, Approved::Run { .. }));
            if progressed {
                state.note_progress();
            }

            let results = self.run_tools_concurrently(approved).await;
            self.history.append(tool_results_message(results));

            if state.exceeded_request_cap() {
                return Ok(turn_outcome(&state, AcpStopReason::MaxTurnRequests));
            }
        }
    }

    fn build_request(&self) -> CompletionRequest {
        CompletionRequest {
            model: self.config.model.clone(),
            system: self.system_prompt.clone(),
            messages: self.history.snapshot(),
            tools: self.tools.schemas(),
            tool_choice: ToolChoice::Auto,
            sampling: self.config.sampling.clone(),
            hosted_capabilities: self.hosted_capabilities,
        }
    }

    async fn maybe_compact(&self) -> Result<(), TurnError> {
        let Some(threshold) = self.config.compact_threshold_tokens else {
            return Ok(());
        };
        let Some(estimate) = self.history.token_estimate() else {
            return Ok(());
        };
        if estimate < threshold {
            return Ok(());
        }
        let report = self.history.compact().await.map_err(TurnError::Internal)?;
        self.events
            .emit(AgentEvent::ContextCompressed {
                tokens_before: report.tokens_before,
                tokens_after: report.tokens_after,
            })
            .await;
        Ok(())
    }

    async fn call_llm_with_retry(
        &self,
        req: &CompletionRequest,
        state: &mut TurnState,
    ) -> Result<ProviderStream, TurnError> {
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
                LlmAttempt::Done(stream) => return Ok(stream),
                LlmAttempt::Failed(err) => return Err(TurnError::Provider(err)),
                LlmAttempt::Cancelled => return Ok(empty_stream()),
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
            })
            .await;

        match self
            .provider
            .complete(req.clone(), self.cancel.clone())
            .await
        {
            Ok(stream) => {
                self.events
                    .emit(AgentEvent::LlmCallFinished {
                        model: req.model.clone(),
                        attempt,
                        usage: Usage::default(),
                        error: None,
                    })
                    .await;
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

    async fn drain_provider_stream(
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

    async fn decide_permissions(
        &self,
        tool_uses: &[ToolUseAccumulated],
    ) -> Result<DecisionFlow, TurnError> {
        let mut approved: Vec<Approved> = Vec::with_capacity(tool_uses.len());

        for tu in tool_uses {
            let id = ToolCallId::new(tu.id.clone());

            let Some(tool) = self.tools.get(&tu.name) else {
                let reason = format!("tool not found: {}", tu.name);
                self.emit_tool_failed(&id, reason.clone()).await;
                approved.push(Approved::FailedArgs {
                    tool_use_id: tu.id.clone(),
                    reason,
                });
                continue;
            };

            let args: JsonValue = match parse_args(&tu.args_buf) {
                Ok(v) => v,
                Err(reason) => {
                    let reason = format!("invalid args: {reason}");
                    self.emit_tool_failed(&id, reason.clone()).await;
                    approved.push(Approved::FailedArgs {
                        tool_use_id: tu.id.clone(),
                        reason,
                    });
                    continue;
                }
            };

            let describe_ctx = ToolContext::new(
                self.cwd,
                self.cancel.clone(),
                self.fs.clone(),
                self.shell.clone(),
            );
            let description = tool.describe(&args, describe_ctx).await;
            self.events
                .emit(AgentEvent::ToolCallStarted {
                    id: id.clone(),
                    name: tu.name.clone(),
                    fields: with_status(description.fields.clone(), ToolCallStatus::Pending),
                })
                .await;

            let safety_hint = tool.safety_hint(&args);
            let decision =
                self.policy
                    .classify(PolicyCtx::new(&tu.name, safety_hint, &args, self.cwd));
            self.events
                .emit(AgentEvent::PolicyDecision {
                    id: id.clone(),
                    decision: decision.clone(),
                })
                .await;

            match decision {
                PolicyDecision::Allow => approved.push(Approved::Run {
                    id,
                    tool_use_id: tu.id.clone(),
                    tool: tool.clone(),
                    args,
                }),
                PolicyDecision::Deny => {
                    self.emit_tool_failed(&id, "denied by policy".to_string())
                        .await;
                    approved.push(Approved::Denied {
                        tool_use_id: tu.id.clone(),
                    });
                }
                PolicyDecision::Ask(ask) => {
                    if ask.options.is_empty() {
                        // 空 options 等价 Deny（见 sandbox-policy.md §2）
                        self.emit_tool_failed(&id, "denied by policy".to_string())
                            .await;
                        approved.push(Approved::Denied {
                            tool_use_id: tu.id.clone(),
                        });
                        continue;
                    }
                    let outcome = self.permissions.wait(id.clone(), self.cancel.clone()).await;
                    self.events
                        .emit(AgentEvent::PermissionResolved {
                            id: id.clone(),
                            outcome: outcome.clone(),
                        })
                        .await;
                    match outcome {
                        PermissionResolution::Selected { option_id } => {
                            let allows = ask
                                .options
                                .iter()
                                .find(|o| o.id == option_id)
                                .map(|o| o.allows)
                                .unwrap_or(false);
                            self.policy.record(
                                PolicyCtx::new(&tu.name, safety_hint, &args, self.cwd),
                                RecordedOutcome::Selected { option_id, allows },
                            );
                            if allows {
                                approved.push(Approved::Run {
                                    id,
                                    tool_use_id: tu.id.clone(),
                                    tool: tool.clone(),
                                    args,
                                });
                            } else {
                                self.emit_tool_failed(&id, "denied by user".to_string())
                                    .await;
                                approved.push(Approved::Denied {
                                    tool_use_id: tu.id.clone(),
                                });
                            }
                        }
                        PermissionResolution::Cancelled => {
                            self.policy.record(
                                PolicyCtx::new(&tu.name, safety_hint, &args, self.cwd),
                                RecordedOutcome::Cancelled,
                            );
                            return Ok(DecisionFlow::Cancelled);
                        }
                    }
                }
            }
        }

        Ok(DecisionFlow::Continue(approved))
    }

    async fn emit_tool_failed(&self, id: &ToolCallId, text: String) {
        let fields = failed_fields_text(text);
        self.events
            .emit(AgentEvent::ToolCallStarted {
                id: id.clone(),
                name: String::new(),
                fields: fields.clone(),
            })
            .await;
        self.events
            .emit(AgentEvent::ToolCallFinished {
                id: id.clone(),
                fields,
            })
            .await;
    }

    async fn run_tools_concurrently(&self, approved: Vec<Approved>) -> Vec<ToolResult> {
        let mut joinset: JoinSet<ToolResult> = JoinSet::new();
        let mut results: Vec<ToolResult> = Vec::with_capacity(approved.len());

        for a in approved {
            match a {
                Approved::Run {
                    id,
                    tool_use_id,
                    tool,
                    args,
                } => {
                    let cancel = self.cancel.child_token();
                    let events = self.events.clone();
                    let cwd = self.cwd.to_path_buf();
                    let fs = self.fs.clone();
                    let shell = self.shell.clone();
                    let span = tracing::info_span!(
                        "tool_call",
                        tool = %tool.schema().name,
                        tool_call_id = %id,
                    );
                    joinset.spawn(
                        drive_tool_stream(
                            id,
                            tool_use_id,
                            tool,
                            args,
                            cwd,
                            cancel,
                            events,
                            fs,
                            shell,
                        )
                        .instrument(span),
                    );
                }
                Approved::Denied { tool_use_id } => {
                    results.push(ToolResult {
                        tool_use_id,
                        body: ToolResultBody::Text {
                            text: "denied".to_string(),
                        },
                        is_error: true,
                    });
                }
                Approved::FailedArgs {
                    tool_use_id,
                    reason,
                } => {
                    results.push(ToolResult {
                        tool_use_id,
                        body: ToolResultBody::Text { text: reason },
                        is_error: true,
                    });
                }
            }
        }

        while let Some(res) = joinset.join_next().await {
            match res {
                Ok(r) => results.push(r),
                Err(join_err) => {
                    tracing::error!(error = ?join_err, "tool task panicked");
                    results.push(ToolResult {
                        tool_use_id: String::new(),
                        body: ToolResultBody::Text {
                            text: format!("tool task crashed: {join_err}"),
                        },
                        is_error: true,
                    });
                }
            }
        }

        results
    }
}

// ----- internal types -----

/// 一次 LLM 调用 attempt 的结果（包给 `.instrument(span).await` 的最小分支）。
enum LlmAttempt {
    Done(ProviderStream),
    Failed(crate::llm::ProviderError),
    Cancelled,
    Retry,
}

struct DrainOutcome {
    saw_stop: bool,
    stop: LlmStopReason,
    text_buf: String,
    thinking_buf: String,
    thinking_signature: Option<String>,
    tool_uses: Vec<ToolUseAccumulated>,
    usage: Usage,
    cancelled: bool,
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

struct ToolUseAccumulated {
    id: String,
    name: String,
    args_buf: String,
}

enum Approved {
    Run {
        id: ToolCallId,
        tool_use_id: String,
        tool: Arc<dyn Tool>,
        args: JsonValue,
    },
    Denied {
        tool_use_id: String,
    },
    FailedArgs {
        tool_use_id: String,
        reason: String,
    },
}

/// `decide_permissions` 的返回：要么继续把 approved 列表交给执行阶段，
/// 要么用户在 `Ask` 阶段取消了 turn。
enum DecisionFlow {
    Continue(Vec<Approved>),
    Cancelled,
}

#[derive(Clone, Copy)]
struct TurnOutcome {
    reason: AcpStopReason,
    usage: Usage,
}

struct ToolResult {
    tool_use_id: String,
    body: ToolResultBody,
    is_error: bool,
}

struct TurnState {
    request_count: u32,
    usage: Usage,
    cap: Option<u32>,
    expand_on_progress: bool,
}

impl TurnState {
    fn new(limit: TurnRequestLimit) -> Self {
        Self {
            request_count: 0,
            usage: Usage::default(),
            cap: limit.initial_cap(),
            expand_on_progress: limit.expand_on_progress(),
        }
    }

    fn note_progress(&mut self) {
        if self.expand_on_progress
            && let Some(cap) = self.cap.as_mut()
        {
            *cap = cap.saturating_add(1);
        }
    }

    fn exceeded_request_cap(&self) -> bool {
        match self.cap {
            None => false,
            Some(cap) => self.request_count >= cap,
        }
    }
}

// ----- helpers -----

fn content_block_to_message_content(cb: ContentBlock) -> Result<Vec<MessageContent>, TurnError> {
    match cb {
        ContentBlock::Text(TextContent { text, .. }) => Ok(vec![MessageContent::Text { text }]),
        ContentBlock::Image(image) => Ok(vec![image_content_to_message_content(image)?]),
        ContentBlock::ResourceLink(link) => Ok(vec![MessageContent::Text {
            text: resource_link_to_text(link),
        }]),
        ContentBlock::Resource(resource) => resource_to_message_content(resource),
        ContentBlock::Audio(_) => Err(invalid_prompt_content(
            "ACP audio content is not supported yet",
        )),
        _ => Err(invalid_prompt_content(
            "unsupported ACP content block variant",
        )),
    }
}

fn image_content_to_message_content(image: ImageContent) -> Result<MessageContent, TurnError> {
    let data = if image.data.is_empty() {
        let Some(uri) = image.uri else {
            return Err(invalid_prompt_content(
                "ACP image content must include data or uri",
            ));
        };
        crate::llm::ImageData::Url { url: uri }
    } else {
        crate::llm::ImageData::Base64 {
            encoded: image.data,
        }
    };

    Ok(MessageContent::Image {
        mime: image.mime_type,
        data,
    })
}

fn resource_to_message_content(
    resource: EmbeddedResource,
) -> Result<Vec<MessageContent>, TurnError> {
    match resource.resource {
        EmbeddedResourceResource::TextResourceContents(text) => Ok(vec![MessageContent::Text {
            text: text_resource_to_text(text),
        }]),
        EmbeddedResourceResource::BlobResourceContents(blob) => {
            Err(invalid_prompt_content(&format!(
                "embedded binary resource is not supported yet: {}",
                blob.uri
            )))
        }
        _ => Err(invalid_prompt_content(
            "unsupported embedded resource variant",
        )),
    }
}

fn resource_link_to_text(link: ResourceLink) -> String {
    let mut lines = vec![format!("resource: {}", link.name)];
    if let Some(title) = link.title {
        lines.push(format!("title: {title}"));
    }
    if let Some(description) = link.description {
        lines.push(format!("description: {description}"));
    }
    if let Some(mime_type) = link.mime_type {
        lines.push(format!("mime_type: {mime_type}"));
    }
    if let Some(size) = link.size {
        lines.push(format!("size: {size}"));
    }
    lines.push(format!("uri: {}", link.uri));
    lines.join("\n")
}

fn text_resource_to_text(resource: TextResourceContents) -> String {
    let mut text = format!("resource: {}", resource.uri);
    if let Some(mime_type) = resource.mime_type {
        text.push_str(&format!("\nmime_type: {mime_type}"));
    }
    text.push_str("\n\n");
    text.push_str(&resource.text);
    text
}

fn invalid_prompt_content(message: &str) -> TurnError {
    TurnError::Internal(BoxError::new(io::Error::other(message)))
}

fn assistant_message(outcome: &DrainOutcome) -> Message {
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
        content,
    }
}

fn tool_results_message(results: Vec<ToolResult>) -> Message {
    Message {
        role: Role::User,
        content: results
            .into_iter()
            .map(|r| MessageContent::ToolResult {
                tool_use_id: r.tool_use_id,
                output: r.body,
                is_error: r.is_error,
            })
            .collect(),
    }
}

fn parse_args(buf: &str) -> Result<JsonValue, String> {
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

fn add_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn turn_outcome(state: &TurnState, reason: AcpStopReason) -> TurnOutcome {
    TurnOutcome {
        reason,
        usage: state.usage,
    }
}

fn with_status(mut f: ToolCallUpdateFields, status: ToolCallStatus) -> ToolCallUpdateFields {
    f.status = Some(status);
    f
}

fn failed_fields_text(text: String) -> ToolCallUpdateFields {
    let mut f = ToolCallUpdateFields::default();
    f.status = Some(ToolCallStatus::Failed);
    f.content = Some(vec![ToolCallContent::Content(AcpContent::new(text))]);
    f
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

#[cfg(test)]
mod test;

/// 把第一段文本从 [`ToolCallUpdateFields::content`] 抽出来当 tool_result。
fn extract_text(fields: &ToolCallUpdateFields) -> Option<String> {
    let blocks = fields.content.as_ref()?;
    blocks.iter().find_map(|c| match c {
        ToolCallContent::Content(inner) => match &inner.content {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        },
        _ => None,
    })
}

/// 单个工具流的驱动 task。把 [`ToolEvent`] 转发为 [`AgentEvent`]，最后产出
/// [`ToolResult`] 喂回 LLM。
#[allow(clippy::too_many_arguments)]
async fn drive_tool_stream(
    id: ToolCallId,
    tool_use_id: String,
    tool: Arc<dyn Tool>,
    args: JsonValue,
    cwd: std::path::PathBuf,
    cancel: CancellationToken,
    events: Arc<EventEmitter>,
    fs: Arc<dyn FsBackend>,
    shell: Arc<dyn ShellBackend>,
) -> ToolResult {
    let ctx = ToolContext::new(&cwd, cancel.clone(), fs.clone(), shell.clone());
    let mut stream = tool.execute(args, ctx);

    let mut last_text: Option<String> = None;

    // 注意：cancel 通过 ctx.cancel 注入工具内部，由工具自己感知并产出
    // [`ToolEvent::Failed(ToolError::Canceled)`]——不要在驱动层加 cancel arm。
    // 一旦驱动层 select 里 drop 掉 stream，工具内部任何在飞的 ACP 反向请求
    // 的 oneshot::Receiver 都会被 drop，server 把"无人接收"映射成 internal_error
    // 并撕掉整条连接（详见 `agent_client_protocol::jsonrpc::incoming_actor`
    // 里 `router.respond_with_result` 的 ?）。Tool trait 契约：必须感知 cancel。
    while let Some(ev) = stream.next().await {
        match ev {
            ToolEvent::Progress(fields) => {
                if let Some(text) = extract_text(&fields) {
                    last_text = Some(text);
                }
                events
                    .emit(AgentEvent::ToolCallProgress {
                        id: id.clone(),
                        fields: with_status(fields, ToolCallStatus::InProgress),
                    })
                    .await;
            }
            ToolEvent::Completed(fields) => {
                if let Some(text) = extract_text(&fields) {
                    last_text = Some(text);
                }
                events
                    .emit(AgentEvent::ToolCallFinished {
                        id: id.clone(),
                        fields: with_status(fields, ToolCallStatus::Completed),
                    })
                    .await;
                return ToolResult {
                    tool_use_id,
                    body: ToolResultBody::Text {
                        text: last_text.unwrap_or_default(),
                    },
                    is_error: false,
                };
            }
            ToolEvent::Failed(err) => {
                let text = err.to_string();
                let is_cancel = matches!(err, ToolError::Canceled);
                events
                    .emit(AgentEvent::ToolCallFinished {
                        id: id.clone(),
                        fields: failed_fields_text(text.clone()),
                    })
                    .await;
                return ToolResult {
                    tool_use_id,
                    body: ToolResultBody::Text { text },
                    is_error: !is_cancel,
                };
            }
        }
    }

    events
        .emit(AgentEvent::ToolCallFinished {
            id: id.clone(),
            fields: failed_fields_text("tool stream closed without terminal event".to_string()),
        })
        .await;
    ToolResult {
        tool_use_id,
        body: ToolResultBody::Text {
            text: "tool stream closed without terminal event".to_string(),
        },
        is_error: true,
    }
}
