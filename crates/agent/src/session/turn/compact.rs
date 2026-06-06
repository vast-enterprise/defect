//! 上下文压缩编排。
//!
//! 压缩**不**在 [`crate::session::History`] 里做——摘要要调 LLM，存储抽象够不到
//! provider。所以编排放在 turn 主循环这层（对齐 codex `compact.rs` /
//! opencode `compaction.ts` / Claude Code `services/compact`）。
//!
//! 一次压缩：
//! 1. [`select_boundary`]：把历史切成「待摘要前缀 head」+「原样保留尾部 tail」。
//!    边界**对齐到轮次起点**（真实 user 消息），保证 tail 以合法 user 轮开头、
//!    且绝不切散 `tool_use`↔`tool_result` 配对（两个 wire codec 都不校验配对，
//!    必须由我们自己保证，详见 `crates/llm/src/protocol/*`）。
//! 2. [`summarize`]：用当前 provider/model 对 head 跑一次「只产文本」的子请求，
//!    要求按固定结构化模板输出摘要；检出旧摘要时走增量合并。
//! 3. 重建历史：`[合成 assistant 摘要消息] ++ tail`，经 [`History::replace`](crate::session::History::replace) 回写。
//!
//! 失败（无安全边界 / provider 出错 / 摘要为空 / 取消）一律**最佳努力**降级：
//! 跳过本次压缩、不杀 turn——大不了下一次真实调用自己撞上下文上限。

use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::llm::{
    CompletionRequest, HostedCapabilities, LlmProvider, Message, MessageContent, ProviderChunk,
    Role, SamplingParams, StopReason, ThinkingConfig, ToolChoice, ToolResultBody,
    ToolResultContent,
};
use crate::session::CompactionReport;
use crate::session::history::estimate_message_tokens;
use crate::tool::ToolSchema;

/// 保留尾部的 token 预算下限 / 上限（对齐 opencode 的 2k–8k）。
const MIN_TAIL_TOKENS: u64 = 2_000;
const MAX_TAIL_TOKENS: u64 = 8_000;

/// head 中单条 tool_result 喂给摘要模型时截断到的字符上限——避免一份巨型工具
/// 输出把摘要请求本身撑爆（对齐 opencode `toolOutputMaxChars: 2000`）。
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// 合成摘要消息的自描述前缀。既给摘要模型一个语境，也让**后续压缩**能识别出
/// 「这条是上一轮的压缩摘要」从而走增量合并、不把它当普通历史重复保留。
pub(super) const SUMMARY_PREFIX: &str =
    "[Compacted context summary — earlier conversation was condensed to save context.]";

/// 摘要子请求的 system prompt（固定）。
const SUMMARIZER_SYSTEM: &str = "\
You are a context-summarization assistant for a coding agent session. You are given the \
earlier part of a conversation that is about to be dropped to free up context. Summarize \
ONLY what you are given. The newest turns are kept verbatim outside your summary, so focus \
on older context that still matters for continuing the work.

If a <previous-summary> block is present, treat it as the current anchored summary and UPDATE \
it: keep still-true facts, drop stale ones, merge in new facts. Always follow the exact \
section structure the user asks for, keep every section even if empty, preserve exact file \
paths / identifiers / commands / error strings, and prefer terse bullets over prose. Do not \
answer or continue the task itself, and do not mention that you are summarizing. Respond in \
the same language as the conversation.";

/// 结构化摘要模板（user prompt 末尾追加）。
const SUMMARY_TEMPLATE: &str = "\
Summarize the conversation above into the following Markdown structure. Keep every heading \
even if a section is empty (write `(none)`):

## Goal
The user's overall objective and the current concrete task.

## Constraints & Preferences
Hard requirements, user preferences, and conventions to respect.

## Progress
### Done
### In Progress
### Blocked

## Key Decisions
Important choices made and why.

## Next Steps
Concrete, ordered next actions to continue the work.

## Key Context
Critical facts, data, snippets, or references needed to continue.

## Relevant Files
`path` — why it matters (one per line).";

/// 压缩任务的不可变上下文——从 [`super::TurnRunner`] 抽出的、做一次摘要所需的
/// 最小依赖集，全为 owned / `Arc`，故可被 `tokio::spawn` 的**后台**压缩任务持有
/// （`'static`）。同步兜底路径也走它，两条路径共用同一份摘要逻辑。
#[derive(Clone)]
pub(crate) struct CompactionCtx {
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
    pub sampling: SamplingParams,
    pub tools: Vec<ToolSchema>,
    pub cancel: CancellationToken,
}

/// 一次压缩的计划：在某个 snapshot 上选好边界后的产物。`drop_count` 即 head 长度
/// （= 待摘要并丢弃的前缀消息数），回写时交给 `History::splice_prefix`。
pub(super) struct CompactionPlan {
    /// 待摘要的前缀（head）。
    pub head: Vec<Message>,
    /// 上一轮压缩摘要（若 head 里检出），用于增量合并。
    pub prev_summary: Option<String>,
    /// 丢弃的前缀长度。
    pub drop_count: usize,
    /// 压缩前整段（head+tail）的 token 估算。
    pub tokens_before: u64,
}

/// 纯计算：在 `messages` 上按 `threshold` 选边界，切出 head。`None` = 无安全边界
/// （如单个超长轮次 / 仅一个轮次），调用方据此跳过。不碰 `History`、不调 LLM。
pub(super) fn plan(messages: &[Message], threshold: u64) -> Option<CompactionPlan> {
    let tail_budget = (threshold / 4).clamp(MIN_TAIL_TOKENS, MAX_TAIL_TOKENS);
    let Some(boundary) = select_boundary(messages, tail_budget) else {
        tracing::warn!(
            messages = messages.len(),
            tail_budget,
            "compaction skipped: no safe turn boundary to summarize before"
        );
        return None;
    };
    let (head, _tail) = messages.split_at(boundary);
    let prev_summary = extract_previous_summary(head);
    Some(CompactionPlan {
        head: head.to_vec(),
        prev_summary,
        drop_count: boundary,
        tokens_before: estimate_total(messages),
    })
}

/// 把摘要文本包成合成的 assistant 摘要消息（带 [`SUMMARY_PREFIX`]）。
pub(super) fn summary_message(summary: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![MessageContent::Text {
            text: format!("{SUMMARY_PREFIX}\n{summary}"),
        }]
        .into(),
    }
}

/// 同步压缩（hard 水位兜底 / 后台关闭时）：在 turn 主循环里阻塞跑完一次压缩并回写。
/// 返回 `Some(report)` 表示成功（调用方发 `ContextCompressed`）；`None` 最佳努力跳过。
///
/// 用 `splice_prefix(plan.drop_count, ..)` 而非 `replace`：与后台路径同一回写原语，
/// 语义统一——这里 snapshot 与回写之间没有并发尾插，`drop_count` 等价于整表前缀。
pub(super) async fn run_sync(
    history: &dyn crate::session::History,
    ctx: &CompactionCtx,
    threshold: u64,
) -> Option<CompactionReport> {
    let messages = history.snapshot();
    let plan = plan(&messages, threshold)?;
    let summary = summarize(ctx, &plan.head, plan.prev_summary.as_deref()).await?;
    let summary_msg = summary_message(&summary);

    history.splice_prefix(plan.drop_count, summary_msg);
    let tokens_after = estimate_total(&history.snapshot());

    tracing::info!(
        drop_count = plan.drop_count,
        tokens_before = plan.tokens_before,
        tokens_after,
        "context compacted (sync)"
    );
    Some(CompactionReport {
        tokens_before: plan.tokens_before,
        tokens_after,
    })
}

/// 选保留边界：返回**第一条要保留**的消息下标（tail 起点）。
///
/// - 「轮次起点」= role==User 且至少含一个非 `ToolResult` 内容块的消息
///   （即真实用户输入，而非工具结果回填消息）。
/// - 从最新轮次向旧走，按字符启发式累加 tail 体积，整轮保留直到超 `tail_budget`。
/// - 边界必须 `> 0`（head 非空才有得摘要）。若全程只有一个轮次（最新轮次起点
///   就是 0）→ 返回 `None`（没有更早的历史可摘要）。
/// - 若连最新一个轮次都超预算（单个超长轮次），仍把该轮次起点作为边界（不在
///   user 消息内部切），把它之前的统统摘要掉——前提是该起点 `> 0`。
fn select_boundary(messages: &[Message], tail_budget: u64) -> Option<usize> {
    let turn_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_turn_start(m))
        .map(|(i, _)| i)
        .collect();

    let last_start = *turn_starts.last()?;
    // 只有一个轮次（或最新轮次就在开头）→ 无更早历史可摘要。
    if last_start == 0 {
        return None;
    }

    // 从最新轮次起点向旧累加；记录「仍能装下且 >0」的最旧起点。
    let mut best: Option<usize> = None;
    let mut acc: u64 = 0;
    let mut next_boundary = messages.len();
    for &start in turn_starts.iter().rev() {
        acc = acc.saturating_add(estimate_range(messages, start, next_boundary));
        next_boundary = start;
        if start == 0 {
            break;
        }
        if acc <= tail_budget {
            best = Some(start);
        } else {
            break;
        }
    }

    // best 命中 → 用它；否则连最新轮次都超预算，回退到最新轮次起点
    // （last_start 已确认 > 0）。
    Some(best.unwrap_or(last_start))
}

/// 是否「轮次起点」：真实用户输入消息。
///
/// `pub(super)`：微压缩（`session/turn/microcompact.rs`）复用同一把轮次尺子，
/// 避免两处「轮次起点」判定漂移。
pub(super) fn is_turn_start(msg: &Message) -> bool {
    msg.role == Role::User
        && msg
            .content
            .iter()
            .any(|c| !matches!(c, MessageContent::ToolResult { .. }))
}

fn estimate_range(messages: &[Message], start: usize, end: usize) -> u64 {
    messages
        .iter()
        .take(end)
        .skip(start)
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

fn estimate_total(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

/// 在 head 里找上一轮的压缩摘要（以 [`SUMMARY_PREFIX`] 起头的 assistant 文本），
/// 返回其正文（去掉前缀）。用于增量合并。
fn extract_previous_summary(head: &[Message]) -> Option<String> {
    head.iter()
        .filter(|m| m.role == Role::Assistant)
        .find_map(|m| {
            m.content.iter().find_map(|c| match c {
                MessageContent::Text { text } => text
                    .strip_prefix(SUMMARY_PREFIX)
                    .map(|rest| rest.trim_start().to_string()),
                _ => None,
            })
        })
}

/// 对 head 跑一次「只产文本」的摘要子请求，返回摘要正文。
/// 任何失败（取消 / provider 错 / 空）→ `None`（调用方降级跳过）。
pub(super) async fn summarize(
    ctx: &CompactionCtx,
    head: &[Message],
    prev_summary: Option<&str>,
) -> Option<String> {
    let mut messages: Vec<Message> = head.iter().map(prepare_head_message).collect();
    messages.push(Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: build_prompt(prev_summary),
        }]
        .into(),
    });
    // head 切片可能含孤儿 tool_use（中断留下的）——发摘要子请求前同样要补全，
    // 否则摘要调用也会被 provider 拒。与 `build_request` 同一道工序。
    let messages = super::sanitize::sanitize_tool_pairing(messages);

    let req = CompletionRequest {
        model: ctx.model.clone(),
        system: Some(SUMMARIZER_SYSTEM.into()),
        messages,
        // 带上 tools schema 让 head 里的 tool_use/tool_result 历史在 wire 上合法，
        // 但 tool_choice=None 禁止摘要模型真去调工具——它只该产文本。
        tools: ctx.tools.clone(),
        tool_choice: ToolChoice::None,
        sampling: SamplingParams {
            // 摘要不需要思考链，关掉省 token。
            thinking: ThinkingConfig::Disabled,
            ..ctx.sampling.clone()
        },
        hosted_capabilities: HostedCapabilities::default(),
    };

    let mut stream = match ctx.provider.complete(req, ctx.cancel.clone()).await {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(error = %err, "compaction summarize failed: provider error");
            return None;
        }
    };

    let mut text = String::new();
    loop {
        tokio::select! {
            biased;
            () = ctx.cancel.cancelled() => {
                tracing::warn!("compaction summarize cancelled");
                return None;
            }
            next = stream.next() => match next {
                None => break,
                Some(Ok(ProviderChunk::TextDelta { text: delta })) => text.push_str(&delta),
                Some(Ok(ProviderChunk::Stop { reason })) => {
                    if matches!(reason, StopReason::Refusal) {
                        tracing::warn!("compaction summarize refused by model");
                        return None;
                    }
                    // 其余 chunk（thinking / tool_use / usage / message_start）忽略。
                }
                Some(Ok(_)) => {}
                Some(Err(err)) => {
                    tracing::warn!(error = %err, "compaction summarize failed: stream error");
                    return None;
                }
            }
        }
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        tracing::warn!("compaction summarize produced empty summary");
        return None;
    }
    Some(text)
}

/// 拼摘要 user prompt：检出旧摘要则前置 `<previous-summary>` 增量块。
fn build_prompt(prev_summary: Option<&str>) -> String {
    match prev_summary {
        Some(prev) => format!(
            "Update the anchored summary below with the new conversation history.\n\n\
             <previous-summary>\n{prev}\n</previous-summary>\n\n{SUMMARY_TEMPLATE}"
        ),
        None => SUMMARY_TEMPLATE.to_string(),
    }
}

/// 把 head 里的一条消息整备成喂摘要模型的形态：截断超长 tool_result、剥离图片。
fn prepare_head_message(msg: &Message) -> Message {
    let content: Vec<MessageContent> = msg
        .content
        .iter()
        .map(|c| match c {
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => MessageContent::ToolResult {
                tool_use_id: tool_use_id.clone(),
                output: truncate_tool_output(output),
                is_error: *is_error,
            },
            // 图片对文本摘要无意义且占带宽——换成占位文本。
            MessageContent::Image { .. } => MessageContent::Text {
                text: "[image omitted from summary]".to_string(),
            },
            other => other.clone(),
        })
        .collect();
    Message {
        role: msg.role,
        content: content.into(),
    }
}

fn truncate_tool_output(output: &ToolResultBody) -> ToolResultBody {
    match output {
        ToolResultBody::Text { text } => ToolResultBody::Text {
            text: truncate_chars(text, TOOL_RESULT_MAX_CHARS),
        },
        ToolResultBody::Json { value } => {
            let s = value.to_string();
            if s.len() <= TOOL_RESULT_MAX_CHARS {
                ToolResultBody::Json {
                    value: value.clone(),
                }
            } else {
                // 超长 JSON 降级成截断后的文本——摘要只需大意，不需结构完整。
                ToolResultBody::Text {
                    text: truncate_chars(&s, TOOL_RESULT_MAX_CHARS),
                }
            }
        }
        // 多模态结果在摘要里降级成纯文本：保留文本块（截断），图片块换成
        // 占位标注——base64 喂进摘要既无意义又昂贵。
        ToolResultBody::Content { blocks } => {
            let mut text = String::new();
            for block in blocks {
                match block {
                    ToolResultContent::Text { text: t } => text.push_str(t),
                    ToolResultContent::Image { mime, .. } => {
                        text.push_str(&format!("\n[image: {mime}]"));
                    }
                }
            }
            ToolResultBody::Text {
                text: truncate_chars(&text, TOOL_RESULT_MAX_CHARS),
            }
        }
    }
}

/// 按**字符边界**截断（不在多字节 UTF-8 中间切），超长则补省略标注。
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let kept: String = s.chars().take(max_chars).collect();
    format!("{kept}\n…[truncated for summary]")
}

#[cfg(test)]
mod tests;
