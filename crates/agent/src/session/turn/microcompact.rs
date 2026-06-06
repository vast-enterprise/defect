//! 微压缩（microcompact）——便宜的第一道上下文防线。
//!
//! 与全量压缩（`compact.rs`）的本质区别：**不调 LLM**，**不删消息**，只把**较旧
//! 轮次**里体积超标的 `tool_result` 正文换成占位符。结构（消息条数、role、
//! `tool_use`↔`tool_result` 配对）一律不动——故 wire 上永远合法，且模型仍看得到
//! 「自己调过哪些工具」，只是看不到那些（多半已不相关的）庞大工具输出正文。
//!
//! 纯 `O(n)` 数据变换、零网络往返，可在 turn 主循环里同步跑（对齐 Claude Code 的
//! microcompact）。它常常就能把工具输出大户削下去，把昂贵的全量压缩推后。
//!
//! ## 安全约束
//!
//! 1. **不增删消息**——只重写 `ToolResult` 的 `output` 字段。这维持了 `History`
//!    的并发不变式（见 `splice_prefix`），与飞行中的后台全量压缩共存。
//! 2. **保留窗口**：最近 [`KEEP_RECENT_TURNS`] 个轮次的工具结果原样留——大概率
//!    还在用。只清更老轮次的。
//! 3. **尺寸地板**：小于 [`MIN_CLEAR_TOKENS`] 的工具结果不动（不值当）。
//! 4. **幂等**：占位文本本身即 sentinel；已清过的再跑直接跳过。

use crate::llm::{Message, MessageContent, ToolResultBody};
use crate::session::history::estimate_message_tokens;

use super::compact::is_turn_start;

/// 保留窗口：最近 N 个轮次的 tool_result 不清。
const KEEP_RECENT_TURNS: usize = 3;

/// 尺寸地板：估算 token 数不超过它的 tool_result 不值得清。
const MIN_CLEAR_TOKENS: u64 = 512;

/// 清理后填入的占位文本。既告知模型该输出已被回收，也充当幂等 sentinel。
pub(super) const CLEARED_PLACEHOLDER: &str =
    "[tool output cleared to save context — re-run the tool if its result is needed again]";

/// 微压缩的报告：清理前后的整段 token 估算。`cleared` = 实际清理的 tool_result 条数。
pub(super) struct MicrocompactReport {
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub cleared: usize,
}

/// 对 `messages` 跑一次微压缩，返回 `Some(rebuilt, report)`（确有清理）或 `None`
/// （无可清理项——调用方据此跳过回写、不发事件）。
///
/// 纯函数：不碰 `History`，由调用方决定如何回写（当前走 `replace`）。
pub(super) fn run(messages: &[Message]) -> Option<(Vec<Message>, MicrocompactReport)> {
    // 找保留边界：从末尾数 KEEP_RECENT_TURNS 个轮次起点，其后（含）全部保留。
    let turn_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_turn_start(m))
        .map(|(i, _)| i)
        .collect();

    // 保留边界下标：轮次数不足保留窗口 → 无更老轮次可清 → 直接跳过。
    // 倒数第 KEEP_RECENT_TURNS 个轮次起点即保留边界；`nth_back(K-1)` 取它，
    // 不足 K 个时 `None` → 跳过（避免裸索引可能 panic）。
    let keep_from = *turn_starts.iter().rev().nth(KEEP_RECENT_TURNS - 1)?;

    let tokens_before = estimate_total(messages);
    let mut cleared = 0usize;

    let rebuilt: Vec<Message> = messages
        .iter()
        .enumerate()
        .map(|(idx, msg)| {
            // 保留窗口内的消息原样放行。
            if idx >= keep_from {
                return msg.clone();
            }
            clear_oversized_results(msg, &mut cleared)
        })
        .collect();

    if cleared == 0 {
        return None;
    }

    let tokens_after = estimate_total(&rebuilt);
    Some((
        rebuilt,
        MicrocompactReport {
            tokens_before,
            tokens_after,
            cleared,
        },
    ))
}

/// 把一条消息里**超标且未清过**的 `ToolResult` 正文换成占位符；其余内容块原样。
/// 命中即 `cleared += 1`。
fn clear_oversized_results(msg: &Message, cleared: &mut usize) -> Message {
    // 先看这条消息有没有任何 ToolResult——绝大多数消息没有，避免无谓 clone。
    let has_tool_result = msg
        .content
        .iter()
        .any(|c| matches!(c, MessageContent::ToolResult { .. }));
    if !has_tool_result {
        return msg.clone();
    }

    let content: Vec<MessageContent> = msg
        .content
        .iter()
        .map(|c| match c {
            MessageContent::ToolResult {
                tool_use_id,
                output,
                is_error,
            } if should_clear(output) => {
                *cleared += 1;
                MessageContent::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    output: ToolResultBody::Text {
                        text: CLEARED_PLACEHOLDER.to_string(),
                    },
                    // 保留 is_error：模型仍知道当初这次工具调用是失败的。
                    is_error: *is_error,
                }
            }
            other => other.clone(),
        })
        .collect();

    Message {
        role: msg.role,
        content: content.into(),
    }
}

/// 该 tool_result 是否该被清：超尺寸地板，且尚未被清过（幂等）。
fn should_clear(output: &ToolResultBody) -> bool {
    if already_cleared(output) {
        return false;
    }
    estimate_tool_result_tokens(output) > MIN_CLEAR_TOKENS
}

/// 是否已是占位符（幂等 sentinel）。
fn already_cleared(output: &ToolResultBody) -> bool {
    matches!(output, ToolResultBody::Text { text } if text == CLEARED_PLACEHOLDER)
}

/// 单个 tool_result 的 token 估算——借一条只含它的临时消息复用 `estimate_message_tokens`，
/// 与压缩判定 / 触发判定同一把尺子，不另立口径。
fn estimate_tool_result_tokens(output: &ToolResultBody) -> u64 {
    let probe = Message {
        role: crate::llm::Role::User,
        content: vec![MessageContent::ToolResult {
            tool_use_id: String::new(),
            output: output.clone(),
            is_error: false,
        }]
        .into(),
    };
    estimate_message_tokens(&probe)
}

fn estimate_total(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u64, u64::saturating_add)
}

#[cfg(test)]
mod test;
