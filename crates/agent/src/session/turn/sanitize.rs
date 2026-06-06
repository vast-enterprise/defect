//! 发往 provider 前的消息序列修复。
//!
//! ## 为什么需要
//!
//! turn 主循环把 assistant（**含 tool_use**）与紧随的 tool_result 分两步 append 进
//! history（见 `turn.rs`：先 append assistant，工具跑完再 append tool_results）。
//! 这两步之间任何中断——LLM 调用永久失败、`session/cancel`、进程被 SIGKILL、断电——
//! 都会在持久化 history 里留下**孤儿 tool_use**：有 tool_use，却没有对应的
//! tool_result。
//!
//! Anthropic / Bedrock 的硬约束是"每个 tool_use 必须紧跟对应 tool_result"，孤儿一旦
//! 进了请求就被永久拒（`tool_use ids were found without tool_result blocks`），resume
//! 该 session 后每次请求都失败——session 彻底死掉。这种中断**不可避免**（进程随时可能
//! 被杀），所以必须在**读取侧（发请求前）容错**，不能只靠写入侧小心。
//!
//! ## 落点：请求构造处，不是 snapshot
//!
//! 补全**只在即将发给 provider 的请求构造点**调用（`turn.rs` 的 `build_request`、
//! `compact.rs` 的摘要子请求）。**不**放进 `History::snapshot()`：
//!
//! - snapshot 的语义是"忠实读出 history 当前真实状态"——在它里面补全会让"读出的"
//!   ≠"存着的"，破坏纯读契约。
//! - 更致命：micro/full compaction 走 `snapshot() → 处理 → replace()` 把结果**写回**
//!   history。若 snapshot 已注入合成 result，这些"本只该给 provider 看"的合成块就会被
//!   `replace` 固化进持久状态——临时修饰漏进真相源（第二份真相源）。
//!
//! 故：snapshot / replace / storage record / UI 回放全部读到**真实**序列（孤儿如实
//! 存在，UI 上呈现为未闭合的 tool_call，是真实状态的真实呈现）；只有发给 provider 的
//! 那一份被补成 wire 合法。

use crate::llm::{Message, MessageContent, Role, ToolResultBody};

/// 中断的 tool_use 补出的合成 tool_result 文本。
const INTERRUPTED_RESULT_TEXT: &str = "tool call interrupted; no result was recorded";

/// 为每个**无后继 tool_result 的 tool_use** 补一条合成的 error tool_result，使序列满足
/// "每个 tool_use 紧跟对应 tool_result"的 provider 约束。
///
/// 配对判定：assistant 消息里每个 `ToolUse{id}` 的 result，应出现在**紧随其后的那条
/// 消息**（按 Anthropic 形态，tool_result 在下一条 user 消息里）。本函数收集"已被某条
/// 消息满足的 tool_use_id"，对未被满足的，在该 assistant 消息**之后**插入一条
/// `Role::User` 的合成 tool_result 消息。
///
/// 无孤儿时原样返回（仅一次 O(n) 扫描，零分配改动）。
pub(crate) fn sanitize_tool_pairing(messages: Vec<Message>) -> Vec<Message> {
    // 先扫一遍：哪些 tool_use_id 在整个序列里**有**对应 tool_result。
    // 用"全局存在"而非"严格紧邻"判定——已落盘的合法历史里 result 一定紧跟，
    // 只有被中断的才整个缺失；全局判定更宽松、不会误伤合法的多块排布。
    let mut satisfied: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in &messages {
        for content in msg.content.iter() {
            if let MessageContent::ToolResult { tool_use_id, .. } = content {
                satisfied.insert(tool_use_id.clone());
            }
        }
    }

    // 没有任何孤儿则快速返回，避免重建 Vec。
    let has_orphan = messages.iter().any(|msg| {
        msg.content
            .iter()
            .any(|c| matches!(c, MessageContent::ToolUse { id, .. } if !satisfied.contains(id)))
    });
    if !has_orphan {
        return messages;
    }

    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 1);
    for msg in messages {
        // 收集本条 assistant 消息里的孤儿 tool_use id（保持出现顺序）。
        let orphans: Vec<String> = msg
            .content
            .iter()
            .filter_map(|c| match c {
                MessageContent::ToolUse { id, .. } if !satisfied.contains(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        out.push(msg);
        if !orphans.is_empty() {
            out.push(Message {
                role: Role::User,
                content: orphans
                    .into_iter()
                    .map(|id| MessageContent::ToolResult {
                        tool_use_id: id,
                        output: ToolResultBody::Text {
                            text: INTERRUPTED_RESULT_TEXT.to_string(),
                        },
                        is_error: true,
                    })
                    .collect::<Vec<_>>()
                    .into(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests;
