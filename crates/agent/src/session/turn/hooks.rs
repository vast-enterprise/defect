//! Turn 主循环里的 hook 触发逻辑。
//!
//! 从 turn 主流程疏散出来：`decide_turn_end`（before turn-end 续命判定）、prompt 摄入 / 工具
//! 前后的 `fire_*` 触发，以及反馈注入 helper，作为 [`super::TurnRunner`] 的方法实现。
//! step 类型与引擎见 `crate::hooks`。

use agent_client_protocol_schema::{ContentBlock, StopReason as AcpStopReason, ToolCallId};
use serde_json::Value as JsonValue;

use crate::llm::{Message, MessageContent, Role, ToolResultBody, ToolResultContent};

use super::content::content_block_to_message_content;
use super::tools::ToolResult;
use super::{TurnRunner, TurnState};

/// `fire_user_prompt_submit` 的结果。
pub(super) enum UserPromptHookFlow {
    Continue(Vec<ContentBlock>),
    Refused,
}

/// `fire_pre_tool_use` 的结果。
pub(super) enum PreToolHookFlow {
    Continue { args: JsonValue },
    Block(String),
}

impl TurnRunner<'_> {
    /// `before turn-end` 判定点（`docs/internal/hook-step-context.md` §5.7）。
    ///
    /// turn **自愿停止**（LLM 说 EndTurn / 没要工具）前调用。让 hook 决定：放停，还是续命
    /// （注入反馈、不结束、回循环顶再转一轮）。
    ///
    /// 返回 `true` = 续命（调用方 `continue` 回循环顶）；`false` = 放停（正常结束 turn）。
    ///
    /// 续命受**硬上限** [`MAX_STOP_HOOK_CONTINUES`] 约束——达上限后强制放停，防死循环。续命的反馈
    /// 作为 **user 消息**注入 history（与用户 prompt 同一道工序，见设计 §4：末尾兜底交替）。
    pub(super) async fn decide_turn_end(&self, state: &mut TurnState) -> bool {
        // 达到续命硬上限：不再问 hook，强制放停。
        if !state.may_stop_hook_continue() {
            return false;
        }

        let mut step = crate::hooks::step::BeforeTurnEnd {
            stop_reason: AcpStopReason::EndTurn,
            continues_so_far: state.stop_hook_continues,
            voluntary: true,
            feedback: Vec::new(),
        };

        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        match control {
            crate::hooks::step::HookControl::Continue => {
                // 续命：把反馈作为 user 消息注入 history。空反馈也注入一条兜底提示，避免
                // LLM 下一轮立刻又说"我说完了"造成空转（设计 §3 不变量）。
                let blocks = if step.feedback.is_empty() {
                    vec![ContentBlock::from(
                        "Continue working — the stop condition is not yet satisfied.",
                    )]
                } else {
                    step.feedback
                };
                self.append_user_feedback(blocks);
                state.note_stop_hook_continue();
                true
            }
            // Proceed / Break / Skip 在 turn-end 都意味着放停。
            _ => false,
        }
    }

    /// 把一组 content block 作为 user 消息注入 history（续命反馈用）。
    ///
    /// 兜底角色交替（设计 §4）：history 末尾已是 user 角色时，并进同一条而非新增相邻 user，
    /// 防止两家 wire codec 撞上"连续同角色"。无法解码的 block 跳过（最佳努力，不杀 turn）。
    pub(super) fn append_user_feedback(&self, blocks: Vec<ContentBlock>) {
        let content: Vec<MessageContent> = blocks
            .into_iter()
            .filter_map(|b| content_block_to_message_content(b).ok())
            .flatten()
            .collect();
        if content.is_empty() {
            return;
        }
        self.history.append(Message {
            role: Role::User,
            content: content.into(),
        });
    }

    /// 触发 `UserPromptSubmit` hook。
    ///
    /// 处理三类 outcome：
    /// - `block` → 拒绝该 turn（调用方返回 `Refusal`）
    /// - `patch = UserPrompt { prepend, append }` → 改写 prompt 顺序为
    ///   `[prepend, original, append]`，落 history 时按改写后形态
    /// - `append` → 暂未拼到 system prompt（v0 无落点；待 system_prompt
    ///   动态拼接落地后填上，详见 `docs/internal/hooks.md` §3.2）
    pub(super) async fn fire_user_prompt_submit(&self, prompt: Vec<ContentBlock>) -> UserPromptHookFlow {
        // Step 模型：`before Ingest`（输入摄入前）。hook 可改写 input、或 `Break` 拒该 turn。
        // source 由 turn 携带——用户 turn=User，后台续转 turn=Background（§5.1）。
        let mut step = crate::hooks::step::BeforeIngest {
            source: self.ingest_source.clone(),
            input: prompt,
        };
        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;
        match control {
            crate::hooks::step::HookControl::Break { .. } => {
                tracing::info!("user prompt blocked by before-ingest hook");
                UserPromptHookFlow::Refused
            }
            // Proceed / Continue / Skip 在摄入点都意味着"继续"，带 hook 改写后的 input。
            _ => UserPromptHookFlow::Continue(step.input),
        }
    }

    /// 触发 `before ToolApply` hook（每工具）。
    pub(super) async fn fire_pre_tool_use(
        &self,
        id: &ToolCallId,
        name: &str,
        args: &JsonValue,
        safety: crate::tool::SafetyClass,
    ) -> PreToolHookFlow {
        let _ = id;
        // Step 模型：`before ToolApply`。hook 可改 args、填 result（拦工具=合成输出）、或 `Break`。
        let mut step = crate::hooks::step::BeforeToolApply {
            tool_name: name.to_string(),
            safety,
            args: args.clone(),
            result: None,
        };
        let control = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        // 填了 result = 拦掉这个工具（合成输出），turn 继续。映射到现有 Block 流（调用方据此
        // 不执行工具、用 reason 作为被拒文本喂回）。
        if let Some(result) = step.result {
            let reason = match &result.body {
                crate::llm::ToolResultBody::Text { text } => text.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| "blocked by hook".into()),
            };
            tracing::info!(tool = %name, "tool short-circuited by before-tool-apply hook");
            return PreToolHookFlow::Block(reason);
        }
        if let crate::hooks::step::HookControl::Break { .. } = control {
            tracing::info!(tool = %name, "tool blocked by before-tool-apply hook (break)");
            return PreToolHookFlow::Block("blocked by hook".to_string());
        }
        PreToolHookFlow::Continue { args: step.args }
    }

    /// 触发 `after ToolApply` hook（每工具）。把 hook 注入的 `additional_context`
    /// 拼到 `result.body` 末尾——下一轮 LLM 看到 hook 注释作为工具输出的一部分。
    pub(super) async fn fire_post_tool_hook(&self, result: &mut ToolResult) {
        // Step 模型：`after ToolApply`。观察 + 可注入（拼进 tool_result）。
        let mut step = crate::hooks::step::AfterToolApply {
            tool_name: result.name.clone(),
            is_error: result.is_error,
            output: result.body.clone(),
            additional_context: Vec::new(),
        };
        let _ = self.hooks.dispatch(&mut step, self.hook_ctx()).await;

        if step.additional_context.is_empty() {
            return;
        }

        // 把 hook 注入的 ContentBlock（仅取 Text 块）拼到 tool_result body。
        let extra: String = step
            .additional_context
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if extra.is_empty() {
            return;
        }
        match &mut result.body {
            ToolResultBody::Text { text } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&extra);
            }
            // 多模态结果：把追加文本作为新的文本块挂到末尾，不动图片块。
            ToolResultBody::Content { blocks } => {
                blocks.push(ToolResultContent::Text { text: extra });
            }
            ToolResultBody::Json { .. } => {}
        }
    }
}
