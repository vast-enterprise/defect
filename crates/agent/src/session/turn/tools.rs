//! 权限决策与工具并发执行。
//!
//! 从 turn 主流程疏散出来：`decide_permissions` / `emit_tool_failed` /
//! `run_tools_concurrently` 作为 [`super::TurnRunner`] 的方法实现，加上 [`Approved`] /
//! [`DecisionFlow`] / [`ToolResult`] 类型、单个工具流驱动 [`drive_tool_stream`] 与相关 helper。

use std::sync::Arc;

use agent_client_protocol_schema::{
    Content as AcpContent, ContentBlock, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdateFields,
};
use futures::StreamExt;
use serde_json::Value as JsonValue;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use tracing::Instrument;

use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::http::HttpClient;
use crate::llm::{ImageData, Message, MessageContent, Role, ToolResultBody, ToolResultContent};
use crate::policy::{PolicyCtx, PolicyDecision, RecordedOutcome};
use crate::session::TurnError;
use crate::session::events::EventEmitter;
use crate::shell::ShellBackend;
use crate::tool::{Tool, ToolContext, ToolError, ToolEvent};

use super::TurnRunner;
use super::hooks::PreToolHookFlow;
use super::llm_drive::{ToolUseAccumulated, parse_args};

impl TurnRunner<'_> {
    pub(super) async fn decide_permissions(
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
                    id: id.clone(),
                    tool_use_id: tu.id.clone(),
                    name: tu.name.clone(),
                    reason,
                });
                continue;
            };

            let mut args: JsonValue = match parse_args(&tu.args_buf) {
                Ok(v) => v,
                Err(reason) => {
                    let reason = format!("invalid args: {reason}");
                    self.emit_tool_failed(&id, reason.clone()).await;
                    approved.push(Approved::FailedArgs {
                        id: id.clone(),
                        tool_use_id: tu.id.clone(),
                        name: tu.name.clone(),
                        reason,
                    });
                    continue;
                }
            };

            // ② PreToolUse hook（Sync 拦截）
            // 在 policy 之前——hook 可改写 args / 直接 block 让 policy 都不用算。
            // 详见 `docs/internal/hooks.md` §7.1 / §7.3。
            let safety_hint_pre = tool.safety_hint(&args);
            match self
                .fire_pre_tool_use(&id, &tu.name, &args, safety_hint_pre)
                .await
            {
                PreToolHookFlow::Continue { args: new_args } => {
                    args = new_args;
                }
                PreToolHookFlow::Block(reason) => {
                    self.emit_tool_failed(&id, reason).await;
                    approved.push(Approved::Denied {
                        id: id.clone(),
                        tool_use_id: tu.id.clone(),
                        name: tu.name.clone(),
                    });
                    continue;
                }
            }

            let describe_ctx = ToolContext::new(
                self.cwd,
                self.cancel.clone(),
                self.fs.clone(),
                self.shell.clone(),
                self.http.clone(),
                &self.config.model,
            );
            let description = tool.describe(&args, describe_ctx).await;
            // raw_input 由主循环在外层填充原始 args（见 tool.rs 注释：工具自己不塞）。
            // 不填则 ACP wire 上的 tool_call 与 langfuse span 都没有 input。
            let mut started_fields =
                with_status(description.fields.clone(), ToolCallStatus::Pending);
            if started_fields.raw_input.is_none() {
                started_fields.raw_input = Some(args.clone());
            }
            self.events
                .emit(AgentEvent::ToolCallStarted {
                    id: id.clone(),
                    name: tu.name.clone(),
                    fields: started_fields,
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
                        id: id.clone(),
                        tool_use_id: tu.id.clone(),
                        name: tu.name.clone(),
                    });
                }
                PolicyDecision::Ask(ask) => {
                    if ask.options.is_empty() {
                        // 空 options 等价 Deny（见 sandbox-policy.md §2）
                        self.emit_tool_failed(&id, "denied by policy".to_string())
                            .await;
                        approved.push(Approved::Denied {
                            id: id.clone(),
                            tool_use_id: tu.id.clone(),
                            name: tu.name.clone(),
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
                                    id: id.clone(),
                                    tool_use_id: tu.id.clone(),
                                    name: tu.name.clone(),
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

    pub(super) async fn run_tools_concurrently(&self, approved: Vec<Approved>) -> Vec<ToolResult> {
        let mut joinset: JoinSet<ToolResult> = JoinSet::new();
        let mut results: Vec<ToolResult> = Vec::with_capacity(approved.len());

        // `max_concurrent_tools == 0` ⇒ 不限并发（None，快路径，永不 await permit）。
        // 否则所有工具 task 共享一个 `Semaphore`：每个 task 在驱动工具流之前先抢一个
        // permit，跑完（task future 结束）即归还。这给同一 turn 内一次发出 N 个
        // `spawn_agent`（fanout）的场景一个上限，避免 spawn 风暴打爆 provider/资源。
        let semaphore = (self.config.max_concurrent_tools > 0).then(|| {
            Arc::new(tokio::sync::Semaphore::new(
                self.config.max_concurrent_tools,
            ))
        });

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
                    let http = self.http.clone();
                    let model = self.config.model.clone();
                    let background = self.background.clone();
                    let name = tool.schema().name.clone();
                    let span = tracing::info_span!(
                        "tool_call",
                        tool = %name,
                        tool_call_id = %id,
                    );
                    let semaphore = semaphore.clone();
                    joinset.spawn(
                        async move {
                            // 抢 permit；持有到本 task future 结束（drive 跑完）自动归还。
                            // `acquire_owned` 仅在 Semaphore 被 close 时返回 Err——本处
                            // 永不 close，故 unwrap 安全。
                            let _permit = match semaphore {
                                Some(sem) => {
                                    Some(sem.acquire_owned().await.expect("semaphore not closed"))
                                }
                                None => None,
                            };
                            drive_tool_stream(
                                id,
                                tool_use_id,
                                name,
                                tool,
                                args,
                                cwd,
                                cancel,
                                events,
                                fs,
                                shell,
                                http,
                                model,
                                background,
                            )
                            .await
                        }
                        .instrument(span),
                    );
                }
                Approved::Denied {
                    id,
                    tool_use_id,
                    name,
                } => {
                    results.push(ToolResult {
                        id,
                        name,
                        tool_use_id,
                        body: ToolResultBody::Text {
                            text: "denied".to_string(),
                        },
                        is_error: true,
                        fields: None,
                        error: Some("denied".to_string()),
                    });
                }
                Approved::FailedArgs {
                    id,
                    tool_use_id,
                    name,
                    reason,
                } => {
                    results.push(ToolResult {
                        id,
                        name,
                        tool_use_id,
                        body: ToolResultBody::Text {
                            text: reason.clone(),
                        },
                        is_error: true,
                        fields: None,
                        error: Some(reason),
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
                        id: ToolCallId::new(""),
                        name: String::new(),
                        tool_use_id: String::new(),
                        body: ToolResultBody::Text {
                            text: format!("tool task crashed: {join_err}"),
                        },
                        is_error: true,
                        fields: None,
                        error: Some(format!("tool task crashed: {join_err}")),
                    });
                }
            }
        }

        // ③/④ PostToolUse / PostToolUseFailure hook（Sync 拦截）
        // 在 tool_result 落 history 之前给 hook 追加注释的机会。详见
        // `docs/internal/hooks.md` §3.2 / §7.1。
        for result in results.iter_mut() {
            self.fire_post_tool_hook(result).await;
        }

        results
    }
}

// ----- hook helpers -----

impl<'a> TurnRunner<'a> {}

// ----- 类型 -----

pub(super) enum Approved {
    Run {
        id: ToolCallId,
        tool_use_id: String,
        tool: Arc<dyn Tool>,
        args: JsonValue,
    },
    Denied {
        id: ToolCallId,
        tool_use_id: String,
        name: String,
    },
    FailedArgs {
        id: ToolCallId,
        tool_use_id: String,
        name: String,
        reason: String,
    },
}

/// `decide_permissions` 的返回：要么继续把 approved 列表交给执行阶段，
/// 要么用户在 `Ask` 阶段取消了 turn。
pub(super) enum DecisionFlow {
    Continue(Vec<Approved>),
    Cancelled,
}

pub(super) struct ToolResult {
    /// 工具调用的 ACP id。step 模型下 `after ToolApply` 暂未消费，保留供未来事件/审计用。
    #[allow(dead_code)]
    pub(super) id: ToolCallId,
    /// 工具名。`after ToolApply` step 的 matcher / 信封要用。
    pub(super) name: String,
    pub(super) tool_use_id: String,
    pub(super) body: ToolResultBody,
    pub(super) is_error: bool,
    /// 终态字段。旧 `PostToolUse` hook 曾消费；step 模型下暂未用，保留供未来用。
    #[allow(dead_code)]
    pub(super) fields: Option<ToolCallUpdateFields>,
    /// 失败文本。旧 `PostToolUseFailure` hook 曾消费；step 模型下暂未用，保留供未来用。
    #[allow(dead_code)]
    pub(super) error: Option<String>,
}

// ----- helpers -----

/// 从一个 [`Approved`] 取工具名（after Permission hook 信封用）。
pub(super) fn approved_tool_name(a: &Approved) -> String {
    match a {
        Approved::Run { tool, .. } => tool.schema().name.clone(),
        Approved::Denied { name, .. } | Approved::FailedArgs { name, .. } => name.clone(),
    }
}

pub(super) fn tool_results_message(results: Vec<ToolResult>) -> Message {
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

/// 把 [`ToolCallUpdateFields::content`] 收成喂回 LLM 的 [`ToolResultBody`]。
///
/// 规则：扫描所有 `ToolCallContent::Content` 块——
/// - 纯文本（含零图片）：拼成单条 [`ToolResultBody::Text`]，与历史行为一致，
///   也让 `fire_post_tool_hook` 的文本追加、OpenAI tool message 走简单路径
/// - 一旦出现图片块：升级为 [`ToolResultBody::Content`]，文本与图片块按原序
///   混排，交给 codec 按 provider 物化
///
/// `Diff` / `ResourceLink` 等非文本非图片块不进 tool_result（它们是给 UI 看的
/// ACP 展示内容，不喂模型）；返回 `None` 表示没有可喂回的内容。
fn extract_body(fields: &ToolCallUpdateFields) -> Option<ToolResultBody> {
    let raw = fields.content.as_ref()?;
    let mut blocks: Vec<ToolResultContent> = Vec::new();
    let mut has_image = false;
    for c in raw {
        let ToolCallContent::Content(inner) = c else {
            continue;
        };
        match &inner.content {
            ContentBlock::Text(t) => blocks.push(ToolResultContent::Text {
                text: t.text.clone(),
            }),
            ContentBlock::Image(img) => {
                has_image = true;
                blocks.push(ToolResultContent::Image {
                    mime: img.mime_type.clone(),
                    data: ImageData::Base64 {
                        encoded: img.data.clone(),
                    },
                });
            }
            _ => {}
        }
    }
    if blocks.is_empty() {
        return None;
    }
    if has_image {
        return Some(ToolResultBody::Content { blocks });
    }
    // 纯文本：拼成单条 Text。
    let text = blocks
        .into_iter()
        .filter_map(|b| match b {
            ToolResultContent::Text { text } => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    Some(ToolResultBody::Text { text })
}

/// 单个工具流的驱动 task。把 [`ToolEvent`] 转发为 [`AgentEvent`]，最后产出
/// [`ToolResult`] 喂回 LLM。
#[allow(clippy::too_many_arguments)]
async fn drive_tool_stream(
    id: ToolCallId,
    tool_use_id: String,
    name: String,
    tool: Arc<dyn Tool>,
    args: JsonValue,
    cwd: std::path::PathBuf,
    cancel: CancellationToken,
    events: Arc<EventEmitter>,
    fs: Arc<dyn FsBackend>,
    shell: Arc<dyn ShellBackend>,
    http: Arc<dyn HttpClient>,
    model: String,
    background: Option<crate::session::BackgroundTasks>,
) -> ToolResult {
    let mut ctx = ToolContext::new(
        &cwd,
        cancel.clone(),
        fs.clone(),
        shell.clone(),
        http.clone(),
        &model,
    );
    if let Some(bg) = background {
        ctx = ctx.with_background(bg);
    }
    // 注入 subagent 事件桥：让 `spawn_agent` 能把子 turn 事件包成
    // AgentEvent::Subagent 转发回本 session 的事件流（按本次 tool_call_id 嵌套）。
    // 对绝大多数工具是惰性的——只有 spawn_agent 会用到。
    ctx = ctx.with_subagent_bridge(crate::tool::SubagentBridge {
        parent_events: events.clone(),
        parent_tool_call_id: id.clone(),
    });
    let mut stream = tool.execute(args, ctx);

    let mut last_body: Option<ToolResultBody> = None;

    // 注意：cancel 通过 ctx.cancel 注入工具内部，由工具自己感知并产出
    // [`ToolEvent::Failed(ToolError::Canceled)`]——不要在驱动层加 cancel arm。
    // 一旦驱动层 select 里 drop 掉 stream，工具内部任何在飞的 ACP 反向请求
    // 的 oneshot::Receiver 都会被 drop，server 把"无人接收"映射成 internal_error
    // 并撕掉整条连接（详见 `agent_client_protocol::jsonrpc::incoming_actor`
    // 里 `router.respond_with_result` 的 ?）。Tool trait 契约：必须感知 cancel。
    while let Some(ev) = stream.next().await {
        match ev {
            ToolEvent::Progress(fields) => {
                if let Some(body) = extract_body(&fields) {
                    last_body = Some(body);
                }
                events
                    .emit(AgentEvent::ToolCallProgress {
                        id: id.clone(),
                        fields: with_status(fields, ToolCallStatus::InProgress),
                    })
                    .await;
            }
            ToolEvent::Completed(fields) => {
                if let Some(body) = extract_body(&fields) {
                    last_body = Some(body);
                }
                let fields = with_status(fields, ToolCallStatus::Completed);
                events
                    .emit(AgentEvent::ToolCallFinished {
                        id: id.clone(),
                        fields: fields.clone(),
                    })
                    .await;
                return ToolResult {
                    id,
                    name,
                    tool_use_id,
                    body: last_body.unwrap_or(ToolResultBody::Text {
                        text: String::new(),
                    }),
                    is_error: false,
                    fields: Some(fields),
                    error: None,
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
                    id,
                    name,
                    tool_use_id,
                    body: ToolResultBody::Text { text: text.clone() },
                    is_error: !is_cancel,
                    fields: None,
                    error: Some(text),
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
    let text = "tool stream closed without terminal event".to_string();
    ToolResult {
        id,
        name,
        tool_use_id,
        body: ToolResultBody::Text { text: text.clone() },
        is_error: true,
        fields: None,
        error: Some(text),
    }
}
