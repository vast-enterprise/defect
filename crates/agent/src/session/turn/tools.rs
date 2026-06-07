//! Permission decision and concurrent tool execution.
//!
//! Extracted from the turn main flow: `decide_permissions` / `emit_tool_failed` /
//! `run_tools_concurrently` are implemented as methods on [`super::TurnRunner`], along
//! with the
//! [`Approved`] / [`DecisionFlow`] / [`ToolResult`] types, single tool stream driving
//! [`drive_tool_stream`], and related helpers.

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
                self.emit_tool_failed(&id, &tu.name, reason.clone()).await;
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
                    self.emit_tool_failed(&id, &tu.name, reason.clone()).await;
                    approved.push(Approved::FailedArgs {
                        id: id.clone(),
                        tool_use_id: tu.id.clone(),
                        name: tu.name.clone(),
                        reason,
                    });
                    continue;
                }
            };

            // ② PreToolUse hook (sync interception)
            // Before policy evaluation — the hook can rewrite args or block directly, so
            // policy is not needed.
            // Let hooks process the tool result before it lands in history.
            let safety_hint_pre = tool.safety_hint(&args);
            match self
                .fire_pre_tool_use(&id, &tu.name, &args, safety_hint_pre)
                .await
            {
                PreToolHookFlow::Continue { args: new_args } => {
                    args = new_args;
                }
                PreToolHookFlow::Block(reason) => {
                    self.emit_tool_failed(&id, &tu.name, reason).await;
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
            )
            .with_current_provider(&self.config.provider);
            let description = tool.describe(&args, describe_ctx).await;
            // The main loop fills `raw_input` with the original `args` from outside (see
            // the comment in tool.rs: the tool itself does not set it). Without this,
            // neither the ACP wire `tool_call` nor the Langfuse span would have an input.
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
                    self.emit_tool_failed(&id, &tu.name, "denied by policy".to_string())
                        .await;
                    approved.push(Approved::Denied {
                        id: id.clone(),
                        tool_use_id: tu.id.clone(),
                        name: tu.name.clone(),
                    });
                }
                PolicyDecision::Ask(ask) => {
                    if ask.options.is_empty() {
                        // Empty options are equivalent to Deny.
                        self.emit_tool_failed(&id, &tu.name, "denied by policy".to_string())
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
                                self.emit_tool_failed(&id, &tu.name, "denied by user".to_string())
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

    async fn emit_tool_failed(&self, id: &ToolCallId, name: &str, text: String) {
        let fields = failed_fields_text(text);
        self.events
            .emit(AgentEvent::ToolCallStarted {
                id: id.clone(),
                name: name.to_owned(),
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

        // When `max_concurrent_tools == 0`, concurrency is unlimited (`None`, fast path,
        // never awaits a permit).
        // Otherwise all tool tasks share a single `Semaphore`: each task acquires a
        // permit before driving the tool stream and releases it when the task future
        // completes. This caps the number of concurrent tool executions in a single turn
        // when N `spawn_agent` calls are fanned out, preventing a spawn storm from
        // overwhelming the provider or resources.
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
                    let provider = self.config.provider.clone();
                    let background = self.background.clone();
                    let goal = self.goal.clone();
                    // Pass the current active policy to the tool — `spawn_agent` uses it
                    // to give the child agent the parent's actual policy (reflecting the
                    // session's current permission mode).
                    let policy = self.policy.clone();
                    let subagent_depth = self.config.subagent_max_depth;
                    let name = tool.schema().name.clone();
                    let span = tracing::info_span!(
                        "tool_call",
                        tool = %name,
                        tool_call_id = %id,
                    );
                    let semaphore = semaphore.clone();
                    joinset.spawn(
                        async move {
                            // Acquire a permit; it is held until this task's future
                            // completes (drive finishes) and is automatically returned.
                            // `acquire_owned` only returns `Err` when the `Semaphore` is
                            // closed — it is never closed here, so `unwrap` is safe.
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
                                provider,
                                background,
                                goal,
                                policy,
                                subagent_depth,
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

        // PostToolUse / PostToolUseFailure hook (sync interception).
        // Gives hooks a chance to append annotations before tool_result is written to
        // history.
        for result in results.iter_mut() {
            self.fire_post_tool_hook(result).await;
        }

        results
    }
}

// ----- hook helpers -----

impl<'a> TurnRunner<'a> {}

// ----- Types -----

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

/// The return value of `decide_permissions`: either continue with the approved list to
/// the execution phase, or the user cancelled the turn during the `Ask` phase.
pub(super) enum DecisionFlow {
    Continue(Vec<Approved>),
    Cancelled,
}

pub(super) struct ToolResult {
    /// ACP ID of the tool call. Not yet consumed by `after ToolApply` in the step model;
    /// kept for future events/auditing.
    #[allow(dead_code)]
    pub(super) id: ToolCallId,
    /// Tool name. Used by the matcher / envelope of the `after ToolApply` step.
    pub(super) name: String,
    pub(super) tool_use_id: String,
    pub(super) body: ToolResultBody,
    pub(super) is_error: bool,
    /// Final field. Previously consumed by the `PostToolUse` hook; unused in the step
    /// model, kept for future use.
    #[allow(dead_code)]
    pub(super) fields: Option<ToolCallUpdateFields>,
    /// Failure text. Previously consumed by the `PostToolUseFailure` hook; not yet used
    /// in the step model, kept for future use.
    #[allow(dead_code)]
    pub(super) error: Option<String>,
}

// ----- helpers -----

/// Extract the tool name from an [`Approved`] (for the after-Permission-hook envelope).
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

/// Collapse [`ToolCallUpdateFields::content`] into a [`ToolResultBody`] to feed back to
/// the LLM.
///
/// Rules: scan all `ToolCallContent::Content` blocks —
/// - Plain text (zero images): combine into a single [`ToolResultBody::Text`], matching
///   historical behavior,
///   and keeping `fire_post_tool_hook`'s text appending and OpenAI tool messages on the
///   simple path.
/// - If any image block appears: upgrade to [`ToolResultBody::Content`], interleaving
///   text and image blocks
///   in original order, and let the codec materialize them per provider.
///
/// `Diff` / `ResourceLink` and other non-text, non-image blocks are excluded from
/// tool_result (they are ACP
/// display content for the UI, not fed to the model); returns `None` if there is nothing
/// to feed back.
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
    // Plain text: join into a single `Text`.
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

/// Drives a single tool stream task. Forwards [`ToolEvent`] as [`AgentEvent`] and finally
/// produces a [`ToolResult`] to feed back to the LLM.
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
    provider: String,
    background: Option<crate::session::BackgroundTasks>,
    goal: Option<Arc<crate::session::GoalState>>,
    policy: Arc<dyn crate::policy::SandboxPolicy>,
    subagent_depth: u32,
) -> ToolResult {
    let mut ctx = ToolContext::new(
        &cwd,
        cancel.clone(),
        fs.clone(),
        shell.clone(),
        http.clone(),
        &model,
    )
    .with_current_provider(&provider)
    .with_policy(policy)
    // Remaining subagent dispatch depth for this turn — `spawn_agent` uses it to decide
    // whether child agents can continue recursing (0 ⇒ the child toolset contains no
    // `spawn_agent`).
    .with_subagent_depth(subagent_depth);
    if let Some(bg) = background {
        ctx = ctx.with_background(bg);
    }
    if let Some(goal) = goal {
        ctx = ctx.with_goal(goal);
    }
    // Inject a subagent event bridge so that `spawn_agent` can wrap child-turn events as
    // `AgentEvent::Subagent` and forward them back into this session's event stream,
    // nested under the current `tool_call_id`. This is a no-op for most tools—only
    // `spawn_agent` uses it. The bridge is injected for both top-level and nested
    // subagent turns (recursive bridging); each layer of `bridge_task` only prepends its
    // own hop's `tool_call_id`.
    ctx = ctx.with_subagent_bridge(crate::tool::SubagentBridge {
        parent_events: events.clone(),
        parent_tool_call_id: id.clone(),
    });
    let mut stream = tool.execute(args, ctx);

    let mut last_body: Option<ToolResultBody> = None;

    // Note: cancellation is injected into the tool via `ctx.cancel`; the tool itself
    // detects it and produces [`ToolEvent::Failed(ToolError::Canceled)`] — do not add a
    // cancel arm in the driver layer.
    // If the driver drops the stream inside a `select`, any in-flight oneshot::Receiver
    // for an ACP reverse request inside the tool will be dropped, causing the server to
    // map "no receiver" to an internal_error and tear down the entire connection (see the
    // `?` on `router.respond_with_result` in
    // `agent_client_protocol::jsonrpc::incoming_actor`).
    // Tool trait contract: must detect cancellation.
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
