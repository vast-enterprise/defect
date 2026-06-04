//! `spawn_agent`：把任务委派给一个 subagent。
//!
//! subagent 在 **fresh、隔离的上下文**里跑一个嵌套 [`TurnRunner`]，只把最终
//! 那条 assistant 文本作为工具结果回给父 agent——父看不到子 agent 的中间过程。
//! 设计见记忆 `project-subagent-design`。
//!
//! ## 两道闸门
//!
//! - **闸门 A（看得到哪些工具）**：每个 profile 的 `tool_allow` 白名单从父 agent
//!   工具集里裁子集；白名单**永不含 `spawn_agent` 自己**，故结构性禁止递归。
//! - **闸门 B（运行时放行到什么程度）**：子 turn 的 policy 是
//!   [`NonInteractivePolicy`] 包住父 policy——`Ask` 降级为 `Deny`，子 agent
//!   非交互、永不阻塞在 [`PermissionGate`] 上、授权恒 ≤ 父。
//!
//! ## 继承原则
//!
//! 继承"够得着世界的能力"（provider registry / fs / shell / http），不继承
//! "身份与行为"（父的 system prompt / hooks / 任务框架）。子 agent 的 system
//! prompt = 继承的 base_prompt + profile 自己的 `system.md`，**不**走
//! [`resolve_system_prompt`]（那会去爬工作区 `AGENTS.md`，那是父的身份）。

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{
    Content, ContentBlock, SessionId, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::error::BoxError;
use crate::event::AgentEvent;
use crate::hooks::{HookEngine, NoopHookEngine};
use crate::llm::{HostedCapabilities, MessageContent, ProviderRegistry, Role, SamplingParams};
use crate::policy::{NonInteractivePolicy, SandboxPolicy};
use crate::session::{
    EventEmitter, History, PermissionGate, RequestAuditTracker, StaticToolRegistry, ToolRegistry,
    TurnConfig, TurnRequestLimit, TurnRunner, VecHistory,
};
use crate::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};

/// `spawn_agent` 工具的名字。用常量供"裁工具集时排除自己"复用，杜绝拼错。
pub(crate) const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";

/// 一个可被 `spawn_agent` 调用的 subagent profile（agent 侧表示）。
///
/// `defect-config` 的 `ProfileSpec` 是配置侧的真相源；CLI 装配时把它投影成
/// 本结构再交给工具。两边分开是因为 `defect-config` 依赖 `defect-agent`——
/// agent 不能反向依赖 config，否则成环。
#[derive(Clone)]
pub struct SubagentProfile {
    /// 选择期描述，进工具 schema 的 catalog，让 LLM 据此挑 profile。
    pub description: String,
    /// 可选 model 覆盖；`None` ⇒ 回落到父会话当前选中的 model（`ctx.current_model`）。
    pub model: Option<String>,
    /// 该 profile 的 system prompt 全文。
    pub system_prompt: String,
    /// 工具白名单——子 agent 只看得到这些工具（`spawn_agent` 永远被排除）。
    pub tool_allow: Vec<String>,
    /// 可选采样覆盖。
    pub sampling: Option<SamplingParams>,
    /// 该 profile 自己的 hook 引擎——子 agent 跑 turn 时挂的钩子。
    ///
    /// 与"继承世界、不继承身份"原则一致：hook 属于 profile 的身份，由 profile
    /// 自己的配置声明（CLI 装配期把 `ProfileSpec.hooks` 编译成引擎注入），**不**
    /// 从父会话继承。`None` ⇒ 子 agent 无钩子（回落 [`NoopHookEngine`]）——保持
    /// 与改动前完全一致的行为，故现有不挂钩子的 profile 零影响。
    pub hooks: Option<Arc<dyn HookEngine>>,
}

// `Arc<dyn HookEngine>` 不是 `Debug`，手写 `Debug` 跳过它（只标注是否挂了引擎）。
impl std::fmt::Debug for SubagentProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentProfile")
            .field("description", &self.description)
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .field("tool_allow", &self.tool_allow)
            .field("sampling", &self.sampling)
            .field("hooks", &self.hooks.as_ref().map(|_| "<engine>"))
            .finish()
    }
}

/// `spawn_agent` 工具。挂在 `StaticToolRegistry` 上、随 `process_tools` 被所属
/// `AgentCore` 的各 session 共享一份（**不是**进程全局单例——一个进程可装配多个
/// `AgentCore`，各持自己的一份）。构造期捕获跑嵌套 turn 所需的一切——因为
/// [`ToolContext`] 只带 cwd/fs/shell/http/cancel/current_model，不带 provider
/// registry / policy / 工具集。
pub struct SpawnAgentTool {
    schema: ToolSchema,
    profiles: Arc<BTreeMap<String, SubagentProfile>>,
    registry: Arc<ProviderRegistry>,
    /// 父 agent 的 policy（本 core 的所有 session 共享同一份）。子 turn 用
    /// [`NonInteractivePolicy`] 包它。
    policy: Arc<dyn SandboxPolicy>,
    /// 父 agent 工具集——按 profile 白名单裁子集的来源。
    process_tools: Arc<dyn ToolRegistry>,
    /// 继承给子 agent 的 base_prompt 文本（"你是个会用工具的 agent"那段底座）。
    base_prompt: Option<String>,
}

impl SpawnAgentTool {
    /// 构造一个 `spawn_agent` 工具。`profiles` 为空时调用方**不应**注册本工具
    /// （schema 的 `profile` enum 会是空集，永远调用失败）——见
    /// [`Self::has_profiles`]。
    pub fn new(
        profiles: Arc<BTreeMap<String, SubagentProfile>>,
        registry: Arc<ProviderRegistry>,
        policy: Arc<dyn SandboxPolicy>,
        process_tools: Arc<dyn ToolRegistry>,
        base_prompt: Option<String>,
    ) -> Self {
        let schema = build_schema(&profiles);
        Self {
            schema,
            profiles,
            registry,
            policy,
            process_tools,
            base_prompt,
        }
    }

    /// 是否发现到任何 profile。装配方据此决定是否注册本工具。
    pub fn has_profiles(profiles: &BTreeMap<String, SubagentProfile>) -> bool {
        !profiles.is_empty()
    }
}

/// 动态构造 schema：`profile` 是发现到的 profile 名的 enum（硬约束），工具
/// description 内嵌 `- <name>: <description>` 的 catalog（软引导）。两者缺一
/// 不可：光 enum 模型不知用途，光 catalog 模型可能填错名。
fn build_schema(profiles: &BTreeMap<String, SubagentProfile>) -> ToolSchema {
    let names: Vec<&str> = profiles.keys().map(String::as_str).collect();
    let catalog = profiles
        .iter()
        .map(|(name, p)| format!("- {name}: {}", p.description))
        .collect::<Vec<_>>()
        .join("\n");
    let description = format!(
        "Delegate a task to a specialized subagent that runs in a fresh, isolated context. \
         The subagent returns only its final summary, not its intermediate work. \
         Pick the profile whose description best matches the task.\n\n\
         When you have multiple independent pieces of work, emit several `spawn_agent` \
         calls in a single message: they run concurrently (fanout), so the total wait is \
         the slowest subagent rather than their sum. Only spawn one at a time when a later \
         task genuinely depends on an earlier subagent's result.\n\n\
         Available profiles:\n{catalog}"
    );
    ToolSchema {
        name: SPAWN_AGENT_TOOL_NAME.to_string(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": names,
                    "description": "Which subagent to spawn. See the tool description for what each profile does."
                },
                "task": {
                    "type": "string",
                    "description": "The complete task for the subagent, as a self-contained \
                                    natural-language instruction. The subagent has none of this \
                                    conversation's context — include everything it needs."
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override for this subagent. When omitted, \
                                    the profile's configured model is used, falling back to the \
                                    parent session's current model. Only set this when a task \
                                    needs a specifically more or less capable model than the default."
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "When true, spawn the subagent asynchronously and return \
                                    immediately with a task id, without waiting for it to finish. \
                                    The subagent's result is delivered back to you later, on a \
                                    subsequent turn, so you can keep working in the meantime. \
                                    Leave false (the default) when the next step depends on this \
                                    subagent's result — then the call blocks until it completes."
                }
            },
            "required": ["profile", "task"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct SpawnArgs {
    profile: String,
    task: String,
    /// 可选 per-call model 覆盖。优先级最高（高于 profile.model 与父 model）。
    #[serde(default)]
    model: Option<String>,
    /// 后台执行开关。`true` 且上下文支持（`ToolContext::background` 为 `Some`）时，
    /// spawn 后立即返回任务 id，不等子 agent 跑完。默认 `false`（同步阻塞）。
    #[serde(default)]
    run_in_background: bool,
}

impl Tool for SpawnAgentTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // 保守标 Mutating：spawn 本身的"危险"由子 agent 的工具集（闸门 A）
        // 与 NonInteractivePolicy（闸门 B）决定，不在这一层细分。
        SafetyClass::Mutating
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let profile = args.get("profile").and_then(|v| v.as_str()).unwrap_or("?");
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(format!("Spawn subagent `{profile}`"));
            fields.kind = Some(ToolKind::Think);
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        // 把构造期捕获的依赖与 ctx 里的运行时句柄都搬进 'static future——
        // 嵌套 TurnRunner 的所有借用都活在这个 async block 内，不外逃。
        let profiles = self.profiles.clone();
        let registry = self.registry.clone();
        // 优先用本 turn 快照的 active policy（ctx 注入）——它反映 session 当前
        // permission mode；缺省（测试 / 未注入）才回退构造期捕获的 policy。
        let policy = ctx.policy.clone().unwrap_or_else(|| self.policy.clone());
        let process_tools = self.process_tools.clone();
        let base_prompt = self.base_prompt.clone();

        let cwd = ctx.cwd.to_path_buf();
        let fs = ctx.fs.clone();
        let shell = ctx.shell.clone();
        let http = ctx.http.clone();
        let parent_model = ctx.current_model.to_string();
        let background = ctx.background.clone();
        // subagent 事件桥：把子 turn 事件嵌套回父 trace（observability）。
        let bridge = ctx.subagent_bridge.clone();
        // 同步路径用 turn 子 token（turn 结束即取消）；后台路径不用它，改用
        // BackgroundTasks 在 spawn 时 mint 的 session 级子 token（见下）。
        let turn_cancel = ctx.cancel.child_token();

        // 先解析出 run_in_background 与 profile 名，决定走同步还是后台。解析失败
        // 在两条路径里都按 InvalidArgs 处理。
        let parsed: Result<SpawnArgs, _> = serde_json::from_value(args.clone());

        let fut = async move {
            let parsed = match parsed {
                Ok(p) => p,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };

            // 后台路径：需要 ctx 支持后台（顶层 turn 才注入），且 run_in_background=true。
            if parsed.run_in_background {
                let Some(bg) = background else {
                    // 上下文不支持后台（子 agent 嵌套 / 测试）——fail loud，不静默降级成同步，
                    // 否则模型以为是后台、实际阻塞，行为与声明不符。
                    return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(
                        "run_in_background is not available in this context (nested subagents \
                         cannot spawn background tasks)"
                            .to_string(),
                    ))));
                };
                let label = parsed.profile.clone();
                let deps = SubagentDeps {
                    profiles,
                    registry,
                    policy,
                    process_tools,
                    base_prompt,
                    cwd,
                    fs,
                    shell,
                    http,
                    parent_model,
                    // 后台路径**也桥接**——与前台同一套 `AgentEvent::Subagent` 机制。
                    // 发起它的 spawn_agent tool span 会先正常 close（下方"已启动"的
                    // ToolCallFinished），随后子 turn 事件作为一个**相邻**的 subagent span
                    // 挂在同一 parent_tool_call_id 锚点下、自行张开到子 turn 真正结束。
                    // projector 据"tool span 是否还在表里"天然区分前台(嵌套)/后台(相邻)。
                    // bridge 的 parent_events 是 session 级 EventEmitter，后台任务跑时仍活着。
                    bridge,
                };
                // spawn 给任务 mint 一个 session 级子 token——任务的取消生命周期独立于
                // 发起它的 turn，turn 结束不会杀掉它。
                let label_for_log = parsed.profile.clone();
                let task_id = bg.spawn(label, move |task_cancel| async move {
                    match run_subagent_core(parsed, deps, task_cancel).await {
                        Ok(answer) => crate::session::BackgroundResult::Completed(answer),
                        Err(err) => {
                            // fail loud：后台失败此前只被数据化成 Failed 字符串静默流走，
                            // 既不上 langfuse 也无日志。这里补一条 warn，带 task / 错误。
                            tracing::warn!(
                                profile = %label_for_log,
                                error = %err,
                                "background subagent failed"
                            );
                            crate::session::BackgroundResult::Failed(err.to_string())
                        }
                    }
                });
                // 当场同步返回"已启动 id=X"——满足 tool_use↔tool_result 配对契约
                // （docs/proposals/task-arrange.md §2.1）。
                let msg = format!(
                    "Started background subagent `{}`, task id `{}`. Its result will arrive on a \
                     later turn.",
                    parsed_profile_for_msg(&args),
                    task_id
                );
                let mut fields = ToolCallUpdateFields::default();
                fields.content = Some(vec![ToolCallContent::Content(Content::new(
                    ContentBlock::Text(TextContent::new(msg.clone())),
                ))]);
                fields.raw_output = Some(serde_json::Value::String(msg));
                return ToolEvent::Completed(fields);
            }

            // 同步路径：原行为——阻塞到子 turn 跑完，把最终文本作为结果。
            let deps = SubagentDeps {
                profiles,
                registry,
                policy,
                process_tools,
                base_prompt,
                cwd,
                fs,
                shell,
                http,
                parent_model,
                // 同步路径：父 spawn_agent tool span 全程张开（阻塞等子 turn），
                // 子事件可嵌套其下。
                bridge,
            };
            match run_subagent_core(parsed, deps, turn_cancel).await {
                Ok(answer) => {
                    let mut fields = ToolCallUpdateFields::default();
                    fields.content = Some(vec![ToolCallContent::Content(Content::new(
                        ContentBlock::Text(TextContent::new(answer.clone())),
                    ))]);
                    fields.raw_output = Some(serde_json::Value::String(answer));
                    ToolEvent::Completed(fields)
                }
                Err(err) => ToolEvent::Failed(err),
            }
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

/// `run_subagent_core` 的依赖打包——避免十几个位置参数。构造期 + ctx 句柄都搬进来，
/// 全 owned，可跨 await / 进后台 task。
struct SubagentDeps {
    profiles: Arc<BTreeMap<String, SubagentProfile>>,
    registry: Arc<ProviderRegistry>,
    policy: Arc<dyn SandboxPolicy>,
    process_tools: Arc<dyn ToolRegistry>,
    base_prompt: Option<String>,
    cwd: std::path::PathBuf,
    fs: Arc<dyn crate::fs::FsBackend>,
    shell: Arc<dyn crate::shell::ShellBackend>,
    http: Arc<dyn crate::http::HttpClient>,
    parent_model: String,
    /// subagent 事件桥：`Some` 时把子 turn 事件嵌套回父 trace。仅同步路径设置。
    bridge: Option<crate::tool::SubagentBridge>,
}

/// 从原始 args 里尽力取 profile 名（仅供后台启动确认消息用，失败回退占位符）。
fn parsed_profile_for_msg(args: &serde_json::Value) -> String {
    args.get("profile")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string()
}

/// 跑一个子 agent turn，返回最终文本（`Ok`）或错误描述（`Err`）。
///
/// 同步与后台两条路径共用本核心：同步路径把 `Ok/Err` 包成 `ToolEvent::Completed/Failed`，
/// 后台路径包成 `BackgroundResult::Completed/Failed`。`cancel` 由调用方决定生命周期——
/// 同步路径传 turn 子 token，后台路径传 session 级子 token。
async fn run_subagent_core(
    parsed: SpawnArgs,
    deps: SubagentDeps,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<String, ToolError> {
    let SubagentDeps {
        profiles,
        registry,
        policy,
        process_tools,
        base_prompt,
        cwd,
        fs,
        shell,
        http,
        parent_model,
        bridge,
    } = deps;

    let Some(profile) = profiles.get(&parsed.profile) else {
        return Err(ToolError::InvalidArgs(BoxError::new(io_err(format!(
            "unknown profile `{}`; available: {}",
            parsed.profile,
            profiles.keys().cloned().collect::<Vec<_>>().join(", ")
        )))));
    };

    // model 优先级：本次调用入参 > profile 指定 > 父会话当前选中的 model。
    let model = parsed
        .model
        .clone()
        .or_else(|| profile.model.clone())
        .unwrap_or(parent_model);
    let Some(entry) = registry.entry_for_model(&model) else {
        return Err(ToolError::Execution(BoxError::new(io_err(format!(
            "subagent model `{model}` is not declared by any provider entry"
        )))));
    };
    let provider = entry.provider().clone();

    // 闸门 A：按白名单从父工具集裁子集；排除 spawn_agent 自己（禁递归）。
    // 未知工具名 hard fail（fail loud，不静默忽略）。
    let mut builder = StaticToolRegistry::builder();
    for name in &profile.tool_allow {
        if name == SPAWN_AGENT_TOOL_NAME {
            // 即便 profile 误写也忽略——结构性保证不递归。
            continue;
        }
        match process_tools.get(name) {
            Some(tool) => builder = builder.insert(tool),
            None => {
                return Err(ToolError::InvalidArgs(BoxError::new(io_err(format!(
                    "profile `{}` allows unknown tool `{name}`",
                    parsed.profile
                )))));
            }
        }
    }
    let sub_tools = builder.build();

    // system prompt：继承的 base_prompt + profile 自己的 system.md。不走
    // resolve_system_prompt（避免爬工作区 AGENTS.md / provider·model overlay）。
    let mut sections = Vec::new();
    if let Some(bp) = base_prompt.as_deref()
        && !bp.is_empty()
    {
        sections.push(bp.to_string());
    }
    if !profile.system_prompt.is_empty() {
        sections.push(profile.system_prompt.clone());
    }
    let system_prompt: Option<Arc<str>> =
        (!sections.is_empty()).then(|| Arc::from(sections.join("\n\n").as_str()));

    // 子 turn 的局部件——全在本 async block 内，跑完即弃。
    let history = VecHistory::new();
    let events = Arc::new(EventEmitter::new());

    // observability 桥：把子 turn 的每个事件包成 AgentEvent::Subagent 转发回父
    // session 的事件流，让 langfuse 把子 turn 嵌套到父 spawn_agent tool span 下。
    // 仅 observability——隔离契约对 storage / wire / REPL 不变（它们忽略 Subagent）。
    // 桥接 task 订阅子 emitter；子 turn 跑完、本函数返回 drop 掉 `events`（最后一个
    // 强引用）后，子流结束，task 自然退出，无需显式 join。
    let bridge_task = bridge.map(|b| {
        let mut sub_events = events.subscribe();
        let agent_type = parsed.profile.clone();
        tokio::spawn(async move {
            while let Some(inner) = sub_events.next().await {
                b.parent_events
                    .emit(AgentEvent::Subagent {
                        parent_tool_call_id: b.parent_tool_call_id.clone(),
                        agent_type: agent_type.clone(),
                        inner: Box::new(inner),
                    })
                    .await;
            }
        })
    });

    let permissions = PermissionGate::new();
    let sub_policy: Arc<dyn SandboxPolicy> = Arc::new(NonInteractivePolicy::new(policy));
    // profile 自己声明的 hook 引擎；未声明 ⇒ NoopHookEngine（行为同改动前）。
    let noop = NoopHookEngine;
    let hooks: &dyn HookEngine = match &profile.hooks {
        Some(engine) => engine.as_ref(),
        None => &noop,
    };
    let session_id = SessionId::new(format!("subagent-{}", parsed.profile));
    let audit = RequestAuditTracker::new();

    let config = TurnConfig {
        model: model.clone(),
        sampling: profile.sampling.clone().unwrap_or_default(),
        // 子 agent 给个有限步数上限——防失控嵌套循环。
        request_limit: TurnRequestLimit::Fixed(32),
        ..TurnConfig::default()
    };

    let runner = TurnRunner {
        history: &history,
        tools: &sub_tools,
        provider: provider.as_ref(),
        policy: sub_policy,
        events: events.clone(),
        permissions: &permissions,
        cancel: cancel.clone(),
        config: &config,
        system_prompt,
        cwd: &cwd,
        fs,
        shell,
        http,
        hosted_capabilities: HostedCapabilities::default(),
        hooks,
        session_id: &session_id,
        request_audit: &audit,
        // 子 agent turn 不携后台句柄：结构性禁止后台任务自我繁殖
        // （与"白名单永不含 spawn_agent 自己"同一道防递归思路）。
        background: None,
        // 子 agent turn 不做后台压缩：上下文短、生命周期随工具调用结束，无须跨
        // turn 的后台摘要。仍享 hard 水位的同步压缩兜底（compact_hard 路径要求
        // provider_arc）——故给它 provider_arc，其余后台压缩件留空。
        compaction_slot: None,
        history_arc: None,
        provider_arc: Some(provider.clone()),
        session_cancel: None,
        // 子 agent 的 task 是它的"用户输入"。
        ingest_source: crate::hooks::step::IngestSource::User,
    };

    let prompt = vec![ContentBlock::Text(TextContent::new(parsed.task))];
    let run_result = runner.run(prompt).await;

    // 子 turn 结束：drop 掉 runner 与本地 `events` 强引用，让子事件流走向关闭，
    // 桥接 task 把缓冲里剩余事件冲刷给父 emitter 后退出。await 它确保所有子事件
    // 在父 spawn_agent tool span 收尾（本函数返回 → ToolCallFinished）之前到达。
    drop(runner);
    drop(events);
    if let Some(task) = bridge_task {
        let _ = task.await;
    }

    if let Err(err) = run_result {
        return Err(ToolError::Execution(BoxError::new(io_err(format!(
            "subagent turn failed: {err}"
        )))));
    }

    // 取最后一条 assistant 消息的文本作为结果。
    Ok(last_assistant_text(&history.snapshot()))
}

/// 从历史里取**最后一条** [`Role::Assistant`] 消息，拼接其所有 `Text` 片段
/// （跳过 thinking / tool_use）。tool-use 循环会 append 多条 assistant 消息，
/// 取最后一条对应"最终回答"。
fn last_assistant_text(history: &[crate::llm::Message]) -> String {
    history
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|c| match c {
                    MessageContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn io_err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

#[cfg(test)]
#[path = "spawn_agent/test.rs"]
mod test;
