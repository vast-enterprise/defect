//! `spawn_agent`：把任务委派给一个 subagent。
//!
//! subagent 在 **fresh、隔离的上下文**里跑一个嵌套 [`TurnRunner`]，只把最终
//! 那条 assistant 文本作为工具结果回给父 agent——父看不到子 agent 的中间过程。
//! 设计见记忆 `project-subagent-design`。
//!
//! ## 两道闸门
//!
//! - **闸门 A（看得到哪些工具）**：每个 profile 的 `tool_allow` 白名单从父进程
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

use agent_client_protocol::schema::{
    Content, ContentBlock, SessionId, TextContent, ToolCallContent, ToolCallUpdateFields, ToolKind,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::json;

use crate::error::BoxError;
use crate::hooks::NoopHookEngine;
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
#[derive(Debug, Clone)]
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
}

/// `spawn_agent` 工具。进程级共享（挂在 `StaticToolRegistry` 上），构造期捕获
/// 跑嵌套 turn 所需的一切——因为 [`ToolContext`] 只带 cwd/fs/shell/http/cancel/
/// current_model，不带 provider registry / policy / 工具集。
pub struct SpawnAgentTool {
    schema: ToolSchema,
    profiles: Arc<BTreeMap<String, SubagentProfile>>,
    registry: Arc<ProviderRegistry>,
    /// 父进程的 policy（进程级，所有 session 共享同一份）。子 turn 用
    /// [`NonInteractivePolicy`] 包它。
    policy: Arc<dyn SandboxPolicy>,
    /// 父进程工具集——按 profile 白名单裁子集的来源。
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
        let policy = self.policy.clone();
        let process_tools = self.process_tools.clone();
        let base_prompt = self.base_prompt.clone();

        let cwd = ctx.cwd.to_path_buf();
        let cancel = ctx.cancel.child_token();
        let fs = ctx.fs.clone();
        let shell = ctx.shell.clone();
        let http = ctx.http.clone();
        let parent_model = ctx.current_model.to_string();

        let fut = async move {
            run_subagent(
                args,
                profiles,
                registry,
                policy,
                process_tools,
                base_prompt,
                cwd,
                cancel,
                fs,
                shell,
                http,
                parent_model,
            )
            .await
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> =
            Box::pin(futures::stream::once(fut));
        s
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_subagent(
    args: serde_json::Value,
    profiles: Arc<BTreeMap<String, SubagentProfile>>,
    registry: Arc<ProviderRegistry>,
    policy: Arc<dyn SandboxPolicy>,
    process_tools: Arc<dyn ToolRegistry>,
    base_prompt: Option<String>,
    cwd: std::path::PathBuf,
    cancel: tokio_util::sync::CancellationToken,
    fs: Arc<dyn crate::fs::FsBackend>,
    shell: Arc<dyn crate::shell::ShellBackend>,
    http: Arc<dyn crate::http::HttpClient>,
    parent_model: String,
) -> ToolEvent {
    let parsed: SpawnArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    let Some(profile) = profiles.get(&parsed.profile) else {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(format!(
            "unknown profile `{}`; available: {}",
            parsed.profile,
            profiles.keys().cloned().collect::<Vec<_>>().join(", ")
        )))));
    };

    // model：profile 指定优先，否则回落到父会话当前选中的 model。
    let model = profile.model.clone().unwrap_or(parent_model);
    let Some(entry) = registry.entry_for_model(&model) else {
        return ToolEvent::Failed(ToolError::Execution(BoxError::new(io_err(format!(
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
                return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(format!(
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
    let system_prompt: Option<Arc<str>> = (!sections.is_empty())
        .then(|| Arc::from(sections.join("\n\n").as_str()));

    // 子 turn 的局部件——全在本 async block 内，跑完即弃。
    let history = VecHistory::new();
    let events = Arc::new(EventEmitter::new());
    let permissions = PermissionGate::new();
    let sub_policy = NonInteractivePolicy::new(policy);
    let hooks = NoopHookEngine;
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
        policy: &sub_policy,
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
        hooks: &hooks,
        session_id: &session_id,
        request_audit: &audit,
    };

    let prompt = vec![ContentBlock::Text(TextContent::new(parsed.task))];
    if let Err(err) = runner.run(prompt).await {
        return ToolEvent::Failed(ToolError::Execution(BoxError::new(io_err(format!(
            "subagent turn failed: {err}"
        )))));
    }

    // 取最后一条 assistant 消息的文本作为结果。
    let answer = last_assistant_text(&history.snapshot());

    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(answer)),
    ))]);
    ToolEvent::Completed(fields)
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
