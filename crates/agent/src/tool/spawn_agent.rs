//! `spawn_agent`: delegates a task to a subagent.
//!
//! The subagent runs a nested [`TurnRunner`] in a **fresh, isolated context**, and only
//! the final assistant text is returned as the tool result to the parent agent — the
//! parent never sees the subagent's intermediate steps. See the design memo
//! `project-subagent-design`.
//!
//! ## Two Gates
//!
//! - **Gate A (which tools are visible)**: each profile's `tool_allow` whitelist is a
//!   subset of the parent agent's tool set. `spawn_agent` **may** be in the whitelist —
//!   recursion is controlled by the **depth gate** (see below), not unconditionally
//!   excluded.
//! - **Gate B (how much is allowed at runtime)**: the child turn's policy is
//!   [`NonInteractivePolicy`] wrapping the parent policy — `Ask` is downgraded to `Deny`,
//!   the child agent is non-interactive, never blocks on [`PermissionGate`], and its
//!   authorization is always ≤ the parent's.
//!
//! ## Recursion and the Depth Gate
//!
//! A subagent is simply "an agent with a parent" — parent and child run the same
//! [`TurnRunner`]. Recursion depth is controlled by
//! [`crate::tool::ToolContext::subagent_depth`]: the top-level turn injects a configured
//! maximum (`TurnConfig::subagent_max_depth`), decremented by one for each level. If a
//! level's `tool_allow` contains `spawn_agent` **and the remaining child depth > 0**, a
//! freshly constructed `spawn_agent` tool is installed for the child agent (capturing the
//! same base tool set as the subset source, so grandchildren can continue); when depth is
//! exhausted (0), the tool is not installed — a structural cutoff. A turn with `depth ==
//! 0` has no `spawn_agent` in its tool set; calling it fails loudly.
//!
//! ## Inheritance Principle
//!
//! Inherit "ability to reach the world" (provider registry / fs / shell / http), but
//! **not** "identity and behavior" (parent's system prompt / hooks / task framework). The
//! child agent's system prompt = inherited base_prompt + the profile's own `system.md`,
//! and does **not** go through
//! [`resolve_system_prompt`](crate::session::resolve_system_prompt) (which would crawl
//! the workspace `AGENTS.md` — that is the parent's identity).

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

/// The name of the `spawn_agent` tool. A constant so it can be reused when pruning the
/// tool set to exclude itself, preventing typos.
pub(crate) const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";

/// A subagent profile that can be invoked by `spawn_agent` (agent-side representation).
///
/// `ProfileSpec` in `defect-config` is the source of truth on the config side; the CLI
/// projects it into this struct during assembly before handing it to the tool. The two
/// are kept separate because `defect-config` depends on `defect-agent` — the agent cannot
/// depend on config in the opposite direction, or a cycle would result.
#[derive(Clone)]
pub struct SubagentProfile {
    /// Selection-time description that goes into the tool schema's catalog, allowing the
    /// LLM to choose a profile based on it.
    pub description: String,
    /// Optional model override; `None` falls back to the parent session's currently
    /// selected model (`ctx.current_model`).
    pub model: Option<String>,
    /// The full system prompt for this profile.
    pub system_prompt: String,
    /// Tool allowlist — the child agent can only see these tools (`spawn_agent` is always
    /// excluded).
    pub tool_allow: Vec<String>,
    /// Optional sampling overrides.
    pub sampling: Option<SamplingParams>,
    /// The hook engine for this profile — hooks that run when a sub-agent executes a
    /// turn.
    ///
    /// Consistent with the "inherit world, not identity" principle: hooks belong to the
    /// profile's identity and are declared by the profile's own configuration (the CLI
    /// assembles `ProfileSpec.hooks` into an engine at build time). They are **not**
    /// inherited from the parent session. `None` means the sub-agent has no hooks (falls
    /// back to [`NoopHookEngine`]), preserving exactly the same behavior as before —
    /// existing profiles without hooks are unaffected.
    pub hooks: Option<Arc<dyn HookEngine>>,
}

// `Arc<dyn HookEngine>` is not `Debug`; manually implement `Debug` to skip it (only
// indicate whether an engine is attached).
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

/// The `spawn_agent` tool. It is registered on `StaticToolRegistry` and shared across
/// sessions of the owning `AgentCore` via `process_tools` (it is **not** a process-global
/// singleton — a single process may host multiple `AgentCore` instances, each with its
/// own copy). At construction time it captures everything needed to run a nested turn,
/// because [`ToolContext`] only carries cwd/fs/shell/http/cancel/current_model, not the
/// provider registry, policy, or tool set.
pub struct SpawnAgentTool {
    schema: ToolSchema,
    profiles: Arc<BTreeMap<String, SubagentProfile>>,
    registry: Arc<ProviderRegistry>,
    /// The parent agent's policy (shared by all sessions in this core). The child turn
    /// wraps it with [`NonInteractivePolicy`].
    policy: Arc<dyn SandboxPolicy>,
    /// Parent agent tool set — source for subsetting by profile allowlist.
    process_tools: Arc<dyn ToolRegistry>,
    /// The `base_prompt` text inherited by child agents (the "you are an agent that can
    /// use tools" boilerplate).
    base_prompt: Option<String>,
}

impl SpawnAgentTool {
    /// Constructs a `spawn_agent` tool. When `profiles` is empty, the caller **should
    /// not** register this tool (the `profile` enum in the schema will be an empty set,
    /// so calls will always fail) — see [`Self::has_profiles`].
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

    /// Whether any profiles were discovered. The assembler uses this to decide whether to
    /// register this tool.
    pub fn has_profiles(profiles: &BTreeMap<String, SubagentProfile>) -> bool {
        !profiles.is_empty()
    }
}

/// Dynamically build the schema: `profile` is an enum of discovered profile names (hard
/// constraint), and the tool description embeds a catalog of `- <name>: <description>`
/// entries (soft guidance). Both are required: the enum alone gives no usage context,
/// while the catalog alone risks name typos.
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
    /// Optional per-call model override. Takes highest priority (overrides
    /// `profile.model` and parent model).
    #[serde(default)]
    model: Option<String>,
    /// Whether to run in the background. When `true` and the context supports it
    /// (`ToolContext::background` is `Some`), spawn returns the task id immediately
    /// without waiting for the child agent to finish. Defaults to `false` (synchronous
    /// blocking).
    #[serde(default)]
    run_in_background: bool,
}

impl Tool for SpawnAgentTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // Conservatively mark as Mutating: the "danger" of spawn itself is determined by
        // the child agent's tool set (gate A) and `NonInteractivePolicy` (gate B), not
        // subdivided at this layer.
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
        // Move captured dependencies from construction and runtime handles from `ctx`
        // into a `'static` future — all borrows of the nested `TurnRunner` live inside
        // this async block and do not escape.
        let profiles = self.profiles.clone();
        let registry = self.registry.clone();
        // Prefer the active policy from the current turn's snapshot (injected via `ctx`),
        // which reflects the session's current permission mode; fall back to the policy
        // captured at construction time only when none was injected (e.g. in tests or
        // when omitted).
        let policy = ctx.policy.clone().unwrap_or_else(|| self.policy.clone());
        // Prefer the session's fully assembled tool pool (built-in + connected MCP) so the
        // child agent's allowlist can reference `mcp__*` tools. Fall back to the static
        // pool captured at construction only when the turn runner did not inject one
        // (legacy / test paths).
        let process_tools = ctx
            .session_tools
            .clone()
            .unwrap_or_else(|| self.process_tools.clone());
        let base_prompt = self.base_prompt.clone();

        let cwd = ctx.cwd.to_path_buf();
        let fs = ctx.fs.clone();
        let shell = ctx.shell.clone();
        let http = ctx.http.clone();
        let parent_model = ctx.current_model.to_string();
        let parent_provider = ctx.current_provider.to_string();
        let background = ctx.background.clone();
        // Subagent event bridge: nest child-turn events back into the parent trace
        // (observability).
        let bridge = ctx.subagent_bridge.clone();
        // Remaining subagent dispatch depth for this turn. Child turns receive `depth-1`;
        // whether the child toolset includes `spawn_agent` is determined by `child_depth
        // > 0` (see `run_subagent_core`).
        let subagent_depth = ctx.subagent_depth;
        // The synchronous path uses a turn child token (cancelled when the turn ends);
        // the background path does not use it, instead using a session-level child token
        // minted by `BackgroundTasks` at spawn time (see below).
        let turn_cancel = ctx.cancel.child_token();

        // First parse `run_in_background` and the profile name to decide whether to run
        // synchronously or in the background. On parse failure, both paths treat it as
        // `InvalidArgs`.
        let parsed: Result<SpawnArgs, _> = serde_json::from_value(args.clone());

        let fut = async move {
            let parsed = match parsed {
                Ok(p) => p,
                Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
            };

            // Depth guard: the remaining dispatch depth for this turn is exhausted (0),
            // so the `spawn_agent` tool should never have been visible —
            // `run_subagent_core` does not include it in the child tool set when
            // `child_depth == 0`. Reaching this point indicates a malformed `ctx`; fail
            // loudly, do not silently swallow. The top-level turn injects the configured
            // maximum, which is always > 0 under normal conditions.
            if subagent_depth == 0 {
                return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io_err(
                    "subagent recursion depth exhausted: this agent is not allowed to spawn \
                     further subagents"
                        .to_string(),
                ))));
            }

            // Background path: requires `ctx` to support background (only injected at the
            // top-level turn), and `run_in_background=true`.
            if parsed.run_in_background {
                let Some(bg) = background else {
                    // Background context is unavailable (nested subagent / test) — fail
                    // loud, do not silently fall back to synchronous execution, otherwise
                    // the model believes it is running in the background while actually
                    // blocking, contradicting the declared behavior.
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
                    parent_provider,
                    subagent_depth,
                    // The background path also uses the bridge — the same
                    // `AgentEvent::Subagent` mechanism as the foreground. The
                    // `spawn_agent` tool span that initiates it closes normally first
                    // (the `ToolCallFinished` "started" below), then the child turn
                    // events appear as an **adjacent** subagent span under the same
                    // `parent_tool_call_id` anchor, remaining open until the child turn
                    // truly ends. The projector naturally distinguishes foreground
                    // (nested) from background (adjacent) by checking whether the tool
                    // span is still in the table. The bridge's `parent_events` is a
                    // session-level `EventEmitter` that stays alive while the background
                    // task runs.
                    bridge,
                    // Only the background path exposes history — `task_handle` is
                    // obtained inside the spawn closure and injected later (see below).
                    task_handle: None,
                };
                // Spawn mints a session-level child token for the task, so the task's
                // cancellation lifecycle is independent of the turn that spawned it —
                // ending the turn does not kill it. Also obtains a `TaskHandle`, shares
                // the child turn's `history` `Arc` into the task table, and lets the main
                // agent inspect the child agent's **submitted-to-LLM message blocks**
                // (not streaming deltas) via `inspect_background_task`.
                let label_for_log = parsed.profile.clone();
                let task_id = bg.spawn(label, move |task_cancel, task_handle| async move {
                    let mut deps = deps;
                    deps.task_handle = Some(task_handle);
                    match run_subagent_core(parsed, deps, task_cancel).await {
                        Ok(answer) => crate::session::BackgroundResult::Completed(answer),
                        Err(err) => {
                            // Log loudly: background failures were previously silently
                            // reduced to a `Failed` string, with no Langfuse event or log
                            // entry. This adds a `warn` with the task and error details.
                            tracing::warn!(
                                profile = %label_for_log,
                                error = %err,
                                "background subagent failed"
                            );
                            crate::session::BackgroundResult::Failed(err.to_string())
                        }
                    }
                });
                // Return synchronously with "started id=X" to satisfy the tool_use ↔
                // tool_result pairing contract.
                // Subagent profiles are indexed by source name at startup.
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

            // Synchronous path: original behavior — block until the sub-turn finishes,
            // then use the final text as the result.
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
                parent_provider,
                subagent_depth,
                // Synchronous path: the parent `spawn_agent` tool span remains open for
                // the entire duration (blocking until the child turn completes), allowing
                // child events to be nested under it.
                bridge,
                // Synchronous path: no background task, no history exposed (parent call
                // blocks entirely; no need to "peek while running").
                task_handle: None,
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

/// Dependency bundle for `run_subagent_core` — avoids a dozen positional parameters. All
/// construction-time and ctx handles are moved in, fully owned, so they can cross await
/// points or be sent to a background task.
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
    /// The provider vendor currently selected in the parent session. Together with
    /// `parent_model` this forms a `(vendor, model)` selection pair – when the child
    /// agent's model falls back to the parent's choice, the entry is resolved exactly by
    /// this pair. An empty string means the parent context did not inject a vendor
    /// (legacy/test path), in which case the fallback picks the first entry by bare model
    /// id.
    parent_provider: String,
    /// Remaining dispatch depth for this (initiator) turn. Child turns run at
    /// `subagent_depth - 1`; the child toolset includes `spawn_agent` only when that
    /// decremented value is `> 0` (see `run_subagent_core`).
    subagent_depth: u32,
    /// Subagent event bridge: when `Some`, nests child turn events back into the parent
    /// trace. Only set on the synchronous path.
    bridge: Option<crate::tool::SubagentBridge>,
    /// Background task handle: when `Some`, shares the child turn's history `Arc` into
    /// the task table so the main agent can inspect the child agent's **message chunks
    /// submitted to the LLM** via `inspect_background_task`. Only set in the background
    /// path — the synchronous path's parent `spawn_agent` call blocks entirely, so there
    /// is no need to "peek while running".
    task_handle: Option<crate::session::TaskHandle>,
}

/// Extracts the profile name from the raw args (used only for the background-start
/// confirmation message; falls back to a placeholder on failure).
fn parsed_profile_for_msg(args: &serde_json::Value) -> String {
    args.get("profile")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string()
}

/// Runs a sub-agent turn, returning the final text (`Ok`) or an error description
/// (`Err`).
///
/// Both the synchronous and background paths share this core: the synchronous path wraps
/// `Ok/Err` into `ToolEvent::Completed/Failed`, while the background path wraps them into
/// `BackgroundResult::Completed/Failed`. The caller determines the lifecycle of `cancel`
/// — the synchronous path passes a turn-level child token, and the background path passes
/// a session-level child token.
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
        parent_provider,
        subagent_depth,
        bridge,
        task_handle,
    } = deps;

    let Some(profile) = profiles.get(&parsed.profile) else {
        return Err(ToolError::InvalidArgs(BoxError::new(io_err(format!(
            "unknown profile `{}`; available: {}",
            parsed.profile,
            profiles.keys().cloned().collect::<Vec<_>>().join(", ")
        )))));
    };

    // Model priority: call argument > profile > parent session's current model.
    // Only when the model falls back to the parent (no explicit override) do we also
    // inherit the parent's provider vendor, resolving precisely by `(vendor, model)` pair
    // (so multiple gateways with the same model won't pick the wrong provider). When the
    // model is explicitly overridden, there is no provider dimension information — fall
    // back to taking the first entry by bare model id.
    let model_override = parsed.model.clone().or_else(|| profile.model.clone());
    let inherits_parent = model_override.is_none();
    let model = model_override.unwrap_or(parent_model);
    let entry = if inherits_parent && !parent_provider.is_empty() {
        registry.entry_for(&parent_provider, &model)
    } else {
        registry.first_entry_for_model(&model)
    };
    let Some(entry) = entry else {
        return Err(ToolError::Execution(BoxError::new(io_err(format!(
            "subagent model `{model}` is not declared by any provider entry"
        )))));
    };
    let provider = entry.provider().clone();

    // The remaining dispatch depth for the child turn is this layer minus one. By this
    // point `subagent_depth >= 1` (execute already fails loud on 0), so the child depth
    // is >= 0.
    let child_depth = subagent_depth - 1;

    // Gate A: subset the parent tool set by the allowlist. Entries are glob patterns (the
    // same engine as the top-level profile / hook matchers); a bare name is the degenerate
    // case. `spawn_agent` is a virtual matchable member — when a pattern matches it AND a
    // **depth gate** allows (`child_depth > 0`), the child receives a **freshly
    // constructed** `spawn_agent` (capturing the same `process_tools` so grandchildren can
    // recurse). When depth is exhausted, a matched `spawn_agent` is ignored — a structural
    // closure. A pattern matching nothing hard-fails (fail loud).
    let matched = crate::session::match_tool_allowlist(&process_tools, &profile.tool_allow)
        .map_err(|pattern| {
            ToolError::InvalidArgs(BoxError::new(io_err(format!(
                "profile `{}` allows tool pattern `{pattern}` matching nothing in the tool pool",
                parsed.profile
            ))))
        })?;
    let mut builder = StaticToolRegistry::builder();
    for name in &matched.tools {
        if let Some(tool) = process_tools.get(name) {
            builder = builder.insert(tool);
        }
    }
    if matched.spawn_agent && child_depth > 0 {
        let child_spawn = SpawnAgentTool::new(
            profiles.clone(),
            registry.clone(),
            // Parent policy captured as the child's construction-time fallback; the active
            // policy injected via `ctx` still takes precedence at runtime, and the child
            // turn is further wrapped in `NonInteractive`.
            policy.clone(),
            process_tools.clone(),
            base_prompt.clone(),
        );
        builder = builder.insert(Arc::new(child_spawn));
    }
    let sub_tools: Arc<dyn ToolRegistry> = Arc::new(builder.build());

    // System prompt: inherited `base_prompt` + profile's own `system.md`. Does not use
    // `resolve_system_prompt` (to avoid crawling workspace `AGENTS.md` / provider·model
    // overlay).
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

    // All sub-turn state is local to this async block and dropped when it completes.
    // `history` is wrapped in `Arc` so the background path can share the same history
    // with the task table, allowing the control plane to peek at the message blocks the
    // sub-agent submits to the LLM.
    let history: Arc<dyn History> = Arc::new(VecHistory::new());
    if let Some(handle) = &task_handle {
        handle.attach_history(history.clone());
    }
    let events = Arc::new(EventEmitter::new());

    // Observability bridge: wraps each event from the child turn into an
    // `AgentEvent::Subagent` and forwards it back to the parent session's event stream,
    // so that Langfuse can nest the child turn under the parent's `spawn_agent` tool
    // span. This is observability-only — the isolation contract leaves `storage` / `wire`
    // / `REPL` unchanged (they ignore `Subagent`). The bridge task subscribes to the
    // child emitter; once the child turn finishes and this function returns, dropping
    // `events` (the last strong reference) ends the child stream, and the task exits
    // naturally without an explicit join.
    let bridge_task = bridge.map(|b| {
        let mut sub_events = events.subscribe();
        let agent_type = parsed.profile.clone();
        tokio::spawn(async move {
            while let Some(ev) = sub_events.next().await {
                // Recursive flattening: this bridge layer only prepends its own
                // `tool_call_id`.
                //
                // - From a deeper layer that is **already** a `Subagent` (with a partial
                //   ancestor chain) → insert this layer's id at the head of the chain,
                //   keeping the deeper `agent_type` and leaf `inner` unchanged.
                // - A **leaf** event from a child turn → wrap it as `Subagent{[this
                //   layer's id], this layer's profile, leaf}`.
                //
                // After the event passes through N layers, `ancestor_path` is exactly the
                // complete chain from the top layer to the leaf.
                let forwarded = match ev {
                    AgentEvent::Subagent {
                        mut ancestor_path,
                        agent_type: deeper,
                        inner,
                    } => {
                        ancestor_path.insert(0, b.parent_tool_call_id.clone());
                        AgentEvent::Subagent {
                            ancestor_path,
                            agent_type: deeper,
                            inner,
                        }
                    }
                    leaf => AgentEvent::Subagent {
                        ancestor_path: vec![b.parent_tool_call_id.clone()],
                        agent_type: agent_type.clone(),
                        inner: Box::new(leaf),
                    },
                };
                b.parent_events.emit(forwarded).await;
            }
        })
    });

    let permissions = PermissionGate::new();
    let sub_policy: Arc<dyn SandboxPolicy> = Arc::new(NonInteractivePolicy::new(policy));
    // Use the hook engine declared in the profile, or fall back to `NoopHookEngine` (same
    // behavior as before the change).
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
        // Limit subagent to a fixed number of steps to prevent runaway nested loops.
        request_limit: TurnRequestLimit::Fixed(32),
        // Depth decreases by one per level: the child turn's tool driver uses this to
        // decide whether grandchildren can be dispatched. When `child_depth == 0`, the
        // child turn's tool set already lacks `spawn_agent` (gate A above is not
        // installed), so redundantly setting it to 0 here is self-consistent.
        subagent_max_depth: child_depth,
        ..TurnConfig::default()
    };

    let runner = TurnRunner {
        history: history.as_ref(),
        tools: &*sub_tools,
        // Owned clone so a nested spawn_agent (if the child's depth allows) sees the same
        // subset; keeps the MCP-aware injection consistent at every recursion level.
        session_tools: Some(sub_tools.clone()),
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
        // Sub‑agent turns carry no background handle: structurally prevents background
        // tasks from spawning themselves (same anti‑recursion design as "whitelist never
        // contains spawn_agent itself").
        background: None,
        // Sub‑agent does not participate in the parent’s goal loop: the parent’s
        // `goal_done` / `goal‑gate` only apply at the top‑level turn; the sub‑agent has
        // its own finite step limit (`request_limit`) as a safety net.
        goal: None,
        // Sub-agent turns skip background compaction: the context is short and its
        // lifetime ends with the tool call, so no cross-turn background summary is
        // needed. It still benefits from the hard-watermark synchronous compaction
        // fallback (the `compact_hard` path requires `provider_arc`), so we give it
        // `provider_arc` and leave the other background compaction fields empty.
        compaction_slot: None,
        history_arc: None,
        provider_arc: Some(provider.clone()),
        session_cancel: None,
        // The sub-agent's task is its "user input".
        ingest_source: crate::hooks::step::IngestSource::User,
    };

    let prompt = vec![ContentBlock::Text(TextContent::new(parsed.task))];
    let run_result = runner.run(prompt).await;

    // End of sub-turn: drop `runner` and the local strong reference to `events`, allowing
    // the child event stream to close. The bridge task flushes any buffered events to the
    // parent emitter and then exits. Awaiting it ensures all child events arrive before
    // the parent `spawn_agent` tool span finishes (this function returns →
    // `ToolCallFinished`).
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

    // Take the text of the last assistant message as the result.
    Ok(last_assistant_text(&history.snapshot()))
}

/// Take the **last** [`Role::Assistant`] message from the history and concatenate all its
/// `Text` segments (skipping thinking / tool_use). The tool-use loop may append multiple
/// assistant messages; the last one corresponds to the "final answer".
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
mod tests;
