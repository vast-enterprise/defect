//! Tool abstraction.
//!
//! Both builtin tools (`defect-tools`) and MCP adapters (`defect-mcp`) integrate
//! into the agent main loop by implementing the [`Tool`] trait.
//!
//! ## ACP alignment
//!
//! [`Tool::describe`] and [`ToolEvent::Progress`] / [`ToolEvent::Completed`]
//! directly reuse ACP's [`ToolCallUpdateFields`] to avoid duplicating fields.
//! The main loop enriches the fields produced by the tool with metadata such as
//! [`ToolCallId`] and [`raw_input`], then forwards them as `session/update` and
//! `session/request_permission`.
//!
//! [`ToolCallId`]: agent_client_protocol_schema::ToolCallId
//! [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
//! [`raw_input`]: agent_client_protocol_schema::ToolCallUpdateFields::raw_input

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol_schema::{ToolCallId, ToolCallUpdateFields};
use futures::Stream;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::error::BoxError;
use crate::fs::FsBackend;
use crate::http::HttpClient;
use crate::session::EventEmitter;
use crate::shell::ShellBackend;

mod background_tasks;
mod goal_done;
mod skill;
mod spawn_agent;
pub use background_tasks::{CancelBackgroundTaskTool, InspectBackgroundTaskTool};
pub use goal_done::{GOAL_DONE_TOOL_NAME, GoalDoneTool};
pub use skill::{SkillEntry, SkillTool, SkillTriggers};
pub(crate) use spawn_agent::SPAWN_AGENT_TOOL_NAME;
pub use spawn_agent::{SpawnAgentTool, SubagentProfile};

/// Tool's "public face": describes the parameter shape without any execution capability.
///
/// [`crate::llm::CompletionRequest::tools`] accepts `Vec<ToolSchema>`.
/// Providers don't hold `dyn Tool`; they serialize schemas into wire JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the parameters. Uses a subset of Draft 2020-12 (the exact subset
    /// and escaping rules are documented in `tool-trait.md`).
    pub input_schema: serde_json::Value,
}

/// Self-description of a tool call, directly mapping to ACP's [`ToolCallUpdateFields`].
///
/// Purpose (the same data drives three ACP messages):
/// - First push of a `ToolCall` (`status = Pending`)
/// - The `tool_call` field in a `RequestPermission` request
/// - Baseline for incremental updates via [`ToolEvent::Progress`]
///
/// Field conventions:
/// - `tool_call_id` is not in this struct; it is assigned uniformly by the main loop
///   (using the LLM's `tool_use_id` or a self-generated UUID). The tool does not care
///   about it.
/// - `raw_input` is filled by the main loop with the original args. Tool implementations
///   must not set it themselves, to avoid divergence from the real parameters on the
///   wire.
/// - `status` is inferred from the [`ToolEvent`] variant: `Progress` → `InProgress`,
///   `Completed` → `Completed`, `Failed` → `Failed`. Tools must not set it themselves.
///
/// [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
#[derive(Debug, Clone)]
pub struct ToolCallDescription {
    pub fields: ToolCallUpdateFields,
}

/// Safety level for a tool.
///
/// This is only a **hint** fed to the external sandbox policy; the final Allow / Deny /
/// Ask decision is made by the policy (in combination with user configuration, prior
/// authorization, etc.). The trait itself does not enforce any policy.
///
/// The `serde` representation uses `snake_case` (`read_only` / `mutating` / `destructive`
/// / `network`), so that `defect-config` can deserialize it directly from TOML in hook
/// matchers and similar contexts.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyClass {
    /// Read-only: list directories, read files, query metadata.
    ReadOnly,
    /// Mutating: writes files or modifies state; side effects may or may not be
    /// reversible.
    Mutating,
    /// Destructive: deleting files, moving, executing commands.
    Destructive,
    /// Outbound network: HTTP / DNS / any remote I/O.
    Network,
}

/// Elements of the event stream produced by [`Tool::execute`].
///
/// Terminal semantics: the stream contains **at most one** [`ToolEvent::Completed`] or
/// [`ToolEvent::Failed`], and it must be the last event in the stream. When the main
/// loop encounters a terminal event, it considers the tool call finished and does not
/// consume any further elements.
#[non_exhaustive]
#[derive(Debug)]
pub enum ToolEvent {
    /// Progress delta: the main loop forwards this as a `tool_call_update` in an ACP
    /// `session/update`.
    /// Contains only the fields that changed, matching the "patch" semantics of
    /// [`ToolCallUpdateFields`].
    ///
    /// [`ToolCallUpdateFields`]: agent_client_protocol_schema::ToolCallUpdateFields
    Progress(ToolCallUpdateFields),

    /// Successful completion. `fields` contains the remaining final-state fields (e.g.,
    /// final content, locations, raw_output); the main loop is responsible for setting
    /// `status` to `Completed`.
    Completed(ToolCallUpdateFields),

    /// Terminal failure. Carries the Rust-side error so the caller can retry or log it;
    /// when mapping to ACP, the main loop sets `status` to `Failed` and places the
    /// [`ToolError`] text into `content`.
    Failed(ToolError),
}

/// Event stream for [`Tool::execute`]. Type-erased so that `dyn Tool` can be used
/// directly.
pub type ToolStream = Pin<Box<dyn Stream<Item = ToolEvent> + Send>>;

/// The execution environment injected into [`Tool::execute`].
///
/// An explicit struct rather than environment variables or thread-locals, making it easy
/// to construct in tests and avoiding implicit global state. Fields are marked
/// `non_exhaustive` to allow future additions (sandbox handles, ACP backchannels, etc.)
/// without breaking existing implementations.
#[non_exhaustive]
pub struct ToolContext<'a> {
    /// The default working directory for the tool (typically the ACP session's `cwd`).
    pub cwd: &'a Path,
    /// Cancellation token: triggered by upstream `session/cancel`, user Ctrl+C, timeout,
    /// etc.
    /// Tool implementations should check `cancel.is_cancelled()` at long loops or await
    /// points and exit as soon as possible.
    pub cancel: CancellationToken,
    /// Filesystem backend. The `fs` tool family (`read_file` / `write_file` /
    /// `edit_file`) reads and writes files through it. During assembly, `defect-acp`
    /// selects either `LocalFsBackend` or `AcpFsBackend` based on the client-negotiated
    /// [`FileSystemCapabilities`]; tool implementations are completely unaware of this.
    ///
    /// Uses [`Arc`] instead of a borrow: `Tool::execute` returns a `'static` future /
    /// stream, and tools typically `clone` the fs into async tasks. A borrow cannot
    /// survive across `.await`.
    ///
    /// [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities
    pub fs: Arc<dyn FsBackend>,
    /// Shell execution backend. The `bash` tool uses it to create a terminal and run
    /// commands; during assembly, `defect-acp` selects either `LocalShellBackend` or
    /// `AcpShellBackend` based on the client-negotiated [`ClientCapabilities::terminal`],
    /// and tool implementations are unaware of the choice.
    ///
    /// Same `Arc` trade-off as `fs` — `Tool::execute` returns a `'static` future.
    ///
    /// [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities
    pub shell: Arc<dyn ShellBackend>,
    /// HTTP fetch backend. The `fetch` tool uses it to perform network reads; it is set
    /// up at the CLI entry point (constructed from `HttpClientConfig` as a process-level
    /// [`HttpClient`] instance and reused). Tool implementations receive an [`Arc`]
    /// clone; `Tool::execute` is a `'static` future, so borrowing cannot survive across
    /// await points.
    pub http: Arc<dyn HttpClient>,
    /// The model id selected for the current turn. Most tools do not need this; the
    /// `spawn_agent` sub-agent tool uses it to "fall back the model to the parent
    /// session's current selection" — `ToolContext` does not carry a provider registry,
    /// but carrying this string is enough for `spawn_agent` to call `entry_for_model` on
    /// its own captured registry to resolve the provider the parent is currently using.
    /// Populated from `config.model` by [`TurnRunner`](crate::session::TurnRunner) when
    /// constructing the context.
    pub current_model: &'a str,
    /// The provider vendor selected for the current turn. Together with
    /// [`Self::current_model`] this forms a `(vendor, model)` selection pair — when a
    /// `spawn_agent` sub-agent falls back to the parent's choice, it uses this pair to
    /// call `entry_for` on the registry for exact resolution (avoiding provider
    /// mis-selection when multiple gateways serve the same model name). An empty string
    /// means the value was not injected (legacy/test paths); in that case `spawn_agent`
    /// falls back to looking up the first entry by bare model id. Populated by the turn
    /// runner from `config.provider` when constructing the context.
    pub current_provider: &'a str,
    /// Session-level background task handle. When `Some`, tools can fire-and-forget a
    /// task that outlives the current turn (primarily for `spawn_agent {
    /// run_in_background: true }`); `None` means the context does not support background
    /// execution (e.g., nested sub-agent turns or tests), and tools should fall back to
    /// synchronous execution.
    ///
    /// Uses an owned [`Arc`]-backed handle instead of a borrow: `Tool::execute` returns a
    /// `'static` future, and a borrow cannot survive across await. Injected by the
    /// top-level [`TurnRunner`](crate::session::TurnRunner) when constructing the
    /// context; not injected for nested sub-agent turns (structurally prevents background
    /// tasks from spawning themselves).
    pub background: Option<crate::session::BackgroundTasks>,
    /// Subagent event bridge: when `Some`, a tool can wrap internally spawned sub-turn
    /// events as [`crate::event::AgentEvent::Subagent`] and forward them back to the
    /// parent session's event stream for nested observability display. Currently only
    /// used by `spawn_agent`. Injected by the turn runner in `session::turn` for each
    /// tool according to its [`ToolCallId`] — **injected for both top-level and nested
    /// sub-agent turns** (recursive bridging), with the mount point expressed by
    /// [`SubagentBridge::parent_tool_call_id`].
    pub subagent_bridge: Option<SubagentBridge>,
    /// The active sandbox policy for this turn snapshot. `spawn_agent` uses it to pass
    /// the parent's current real policy to child agents — after a `session/set_mode`
    /// switch, newly created turns propagate the new policy through this field, so child
    /// agents never see a stale process-level default. When `None`, `spawn_agent` falls
    /// back to the policy captured at construction time (testing / uninjected scenarios).
    /// Most tools ignore this field.
    pub policy: Option<Arc<dyn crate::policy::SandboxPolicy>>,
    /// Shared state for the `--goal` goal-driven loop. When `Some`, this session runs in
    /// goal mode; the `goal_done` tool calls [`crate::session::GoalState::mark_reached`]
    /// to set the flag, and the `goal-gate` hook uses it to decide whether to release or
    /// extend a turn when it voluntarily stops. `None` means non-goal mode (the default);
    /// the `goal_done` tool is not registered and this field is never read.
    pub goal: Option<Arc<crate::session::GoalState>>,
    /// How many more layers of subagent can be dispatched from the current layer. The
    /// top-level turn starts at the configured initial limit; `spawn_agent` decrements it
    /// by one when injecting a nested turn for a child agent. `0` means the child agent
    /// cannot obtain the `spawn_agent` tool (depth exhausted, structurally preventing
    /// further recursion) — replacing the old hard-coded "whitelist never contains
    /// `spawn_agent`". This is a functional gate, unrelated to observability, so it is
    /// independent of the optional [`Self::subagent_bridge`] and also takes effect in
    /// test / no-bridge scenarios. Defaults to `0` (most conservative: no explicit
    /// injection means no dispatch; the top-level turn must explicitly use
    /// [`Self::with_subagent_depth`]).
    pub subagent_depth: u32,
    /// The current session's **fully assembled** tool pool — the `CompositeRegistry` that
    /// already merged built-in tools with the per-session MCP tools. `spawn_agent` uses
    /// this (rather than a static, MCP-free tool set captured at construction) to build a
    /// child agent's tool subset, so a subagent profile may allow `mcp__*` tools. `None`
    /// in legacy / test paths, where `spawn_agent` falls back to its captured static pool.
    /// Injected by the [`TurnRunner`](crate::session::TurnRunner) when constructing the
    /// context.
    pub session_tools: Option<Arc<dyn crate::session::ToolRegistry>>,
    /// The current turn's [`TurnConfig`](crate::session::TurnConfig). `spawn_agent` uses it
    /// so a child agent **inherits** the parent's turn settings (compaction thresholds,
    /// retry/concurrency limits, sampling incl. `reasoning_effort`, request-limit default)
    /// instead of silently falling back to `TurnConfig::default()`. A profile may still
    /// override individual fields. `None` in legacy / test paths, where `spawn_agent` uses
    /// defaults. Injected by the [`TurnRunner`](crate::session::TurnRunner).
    pub parent_turn_config: Option<Arc<crate::session::TurnConfig>>,
}

/// A handle for bridging sub-turn events (spawned internally by a tool) back into the
/// parent session's event stream.
///
/// Holds the parent session's [`EventEmitter`] and the [`ToolCallId`] that initiated this
/// tool invocation. `Clone` is cheap (internally `Arc` + small string).
///
/// ## Recursive bridging: each layer only prepends its own id
///
/// The full ancestor chain is not stored here — it is accumulated incrementally as events
/// **bubble upward** through each layer's bridge. The bridge subscriber (e.g.,
/// `spawn_agent`'s `bridge_task`) at each layer:
/// - Receives a **leaf** event from the sub-turn → wraps it as
///   `Subagent{ ancestor_path: [parent_tool_call_id], agent_type: <this layer's profile>,
///   inner: leaf }`;
/// - Receives an **already** `Subagent` (from a deeper layer, already carrying a partial
///   chain) → **prepends** `parent_tool_call_id` to the head of its `ancestor_path`,
///   leaving `inner` leaf and deeper `agent_type` unchanged.
///
/// Thus after passing through N layers of bridging, `ancestor_path` is exactly the
/// complete id chain from the top layer down to the leaf. Each layer only needs to know
/// its own hop — this lets frontend, backend, and arbitrary depths share the same logic.
///
/// The recursive **depth gate** is not here — it is functional and must always apply
/// (including in non-observability / test scenarios), so it lives in the separate
/// [`ToolContext::subagent_depth`] field rather than in this optional bridge.
#[derive(Clone)]
pub struct SubagentBridge {
    /// Event bus of the parent session. Wrapped [`crate::event::AgentEvent::Subagent`]
    /// events are emitted here.
    pub parent_events: Arc<EventEmitter>,
    /// The tool call ID that spawned this subagent (the corresponding tool span in the
    /// parent trace). The bridge prepends this ID, serving as the mount point of this
    /// subagent within the parent trace.
    pub parent_tool_call_id: ToolCallId,
}

impl<'a> ToolContext<'a> {
    /// Constructs a minimal `ToolContext`. The `#[non_exhaustive]` attribute prevents
    /// external crates from constructing the struct directly with a literal — this
    /// constructor is the only cross-crate entry point. When adding new fields, add
    /// default values to the signature or provide a new constructor to avoid breaking
    /// existing call sites.
    pub fn new(
        cwd: &'a Path,
        cancel: CancellationToken,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        http: Arc<dyn HttpClient>,
        current_model: &'a str,
    ) -> Self {
        Self {
            cwd,
            cancel,
            fs,
            shell,
            http,
            current_model,
            current_provider: "",
            background: None,
            subagent_bridge: None,
            policy: None,
            goal: None,
            subagent_depth: 0,
            session_tools: None,
            parent_turn_config: None,
        }
    }

    /// Inject the current turn's [`TurnConfig`](crate::session::TurnConfig) so `spawn_agent`
    /// can build a child config that inherits the parent's turn settings. If not called,
    /// `spawn_agent` falls back to `TurnConfig::default()` for non-explicit fields.
    #[must_use]
    pub fn with_parent_turn_config(mut self, config: Arc<crate::session::TurnConfig>) -> Self {
        self.parent_turn_config = Some(config);
        self
    }

    /// Inject the current session's fully assembled tool pool (built-in + MCP composite).
    /// `spawn_agent` uses it to build a child agent's tool subset so subagent profiles can
    /// allow `mcp__*` tools. If not called, `session_tools` is `None` and `spawn_agent`
    /// falls back to the static pool captured at construction.
    #[must_use]
    pub fn with_session_tools(mut self, tools: Arc<dyn crate::session::ToolRegistry>) -> Self {
        self.session_tools = Some(tools);
        self
    }

    /// Inject the provider vendor selected for the current turn, forming a selection pair
    /// with `current_model`.
    /// If not called, defaults to an empty string (`spawn_agent` falls back to picking
    /// the first entry by bare model id).
    #[must_use]
    pub fn with_current_provider(mut self, vendor: &'a str) -> Self {
        self.current_provider = vendor;
        self
    }

    /// Inject the remaining subagent dispatch depth from this layer onward. The tool
    /// driver for the top-level turn calls with the configured initial cap; `spawn_agent`
    /// injects the decremented value for nested child-agent turns. If not called,
    /// defaults to `0` (most conservative: no subagent dispatch allowed).
    #[must_use]
    pub fn with_subagent_depth(mut self, depth: u32) -> Self {
        self.subagent_depth = depth;
        self
    }

    /// Inject the active sandbox policy for this turn snapshot. The top-level turn's tool
    /// driver uses this to pass the parent turn's policy to `spawn_agent`; if not called,
    /// `policy` is `None` (child agent nesting / testing), and `spawn_agent` falls back
    /// to the policy captured at construction time.
    #[must_use]
    pub fn with_policy(mut self, policy: Arc<dyn crate::policy::SandboxPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Inject a session-level background task handle. The top-level turn's tool driver
    /// uses this to enable `run_in_background`; if not called, `background` is `None`
    /// (the default for sub-agents / tests), and tools fall back to synchronous
    /// execution.
    #[must_use]
    pub fn with_background(mut self, background: crate::session::BackgroundTasks) -> Self {
        self.background = Some(background);
        self
    }

    /// Inject shared state for the `--goal` goal-driven loop. The `goal_done` tool sets
    /// `reached` based on this state; if not called, `goal` is `None` (non-goal mode, the
    /// default).
    #[must_use]
    pub fn with_goal(mut self, goal: Arc<crate::session::GoalState>) -> Self {
        self.goal = Some(goal);
        self
    }

    /// Inject a subagent event bridge. The tool driver injects one per tool call in
    /// `session::turn`, keyed by [`ToolCallId`], so that `spawn_agent` can nest child
    /// turn events back into the parent trace.
    #[must_use]
    pub fn with_subagent_bridge(mut self, bridge: SubagentBridge) -> Self {
        self.subagent_bridge = Some(bridge);
        self
    }
}

/// Tools callable by the agent.
///
/// Implementors are typically stateless (each invocation receives all dependencies via
/// `args` + [`ToolContext`]); if you need to hold state such as connections or caches,
/// place the state on `Self` and register an `Arc<Self>` with the main loop.
pub trait Tool: Send + Sync {
    /// Tool metadata. Returns a reference to avoid allocating on every call.
    fn schema(&self) -> &ToolSchema;

    /// Provides a safety-level hint to the sandbox policy without actually executing the
    /// tool.
    ///
    /// `args` is the already-deserialized JSON value — the same tool's safety level may
    /// vary by arguments (e.g., the `bash` tool escalates to [`SafetyClass::Destructive`]
    /// when `command` contains `rm`). The implementation should be a **pure function**
    /// and perform no IO.
    fn safety_hint(&self, args: &serde_json::Value) -> SafetyClass;

    /// Generates a "self-description" before execution, for display to the ACP client.
    ///
    /// The async signature and [`ToolContext`] injection allow implementations to perform
    /// lightweight IO during the describe phase (typical example: `write_file` reads the
    /// old content before requesting authorization, producing a precise old↔new diff for
    /// the client—more reviewable than "entirely new content").
    ///
    /// Performance constraint: `describe` runs before every ACP `ToolCall` push.
    /// Implementations should remain fast and graceful on failure (on IO failure, degrade
    /// to returning basic fields; do not let `describe` itself throw—the signature
    /// provides no error channel).
    ///
    /// See the field conventions on [`ToolCallDescription`] for which fields are filled
    /// by whom.
    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription>;

    /// Initiates a tool call and returns an event stream.
    ///
    /// See [`ToolEvent`] for the stream elements. The stream must end immediately after
    /// the terminal event. Dropping the stream is treated as cancellation (equivalent to
    /// `ctx.cancel.cancel()`).
    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream;
}

/// Tool execution error.
///
/// The granularity is intentionally coarse — finer-grained error types are carried by
/// built-in tools themselves in the `Execution` source. Here we only distinguish the
/// broad categories that the main loop needs to handle differently.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ToolError {
    /// Canceled by the caller (triggered via [`ToolContext::cancel`]).
    #[error("tool canceled")]
    Canceled,

    /// The tool arguments failed JSON parsing or schema validation. The main loop can
    /// send this back to the LLM so the model can fix the parameters and retry.
    #[error("invalid tool arguments: {0}")]
    InvalidArgs(#[source] BoxError),

    /// Runtime error (I/O failure, non-zero subprocess exit, network error, etc.).
    #[error("tool execution failed: {0}")]
    Execution(#[source] BoxError),
}
