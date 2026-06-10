//! Session — state container and lifecycle interface for a single conversation.
//!
//! ## Abstraction layers
//!
//! - [`AgentCore`]: process-level "agent instance", holds the built-in tool set and
//!   global configuration;
//!   it is the root object assembled by `defect-cli` and injected into
//!   `defect-acp::serve`
//! - [`Session`]: lifecycle unit for a single conversation; holds history, per-session
//!   tool
//!   table (including MCP), cancel token, and event stream
//! - [`History`]: wrapper around message history, with hooks reserved for compression,
//!   token counting, and resume
//!
//! All three are **exposed as traits**; concrete implementations live in the `session/`
//! submodule within this crate
//! and at the assembly point in `defect-cli`; `defect-acp` interacts with them only
//! through the traits.

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use futures::future::BoxFuture;

use crate::error::BoxError;
use crate::event::{AgentEvent, PermissionResolution};
use crate::fs::FsBackend;
use crate::llm::{
    Message, ModelCandidate, ModelInfo, ProviderError, ProviderInfo, ReasoningEffort,
};
use crate::shell::ShellBackend;
use crate::tool::{Tool, ToolSchema};

mod background;
mod capabilities;
mod context;
mod default;
mod events;
mod goal;
mod history;
mod permissions;
mod prompt;
mod tool_registry;
mod turn;

pub use background::{
    BackgroundOutcome, BackgroundProgressConfig, BackgroundResult, BackgroundTasks, BlockKind,
    ProgressBlock, TaskHandle, TaskSnapshot, TaskStatus, format_background_outcome,
};
pub use capabilities::{
    ResolvedSessionCapabilities, SessionCapabilitiesConfig, WebSearchCapabilityConfig,
    WebSearchCapabilityMode,
};
pub use context::{Frontend, RunningContext};
pub use default::{DefaultAgentCore, DefaultAgentCoreBuilder, DefaultSession, new_session_id};
pub use events::EventEmitter;
pub use goal::GoalState;
pub use history::VecHistory;
pub use permissions::PermissionGate;
pub use prompt::{load_project_prompt, resolve_system_prompt};
pub use tool_registry::{
    AllowlistMatch, CompositeRegistry, StaticToolRegistry, StaticToolRegistryBuilder,
    filter_registry_by_allowlist, match_tool_allowlist,
};
/// Re-exported for reuse within the crate: the `spawn_agent` sub-agent tool needs a
/// `RequestAuditTracker` instance when constructing a nested [`TurnRunner`]. This type is
/// not public (it exposes internal diagnostic state), but `crate::tool::spawn_agent` in
/// the same crate must be able to call `new()`.
pub(crate) use turn::RequestAuditTracker;
pub use turn::{
    BasePromptConfig, CompactionSlot, PromptConfig, TurnConfig, TurnRequestLimit, TurnRunner,
};

/// Process-level agent root object.
///
/// `defect-cli` constructs a concrete implementation at startup (holding the LLM provider
/// registry, built-in tool set, and configuration) and injects an `Arc<dyn AgentCore>`
/// into `defect-acp::serve`.
///
/// Rationale for extracting a trait:
/// - Allows injecting a mock in tests without spinning up a real LLM.
/// - If an "embedded agent" (library mode called by a host application) emerges in the
///   future, a second concrete implementation can be added without touching the ACP
///   bridge code.
pub trait AgentCore: Send + Sync {
    /// Creates a new session.
    ///
    /// `id` is generated and passed in by the caller (the `defect-acp` `session/new`
    /// handler) — the filesystem backend already needs a `SessionId` when constructed
    /// outside of [`AgentCore::create_session`] (see the ACP filesystem delegation
    /// contract). Concrete implementations treat it as the authoritative external id and
    /// return [`AgentError::DuplicateSessionId`] on duplicates.
    ///
    /// `mcp_servers` is the per-session MCP server list from the `session/new` request;
    /// the concrete implementation spawns subprocesses or establishes SSE connections
    /// during initialization, wrapping each MCP tool as a [`Tool`] and adding it to the
    /// session's tool table.
    ///
    /// `fs` is the session-level filesystem backend — `defect-acp` selects
    /// `LocalFsBackend` or `AcpFsBackend` at assembly time based on the client's
    /// [`FileSystemCapabilities`]. The session holds an `Arc` to it, and all filesystem
    /// tool calls go through it.
    ///
    /// `shell` is the session-level shell backend — `defect-acp` selects
    /// `LocalShellBackend` or `AcpShellBackend` at assembly time based on the client's
    /// [`ClientCapabilities::terminal`]. The session holds an `Arc` to it, and all `bash`
    /// tool calls go through it.
    ///
    /// `frontend` indicates how the agent is being accessed ([`Frontend::Acp`] carries
    /// the fs/shell delegation state negotiated during the ACP handshake) and is used to
    /// inject the `# Environment` section of the system prompt.
    ///
    /// # Errors
    ///
    /// MCP startup failure, missing cwd, duplicate id, etc.
    ///
    /// [`FileSystemCapabilities`]: agent_client_protocol_schema::FileSystemCapabilities
    /// [`ClientCapabilities::terminal`]: agent_client_protocol_schema::ClientCapabilities
    fn create_session(
        &self,
        id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        frontend: Frontend,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    /// Restore an existing session from persistent state.
    ///
    /// `frontend` works the same as in [`AgentCore::create_session`] — the restored
    /// session also uses it to inject runtime environment information.
    ///
    /// # Errors
    ///
    /// The session does not exist, the persisted data is corrupted, the restored `cwd` is
    /// unavailable, etc.
    fn load_session(
        &self,
        id: SessionId,
        fs: Arc<dyn FsBackend>,
        shell: Arc<dyn ShellBackend>,
        frontend: Frontend,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>>;

    /// Look up an existing session by id.
    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>>;
}

/// Abstraction for restoring a session from persistent storage.
///
/// Concrete implementations typically come from `defect-storage`.
pub trait SessionLoader: Send + Sync {
    /// Read back the state needed for recovery by session id.
    ///
    /// # Errors
    ///
    /// The session does not exist, the storage is corrupted, or replay fails.
    fn load_session(&self, id: SessionId) -> BoxFuture<'_, Result<LoadedSession, BoxError>>;
}

/// Abstraction for building an additional tool registry for a single session.
///
/// A typical implementation comes from `defect-mcp`: it connects to the list of MCP
/// servers provided by `session/new` or `session/load`, and wraps the remote tools into a
/// [`ToolRegistry`].
pub trait SessionToolFactory: Send + Sync {
    /// Build a session-level tool registry for the current session.
    ///
    /// # Errors
    ///
    /// Returns an error if the external tool source fails to initialize, the remote
    /// inventory cannot be fetched, or the configuration is unsupported.
    fn build_registry(
        &self,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
    ) -> BoxFuture<'_, Result<Arc<dyn ToolRegistry>, BoxError>>;
}

/// Observer for when `AgentCore::create_session` succeeds.
///
/// Typical uses:
/// - Start `defect-storage` event subscription persistence
/// - Attach per-session sidecar consumers for tracing / metrics
pub trait SessionObserver: Send + Sync {
    /// Called after the session is successfully created.
    ///
    /// # Errors
    ///
    /// Returns an error if initializing the side‑channel consumer fails, preventing the
    /// session from becoming externally visible.
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError>;
}

/// A public description of an optional permission mode. Used by `defect-acp` to construct
/// an ACP `SessionMode`.
///
/// It is a "policy-free" projection of [`crate::policy::PolicyMode`] — exposing only the
/// id/display fields without leaking the internal decision engine.
#[derive(Debug, Clone)]
pub struct ModeDescriptor {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

/// Model selection key: a `(provider vendor, model id)` pair.
///
/// The same model id can be declared by multiple providers (multiple gateways with the
/// same model), so selection must include both the provider vendor and the model id.
/// `provider` refers to [`ProviderInfo::vendor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    pub provider: String,
    pub model: String,
}

/// A single session.
///
/// All methods are trait-object-friendly (`&self` + `BoxFuture`). The `Arc<dyn Session>`
/// is shared between `defect-acp` and the main loop.
pub trait Session: Send + Sync {
    fn id(&self) -> &SessionId;

    /// Provider metadata used by the current session.
    fn provider_info(&self) -> ProviderInfo;

    /// The model ID used by the current session.
    fn current_model(&self) -> String;

    /// List the model candidates available from the current provider for this session.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] if the provider fails to fetch the model list.
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>>;

    /// List the (provider, model) candidate pairs visible to the session. Under a
    /// multi-provider setup, the same session may switch models across providers, so ACP
    /// rendering needs to annotate each candidate with its provider.
    ///
    /// # Errors
    ///
    /// Same as [`Self::list_models`]: returns [`ProviderError`] if fetching the provider
    /// list fails.
    fn list_candidates(&self) -> BoxFuture<'_, Result<Vec<ModelCandidate>, ProviderError>>;

    /// Switches the model for the current session.
    ///
    /// The selection key is a `(provider vendor, model)` pair — the same model id may be
    /// advertised by multiple providers (multiple gateways for the same model), so the
    /// provider must be explicitly specified. The currently in-progress turn retains its
    /// original selection; subsequent turns use the new selection.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the provider fails to fetch its model list, or when
    /// the requested `(provider, model)` pair does not exist.
    fn set_model(&self, selection: ModelSelection) -> BoxFuture<'_, Result<(), ProviderError>>;

    /// The current active permission mode ID. Returns `None` if no mode catalog is
    /// loaded.
    ///
    /// Maps to ACP `SessionModeState::current_mode_id`.
    fn current_mode(&self) -> Option<String>;

    /// The list of permission modes available to this session, in assembly order. Returns
    /// an empty list when no mode directory is mounted. Maps to ACP
    /// `SessionModeState::available_modes`.
    fn available_modes(&self) -> Vec<ModeDescriptor>;

    /// Switch the current permission mode. The change takes effect on subsequent turns;
    /// the in-flight turn retains its original policy (same semantics as
    /// [`Self::set_model`] — the policy is snapshotted when `run_turn` starts).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::ModeNotFound`] if `mode_id` does not match any available
    /// mode, or if the session has no mode directory installed.
    fn set_mode(&self, mode_id: String) -> Result<(), AgentError>;

    /// The current `reasoning_effort` level (`None` = unset, falling back to the provider
    /// default). Maps to the current value of the ACP thought-level configuration item.
    fn current_reasoning_effort(&self) -> Option<ReasoningEffort>;

    /// Sets the `reasoning_effort` level. `None` clears the override (falls back to the
    /// provider default). Takes effect on subsequent turns. Providers that do not support
    /// this concept ignore it when assembling requests.
    fn set_reasoning_effort(&self, effort: Option<ReasoningEffort>);

    /// Subscribe to the event stream. Three independent consumers (acp / storage /
    /// tracing) each call this once without interfering with each other — internally uses
    /// mpsc with fan-out so that slow consumers only experience backpressure without
    /// dropping events.
    fn subscribe(&self) -> EventStream;

    /// A read-only snapshot of the current history, used to replay the transcript to the
    /// client after a session load.
    fn history_snapshot(&self) -> Vec<Message>;

    /// Starts a turn.
    ///
    /// The returned future resolves when the turn ends:
    /// - `Ok(StopReason)` – normal termination (including Cancelled); drives the ACP
    ///   `PromptResponse`
    /// - `Err(TurnError)` – fatal error (auth expiry, model unavailable, etc.);
    ///   drives the ACP JSON-RPC `Error` response
    ///
    /// [`AgentEvent`]s produced during the turn are pushed via [`Session::subscribe`],
    /// **not** through this future. The `TurnEnded` event is still emitted on the event
    /// stream (for storage / tracing), but the ACP bridge uses this future's outcome.
    ///
    /// Only one turn may be in progress per session at a time; concurrent calls return
    /// [`TurnError::TurnInProgress`].
    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>>;

    /// Cancels the current turn. Idempotent: no-op if no turn is in progress.
    fn cancel_turn(&self);

    /// Writes back the client response to the ACP reverse request
    /// `session/request_permission` to the main loop.
    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution);

    /// Current context usage. Read-only and cheap; backs the `/context` slash command.
    fn context_status(&self) -> ContextStatus;

    /// Synchronously compact the session history now (out-of-band `/compact` command),
    /// reusing the same boundary selection + summarization as the turn loop's hard
    /// watermark.
    ///
    /// Returns `Ok(Some(report))` when a compaction ran, `Ok(None)` when there was no safe
    /// boundary to summarize (e.g. a single short turn — nothing to do).
    ///
    /// # Errors
    ///
    /// Returns [`TurnError::TurnInProgress`] if a turn is currently running: compaction
    /// rewrites history and would race the in-flight turn, so the caller must `/cancel` or
    /// wait first.
    fn compact_now(&self) -> BoxFuture<'_, Result<Option<CompactionReport>, TurnError>>;
}

/// Event stream. Type-erased to support trait object return.
pub type EventStream = futures::stream::BoxStream<'static, AgentEvent>;

/// Stable information provided to [`SessionObserver`] after successful creation.
#[derive(Debug, Clone)]
pub struct SessionCreateInfo {
    pub id: SessionId,
    pub cwd: PathBuf,
    pub mcp_servers: Vec<McpServer>,
}

/// Minimal session data restored from persistent storage.
#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub info: SessionCreateInfo,
    pub history: Vec<Message>,
}

/// Abstraction over message history — pure storage + token accounting.
///
/// Compaction is **not** handled here: summarization requires calling the LLM, which the
/// storage abstraction cannot reach.
/// Compaction is orchestrated in the turn main loop (`session/turn/compact.rs`) — it
/// reads [`History::snapshot`], calls the LLM for a summary, then writes back the
/// computed new message list via [`History::replace`]. This trait is only responsible
/// for: appending, snapshotting, wholesale replacement, and providing the main loop with
/// an estimate of "how many tokens the current history is worth."
///
/// Token estimation strategy (see [`VecHistory`]): use the **actual input token** count
/// reported by the last LLM call as a baseline, then add a **character-heuristic**
/// increment for messages appended after that baseline; when no real baseline is
/// available, fall back to a pure character-heuristic estimate for the entire history.
/// The turn main loop compares this estimate against the compaction threshold.
pub trait History: Send + Sync {
    /// Appends a message.
    fn append(&self, msg: Message);

    /// A snapshot of the current history, to be fed into the next LLM call.
    fn snapshot(&self) -> Vec<Message>;

    /// Replace the entire message list after compression. The turn main loop calls this
    /// to write back the new list consisting of a summary plus the retained tail. The
    /// implementation should also reset the token estimation baseline, since the old
    /// actual token counts no longer apply to the new list.
    fn replace(&self, messages: Vec<Message>);

    /// Prefix splice: replaces the first `drop_count` messages in the **current** list
    /// with the single `summary` message, preserving everything after them. Returns the
    /// actual number of messages dropped (`drop_count` is clamped to the current length).
    ///
    /// This is the primitive for **background compression** write-back: a background task
    /// computes `drop_count` (= the prefix length to summarize) and `summary` from a
    /// snapshot taken at some point, but while the summarization LLM call is in flight,
    /// the foreground turn may still be `append`ing to the **tail**. Writing back with
    /// `replace(entire list)` would discard any tail messages added during that time.
    /// `splice_prefix` only touches the first `drop_count` messages of the **current**
    /// list, preserving everything from `drop_count..` onward (including tail messages
    /// added in the meantime), so the write-back is correct.
    ///
    /// **Concurrency invariant** (must be maintained): `drop_count` is computed from an
    /// old snapshot and remains valid for the **current** list provided that during the
    /// flight only tail appends (`append`) and in-place content replacements
    /// (micro-compression `replace` with same-length rebuild) occur — no insertion or
    /// deletion of middle messages. The only operation that removes middle messages is
    /// compression itself, and compression runs **solo** (at most one in flight at a
    /// time), so the invariant holds.
    ///
    /// Like [`Self::replace`], resets the token estimation baseline after write-back (the
    /// true token count of the new prefix is unknown).
    fn splice_prefix(&self, drop_count: usize, summary: Message) -> usize;

    /// Number of messages currently held. Used to record a rollback boundary before a turn
    /// appends its prompt, so [`Self::truncate`] can undo it if the turn fails permanently.
    fn len(&self) -> usize;

    /// Returns whether the history holds no messages.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Truncates the message list to at most `len` messages, dropping any tail beyond it.
    /// A no-op when `len >= current length`.
    ///
    /// Used to roll back a permanently-failed turn: the user prompt (and any hook feedback)
    /// appended at the start of the turn must not linger in history once the turn errors
    /// out, otherwise it would be replayed on reload and re-sent to the model on the next
    /// request. Like [`Self::replace`], resets the token estimation baseline since the
    /// dropped messages may have contributed to the delta estimate.
    fn truncate(&self, len: usize);

    /// Records the actual input token count from the last LLM call
    /// (`input + cache_read + cache_creation`). Serves as the precise baseline for
    /// [`Self::token_estimate`]; subsequent [`Self::append`] messages are accumulated
    /// incrementally using a character heuristic.
    fn record_input_tokens(&self, tokens: u64);

    /// Estimates the token count for the current history. `None` indicates the history is
    /// empty or no estimate is available.
    fn token_estimate(&self) -> Option<u64>;
}

/// Compaction report. The token counts before and after compaction are wrapped into
/// [`AgentEvent::ContextCompressed`] by the main loop.
#[derive(Debug, Clone, Copy)]
pub struct CompactionReport {
    pub tokens_before: u64,
    pub tokens_after: u64,
}

/// Snapshot of the session's context usage, returned by [`Session::context_status`].
/// Powers the `/context` slash command (and any client-side context gauge).
#[derive(Debug, Clone, Copy)]
pub struct ContextStatus {
    /// Estimated tokens currently held in history. `None` when no estimate is available
    /// yet (e.g. an empty session before the first request).
    pub used_tokens: Option<u64>,
    /// The model's context window in tokens, if the provider exposes it.
    pub context_window: Option<u64>,
    /// Fraction of the window in use (`used / window`), only when both are known.
    pub ratio: Option<f64>,
}

/// Process-level agent error.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("invalid working directory: {0}")]
    InvalidCwd(PathBuf),

    /// MCP server failed to start (stdio process could not be launched / SSE connection
    /// could not be established).
    #[error("mcp startup failed for {server}: {source}")]
    McpStartup {
        server: String,
        #[source]
        source: BoxError,
    },

    /// The caller-provided [`SessionId`] already exists in the session table.
    /// A monotonic + timestamp ID generator should theoretically never collide; this is a
    /// safety net.
    #[error("session id already in use: {0}")]
    DuplicateSessionId(SessionId),

    #[error("session observer failed: {0}")]
    Observer(#[source] BoxError),

    #[error("session not found in storage: {0}")]
    SessionNotFound(SessionId),

    /// The `mode_id` received by `set_mode` is not in the session's mode directory (or
    /// the directory is not mounted).
    #[error("permission mode not found: {0}")]
    ModeNotFound(String),

    #[error("session restore failed: {0}")]
    Restore(#[source] BoxError),

    /// Session capability adjudication failed during startup. See [`SessionInitError`].
    #[error(transparent)]
    Init(#[from] SessionInitError),

    #[error(transparent)]
    Other(#[from] BoxError),
}

/// A one-time adjudication failure during session startup.
///
/// See capabilities design.
/// The session is refused when `capabilities.<name>.mode = "delegate"` but the current
/// provider's
/// [`crate::llm::LlmProvider::hosted_capabilities`] does not support that capability.
#[non_exhaustive]
#[derive(Debug)]
pub enum SessionInitError {
    /// The user explicitly chose `Delegate`, but the provider does not support the
    /// corresponding hosted capability.
    CapabilityUnsatisfied {
        /// The name of the problematic capability (e.g. `"web_search"`).
        capability: &'static str,
        /// The name of the provider bound to the current session.
        provider: String,
    },
}

impl std::fmt::Display for SessionInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityUnsatisfied {
                capability,
                provider,
            } => {
                writeln!(
                    f,
                    "{capability} capability is unsatisfied: provider `{provider}` does not support hosted {capability}."
                )?;
                writeln!(f)?;
                writeln!(f, "To fix this, choose one of:")?;
                writeln!(f, "  1. Disable hosted {capability} for this provider:")?;
                writeln!(f, "       [providers.{provider}.capabilities.{capability}]")?;
                writeln!(f, "       mode = \"disabled\"")?;
                writeln!(
                    f,
                    "  2. Change global default to `disabled` and only delegate where supported:"
                )?;
                writeln!(f, "       [capabilities.{capability}]")?;
                writeln!(f, "       mode = \"disabled\"")?;
                writeln!(
                    f,
                    "       [providers.<hosted-supported>.capabilities.{capability}]"
                )?;
                write!(f, "       mode = \"delegate\"")
            }
        }
    }
}

impl std::error::Error for SessionInitError {}

/// Reasons why a turn fails.
///
/// Rule of thumb: **only include errors that make the turn unable to continue**. Internal
/// tool failures within a turn, single LLM retry failures, etc. belong in [`AgentEvent`]
/// and the historical state machine instead.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TurnError {
    /// A turn is already in progress for this session.
    #[error("turn already in progress for this session")]
    TurnInProgress,

    /// Provider error that still fails after retries are exhausted.
    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// Internal invariant broken (should be a bug).
    #[error("internal turn error: {0}")]
    Internal(#[source] BoxError),
}

/// Abstraction for a tool registry.
///
/// Both the process-level registry (owned by [`AgentCore`], for built-in tools) and the
/// session-level registry (owned by [`Session`], for MCP tools) share the same shape; the
/// turn main loop looks up tools through the composite registry exposed by [`Session`].
pub trait ToolRegistry: Send + Sync {
    /// Return the schemas of all tools in the registry, used to populate the `tools`
    /// field of an LLM request.
    fn schemas(&self) -> Vec<ToolSchema>;

    /// Looks up a tool by name.
    fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
}
