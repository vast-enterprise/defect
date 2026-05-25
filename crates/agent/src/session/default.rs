//! [`Session`] / [`AgentCore`] 的 v0 默认实现。
//!
//! 装配关系：
//!
//! ```text
//! DefaultAgentCore
//!   ├── Arc<dyn LlmProvider>          (装配时传入，所有 session 共享)
//!   ├── Arc<dyn ToolRegistry>         (进程级内置工具)
//!   ├── TurnConfig                    (默认配置)
//!   └── DashMap<SessionId, Arc<dyn Session>>
//!
//! DefaultSession
//!   ├── id: SessionId
//!   ├── cwd: PathBuf
//!   ├── history: Box<dyn History>
//!   ├── tools:   Arc<dyn ToolRegistry>   (CompositeRegistry: per-session + process)
//!   ├── provider: Arc<dyn LlmProvider>
//!   ├── events:   Arc<EventEmitter>
//!   ├── permissions: Arc<PermissionGate>
//!   ├── turn_lock: tokio::sync::Mutex<TurnSlot>
//!   └── config: TurnConfig
//! ```
//!
//! turn 互斥用 `Mutex<TurnSlot>`：`run_turn` 在最外层 `try_lock`，失败即返回
//! `TurnError::TurnInProgress`。`TurnSlot` 内部存当前 turn 的
//! [`CancellationToken`]，`cancel_turn` 取出后 `cancel()`。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{ContentBlock, McpServer, SessionId, StopReason, ToolCallId};
use dashmap::DashMap;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::event::PermissionResolution;
use crate::fs::FsBackend;
use crate::llm::LlmProvider;
use crate::policy::{AskWritesPolicy, SandboxPolicy};
use crate::session::events::EventEmitter;
use crate::session::permissions::PermissionGate;
use crate::session::tool_registry::{CompositeRegistry, StaticToolRegistry};
use crate::session::turn::{TurnConfig, TurnRunner};
use crate::session::{
    AgentCore, AgentError, EventStream, History, Session, ToolRegistry, TurnError, VecHistory,
};

/// 默认 [`AgentCore`]。
pub struct DefaultAgentCore {
    provider: Arc<dyn LlmProvider>,
    process_tools: Arc<dyn ToolRegistry>,
    policy: Arc<dyn SandboxPolicy>,
    config: TurnConfig,
    sessions: DashMap<SessionId, Arc<dyn Session>>,
}

impl DefaultAgentCore {
    pub fn builder() -> DefaultAgentCoreBuilder {
        DefaultAgentCoreBuilder::default()
    }
}

#[derive(Default)]
pub struct DefaultAgentCoreBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    process_tools: Option<Arc<dyn ToolRegistry>>,
    policy: Option<Arc<dyn SandboxPolicy>>,
    config: TurnConfig,
}

impl DefaultAgentCoreBuilder {
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn process_tools(mut self, tools: Arc<dyn ToolRegistry>) -> Self {
        self.process_tools = Some(tools);
        self
    }

    pub fn policy(mut self, policy: Arc<dyn SandboxPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn config(mut self, config: TurnConfig) -> Self {
        self.config = config;
        self
    }

    /// # Panics
    /// 如果 `provider` 没有设置。
    pub fn build(self) -> DefaultAgentCore {
        DefaultAgentCore {
            provider: self.provider.expect("DefaultAgentCore requires a provider"),
            process_tools: self
                .process_tools
                .unwrap_or_else(|| Arc::new(StaticToolRegistry::empty()) as Arc<dyn ToolRegistry>),
            policy: self
                .policy
                .unwrap_or_else(|| Arc::new(AskWritesPolicy::new()) as Arc<dyn SandboxPolicy>),
            config: self.config,
            sessions: DashMap::new(),
        }
    }
}

impl AgentCore for DefaultAgentCore {
    fn create_session(
        &self,
        id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        fs: Arc<dyn FsBackend>,
    ) -> BoxFuture<'_, Result<Arc<dyn Session>, AgentError>> {
        Box::pin(async move {
            if !cwd.is_absolute() || !cwd.exists() {
                return Err(AgentError::InvalidCwd(cwd));
            }
            // v0 不实际拉起 MCP server——defect-mcp 接入后这里会变成
            // "为每个 server 起 client、把它的工具注册成 per-session"。
            if !mcp_servers.is_empty() {
                tracing::warn!(
                    count = mcp_servers.len(),
                    "DefaultAgentCore: MCP servers requested but mcp adapter not wired; ignoring"
                );
            }

            if self.sessions.contains_key(&id) {
                return Err(AgentError::DuplicateSessionId(id));
            }

            let session_tools: Arc<dyn ToolRegistry> = Arc::new(StaticToolRegistry::empty());
            let composite: Arc<dyn ToolRegistry> = Arc::new(CompositeRegistry::new(
                session_tools,
                self.process_tools.clone(),
            ));

            let session = Arc::new(DefaultSession {
                id: id.clone(),
                cwd,
                history: Box::new(VecHistory::new()) as Box<dyn History>,
                tools: composite,
                provider: self.provider.clone(),
                policy: self.policy.clone(),
                events: Arc::new(EventEmitter::new()),
                permissions: Arc::new(PermissionGate::new()),
                turn_state: Mutex::new(TurnSlot::default()),
                config: self.config.clone(),
                fs,
            }) as Arc<dyn Session>;

            self.sessions.insert(id, session.clone());
            Ok(session)
        })
    }

    fn session(&self, id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.get(id).map(|r| r.value().clone())
    }
}

pub struct DefaultSession {
    id: SessionId,
    cwd: PathBuf,
    history: Box<dyn History>,
    tools: Arc<dyn ToolRegistry>,
    provider: Arc<dyn LlmProvider>,
    policy: Arc<dyn SandboxPolicy>,
    events: Arc<EventEmitter>,
    permissions: Arc<PermissionGate>,
    /// 单 turn 互斥 + cancel 通道。`Some(token)` 表示有 turn 在跑；
    /// `None` 表示空闲。`std::sync::Mutex` 仅短暂持锁、不跨 await。
    turn_state: Mutex<TurnSlot>,
    config: TurnConfig,
    /// session 级 fs 后端。由 [`AgentCore::create_session`] 注入；
    /// `TurnRunner` 把 `&dyn FsBackend` 借到 [`crate::tool::ToolContext`] 传给工具。
    fs: Arc<dyn FsBackend>,
}

#[derive(Default)]
struct TurnSlot {
    cancel: Option<CancellationToken>,
}

/// `run_turn` 的"占位 / 释放" guard：构造时占用 turn slot、drop 时释放。
struct TurnGuard<'a> {
    state: &'a Mutex<TurnSlot>,
}

impl<'a> Drop for TurnGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.state.lock() {
            slot.cancel = None;
        }
    }
}

impl Session for DefaultSession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn subscribe(&self) -> EventStream {
        self.events.subscribe()
    }

    fn run_turn(&self, prompt: Vec<ContentBlock>) -> BoxFuture<'_, Result<StopReason, TurnError>> {
        // 整个 turn 包在一个 span 里——LLM 调用 / 工具调用 / 权限请求 都
        // 自动挂成子 span，排障时一棵 trace 走到底。session_id 截短，避免
        // 输出整段 uuid 噪音。详见 docs/outbound/tracing.md §2.2。
        let span = tracing::info_span!(
            "turn",
            session_id = %short_id(self.id.0.as_ref()),
            model = %self.config.model,
        );
        Box::pin(
            async move {
                let cancel = {
                    let mut slot = self
                        .turn_state
                        .lock()
                        .expect("DefaultSession turn_state mutex poisoned");
                    if slot.cancel.is_some() {
                        return Err(TurnError::TurnInProgress);
                    }
                    let cancel = CancellationToken::new();
                    slot.cancel = Some(cancel.clone());
                    cancel
                };

                // RAII：函数任意路径退出（含 await 内 panic）都释放 slot
                let _guard = TurnGuard {
                    state: &self.turn_state,
                };

                let runner = TurnRunner {
                    history: self.history.as_ref(),
                    tools: self.tools.as_ref(),
                    provider: self.provider.as_ref(),
                    policy: self.policy.as_ref(),
                    events: self.events.clone(),
                    permissions: self.permissions.as_ref(),
                    cancel,
                    config: &self.config,
                    cwd: &self.cwd,
                    fs: self.fs.clone(),
                };

                runner.run(prompt).await
            }
            .instrument(span),
        )
    }

    fn cancel_turn(&self) {
        let token = {
            let slot = self
                .turn_state
                .lock()
                .expect("DefaultSession turn_state mutex poisoned");
            slot.cancel.clone()
        };
        if let Some(token) = token {
            token.cancel();
        }
        // 没 turn 在跑 → no-op（幂等）
    }

    fn resolve_permission(&self, id: ToolCallId, outcome: PermissionResolution) {
        self.permissions.resolve(&id, outcome);
    }
}

/// v0 的 session id 生成：进程内单调递增 + 时间戳。引入 uuid crate 时再换。
///
/// `defect-acp` 的 `session/new` handler 在调用
/// [`AgentCore::create_session`] 之前需要 `SessionId`（用于构造
/// `AcpFsBackend`，详见 `docs/inbound/acp-fs.md` §3.2）；这个函数对外公开，
/// 让 acp / 测试都能拿到一致格式的 id。
pub fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("session-{ts:x}-{n:x}")
}

/// 给 tracing span 用的 session id 短形：按字符取前 12 个。仅诊断用。
fn short_id(s: &str) -> &str {
    match s.char_indices().nth(12) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
