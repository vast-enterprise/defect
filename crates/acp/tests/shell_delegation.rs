//! ACP shell (terminal) delegation e2e tests.
//!
//! Same pattern as `fs_delegation.rs`: `Channel::duplex` connects the ACP server and
//! client in-process. The server runs `defect_acp::serve_on` (injecting a
//! `ScriptedProvider` plus a [`StaticToolRegistry`] containing `BashTool`). The client
//! uses a builder that declares `terminal` capability and registers a reverse-request
//! handler for `terminal/*`.
//!
//! The LLM provider script is configured per test case via `Round`: round 1 emits a
//! `tool_use` (with the specified args JSON), round 2 emits the text `"done"`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::{
    ClientCapabilities, ContentBlock, CreateTerminalRequest, CreateTerminalResponse,
    InitializeRequest, KillTerminalRequest, KillTerminalResponse, NewSessionRequest, PromptRequest,
    ProtocolVersion, ReleaseTerminalRequest, ReleaseTerminalResponse, SessionNotification,
    SessionUpdate, StopReason as AcpStopReason, TerminalExitStatus as AcpTerminalExitStatus,
    TerminalId as AcpTerminalId, TerminalOutputRequest, TerminalOutputResponse, TextContent,
    ToolCallStatus, WaitForTerminalExitRequest, WaitForTerminalExitResponse,
};
use agent_client_protocol::{Agent, Channel, Client, ConnectTo, ConnectionTo, Role};
use defect_acp::serve_on;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderInfo, ProviderStream, StopReason as LlmStopReason,
    ThinkingEcho,
};
use defect_agent::policy::{OpenPolicy, SandboxPolicy};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, StaticToolRegistry, ToolRegistry, TurnConfig,
};
use defect_tools::BashTool;
use futures::future::BoxFuture;
use futures::stream;
use serde_json::json;
use tokio_util::sync::CancellationToken;

// Transport wrapper

struct ChannelTransport<R: Role> {
    inner: Channel,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Role> ChannelTransport<R> {
    fn new(inner: Channel) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<R: Role> ConnectTo<R> for ChannelTransport<R> {
    async fn connect_to(
        self,
        client: impl ConnectTo<R::Counterpart>,
    ) -> Result<(), agent_client_protocol::Error> {
        <Channel as ConnectTo<R>>::connect_to(self.inner, client).await
    }

    fn into_channel_and_future(
        self,
    ) -> (
        Channel,
        agent_client_protocol::BoxFuture<'static, Result<(), agent_client_protocol::Error>>,
    ) {
        <Channel as ConnectTo<R>>::into_channel_and_future(self.inner)
    }
}

// ---------- programmable LLM provider ----------

#[derive(Clone)]
enum Round {
    ToolUse {
        id: String,
        name: String,
        args_json: String,
    },
    EndTurn {
        text: String,
    },
}

struct ScriptedProvider {
    rounds: Mutex<Vec<Round>>,
    cursor: Mutex<usize>,
}

impl ScriptedProvider {
    fn new(rounds: Vec<Round>) -> Self {
        Self {
            rounds: Mutex::new(rounds),
            cursor: Mutex::new(0),
        }
    }
}

impl LlmProvider for ScriptedProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "scripted".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Scripted".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tool_calls: FeatureSupport::Supported,
            parallel_tool_calls: FeatureSupport::Supported,
            thinking: FeatureSupport::Unsupported,
            vision: FeatureSupport::Unsupported,
            prompt_cache: FeatureSupport::Unsupported,
            thinking_echo: ThinkingEcho::Forbidden,
        }
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn model_info(&self, _id: &str) -> Option<ModelInfo> {
        None
    }

    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let round = {
            let mut cursor = self.cursor.lock().expect("cursor lock");
            let rounds = self.rounds.lock().expect("rounds lock");
            let idx = *cursor;
            *cursor = (*cursor + 1).min(rounds.len());
            rounds.get(idx).cloned()
        };
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = match round {
                Some(Round::ToolUse {
                    id,
                    name,
                    args_json,
                }) => vec![
                    Ok(ProviderChunk::MessageStart {
                        id: format!("msg-{id}"),
                        model: "scripted".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseStart {
                        id: id.clone(),
                        name,
                    }),
                    Ok(ProviderChunk::ToolUseArgsDelta {
                        id: id.clone(),
                        fragment: args_json,
                    }),
                    Ok(ProviderChunk::ToolUseEnd { id }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::ToolUse,
                    }),
                ],
                Some(Round::EndTurn { text }) => vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "msg-end".to_string(),
                        model: "scripted".to_string(),
                    }),
                    Ok(ProviderChunk::TextDelta { text }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ],
                None => vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "msg-end".to_string(),
                        model: "scripted".to_string(),
                    }),
                    Ok(ProviderChunk::TextDelta {
                        text: "done".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ],
            };
            let s: Pin<
                Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
            > = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

// ---------- harness ----------

#[derive(Default)]
struct ClientObservations {
    /// Received `terminal/create` parameters (in arrival order)
    creates: Vec<CreateTerminalRequest>,
    /// Counts of `terminal/output` (no argument details, only frequency)
    outputs: Vec<AcpTerminalId>,
    /// `terminal/wait_for_exit` arguments
    waits: Vec<AcpTerminalId>,
    /// Received `terminal/release` arguments
    releases: Vec<AcpTerminalId>,
    /// The `terminal/kill` argument
    kills: Vec<AcpTerminalId>,
    /// List of received `SessionNotification`s
    updates: Vec<SessionUpdate>,
}

type SharedObs = Arc<Mutex<ClientObservations>>;

/// Script for a single terminal: what `output` / `wait_for_exit` should return.
#[derive(Clone, Default)]
struct TerminalScript {
    /// Accumulated output text.
    output: String,
    truncated: bool,
    /// Exit status — `None` means the terminal has not exited yet (the output side does
    /// not report it; however, the blocking behavior of `wait_for_exit` is not covered by
    /// v0 tests, so if `None` we fall back to `exit_code=0`).
    exit: Option<AcpTerminalExitStatus>,
    /// Wait time before `wait_for_exit` (used for the "client returns late" path in case3
    /// / case7).
    wait_delay: Duration,
    /// Whether to rewrite the exit status as SIGKILL after receiving a `kill`.
    kill_marks_signal: bool,
    /// Causes `create` to return a wire error directly (used for case7).
    create_error: Option<agent_client_protocol::Error>,
}

#[derive(Clone, Default)]
struct ClientScript {
    /// Maps command strings (`CreateTerminalRequest.args[1]`) to `TerminalScript`.
    by_command: HashMap<String, TerminalScript>,
    /// Fallback script used for commands not found in `by_command`.
    default: TerminalScript,
}

impl ClientScript {
    fn for_command(&self, command: &str) -> TerminalScript {
        self.by_command
            .get(command)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
}

/// Held by the client during e2e: the current state of each created terminal.
struct LiveTerminal {
    script: TerminalScript,
    /// Set by the client after receiving a kill; `wait_for_exit` uses this to select the
    /// SIGKILL exit state.
    killed: bool,
}

type LiveTerminals = Arc<Mutex<HashMap<String, LiveTerminal>>>;

fn build_server(rounds: Vec<Round>) -> Arc<dyn AgentCore> {
    let provider = Arc::new(ScriptedProvider::new(rounds));
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(BashTool::new()))
            .build(),
    );
    let core = DefaultAgentCore::builder()
        .provider(provider)
        .process_tools(tools)
        .policy(Arc::new(OpenPolicy) as Arc<dyn SandboxPolicy>)
        .config(TurnConfig {
            model: "scripted".to_string(),
            ..TurnConfig::default()
        })
        .build();
    Arc::new(core)
}

fn full_terminal_caps() -> ClientCapabilities {
    ClientCapabilities::new().terminal(true)
}

/// Assembles a fake client: all `terminal/*` handlers are wired here, runs init →
/// session/new → a custom prompt flow closure, and finally returns a stop reason.
///
/// A macro is used instead of a function because the intermediate type returned by
/// `Client.builder()` chaining is not publicly exported, making the function signature
/// impossible to write. The macro expands at the call site, letting the compiler infer
/// all types.
macro_rules! build_and_run_client {
    (
        obs = $obs:expr,
        live = $live:expr,
        counter = $counter:expr,
        script = $script:expr,
        channel = $channel:expr,
        flow = $flow:expr $(,)?
    ) => {{
        let obs_for_notif = $obs.clone();
        let obs_for_create = $obs.clone();
        let obs_for_output = $obs.clone();
        let obs_for_wait = $obs.clone();
        let obs_for_release = $obs.clone();
        let obs_for_kill = $obs.clone();
        let live_for_create = $live.clone();
        let live_for_output = $live.clone();
        let live_for_wait = $live.clone();
        let live_for_release = $live.clone();
        let live_for_kill = $live.clone();
        let counter_for_create = $counter;
        let script_arc = ::std::sync::Arc::new($script);
        let script_for_create = script_arc.clone();

        Client
            .builder()
            .name("shell-delegation-test")
            .on_receive_notification(
                async move |notif: SessionNotification, _cx| {
                    obs_for_notif
                        .lock()
                        .expect("obs lock")
                        .updates
                        .push(notif.update);
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                async move |req: CreateTerminalRequest, responder, _cx| {
                    let command = req
                        .args
                        .get(1)
                        .cloned()
                        .unwrap_or_else(|| req.command.clone());
                    obs_for_create
                        .lock()
                        .expect("obs lock")
                        .creates
                        .push(req.clone());
                    let term_script = script_for_create.for_command(&command);

                    if let Some(err) = term_script.create_error.clone() {
                        return responder.respond_with_error(err);
                    }

                    let n = counter_for_create.fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
                    let id_str = format!("term-{n}");
                    live_for_create.lock().expect("live lock").insert(
                        id_str.clone(),
                        LiveTerminal {
                            script: term_script,
                            killed: false,
                        },
                    );
                    responder.respond(CreateTerminalResponse::new(AcpTerminalId::new(
                        id_str.as_str(),
                    )))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |req: TerminalOutputRequest, responder, _cx| {
                    obs_for_output
                        .lock()
                        .expect("obs lock")
                        .outputs
                        .push(req.terminal_id.clone());
                    let key = req.terminal_id.0.to_string();
                    let snapshot = live_for_output
                        .lock()
                        .expect("live lock")
                        .get(&key)
                        .map(|t| {
                            (
                                t.script.output.clone(),
                                t.script.truncated,
                                t.script.exit.clone(),
                            )
                        });
                    let (text, truncated, exit) =
                        snapshot.unwrap_or_else(|| (String::new(), false, None));
                    let resp = TerminalOutputResponse::new(text, truncated).exit_status(exit);
                    responder.respond(resp)
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |req: WaitForTerminalExitRequest, responder, _cx| {
                    obs_for_wait
                        .lock()
                        .expect("obs lock")
                        .waits
                        .push(req.terminal_id.clone());
                    let key = req.terminal_id.0.to_string();
                    let (delay, exit, killed, kill_marks_signal) = {
                        let guard = live_for_wait.lock().expect("live lock");
                        match guard.get(&key) {
                            Some(t) => (
                                t.script.wait_delay,
                                t.script.exit.clone(),
                                t.killed,
                                t.script.kill_marks_signal,
                            ),
                            None => (Duration::ZERO, None, false, false),
                        }
                    };
                    if delay > Duration::ZERO {
                        tokio::time::sleep(delay).await;
                    }
                    let exit = if killed && kill_marks_signal {
                        AcpTerminalExitStatus::new()
                            .exit_code(None)
                            .signal(Some("SIGKILL".to_string()))
                    } else if let Some(e) = exit {
                        e
                    } else {
                        // If not explicitly declared, default to a normal exit code of 0.
                        AcpTerminalExitStatus::new().exit_code(Some(0_u32))
                    };
                    responder.respond(WaitForTerminalExitResponse::new(exit))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |req: ReleaseTerminalRequest, responder, _cx| {
                    obs_for_release
                        .lock()
                        .expect("obs lock")
                        .releases
                        .push(req.terminal_id.clone());
                    let key = req.terminal_id.0.to_string();
                    live_for_release.lock().expect("live lock").remove(&key);
                    responder.respond(ReleaseTerminalResponse::default())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |req: KillTerminalRequest, responder, _cx| {
                    obs_for_kill
                        .lock()
                        .expect("obs lock")
                        .kills
                        .push(req.terminal_id.clone());
                    let key = req.terminal_id.0.to_string();
                    if let Some(t) = live_for_kill.lock().expect("live lock").get_mut(&key) {
                        t.killed = true;
                    }
                    responder.respond(KillTerminalResponse::default())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(ChannelTransport::<Client>::new($channel), $flow)
            .await
            .expect("client connection completed")
    }};
}

/// Runs the init → session/new → prompt path, returning the stop reason and client
/// observations.
async fn run_e2e(
    cwd: PathBuf,
    rounds: Vec<Round>,
    client_caps: ClientCapabilities,
    script: ClientScript,
) -> (AcpStopReason, ClientObservations) {
    let agent_core = build_server(rounds);
    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let obs: SharedObs = Arc::new(Mutex::new(ClientObservations::default()));
    let live: LiveTerminals = Arc::new(Mutex::new(HashMap::new()));
    let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let stop = build_and_run_client! {
        obs = obs,
        live = live,
        counter = counter,
        script = script,
        channel = channel_a,
        flow = async move |cx: ConnectionTo<Agent>| {
            cx.send_request(
                InitializeRequest::new(ProtocolVersion::V1).client_capabilities(client_caps),
            )
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let prompt_resp = cx
                .send_request(PromptRequest::new(
                    new_session.session_id,
                    vec![ContentBlock::Text(TextContent::new("go"))],
                ))
                .block_task()
                .await?;
            Ok(prompt_resp.stop_reason)
        },
    };

    server_handle.abort();
    let _ = server_handle.await;

    let final_obs = std::mem::take(&mut *obs.lock().expect("obs lock"));
    (stop, final_obs)
}

// ========== test cases ==========

/// #1 delegated mode + bash echo → terminal/create + wait_for_exit + output all hit
#[tokio::test]
async fn case1_delegated_bash_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "echo hello"}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    let mut by_cmd = HashMap::new();
    by_cmd.insert(
        "echo hello".to_string(),
        TerminalScript {
            output: "hello\n".to_string(),
            exit: Some(AcpTerminalExitStatus::new().exit_code(Some(0_u32))),
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    let (stop, obs) = run_e2e(cwd, rounds, full_terminal_caps(), script).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.creates.len(), 1, "expected one terminal/create");
    let create = &obs.creates[0];
    assert_eq!(create.command, "/bin/sh");
    assert_eq!(
        create.args,
        vec!["-c".to_string(), "echo hello".to_string()]
    );
    assert!(create.cwd.is_some(), "create.cwd must be set");
    assert_eq!(obs.waits.len(), 1, "expected one wait_for_exit");
    assert!(!obs.outputs.is_empty(), "expected at least one output");
    assert!(!obs.releases.is_empty(), "release should be called");

    let any_completed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Completed)));
    assert!(
        any_completed,
        "expected ToolCallUpdate Completed; updates={:?}",
        obs.updates
    );
}

/// #2 command exits with non-zero → tool is still Completed (not Failed), exit code goes
/// into raw output
#[tokio::test]
async fn case2_nonzero_exit_yields_completed_with_marker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "false"}).to_string(),
        },
        Round::EndTurn { text: "ack".into() },
    ];

    let mut by_cmd = HashMap::new();
    by_cmd.insert(
        "false".to_string(),
        TerminalScript {
            output: String::new(),
            exit: Some(AcpTerminalExitStatus::new().exit_code(Some(7_u32))),
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    let (stop, obs) = run_e2e(cwd, rounds, full_terminal_caps(), script).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    let any_completed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Completed)));
    assert!(
        any_completed,
        "non-zero exit should still finish as Completed; updates={:?}",
        obs.updates
    );
    let any_failed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Failed)));
    assert!(!any_failed, "should not be Failed for non-zero exit");
}

/// #3 timeout → agent sends terminal/kill; tool Completed includes timeout flag
#[tokio::test]
async fn case3_timeout_invokes_kill() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "sleep 10", "timeout_ms": 100}).to_string(),
        },
        Round::EndTurn {
            text: "ack timeout".into(),
        },
    ];

    let mut by_cmd = HashMap::new();
    by_cmd.insert(
        "sleep 10".to_string(),
        TerminalScript {
            // Simulates a "long run" scenario: `wait_for_exit` responds slowly so the
            // agent side hits the timeout first and triggers a kill. When the kill
            // arrives, the client changes the exit status to `SIGKILL` in the wait path.
            wait_delay: Duration::from_millis(500),
            kill_marks_signal: true,
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    // 8s safety net: if the cancel/timeout path regresses into the deadlock where
    // dropping `wait_fut` inside `select` causes the server to tear down the connection,
    // make the test fail fast instead of exhausting cargo's default 60s timeout.
    let fut = run_e2e(cwd, rounds, full_terminal_caps(), script);
    let (stop, obs) = match tokio::time::timeout(Duration::from_secs(8), fut).await {
        Ok(v) => v,
        Err(_) => panic!("case3 timed out after 8s — likely deadlock in dispatch loop"),
    };

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.creates.len(), 1);
    assert_eq!(obs.kills.len(), 1, "agent should have sent terminal/kill");
    let any_completed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Completed)));
    assert!(
        any_completed,
        "tool should reach Completed (with timeout marker)"
    );
}

/// #4: cancel mid-turn → agent sends kill (no hang)
#[tokio::test]
async fn case4_cancel_invokes_kill() {
    use agent_client_protocol::schema::CancelNotification;

    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "sleep 10"}).to_string(),
        },
        Round::EndTurn {
            text: "should not reach".into(),
        },
    ];

    let mut by_cmd = HashMap::new();
    by_cmd.insert(
        "sleep 10".to_string(),
        TerminalScript {
            wait_delay: Duration::from_secs(2),
            kill_marks_signal: true,
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    let agent_core = build_server(rounds);
    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let obs: SharedObs = Arc::new(Mutex::new(ClientObservations::default()));
    let live: LiveTerminals = Arc::new(Mutex::new(HashMap::new()));
    let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let stop = build_and_run_client! {
        obs = obs,
        live = live,
        counter = counter,
        script = script,
        channel = channel_a,
        flow = async move |cx: ConnectionTo<Agent>| {
            cx.send_request(
                InitializeRequest::new(ProtocolVersion::V1)
                    .client_capabilities(full_terminal_caps()),
            )
            .block_task()
            .await?;
            let new_session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let session_id = new_session.session_id.clone();
            let cx_for_cancel = cx.clone();
            let cancel_task = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(80)).await;
                let _ = cx_for_cancel.send_notification(CancelNotification::new(session_id));
            });
            let prompt_resp = cx
                .send_request(PromptRequest::new(
                    new_session.session_id,
                    vec![ContentBlock::Text(TextContent::new("go"))],
                ))
                .block_task()
                .await?;
            let _ = cancel_task.await;
            Ok(prompt_resp.stop_reason)
        },
    };

    server_handle.abort();
    let _ = server_handle.await;

    let obs = std::mem::take(&mut *obs.lock().expect("obs lock"));

    assert!(
        matches!(stop, AcpStopReason::Cancelled | AcpStopReason::EndTurn),
        "stop should be Cancelled or EndTurn, got {stop:?}"
    );
    assert_eq!(obs.creates.len(), 1, "create should still be issued");
    assert!(
        !obs.kills.is_empty(),
        "agent should have sent terminal/kill on cancel"
    );
}

/// #5 client does not declare terminal capability → falls back to LocalShellBackend, does
/// **not** send reverse request
#[tokio::test]
async fn case5_no_terminal_caps_falls_back_to_local() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "echo from_local"}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    // client_caps without terminal=true → ShellMode::Local
    let (stop, obs) = run_e2e(
        cwd,
        rounds,
        ClientCapabilities::new(),
        ClientScript::default(),
    )
    .await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(
        obs.creates.is_empty(),
        "client should not see terminal/create"
    );
    assert!(obs.outputs.is_empty());
    assert!(obs.waits.is_empty());
    assert!(obs.kills.is_empty());
    assert!(obs.releases.is_empty());
}

/// #6 Delegation mode: workdir escape blocked by agent, no terminal/create sent
#[tokio::test]
async fn case6_workdir_escape_blocked_before_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "pwd", "workdir": "../../../etc"}).to_string(),
        },
        Round::EndTurn { text: "ack".into() },
    ];

    let (stop, obs) = run_e2e(cwd, rounds, full_terminal_caps(), ClientScript::default()).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(
        obs.creates.is_empty(),
        "boundary should block before terminal/create"
    );
    let saw_failed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Failed)));
    assert!(
        saw_failed,
        "expected tool Failed; updates={:?}",
        obs.updates
    );
}

/// #7 Delegation mode: client returns wire error on create → tool Failed, turn continues
#[tokio::test]
async fn case7_client_create_error_marks_tool_failed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = std::fs::canonicalize(dir.path()).expect("canon cwd");

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "bash".into(),
            args_json: json!({"command": "echo blocked"}).to_string(),
        },
        Round::EndTurn {
            text: "ack failure".into(),
        },
    ];

    let mut by_cmd = HashMap::new();
    by_cmd.insert(
        "echo blocked".to_string(),
        TerminalScript {
            create_error: Some(agent_client_protocol::Error::new(-32000, "client says no")),
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    let (stop, obs) = run_e2e(cwd, rounds, full_terminal_caps(), script).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.creates.len(), 1, "create was attempted");
    assert!(obs.outputs.is_empty(), "no output after failed create");
    let saw_failed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Failed)));
    assert!(
        saw_failed,
        "expected tool Failed when create errors; updates={:?}",
        obs.updates
    );
}
