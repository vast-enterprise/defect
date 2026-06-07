//! ACP shell (terminal) delegation e2e tests.
//!
//! 形态：与 `fs_delegation.rs` 同款——`Channel::duplex` 把 ACP server / client
//! 在进程内对接，server 跑 `defect_acp::serve_on`（注入 `ScriptedProvider` +
//! 一个含 `BashTool` 的 [`StaticToolRegistry`]），client 用一个声明 `terminal`
//! 能力的 builder 注册 `terminal/*` 反向请求 handler。
//!
//! LLM provider 的脚本由测试 case 自行配 `Round`：第 1 轮 emit tool_use
//! （指定 args JSON），第 2 轮 emit "done" 文本。
//!

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

// ---------- transport wrapper ----------

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
    /// 收到的 `terminal/create` 入参（按到达顺序）
    creates: Vec<CreateTerminalRequest>,
    /// `terminal/output` 的次数（无入参细节，只看出现频率）
    outputs: Vec<AcpTerminalId>,
    /// `terminal/wait_for_exit` 入参
    waits: Vec<AcpTerminalId>,
    /// `terminal/release` 入参
    releases: Vec<AcpTerminalId>,
    /// `terminal/kill` 入参
    kills: Vec<AcpTerminalId>,
    /// 收到的 SessionNotification 列表
    updates: Vec<SessionUpdate>,
}

type SharedObs = Arc<Mutex<ClientObservations>>;

/// 单个 terminal 的脚本：`output` / `wait_for_exit` 应该回什么。
#[derive(Clone, Default)]
struct TerminalScript {
    /// 累积输出文本。
    output: String,
    truncated: bool,
    /// 退出态——None 表示还没退出（output 端不上报；但 wait_for_exit 阻塞
    /// 行为不在 v0 测试里覆盖，这里若 None 则按 exit_code=0 兜底）。
    exit: Option<AcpTerminalExitStatus>,
    /// `wait_for_exit` 之前的等待时间（用于 case3 / case7 的"客户端晚回"路径）。
    wait_delay: Duration,
    /// 是否在收到 `kill` 后把 exit 改写成 SIGKILL。
    kill_marks_signal: bool,
    /// 让 create 直接返回 wire error（用于 case7）。
    create_error: Option<agent_client_protocol::Error>,
}

#[derive(Clone, Default)]
struct ClientScript {
    /// 命令字符串（CreateTerminalRequest.args[1]）→ TerminalScript
    by_command: HashMap<String, TerminalScript>,
    /// 兜底脚本，未在 `by_command` 命中的命令使用。
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

/// 进入 e2e 时由 client 端持有：每个 create 出来的 terminal 的当前态。
struct LiveTerminal {
    script: TerminalScript,
    /// 客户端在收到 kill 之后置位，wait_for_exit 据此选 SIGKILL 退出态。
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

/// 装配 fake client：所有 `terminal/*` handler 在这里挂上，跑 init →
/// session/new → 一段自定义 prompt 流程闭包，最终返回 stop reason。
///
/// 用宏而非函数：`Client.builder()` 链返回的中间类型未公开导出，函数签名
/// 写不出来，宏在调用点展开后所有类型都由编译器推断。
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
                        // 没显式声明就按正常 exit_code=0 收尾。
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

/// 跑 init→session/new→prompt 路径，返回 stop reason 与 client 观察记录。
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

// ---------- cases ----------

/// #1 委托模式 + bash echo → terminal/create + wait_for_exit + output 全命中
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

/// #2 命令以非零退出 → tool 仍然 Completed（不是 Failed），exit code 进 raw output
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

/// #3 超时 → agent 发 terminal/kill；tool Completed 含 timeout 标记
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
            // 客户端按"长跑"模拟：wait_for_exit 慢回，让 agent 端先撞 timeout
            // 触发 kill。kill 来后客户端在 wait 路径里把 exit 改成 SIGKILL。
            wait_delay: Duration::from_millis(500),
            kill_marks_signal: true,
            ..TerminalScript::default()
        },
    );
    let script = ClientScript {
        by_command: by_cmd,
        default: TerminalScript::default(),
    };

    // 8s 兜底：cancel/timeout 路径如果再次回归"在 select 里 drop wait_fut →
    // server 撕连接"的死锁，让测试快速失败而不是耗尽 cargo 的默认 60s。
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

/// #4 turn 中途 cancel → agent 发 kill（不 hang）
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

/// #5 client 没声明 terminal 能力 → 退回 LocalShellBackend，**不**发反向请求
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

    // client_caps 没置 terminal=true → ShellMode::Local
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

/// #6 委托模式下 workdir 越界 → agent 自己拦下，**不**发 terminal/create
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

/// #7 委托模式下 client 在 create 上返回 wire error → 工具 Failed，turn 继续
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
