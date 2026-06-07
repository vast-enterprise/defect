//! ACP filesystem delegation e2e tests.
//!
//! Architecture: within the process, `Channel::duplex` connects the ACP server and
//! client. The server runs `defect_acp::serve_on` (injecting a `ScriptedProvider` plus a
//! [`StaticToolRegistry`] of three fs tools). The client uses a builder that declares fs
//! capabilities to register reverse-request handlers for `fs/read_text_file` and
//! `fs/write_text_file`.
//!
//! The LLM provider script is configured per test case via [`Round`]:
//! - Round 1: emit `tool_use` (specifying tool name + args JSON), `Stop=ToolUse`
//! - Round 2: emit `"done"` text, `Stop=EndTurn`

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    ClientCapabilities, ContentBlock, FileSystemCapabilities, InitializeRequest, NewSessionRequest,
    PromptRequest, ProtocolVersion, ReadTextFileRequest, ReadTextFileResponse, SessionNotification,
    SessionUpdate, StopReason as AcpStopReason, TextContent, ToolCallStatus, WriteTextFileRequest,
    WriteTextFileResponse,
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
use defect_tools::{EditFileTool, ReadFileTool, WriteFileTool};
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
            let mut cursor = self.cursor.lock().unwrap();
            let rounds = self.rounds.lock().unwrap();
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
    reads: Vec<PathBuf>,
    writes: Vec<(PathBuf, String)>,
    updates: Vec<SessionUpdate>,
}

type SharedObs = Arc<Mutex<ClientObservations>>;

/// Optional body for a fake-client reverse-request handler; returning `None` means the
/// case is expected never to be invoked (and will panic if it is).
type ReadFn = dyn Fn(
        &ReadTextFileRequest,
        &SharedObs,
    ) -> Result<ReadTextFileResponse, agent_client_protocol::Error>
    + Send
    + Sync;
type WriteFn = dyn Fn(
        &WriteTextFileRequest,
        &SharedObs,
    ) -> Result<WriteTextFileResponse, agent_client_protocol::Error>
    + Send
    + Sync;

fn build_server(rounds: Vec<Round>) -> Arc<dyn AgentCore> {
    let provider = Arc::new(ScriptedProvider::new(rounds));
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(ReadFileTool::new()))
            .insert(Arc::new(WriteFileTool::new()))
            .insert(Arc::new(EditFileTool::new()))
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

/// Run an init → session/new → prompt flow, returning the stop reason and client-side
/// observations.
async fn run_e2e(
    cwd: PathBuf,
    rounds: Vec<Round>,
    client_caps: ClientCapabilities,
    read: Option<Arc<ReadFn>>,
    write: Option<Arc<WriteFn>>,
) -> (AcpStopReason, ClientObservations) {
    let agent_core = build_server(rounds);
    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let obs: SharedObs = Arc::new(Mutex::new(ClientObservations::default()));
    let obs_for_notif = obs.clone();
    let obs_for_read = obs.clone();
    let obs_for_write = obs.clone();
    let read_fn = read;
    let write_fn = write;

    let stop = Client
        .builder()
        .name("fs-delegation-test")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                obs_for_notif.lock().unwrap().updates.push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: ReadTextFileRequest, responder, _cx| {
                let path = req.path.clone();
                let f = read_fn
                    .as_ref()
                    .unwrap_or_else(|| panic!("did not expect fs/read_text_file; got {path:?}"))
                    .clone();
                let res = f(&req, &obs_for_read);
                obs_for_read.lock().unwrap().reads.push(path);
                match res {
                    Ok(resp) => responder.respond(resp),
                    Err(err) => responder.respond_with_error(err),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: WriteTextFileRequest, responder, _cx| {
                let path = req.path.clone();
                let content = req.content.clone();
                let f = write_fn
                    .as_ref()
                    .unwrap_or_else(|| panic!("did not expect fs/write_text_file; got {path:?}"))
                    .clone();
                let res = f(&req, &obs_for_write);
                obs_for_write.lock().unwrap().writes.push((path, content));
                match res {
                    Ok(resp) => responder.respond(resp),
                    Err(err) => responder.respond_with_error(err),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx: ConnectionTo<Agent>| {
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
        )
        .await
        .expect("client connection completed");

    server_handle.abort();
    let _ = server_handle.await;

    let final_obs = std::mem::take(&mut *obs.lock().unwrap());
    (stop, final_obs)
}

fn full_fs_caps() -> ClientCapabilities {
    ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(true)
        .write_text_file(true))
}

fn read_only_fs_caps() -> ClientCapabilities {
    ClientCapabilities::new().fs(FileSystemCapabilities::new()
        .read_text_file(true)
        .write_text_file(false))
}

// ---------- cases ----------

/// #1 Delegation pattern + read_file → fs/read_text_file reverse request is hit
#[tokio::test]
async fn case1_delegated_read_file_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("hello.txt");
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": target_str}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    let read: Arc<ReadFn> =
        Arc::new(|_req, _obs| Ok(ReadTextFileResponse::new("alpha\nbeta\n".to_string())));

    let (stop, obs) = run_e2e(cwd, rounds, full_fs_caps(), Some(read), None).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.reads.len(), 1, "expected one fs/read_text_file");
    assert_eq!(obs.reads[0], target);
    let any_tool_completed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Completed)));
    assert!(
        any_tool_completed,
        "no tool call reached Completed; updates={:?}",
        obs.updates
    );
}

/// #2 delegated write_file → fs/write_text_file round-trip hit
#[tokio::test]
async fn case2_delegated_write_file_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("out.txt");
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "write_file".into(),
            args_json: json!({"path": target_str, "content": "hi"}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    // The `write_file` tool best-effort reads old content first; the client returns
    // `ResourceNotFound` for a new file.
    let read: Arc<ReadFn> = Arc::new(|_req, _obs| {
        Err(agent_client_protocol::Error::resource_not_found(Some(
            "not found".to_string(),
        )))
    });
    let write: Arc<WriteFn> = Arc::new(|_req, _obs| Ok(WriteTextFileResponse::default()));

    let (stop, obs) = run_e2e(cwd, rounds, full_fs_caps(), Some(read), Some(write)).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.writes.len(), 1, "expected one fs/write_text_file");
    assert_eq!(obs.writes[0].0, target);
    assert_eq!(obs.writes[0].1, "hi");
}

/// #3 delegated edit_file: read before write order is correct
#[tokio::test]
async fn case3_delegated_edit_file_reads_then_writes() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("e.txt");
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "edit_file".into(),
            args_json: json!({
                "path": target_str,
                "old_string": "BETA",
                "new_string": "delta",
            })
            .to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    let read: Arc<ReadFn> =
        Arc::new(|_req, _obs| Ok(ReadTextFileResponse::new("alpha BETA gamma\n".to_string())));
    let write: Arc<WriteFn> = Arc::new(|req, _obs| {
        assert_eq!(req.content, "alpha delta gamma\n");
        Ok(WriteTextFileResponse::default())
    });

    let (stop, obs) = run_e2e(cwd, rounds, full_fs_caps(), Some(read), Some(write)).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(!obs.reads.is_empty(), "expected at least one fs/read");
    assert!(!obs.writes.is_empty(), "expected at least one fs/write");
}

/// #4 client declares read only (write=false) → entire group falls back to local, **no**
/// reverse request sent
#[tokio::test]
async fn case4_partial_caps_falls_back_to_local() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("local.txt");
    std::fs::write(&target, "actual disk\n").unwrap();
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": target_str}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    let (stop, obs) = run_e2e(cwd, rounds, read_only_fs_caps(), None, None).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(obs.reads.is_empty(), "client should not have seen fs/read");
    assert!(
        obs.writes.is_empty(),
        "client should not have seen fs/write"
    );
}

/// #5 client does not declare fs → falls back to local (same as #4, but caps are
/// completely missing rather than half-declared)
#[tokio::test]
async fn case5_no_caps_falls_back_to_local() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("local.txt");
    std::fs::write(&target, "x\n").unwrap();
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": target_str}).to_string(),
        },
        Round::EndTurn {
            text: "done".into(),
        },
    ];

    let (stop, obs) = run_e2e(cwd, rounds, ClientCapabilities::new(), None, None).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(obs.reads.is_empty());
    assert!(obs.writes.is_empty());
}

/// #6 delegated client error → tool Failed, but turn continues to EndTurn
#[tokio::test]
async fn case6_delegated_client_error_marks_tool_failed() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("forbidden.txt");
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": target_str}).to_string(),
        },
        Round::EndTurn {
            text: "ack failure".into(),
        },
    ];

    let read: Arc<ReadFn> =
        Arc::new(|_req, _obs| Err(agent_client_protocol::Error::new(-32000, "client says no")));

    let (stop, obs) = run_e2e(cwd, rounds, full_fs_caps(), Some(read), None).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert_eq!(obs.reads.len(), 1);
    let saw_failed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Failed)));
    assert!(
        saw_failed,
        "expected ToolCallUpdate Failed; updates={:?}",
        obs.updates
    );
}

/// #7 delegated cancel mid-turn → CancelNotification cuts the turn (no hang)
#[tokio::test]
async fn case7_delegated_cancel_short_circuits() {
    use agent_client_protocol::schema::CancelNotification;

    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let target = cwd.join("slow.txt");
    let target_str = target.to_string_lossy().into_owned();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": target_str}).to_string(),
        },
        Round::EndTurn {
            text: "should not reach".into(),
        },
    ];

    // Make the client's `read` fail immediately, so the reverse-request does not stay
    // pending forever. This case mainly verifies that delegation mode with a concurrent
    // cancel does not hang. The real cancel path is covered by the `e2e_turn` test suite.
    let read: Arc<ReadFn> = Arc::new(|_req, _obs| {
        Err(agent_client_protocol::Error::resource_not_found(Some(
            "missing".to_string(),
        )))
    });

    let agent_core = build_server(rounds);
    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let obs: SharedObs = Arc::new(Mutex::new(ClientObservations::default()));
    let obs_for_notif = obs.clone();
    let obs_for_read = obs.clone();
    let read_fn_arc = read.clone();

    let stop = Client
        .builder()
        .name("cancel-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                obs_for_notif.lock().unwrap().updates.push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: ReadTextFileRequest, responder, _cx| {
                let path = req.path.clone();
                let res = (read_fn_arc)(&req, &obs_for_read);
                obs_for_read.lock().unwrap().reads.push(path);
                match res {
                    Ok(resp) => responder.respond(resp),
                    Err(err) => responder.respond_with_error(err),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx: ConnectionTo<Agent>| {
                cx.send_request(
                    InitializeRequest::new(ProtocolVersion::V1).client_capabilities(full_fs_caps()),
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
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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
        )
        .await
        .expect("client connection completed");

    server_handle.abort();
    let _ = server_handle.await;

    assert!(
        matches!(stop, AcpStopReason::EndTurn | AcpStopReason::Cancelled),
        "stop should be EndTurn or Cancelled, got {stop:?}"
    );
}

/// #8 In delegated mode, path escape is blocked by the agent itself, **without** sending
/// a reverse request
#[tokio::test]
async fn case8_path_escape_blocked_before_reverse_request() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();

    let rounds = vec![
        Round::ToolUse {
            id: "tu-1".into(),
            name: "read_file".into(),
            args_json: json!({"path": "/etc/passwd"}).to_string(),
        },
        Round::EndTurn { text: "ack".into() },
    ];

    let (stop, obs) = run_e2e(cwd, rounds, full_fs_caps(), None, None).await;

    assert_eq!(stop, AcpStopReason::EndTurn);
    assert!(obs.reads.is_empty(), "boundary should block before fs/read");
    let saw_failed = obs.updates.iter().any(|u| matches!(u, SessionUpdate::ToolCallUpdate(upd) if upd.fields.status == Some(ToolCallStatus::Failed)));
    assert!(
        saw_failed,
        "expected tool Failed; updates={:?}",
        obs.updates
    );
}
