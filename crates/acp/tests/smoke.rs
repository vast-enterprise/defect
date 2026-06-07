//! E2E smoke test: connects an ACP client and server in-process via [`Channel::duplex`],
//! then runs the minimal path of initialize → session/new → session/prompt.
//!
//! Verifies three things:
//! 1. `serve_on` correctly handles `initialize` / `session/new` / `session/prompt`
//! 2. [`EchoProvider`] projects `AgentMessageChunk` through [`crate::project`]
//! 3. `PromptResponse` yields `EndTurn` as the stop reason

use std::sync::Arc;
use std::sync::Mutex;

use agent_client_protocol::schema::{
    ContentBlock, InitializeRequest, LoadSessionRequest, NewSessionRequest, PromptRequest,
    ProtocolVersion, SessionId, SessionNotification, SessionUpdate, StopReason as AcpStopReason,
    TextContent,
};
use agent_client_protocol::{Agent, Channel, Client, ConnectTo, Role};
use defect_acp::{EchoProvider, serve_on, serve_on_with_resume};
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderErrorKind, ProviderInfo, ProviderStream,
    StopReason as LlmStopReason, ThinkingEcho,
};
use defect_agent::session::{AgentCore, DefaultAgentCore, TurnConfig};
use defect_storage::StorageObserver;
use futures::future::BoxFuture;
use futures::stream;
use tokio_util::sync::CancellationToken;

/// `Channel` implements `ConnectTo<R>` for any `R`, but `serve_on` requires
/// `T: ConnectTo<Agent>`. This wrapper simply declares the role explicitly to aid type
/// inference.
struct ChannelTransport<R: Role> {
    inner: Channel,
    _marker: std::marker::PhantomData<R>,
}

struct SwitchableProvider;

impl LlmProvider for SwitchableProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "switchable".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Switchable Test Provider".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tool_calls: FeatureSupport::Unsupported,
            parallel_tool_calls: FeatureSupport::Unsupported,
            thinking: FeatureSupport::Unsupported,
            vision: FeatureSupport::Unsupported,
            prompt_cache: FeatureSupport::Unsupported,
            thinking_echo: ThinkingEcho::Forbidden,
        }
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async {
            Ok(vec![
                ModelInfo {
                    id: "alpha".to_string(),
                    display_name: Some("Alpha".to_string()),
                    context_window: None,
                    max_output_tokens: None,
                    deprecated: false,
                    capabilities_overrides: Default::default(),
                },
                ModelInfo {
                    id: "beta".to_string(),
                    display_name: Some("Beta".to_string()),
                    context_window: None,
                    max_output_tokens: None,
                    deprecated: false,
                    capabilities_overrides: Default::default(),
                },
            ])
        })
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        match model_id {
            "alpha" => Some(ModelInfo {
                id: "alpha".to_string(),
                display_name: Some("Alpha".to_string()),
                context_window: None,
                max_output_tokens: None,
                deprecated: false,
                capabilities_overrides: Default::default(),
            }),
            "beta" => Some(ModelInfo {
                id: "beta".to_string(),
                display_name: Some("Beta".to_string()),
                context_window: None,
                max_output_tokens: None,
                deprecated: false,
                capabilities_overrides: Default::default(),
            }),
            _ => None,
        }
    }

    fn complete(
        &self,
        req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let model = req.model.clone();
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
                Ok(ProviderChunk::MessageStart {
                    id: "switchable-0".to_string(),
                    model: model.clone(),
                }),
                Ok(ProviderChunk::TextDelta {
                    text: format!("model={model}"),
                }),
                Ok(ProviderChunk::Stop {
                    reason: LlmStopReason::EndTurn,
                }),
            ];
            let s: ProviderStream = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

struct FlakyModelProvider;

impl LlmProvider for FlakyModelProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "flaky".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Flaky Model Provider".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tool_calls: FeatureSupport::Unsupported,
            parallel_tool_calls: FeatureSupport::Unsupported,
            thinking: FeatureSupport::Unsupported,
            vision: FeatureSupport::Unsupported,
            prompt_cache: FeatureSupport::Unsupported,
            thinking_echo: ThinkingEcho::Forbidden,
        }
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async {
            Err(ProviderError::new(ProviderErrorKind::Other(
                defect_agent::error::BoxError::new(std::io::Error::other(
                    "models endpoint unavailable",
                )),
            )))
        })
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        Some(ModelInfo {
            id: model_id.to_string(),
            display_name: Some(model_id.to_string()),
            context_window: None,
            max_output_tokens: None,
            deprecated: false,
            capabilities_overrides: Default::default(),
        })
    }

    fn complete(
        &self,
        req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let model = req.model.clone();
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
                Ok(ProviderChunk::MessageStart {
                    id: "flaky-0".to_string(),
                    model: model.clone(),
                }),
                Ok(ProviderChunk::TextDelta {
                    text: format!("model={model}"),
                }),
                Ok(ProviderChunk::Stop {
                    reason: LlmStopReason::EndTurn,
                }),
            ];
            let s: ProviderStream = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
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

#[tokio::test]
async fn echo_round_trip() {
    let provider = Arc::new(EchoProvider::new());
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    // The server uses `channel_b` (from the agent's perspective), and the client uses
    // `channel_a` (from the client's perspective).
    let (channel_a, channel_b) = Channel::duplex();

    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();

    let cwd = std::env::current_dir().expect("cwd available");
    let prompt_text = "hello echo";

    let client_result = Client
        .builder()
        .name("smoke-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                assert_eq!(select_current_value(model_opt), "defect::echo");
                let values = select_option_values(model_opt);
                assert_eq!(values, vec!["defect::echo".to_string()]);

                let prompt_resp = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new(
                            prompt_text.to_string(),
                        ))],
                    ))
                    .block_task()
                    .await?;

                Ok(prompt_resp.stop_reason)
            },
        )
        .await
        .expect("client connection completed");

    assert_eq!(
        client_result,
        AcpStopReason::EndTurn,
        "echo provider should drive a clean EndTurn"
    );

    // `serve` uses `connect_to` (which internally calls `future::pending`), so it won't
    // exit automatically when `main_fn` ends; aborting it directly in the test is
    // sufficient.
    server_handle.abort();
    let _ = server_handle.await;

    let updates = updates.lock().expect("updates mutex");
    let assistant_text: String = updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(chunk) => Some(&chunk.content),
            _ => None,
        })
        .filter_map(|content| match content {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    assert!(
        assistant_text.contains(prompt_text),
        "echo response should include user's prompt; got {assistant_text:?}; updates {updates:?}",
    );
}

#[tokio::test]
async fn load_session_round_trip() {
    let provider = Arc::new(EchoProvider::new());
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let sessions_dir = tempfile::tempdir().expect("tempdir");
    let storage = Arc::new(StorageObserver::new(sessions_dir.path().to_path_buf()));
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .observe_session(storage.clone())
        .session_loader(storage)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let prompt_text = "resume me";
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();

    let client_result = Client
        .builder()
        .name("load-session-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                let init = cx
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                assert!(
                    init.agent_capabilities.load_session,
                    "agent should advertise load_session capability"
                );

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd.clone()))
                    .block_task()
                    .await?;
                let new_options = new_session
                    .config_options
                    .expect("new session should include config options");
                let new_model_opt = new_options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                assert_eq!(select_current_value(new_model_opt), "defect::echo");

                let first = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id.clone(),
                        vec![ContentBlock::Text(TextContent::new(
                            prompt_text.to_string(),
                        ))],
                    ))
                    .block_task()
                    .await?;
                assert_eq!(first.stop_reason, AcpStopReason::EndTurn);

                let loaded = cx
                    .send_request(LoadSessionRequest::new(
                        new_session.session_id.clone(),
                        cwd.clone(),
                    ))
                    .block_task()
                    .await?;
                assert!(
                    loaded
                        .config_options
                        .iter()
                        .flatten()
                        .any(|o| o.id.0.as_ref() == "model"),
                    "loaded session should include model config option"
                );

                let replayed_user_text = updates
                    .lock()
                    .expect("updates mutex")
                    .iter()
                    .filter_map(|update| match update {
                        SessionUpdate::UserMessageChunk(chunk) => Some(&chunk.content),
                        _ => None,
                    })
                    .filter_map(|content| match content {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .any(|text| text == prompt_text);
                assert!(
                    replayed_user_text,
                    "session/load should replay previous user transcript"
                );

                let second = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new(
                            "after load".to_string(),
                        ))],
                    ))
                    .block_task()
                    .await?;

                Ok(second.stop_reason)
            },
        )
        .await
        .expect("client connection completed");

    assert_eq!(client_result, AcpStopReason::EndTurn);

    server_handle.abort();
    let _ = server_handle.await;
}

/// With `--resume`, the first `session/new` is transparently rewritten to `load_session`
/// — it returns the restored old session id and replays the old transcript before
/// responding. After that one-time consumption, a second `session/new` falls back to the
/// normal creation path.
#[tokio::test]
async fn resume_intercepts_first_session_new() {
    let sessions_dir = tempfile::tempdir().expect("tempdir");

    // Phase 1: create a normal session, run one turn, and persist to disk. Use a separate
    // server instance.
    let prompt_text = "remember this";
    let persisted_id = {
        let provider = Arc::new(EchoProvider::new());
        let config = TurnConfig {
            model: "echo".to_string(),
            ..TurnConfig::default()
        };
        let storage = Arc::new(StorageObserver::new(sessions_dir.path().to_path_buf()));
        let agent_core = DefaultAgentCore::builder()
            .provider(provider)
            .observe_session(storage.clone())
            .session_loader(storage)
            .config(config)
            .build();
        let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

        let (channel_a, channel_b) = Channel::duplex();
        let server_handle = tokio::spawn(serve_on(
            agent_core,
            ChannelTransport::<Agent>::new(channel_b),
        ));

        let cwd = std::env::current_dir().expect("cwd available");
        let id = Client
            .builder()
            .name("seed-client")
            .connect_with(
                ChannelTransport::<Client>::new(channel_a),
                async move |cx| {
                    cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    let new_session = cx
                        .send_request(NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    let resp = cx
                        .send_request(PromptRequest::new(
                            new_session.session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new(
                                prompt_text.to_string(),
                            ))],
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(resp.stop_reason, AcpStopReason::EndTurn);
                    Ok(new_session.session_id)
                },
            )
            .await
            .expect("seed connection completed");

        server_handle.abort();
        let _ = server_handle.await;
        id
    };

    // Phase 2: new server with resume target. The first session/new should restore the
    // old session.
    let provider = Arc::new(EchoProvider::new());
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let storage = Arc::new(StorageObserver::new(sessions_dir.path().to_path_buf()));
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .observe_session(storage.clone())
        .session_loader(storage)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let resume_target = SessionId::new(persisted_id.0.to_string());
    let server_handle = tokio::spawn(serve_on_with_resume(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
        Some(resume_target),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();
    let expected_id = persisted_id.0.to_string();

    Client
        .builder()
        .name("resume-client")
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notif.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                // First session/new → transparent resume: returns the old id.
                let resumed = cx
                    .send_request(NewSessionRequest::new(cwd.clone()))
                    .block_task()
                    .await?;
                assert_eq!(
                    resumed.session_id.0.as_ref(),
                    expected_id.as_str(),
                    "first session/new under --resume should return the resumed session id"
                );

                let replayed = updates
                    .lock()
                    .expect("updates mutex")
                    .iter()
                    .filter_map(|update| match update {
                        SessionUpdate::UserMessageChunk(chunk) => Some(&chunk.content),
                        _ => None,
                    })
                    .filter_map(|content| match content {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .any(|text| text == prompt_text);
                assert!(replayed, "resume should replay prior transcript");

                // Second session/new → one-time resume consumed, falls back to normal
                // creation (new id).
                let fresh = cx
                    .send_request(NewSessionRequest::new(cwd.clone()))
                    .block_task()
                    .await?;
                assert_ne!(
                    fresh.session_id.0.as_ref(),
                    expected_id.as_str(),
                    "second session/new should create a fresh session, not resume again"
                );
                Ok(())
            },
        )
        .await
        .expect("resume connection completed");

    server_handle.abort();
    let _ = server_handle.await;
}

#[tokio::test]
async fn set_model_updates_next_turn_model() {
    let provider = Arc::new(SwitchableProvider);
    let config = TurnConfig {
        model: "alpha".to_string(),
        allowed_models: Some(vec!["alpha".to_string(), "beta".to_string()]),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let client_result = Client
        .builder()
        .name("set-model-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                assert_eq!(select_current_value(model_opt), "switchable::alpha");
                assert_eq!(select_option_values(model_opt).len(), 2);

                cx.send_request(
                    agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                        new_session.session_id.clone(),
                        agent_client_protocol::schema::SessionConfigId::new("model"),
                        agent_client_protocol::schema::SessionConfigValueId::new(
                            "switchable::beta",
                        ),
                    ),
                )
                .block_task()
                .await?;

                let prompt_resp = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new("switch".to_string()))],
                    ))
                    .block_task()
                    .await?;

                Ok(prompt_resp.stop_reason)
            },
        )
        .await
        .expect("client connection completed");

    assert_eq!(client_result, AcpStopReason::EndTurn);

    server_handle.abort();
    let _ = server_handle.await;
}

#[tokio::test]
async fn set_model_rejects_model_outside_configured_candidates() {
    let provider = Arc::new(SwitchableProvider);
    let config = TurnConfig {
        model: "alpha".to_string(),
        allowed_models: Some(vec!["alpha".to_string()]),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let client_result = Client
        .builder()
        .name("set-model-reject-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                let values = select_option_values(model_opt);
                assert_eq!(values, vec!["switchable::alpha".to_string()]);

                let err = cx
                    .send_request(
                        agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                            new_session.session_id,
                            agent_client_protocol::schema::SessionConfigId::new("model"),
                            agent_client_protocol::schema::SessionConfigValueId::new(
                                "switchable::beta",
                            ),
                        ),
                    )
                    .block_task()
                    .await
                    .expect_err("beta should be rejected by configured candidate filter");

                Ok(err.message)
            },
        )
        .await
        .expect("client connection completed");

    assert!(
        client_result.contains("beta") && client_result.contains("model"),
        "expected config-option rejection for filtered model, got {client_result:?}"
    );

    server_handle.abort();
    let _ = server_handle.await;
}

#[tokio::test]
async fn model_candidates_fall_back_to_configured_whitelist_when_provider_list_fails() {
    let provider = Arc::new(FlakyModelProvider);
    let config = TurnConfig {
        model: "deepseek-v4-pro".to_string(),
        allowed_models: Some(vec![
            "deepseek-v4-pro".to_string(),
            "deepseek-v4-flash".to_string(),
        ]),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(provider)
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let client_result = Client
        .builder()
        .name("flaky-model-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let loaded = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = loaded
                    .config_options
                    .expect("session should advertise config options");
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                Ok(select_option_values(model_opt).len() as u64)
            },
        )
        .await
        .expect("client connection completed");

    assert_eq!(client_result, 2);

    server_handle.abort();
    let _ = server_handle.await;
}

/// Set up a two-mode catalog; `session/new` should expose them via `config_options`
/// (category=mode), and switching to the second mode via `session/set_config_option`
/// should succeed without affecting subsequent prompt completion (verifying that mode
/// switching takes effect at turn boundaries and does not interrupt the session).
#[tokio::test]
async fn set_mode_switches_session_permission_mode() {
    use defect_agent::policy::{ModeCatalog, OpenPolicy, PolicyMode, ReadOnlyPolicy};

    let modes = ModeCatalog::new(
        vec![
            PolicyMode {
                id: "ask-writes".to_string(),
                name: "Ask before writes".to_string(),
                description: None,
                policy: Arc::new(OpenPolicy),
            },
            PolicyMode {
                id: "read-only".to_string(),
                name: "Read-only".to_string(),
                description: Some("Read-only".to_string()),
                policy: Arc::new(ReadOnlyPolicy),
            },
        ],
        "ask-writes",
    )
    .expect("catalog construction");

    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(Arc::new(EchoProvider::new()))
        .config(config)
        .modes(modes)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    let client_result = Client
        .builder()
        .name("set-mode-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                let mode_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "permission_mode")
                    .expect("permission_mode config option present");
                assert_eq!(select_current_value(mode_opt), "ask-writes");
                assert_eq!(select_option_values(mode_opt).len(), 2);

                cx.send_request(
                    agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                        new_session.session_id.clone(),
                        agent_client_protocol::schema::SessionConfigId::new("permission_mode"),
                        agent_client_protocol::schema::SessionConfigValueId::new("read-only"),
                    ),
                )
                .block_task()
                .await?;

                let prompt_resp = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new("hi".to_string()))],
                    ))
                    .block_task()
                    .await?;

                Ok(prompt_resp.stop_reason)
            },
        )
        .await
        .expect("client connection completed");

    assert_eq!(client_result, AcpStopReason::EndTurn);

    server_handle.abort();
    let _ = server_handle.await;
}

/// `session/new` should expose the thought-level selector via `config_options` (currently
/// `default`); `session/set_config_option` to `high` should succeed, with the response
/// returning the updated current value. Unknown values should be rejected.
#[tokio::test]
async fn set_config_option_switches_reasoning_effort() {
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(Arc::new(EchoProvider::new()))
        .config(config)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    Client
        .builder()
        .name("set-config-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                // No-mode directory: two selectors for model and thought-level (no
                // `permission_mode`).
                assert_eq!(options.len(), 2);
                assert!(options.iter().any(|o| o.id.0.as_ref() == "model"));
                assert!(
                    options
                        .iter()
                        .any(|o| o.id.0.as_ref() == "reasoning_effort")
                );
                assert!(!options.iter().any(|o| o.id.0.as_ref() == "permission_mode"));

                // Valid option: high.
                cx.send_request(
                    agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                        new_session.session_id.clone(),
                        agent_client_protocol::schema::SessionConfigId::new("reasoning_effort"),
                        agent_client_protocol::schema::SessionConfigValueId::new("high"),
                    ),
                )
                .block_task()
                .await?;

                // Invalid value: must produce an error.
                let bad = cx
                    .send_request(
                        agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                            new_session.session_id.clone(),
                            agent_client_protocol::schema::SessionConfigId::new("reasoning_effort"),
                            agent_client_protocol::schema::SessionConfigValueId::new("bogus"),
                        ),
                    )
                    .block_task()
                    .await;
                assert!(bad.is_err(), "unknown value must be rejected");

                Ok(())
            },
        )
        .await
        .expect("client connection completed");

    server_handle.abort();
    let _ = server_handle.await;
}

/// Returns the current value of a select-type config option as a string (test helper).
fn select_current_value(opt: &agent_client_protocol::schema::SessionConfigOption) -> String {
    match &opt.kind {
        agent_client_protocol::schema::SessionConfigKind::Select(select) => {
            select.current_value.0.to_string()
        }
        #[allow(unreachable_patterns)]
        _ => panic!("expected a select config option"),
    }
}

/// Extract all candidate values from a select config option (in order, test helper).
fn select_option_values(opt: &agent_client_protocol::schema::SessionConfigOption) -> Vec<String> {
    use agent_client_protocol::schema::{SessionConfigKind, SessionConfigSelectOptions};
    match &opt.kind {
        SessionConfigKind::Select(select) => match &select.options {
            SessionConfigSelectOptions::Ungrouped(opts) => {
                opts.iter().map(|o| o.value.0.to_string()).collect()
            }
            SessionConfigSelectOptions::Grouped(groups) => groups
                .iter()
                .flat_map(|g| g.options.iter().map(|o| o.value.0.to_string()))
                .collect(),
            _ => panic!("unexpected SessionConfigSelectOptions variant"),
        },
        #[allow(unreachable_patterns)]
        _ => panic!("expected a select config option"),
    }
}

/// Permission modes must also be exposed as a config option with `category = Mode`
/// (modern clients like Zed read `config_options` and ignore the deprecated `modes`
/// field). For a session with a mode catalog: `session/new`'s `config_options` should
/// include `permission_mode` plus a thought-level item; `session/set_config_option`
/// switching `permission_mode` to a valid mode id should succeed, and an unknown id
/// should be rejected.
#[tokio::test]
async fn mode_exposed_as_config_option_and_set_via_config() {
    use defect_agent::policy::{ModeCatalog, OpenPolicy, PolicyMode, ReadOnlyPolicy};

    let modes = ModeCatalog::new(
        vec![
            PolicyMode {
                id: "ask-writes".to_string(),
                name: "Ask before writes".to_string(),
                description: None,
                policy: Arc::new(OpenPolicy),
            },
            PolicyMode {
                id: "read-only".to_string(),
                name: "Read-only".to_string(),
                description: Some("Read-only".to_string()),
                policy: Arc::new(ReadOnlyPolicy),
            },
        ],
        "ask-writes",
    )
    .expect("catalog construction");

    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(Arc::new(EchoProvider::new()))
        .config(config)
        .modes(modes)
        .build();
    let agent_core: Arc<dyn AgentCore> = Arc::new(agent_core);

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let cwd = std::env::current_dir().expect("cwd available");
    Client
        .builder()
        .name("mode-config-client")
        .connect_with(
            ChannelTransport::<Client>::new(channel_a),
            async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let new_session = cx
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                let options = new_session
                    .config_options
                    .expect("agent should advertise config options");
                // Three selectors: model, mode, and thought-level.
                assert_eq!(options.len(), 3);
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                assert_eq!(
                    model_opt.category,
                    Some(agent_client_protocol::schema::SessionConfigOptionCategory::Model)
                );
                assert_eq!(select_current_value(model_opt), "defect::echo");
                let mode_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "permission_mode")
                    .expect("permission_mode config option present");
                assert_eq!(
                    mode_opt.category,
                    Some(agent_client_protocol::schema::SessionConfigOptionCategory::Mode)
                );
                assert_eq!(select_current_value(mode_opt), "ask-writes");

                // Valid mode: read-only. Response includes the refreshed current value.
                let resp = cx
                    .send_request(
                        agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                            new_session.session_id.clone(),
                            agent_client_protocol::schema::SessionConfigId::new("permission_mode"),
                            agent_client_protocol::schema::SessionConfigValueId::new("read-only"),
                        ),
                    )
                    .block_task()
                    .await?;
                let refreshed = resp
                    .config_options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "permission_mode")
                    .expect("permission_mode in refreshed options");
                assert_eq!(select_current_value(refreshed), "read-only");

                // Invalid mode id: must produce an error.
                let bad = cx
                    .send_request(
                        agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                            new_session.session_id.clone(),
                            agent_client_protocol::schema::SessionConfigId::new("permission_mode"),
                            agent_client_protocol::schema::SessionConfigValueId::new("bogus"),
                        ),
                    )
                    .block_task()
                    .await;
                assert!(bad.is_err(), "unknown mode id must be rejected");

                Ok(())
            },
        )
        .await
        .expect("client connection completed");

    server_handle.abort();
    let _ = server_handle.await;
}
