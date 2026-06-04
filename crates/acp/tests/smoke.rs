//! E2E smoke test：在进程内用 [`Channel::duplex`] 把 ACP 客户端 / 服务端对接起来，
//! 跑一遍 initialize → session/new → session/prompt 的最小路径。
//!
//! 校验三件事：
//! 1. `serve_on` 正确处理 `initialize` / `session/new` / `session/prompt`
//! 2. [`EchoProvider`] 通过 [`crate::project`] 投射出 `AgentMessageChunk`
//! 3. `PromptResponse` 拿到 `EndTurn` stop reason

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

/// `Channel` 实现的是 `ConnectTo<R>` for 任意 R，但 `serve_on` 需要
/// `T: ConnectTo<Agent>`。这里的 wrapper 仅是显式声明 role，方便类型推导。
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

    // server 用 channel_b（agent 视角），client 用 channel_a（client 视角）。
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
                assert_eq!(select_current_value(model_opt), "echo");
                let values = select_option_values(model_opt);
                assert_eq!(values, vec!["echo".to_string()]);

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

    // serve 用的是 `connect_to`（内部 `future::pending`），不会因为 main_fn
    // 结束自动退出；测试里直接 abort 即可。
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
                assert_eq!(select_current_value(new_model_opt), "echo");

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

/// `--resume`：首个 `session/new` 被透明改写成 load_session——返回的是被恢复
/// 的旧 session id，且响应前回放了旧 transcript；一次性消费后，第二个
/// `session/new` 回到正常的新建路径。
#[tokio::test]
async fn resume_intercepts_first_session_new() {
    let sessions_dir = tempfile::tempdir().expect("tempdir");

    // 阶段一：正常建一个 session 跑一轮，落盘。用独立 server 实例。
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

    // 阶段二：新 server 带 resume 目标。首个 session/new 应恢复旧 session。
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

                // 首个 session/new → 透明 resume：返回旧 id。
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

                // 第二个 session/new → 一次性已消费，回到正常新建（新 id）。
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
                assert_eq!(select_current_value(model_opt), "alpha");
                assert_eq!(select_option_values(model_opt).len(), 2);

                cx.send_request(
                    agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                        new_session.session_id.clone(),
                        agent_client_protocol::schema::SessionConfigId::new("model"),
                        agent_client_protocol::schema::SessionConfigValueId::new("beta"),
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
                assert_eq!(values, vec!["alpha".to_string()]);

                let err = cx
                    .send_request(
                        agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                            new_session.session_id,
                            agent_client_protocol::schema::SessionConfigId::new("model"),
                            agent_client_protocol::schema::SessionConfigValueId::new("beta"),
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

/// 装一个两模式目录，`session/new` 应通过 `config_options`（category=mode）暴露
/// 它们；经 `session/set_config_option` 切到第二个应成功，且不影响后续 prompt
/// 跑通（验证模式切换是 turn 边界生效、不打断会话）。
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
                description: Some("只读".to_string()),
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
        .provider(Arc::new(EchoProvider::default()))
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

/// `session/new` 应通过 `config_options` 暴露 thought-level 选择器（当前
/// `default`）；`session/set_config_option` 改成 `high` 应成功，且响应回带
/// 更新后的当前值。未知 value 应被拒。
#[tokio::test]
async fn set_config_option_switches_reasoning_effort() {
    let config = TurnConfig {
        model: "echo".to_string(),
        ..TurnConfig::default()
    };
    let agent_core = DefaultAgentCore::builder()
        .provider(Arc::new(EchoProvider::default()))
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
                // 无模式目录：model + thought-level 两个选择器（无 permission_mode）。
                assert_eq!(options.len(), 2);
                assert!(options.iter().any(|o| o.id.0.as_ref() == "model"));
                assert!(options.iter().any(|o| o.id.0.as_ref() == "reasoning_effort"));
                assert!(!options.iter().any(|o| o.id.0.as_ref() == "permission_mode"));

                // 合法档位：high。
                cx.send_request(
                    agent_client_protocol::schema::SetSessionConfigOptionRequest::new(
                        new_session.session_id.clone(),
                        agent_client_protocol::schema::SessionConfigId::new("reasoning_effort"),
                        agent_client_protocol::schema::SessionConfigValueId::new("high"),
                    ),
                )
                .block_task()
                .await?;

                // 非法 value：必须报错。
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

    assert_eq!(client_result, ());

    server_handle.abort();
    let _ = server_handle.await;
}

/// 从一个 select 型 config option 取当前值字符串（测试辅助）。
fn select_current_value(opt: &agent_client_protocol::schema::SessionConfigOption) -> String {
    match &opt.kind {
        agent_client_protocol::schema::SessionConfigKind::Select(select) => {
            select.current_value.0.to_string()
        }
        #[allow(unreachable_patterns)]
        _ => panic!("expected a select config option"),
    }
}

/// 从一个 select 型 config option 取全部候选值（按序，测试辅助）。
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

/// 权限模式必须**也**作为 `category = Mode` 的 config option 暴露（现代客户端
/// 如 Zed 只读 config_options、忽略 deprecated `modes` 字段）。装了模式目录的
/// session：`session/new` 的 `config_options` 应含 `permission_mode` + 一个
/// thought-level 项；`session/set_config_option` 切 `permission_mode` 到合法
/// mode id 应成功，未知 id 应被拒。
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
                description: Some("只读".to_string()),
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
        .provider(Arc::new(EchoProvider::default()))
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
                // model + mode + thought-level 三个选择器。
                assert_eq!(options.len(), 3);
                let model_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "model")
                    .expect("model config option present");
                assert_eq!(
                    model_opt.category,
                    Some(agent_client_protocol::schema::SessionConfigOptionCategory::Model)
                );
                assert_eq!(select_current_value(model_opt), "echo");
                let mode_opt = options
                    .iter()
                    .find(|o| o.id.0.as_ref() == "permission_mode")
                    .expect("permission_mode config option present");
                assert_eq!(
                    mode_opt.category,
                    Some(agent_client_protocol::schema::SessionConfigOptionCategory::Mode)
                );
                assert_eq!(select_current_value(mode_opt), "ask-writes");

                // 合法 mode：read-only。响应回带刷新后的当前值。
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

                // 非法 mode id：必须报错。
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

    assert_eq!(client_result, ());

    server_handle.abort();
    let _ = server_handle.await;
}
