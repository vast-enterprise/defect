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
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion,
    SessionNotification, SessionUpdate, StopReason as AcpStopReason, TextContent,
};
use agent_client_protocol::{Agent, Channel, Client, ConnectTo, Role};
use defect_acp::{serve_on, EchoProvider};
use defect_agent::session::{AgentCore, DefaultAgentCore, TurnConfig};

/// `Channel` 实现的是 `ConnectTo<R>` for 任意 R，但 `serve_on` 需要
/// `T: ConnectTo<Agent>`。这里的 wrapper 仅是显式声明 role，方便类型推导。
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

                let prompt_resp = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new(prompt_text.to_string()))],
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
