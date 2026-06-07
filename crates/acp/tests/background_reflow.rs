//! E2E wire-level test: `run_in_background` auto-continue through the ACP protocol layer.
//!
//! Proves that after a background task completes, the session driver autonomously
//! continues the turn and delivers `session/update` notifications to the client
//! via the session-level persistent event pump — no second `session/prompt` needed.
//!
//! Setup:
//! - 自定义工具 `bg_tool`：调 `ctx.background.spawn` 起一个立即完成的后台任务，**立刻**返回。
//! - 脚本化 provider：turn 1 调 `bg_tool` 后 EndTurn；其后每轮发一段可识别文本 + EndTurn。
//! - 客户端只发**一次** prompt，然后被动收通知——断言能收到含后台答案标记的 AgentMessageChunk。

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use agent_client_protocol::schema::{
    Content, ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion,
    SessionNotification, SessionUpdate, TextContent, ToolCallContent, ToolCallUpdateFields,
};
use agent_client_protocol::{Agent, Channel, Client, ConnectTo, Role};
use defect_acp::serve_on;
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderInfo, ProviderStream, StopReason as LlmStopReason,
    ThinkingEcho,
};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, StaticToolRegistry, ToolRegistry, TurnConfig,
};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolEvent, ToolSchema, ToolStream,
};
use futures::future::BoxFuture;
use futures::stream;
use tokio_util::sync::CancellationToken;

/// 后台答案里嵌一个独特标记，客户端据它确认"来自自主续转 turn 的事件确实到达"。
const MARKER: &str = "BG-MARKER-7f3a";

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

fn caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

/// turn 1：调 bg_tool 后 Stop=ToolUse；之后每轮：发一段含 MARKER 的文本 + EndTurn。
/// 自主续转 turn（携带后台答案）跑的就是"之后"分支——它的 AgentMessageChunk 含 MARKER。
struct BgWireProvider {
    calls: Mutex<u32>,
}

impl LlmProvider for BgWireProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "bg-wire".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Background Wire Provider".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _: CompletionRequest,
        _: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let mut calls = self.calls.lock().expect("calls poisoned");
        *calls += 1;
        let nth = *calls;
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = if nth == 1 {
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "m1".to_string(),
                        model: "bg-wire-001".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseStart {
                        id: "tu-1".to_string(),
                        name: "bg_tool".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseArgsDelta {
                        id: "tu-1".to_string(),
                        fragment: "{}".to_string(),
                    }),
                    Ok(ProviderChunk::ToolUseEnd {
                        id: "tu-1".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::ToolUse,
                    }),
                ]
            } else {
                // 自主续转 turn：把 MARKER 回成 assistant 文本——它会经 pump 投成
                // AgentMessageChunk 送到客户端。
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "m2".to_string(),
                        model: "bg-wire-001".to_string(),
                    }),
                    Ok(ProviderChunk::TextDelta {
                        text: format!("autonomous turn saw {MARKER}"),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ]
            };
            let s: ProviderStream = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

/// 后台 spawn 工具：起一个立即完成、结果含 MARKER 的后台任务，立刻返回。
struct BgTool {
    schema: ToolSchema,
}

impl Tool for BgTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    fn describe<'a>(
        &'a self,
        _args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async {
            ToolCallDescription {
                fields: ToolCallUpdateFields::default(),
            }
        })
    }
    fn execute(&self, _args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let bg = ctx.background.clone();
        let fut = async move {
            let mut fields = ToolCallUpdateFields::default();
            let text = match bg {
                Some(bg) => {
                    bg.spawn("worker".to_string(), |_cancel, _progress| async move {
                        defect_agent::session::BackgroundResult::Completed(MARKER.to_string())
                    });
                    "started background task".to_string()
                }
                None => "no background".to_string(),
            };
            fields.content = Some(vec![ToolCallContent::Content(Content::new(text))]);
            ToolEvent::Completed(fields)
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

/// 客户端只发一次 prompt；断言它随后被动收到由自主续转 turn 产生的、含 MARKER 的
/// AgentMessageChunk——证明主动续转穿过 ACP 协议层送达客户端。
#[tokio::test]
async fn background_result_reaches_client_via_autonomous_turn() {
    let provider = Arc::new(BgWireProvider {
        calls: Mutex::new(0),
    }) as Arc<dyn LlmProvider>;
    let tools: Arc<dyn ToolRegistry> = Arc::new(
        StaticToolRegistry::builder()
            .insert(Arc::new(BgTool {
                schema: ToolSchema {
                    name: "bg_tool".to_string(),
                    description: "spawn a background task".to_string(),
                    input_schema: serde_json::json!({"type":"object"}),
                },
            }))
            .build(),
    );
    let agent_core: Arc<dyn AgentCore> = Arc::new(
        DefaultAgentCore::builder()
            .provider(provider)
            .process_tools(tools)
            .config(TurnConfig {
                model: "bg-wire-001".to_string(),
                ..TurnConfig::default()
            })
            .build(),
    );

    let (channel_a, channel_b) = Channel::duplex();
    let server_handle = tokio::spawn(serve_on(
        agent_core,
        ChannelTransport::<Agent>::new(channel_b),
    ));

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = updates.clone();
    let cwd = std::env::current_dir().expect("cwd");

    Client
        .builder()
        .name("bg-reflow-client")
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
                // 只发一次 prompt——turn 1 调 bg_tool 起后台任务后即结束。
                let _ = cx
                    .send_request(PromptRequest::new(
                        new_session.session_id,
                        vec![ContentBlock::Text(TextContent::new("kick off".to_string()))],
                    ))
                    .block_task()
                    .await?;
                Ok(())
            },
        )
        .await
        .expect("client connection completed");

    // prompt 已 respond，但自主续转 turn 是 prompt respond **之后**异步发生的。
    // 轮询通知，等含 MARKER 的 AgentMessageChunk 到达（带超时上限）。
    let mut saw_marker = false;
    for _ in 0..200 {
        {
            let updates = updates.lock().expect("updates mutex");
            saw_marker = updates.iter().any(|u| match u {
                SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                    ContentBlock::Text(t) => t.text.contains(MARKER),
                    _ => false,
                },
                _ => false,
            });
        }
        if saw_marker {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    server_handle.abort();
    let _ = server_handle.await;

    assert!(
        saw_marker,
        "client should receive an AgentMessageChunk from the autonomous re-invoke turn carrying \
         the background result marker, without sending a second prompt"
    );
}
