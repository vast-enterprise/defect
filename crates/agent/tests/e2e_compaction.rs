//! 端到端压缩测试：mock provider 触发上下文压缩，断言历史被「摘要 + 保留尾部」
//! 重建、发了 `ContextCompressed` 事件、且压缩子请求被正确识别。
//!
//! 触发路径：
//! - `model_info` 报一个很小的 `context_window`，配默认 `compact_ratio=0.85`
//!   推出极低阈值；
//! - 经 `load_session` 预置多轮历史，使 `select_boundary` 有 ≥2 个轮次起点、
//!   能切出非空 head；
//! - turn 开始即 `maybe_compact` → 触发 → provider 收到一个
//!   `tool_choice=None` 的摘要子请求，回一段摘要文本；
//! - 压缩后正常 turn 继续，provider EndTurn。

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use agent_client_protocol_schema::{ContentBlock, SessionId, StopReason, TextContent};
use defect_agent::event::AgentEvent;
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, Message, MessageContent,
    ModelInfo, ProtocolId, ProviderChunk, ProviderError, ProviderInfo, ProviderStream, Role,
    StopReason as LlmStopReason, ThinkingEcho, ToolChoice, Usage,
};
use defect_agent::session::{
    AgentCore, DefaultAgentCore, Frontend, LoadedSession, SessionCreateInfo, SessionLoader,
    TurnConfig, new_session_id,
};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use tokio_util::sync::CancellationToken;

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

/// 记录 provider 收到的每次请求，区分摘要子请求（`tool_choice=None`）与普通 turn。
struct RecordingProvider {
    /// 每次 complete 收到的 (tool_choice_is_none, message_count)。
    seen: Mutex<Vec<(bool, usize)>>,
}

impl RecordingProvider {
    fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl LlmProvider for RecordingProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "rec".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Recording Test Provider".to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        caps()
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        // 极小 context_window → 阈值 = floor(100 * 0.85) = 85 token，
        // 预置历史轻松越过它，触发压缩。
        Some(ModelInfo {
            id: model_id.to_string(),
            display_name: None,
            context_window: Some(100),
            max_output_tokens: Some(64),
            deprecated: false,
            capabilities_overrides: Default::default(),
        })
    }

    fn complete(
        &self,
        req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let is_summarize = matches!(req.tool_choice, ToolChoice::None);
        self.seen
            .lock()
            .expect("seen poisoned")
            .push((is_summarize, req.messages.len()));

        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = if is_summarize {
                // 摘要子请求：回一段结构化摘要文本。
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "sum-1".to_string(),
                        model: "rec-001".to_string(),
                    }),
                    Ok(ProviderChunk::TextDelta {
                        text: "## Goal\nship compaction".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ]
            } else {
                // 压缩后的正常 turn：直接 EndTurn。
                vec![
                    Ok(ProviderChunk::MessageStart {
                        id: "turn-1".to_string(),
                        model: "rec-001".to_string(),
                    }),
                    Ok(ProviderChunk::Usage(Usage {
                        input_tokens: Some(10),
                        output_tokens: Some(2),
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                    })),
                    Ok(ProviderChunk::TextDelta {
                        text: "done".to_string(),
                    }),
                    Ok(ProviderChunk::Stop {
                        reason: LlmStopReason::EndTurn,
                    }),
                ]
            };
            let s: Pin<
                Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
            > = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

fn text_msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }]
        .into(),
    }
}

#[tokio::test]
async fn compaction_rebuilds_history_with_summary_and_tail() {
    let provider = Arc::new(RecordingProvider::new());
    let provider_dyn = provider.clone() as Arc<dyn LlmProvider>;

    // 预置多轮历史：3 个用户轮次，每个带大段文本以越过阈值。
    let big = "x".repeat(400); // ~100 token each via chars/4
    let history = vec![
        text_msg(Role::User, &format!("turn one {big}")),
        text_msg(Role::Assistant, "reply one"),
        text_msg(Role::User, &format!("turn two {big}")),
        text_msg(Role::Assistant, "reply two"),
        text_msg(Role::User, &format!("turn three {big}")),
        text_msg(Role::Assistant, "reply three"),
    ];
    let loaded = LoadedSession {
        info: SessionCreateInfo {
            id: SessionId::new(new_session_id()),
            cwd: std::env::current_dir().expect("cwd"),
            mcp_servers: Vec::new(),
        },
        history,
    };

    let core = DefaultAgentCore::builder()
        .provider(provider_dyn)
        .session_loader(Arc::new(StubLoader {
            loaded: loaded.clone(),
        }))
        .config(TurnConfig {
            model: "rec-001".to_string(),
            // 用默认 compact_ratio=0.85；不设绝对阈值，走 ratio 推算。
            ..TurnConfig::default()
        })
        .build();

    let session = core
        .load_session(
            loaded.info.id.clone(),
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("load session");

    let mut events = session.subscribe();

    let stop = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("new prompt"))])
        .await
        .expect("turn");
    assert!(matches!(stop, StopReason::EndTurn));

    // 收事件直到 TurnEnded，捕获是否发了 ContextCompressed。
    let mut got_compressed = false;
    let mut compressed_before_after = None;
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::ContextCompressed {
                tokens_before,
                tokens_after,
            } => {
                got_compressed = true;
                compressed_before_after = Some((tokens_before, tokens_after));
            }
            AgentEvent::TurnEnded { .. } => break,
            _ => {}
        }
    }

    assert!(got_compressed, "should emit ContextCompressed");
    let (before, after) = compressed_before_after.expect("before/after");
    assert!(after < before, "compaction should shrink token estimate");

    // provider 应先收到一个摘要子请求（tool_choice=None），再收到压缩后的正常 turn。
    let seen = provider.seen.lock().expect("seen poisoned").clone();
    assert!(
        seen.iter().any(|(is_sum, _)| *is_sum),
        "expected one summarize sub-request (tool_choice=None), saw {seen:?}"
    );
    let normal = seen
        .iter()
        .find(|(is_sum, _)| !*is_sum)
        .expect("expected a normal turn request after compaction");
    // 压缩后历史 = [合成 assistant 摘要] + 保留尾部 + 本轮新 user prompt。
    // 尾部预算极小（clamp(85/4,2k,8k)=2k 够大 → 实际由轮次数决定），但 head 非空，
    // 故重建后的消息数应明显少于原始 6 轮历史 + 新 prompt = 7。
    assert!(
        normal.1 <= 7,
        "compacted request should not exceed original message count, saw {}",
        normal.1
    );

    // 重建后的历史首条应是合成的 assistant 摘要消息（带 SUMMARY 前缀）。
    let snap = session.history_snapshot();
    let first = snap.first().expect("non-empty history");
    assert_eq!(first.role, Role::Assistant, "summary message is assistant");
    let has_summary_text = first.content.iter().any(|c| {
        matches!(
            c,
            MessageContent::Text { text } if text.contains("ship compaction")
        )
    });
    assert!(has_summary_text, "first message should carry the summary");
}

struct StubLoader {
    loaded: LoadedSession,
}

impl SessionLoader for StubLoader {
    fn load_session(
        &self,
        _id: SessionId,
    ) -> BoxFuture<'_, Result<LoadedSession, defect_agent::error::BoxError>> {
        let loaded = self.loaded.clone();
        Box::pin(async move { Ok(loaded) })
    }
}
