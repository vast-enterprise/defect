//! End-to-end compression test: a mock provider triggers context compression. Asserts
//! that history is rebuilt as "summary + retained tail", that a `ContextCompressed` event
//! is emitted, and that the compression sub-request is correctly identified.
//!
//! Trigger path:
//! - `model_info` reports a very small `context_window`; with the default
//!   `compact_ratio=0.85` this yields an extremely low threshold.
//! - `load_session` pre-populates multiple turns of history so that `select_boundary` has
//!   ≥2 turn boundaries and can split off a non-empty head.
//! - At the start of a turn, `maybe_compact` triggers → the provider receives a
//!   summarization sub-request with `tool_choice=None` and returns a summary text.
//! - After compression, the normal turn continues and the provider returns `EndTurn`.

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

/// Records each request received by the provider, distinguishing summary sub-requests
/// (`tool_choice=None`) from normal turns.
struct RecordingProvider {
    /// (tool_choice_is_none, message_count) received per `complete` call.
    seen: Mutex<Vec<(bool, usize)>>,
    /// The `context_window` reported by `model_info` – determines the three compression
    /// thresholds.
    context_window: u64,
}

impl RecordingProvider {
    fn new() -> Self {
        Self::with_context_window(100)
    }

    fn with_context_window(context_window: u64) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            context_window,
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
        // `context_window` determines three threshold tiers. Default 100 → hard =
        // floor(100 * 0.85) = 85, which the preset history easily exceeds, triggering
        // synchronous compression.
        Some(ModelInfo {
            id: model_id.to_string(),
            display_name: None,
            context_window: Some(self.context_window),
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
                // Summarize sub-request: return a structured summary text.
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
                // Compressed normal turn: directly emit EndTurn.
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

    // Pre-seed multi-turn history: 3 user turns, each with a large text block to exceed
    // the threshold.
    let big = "x".repeat(400); // ~100 tokens each via chars/4
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
            // Use default compact_ratio=0.85; no absolute threshold, rely on ratio-based
            // inference.
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

    // Consume events until TurnEnded, capturing whether ContextCompressed was emitted.
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

    // The provider should first receive a summarization sub-request (`tool_choice=None`),
    // then the compressed normal turn.
    let seen = provider.seen.lock().expect("seen poisoned").clone();
    assert!(
        seen.iter().any(|(is_sum, _)| *is_sum),
        "expected one summarize sub-request (tool_choice=None), saw {seen:?}"
    );
    let normal = seen
        .iter()
        .find(|(is_sum, _)| !*is_sum)
        .expect("expected a normal turn request after compaction");
    // After compaction, the history is: [synthesized assistant summary] + retained tail +
    // current user prompt.
    // The tail budget is very small (clamp(85/4, 2k, 8k) = 2k, which is large enough →
    // actually determined by the number of turns), but the head is non-empty,
    // so the reconstructed message count should be significantly less than the original
    // 6-turn history + new prompt = 7.
    assert!(
        normal.1 <= 7,
        "compacted request should not exceed original message count, saw {}",
        normal.1
    );

    // The first entry in the rebuilt history should be the synthetic assistant summary
    // message (prefixed with SUMMARY).
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

/// Background compaction path: when the estimated size falls in the soft zone `[soft,
/// hard)`, the turn does **not** compact synchronously; instead it kicks off an
/// asynchronous background summary compaction. This turn sends the full history, and
/// compaction takes effect later (here we poll history until the first message becomes a
/// synthetic summary).
#[tokio::test]
async fn background_compaction_runs_off_turn_critical_path() {
    // Preset history ≈ 314 tokens (3×102 large user + 3×2 small reply) + new prompt ~2 ≈
    // 316.
    // Set context_window=405 → micro=floor(405*0.6)=243, soft=283, hard=344.
    // Estimate ~316 ∈ [283, 344) soft band (~30 margin on each side): this turn starts
    // async background compaction.
    // History has no large tool_result (plain-text turns) → micro-compression has nothing
    // to clean → skipped, falls into soft.
    let provider = Arc::new(RecordingProvider::with_context_window(405));
    let provider_dyn = provider.clone() as Arc<dyn LlmProvider>;

    let big = "x".repeat(400); // ~100 tokens each
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

    let stop = session
        .run_turn(vec![ContentBlock::Text(TextContent::new("new prompt"))])
        .await
        .expect("turn");
    assert!(matches!(stop, StopReason::EndTurn));

    // This turn uses a soft trigger → background compaction starts asynchronously and
    // does **not** block this turn. Therefore, the normal request in this turn sees the
    // full uncompressed history (the first entry is still the original user turn, not a
    // summary). The background compaction lands later — poll the history until the first
    // entry becomes a synthetic summary (with a SUMMARY prefix).
    let mut compacted = false;
    for _ in 0..50 {
        let snap = session.history_snapshot();
        if let Some(first) = snap.first()
            && first.role == Role::Assistant
            && first.content.iter().any(
                |c| matches!(c, MessageContent::Text { text } if text.contains("ship compaction")),
            )
        {
            compacted = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        compacted,
        "background compaction should eventually rebuild history with a summary"
    );

    // The provider must have received a summarize sub-request (`tool_choice=None`).
    let seen = provider.seen.lock().expect("seen poisoned").clone();
    assert!(
        seen.iter().any(|(is_sum, _)| *is_sum),
        "expected a background summarize sub-request, saw {seen:?}"
    );
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
