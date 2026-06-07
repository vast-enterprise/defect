use super::*;
use crate::llm::{
    Capabilities, FeatureSupport, ModelInfo, ProtocolId, ProviderChunk, ProviderError,
    ProviderErrorKind, ProviderInfo, ProviderStream, StopReason, ThinkingEcho,
};
use agent_client_protocol_schema::SessionId;
use futures::stream;
use std::path::Path;
use tokio_util::sync::CancellationToken;

fn ctx<'a>(session_id: &'a SessionId, cwd: &'a Path) -> HookCtx<'a> {
    HookCtx::new(session_id, cwd, CancellationToken::new())
}

/// Fake provider that always returns a single `TextDelta` followed by `Stop`.
struct FakeProvider {
    text: String,
}

fn fake_caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Unsupported,
        parallel_tool_calls: FeatureSupport::Unsupported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

fn fake_info(vendor: &str) -> ProviderInfo {
    ProviderInfo {
        vendor: vendor.to_string(),
        protocol: ProtocolId::OpenAiChat,
        display_name: vendor.to_string(),
    }
}

impl LlmProvider for FakeProvider {
    fn info(&self) -> ProviderInfo {
        fake_info("fake")
    }
    fn capabilities(&self) -> Capabilities {
        fake_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _model_id: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let chunks = vec![
            Ok(ProviderChunk::MessageStart {
                id: "fake".into(),
                model: "fake-1".into(),
            }),
            Ok(ProviderChunk::TextDelta {
                text: self.text.clone(),
            }),
            Ok(ProviderChunk::Stop {
                reason: StopReason::EndTurn,
            }),
        ];
        let s: ProviderStream = Box::pin(stream::iter(chunks));
        Box::pin(async move { Ok(s) })
    }
}

/// A fake provider whose `complete()` always returns `Err`.
struct FailingProvider;

impl LlmProvider for FailingProvider {
    fn info(&self) -> ProviderInfo {
        fake_info("failing")
    }
    fn capabilities(&self) -> Capabilities {
        fake_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _model_id: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        Box::pin(async {
            Err(ProviderError::new(ProviderErrorKind::Transport(
                BoxError::new(std::io::Error::other("boom")),
            )))
        })
    }
}

fn after_session_enter_env() -> serde_json::Value {
    serde_json::json!({"cwd": "/repo", "source": "new"})
}

/// Step model: LLM output text → `additional_context` verdict.
#[tokio::test]
async fn prompt_step_injects_additional_context() {
    let provider = Arc::new(FakeProvider {
        text: "preload-summary".into(),
    });
    let h = PromptHandler::new(PromptSpec {
        provider,
        model: None,
        fallback_model: "fake-1".into(),
        system: "summarize".into(),
        render: PromptRender::Template {
            template: "cwd={{cwd}}".into(),
        },
        timeout_sec: None,
    });
    let session_id = SessionId::new("s1");
    let cwd = Path::new("/repo");
    let verdict = h
        .handle_step(&after_session_enter_env(), ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    let arr = verdict["additional_context"].as_array().expect("array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0], "preload-summary");
}

/// Provider error → HandlerFailed.
#[tokio::test]
async fn prompt_step_propagates_provider_error() {
    let h = PromptHandler::new(PromptSpec {
        provider: Arc::new(FailingProvider),
        model: None,
        fallback_model: "fake-1".into(),
        system: "x".into(),
        render: PromptRender::Json,
        timeout_sec: None,
    });
    let session_id = SessionId::new("s1");
    let cwd = Path::new("/");
    let err = h
        .handle_step(&after_session_enter_env(), ctx(&session_id, cwd))
        .await
        .expect_err("expected error");
    assert!(matches!(err, HookError::HandlerFailed(_)));
}

// ----- render_envelope (envelope template rendering) -----

#[test]
fn envelope_template_replaces_known_keys() {
    let env = serde_json::json!({"cwd": "/repo", "source": "new"});
    let r = render_envelope(
        &env,
        &PromptRender::Template {
            template: "cwd={{cwd}} src={{source}}".into(),
        },
    );
    assert_eq!(r, "cwd=/repo src=new");
}

#[test]
fn envelope_template_missing_key_becomes_empty() {
    let env = serde_json::json!({"tool": "bash"});
    let r = render_envelope(
        &env,
        &PromptRender::Template {
            template: "before/{{nonexistent}}/after".into(),
        },
    );
    assert_eq!(r, "before//after");
}

#[test]
fn envelope_template_unclosed_passes_literally() {
    let env = serde_json::json!({});
    let r = render_envelope(
        &env,
        &PromptRender::Template {
            template: "hello {{ unclosed".into(),
        },
    );
    assert_eq!(r, "hello {{ unclosed");
}

#[test]
fn envelope_json_render_serializes_envelope() {
    let env = serde_json::json!({"tool": "bash", "args": {"k": 1}});
    let r = render_envelope(&env, &PromptRender::Json);
    let parsed: serde_json::Value = serde_json::from_str(&r).expect("valid json");
    assert_eq!(parsed["tool"], "bash");
}
