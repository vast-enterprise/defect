//! Prompt hook handler — feeds a step envelope into a single LLM call.

//! ## Not counted in the main loop's LLM call count
//!
//! The handler directly uses an [`Arc<dyn LlmProvider>`] to call
//! [`LlmProvider::complete`],
//! **without entering the history or counting toward `turn_request_count`** — this
//! prevents
//! a `SessionStart` hook from consuming one of the user's `max_turn_requests`.
//!
//! ## No nested Prompt handlers
//!
//! Design doc §4.3.1, rule 3: internal LLM calls must not emit hook events, to avoid
//! infinite recursion. This is guaranteed by the caller (the hook engine) — events
//! entered via `fire` will not trigger hooks again due to LLM calls made inside the
//! handler (there is no back-channel between the hook engine and the LLM provider;
//! `provider.complete` is unaware of the hook system). No additional protection is
//! needed on the handler side.
//!
//! ## Cold-start degradation
//!
//! If the LLM call on `SessionStart` fails, degrade per §3.5's table — `SessionStart`
//! must not block; errors are downgraded to warnings and the pipeline continues. This
//! invariant is enforced by [`super::DefaultHookEngine`]; the handler only needs to
//! propagate the error faithfully.
//!
//! [`LlmProvider::complete`]: crate::llm::LlmProvider::complete

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::future::BoxFuture;

use crate::error::BoxError;
use crate::llm::{
    CompletionRequest, LlmProvider, Message, MessageContent, ProviderChunk, Role, SamplingParams,
    ToolChoice,
};

use super::{HookCtx, HookError, StepHandler};

/// Template rendering strategy.
///
/// The `Template` variant performs simple `{{key}}` string substitution without
/// introducing heavy dependencies like handlebars or tera. Recognized keys are
/// documented in the `render_envelope` implementation:
/// - All events: `{{event}}` / `{{cwd}}` / `{{session_id}}`
/// - PreToolUse / Post*: `{{tool}}` / `{{tool_input}}` / `{{tool_error}}`
/// - UserPromptSubmit: `{{prompt}}`
/// - SessionStart: `{{session_source}}`
///
/// Unrecognized keys are replaced with an empty string (conservative semantics to
/// avoid sending raw `{{...}}` to the model when the template is misconfigured).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptRender {
    /// Feeds the JSON-serialized step envelope directly.
    Json,
    /// Replaces `{{key}}` placeholders with values from the event fields.
    Template { template: String },
}

/// Configuration for the prompt handler.
#[derive(Clone)]
pub struct PromptSpec {
    pub provider: Arc<dyn LlmProvider>,
    /// `None` = use [`Self::fallback_model`] (the session default model).
    pub model: Option<String>,
    /// Used when `model` is `None` — the CLI assembly phase feeds in `TurnConfig::model`.
    pub fallback_model: String,
    pub system: String,
    pub render: PromptRender,
    pub timeout_sec: Option<u64>,
}

impl std::fmt::Debug for PromptSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptSpec")
            .field("provider", &self.provider.info())
            .field("model", &self.model)
            .field("fallback_model", &self.fallback_model)
            .field("system", &self.system)
            .field("render", &self.render)
            .field("timeout_sec", &self.timeout_sec)
            .finish()
    }
}

/// Implementation of the `Prompt` handler.
pub struct PromptHandler {
    spec: PromptSpec,
}

impl PromptHandler {
    #[must_use]
    pub fn new(spec: PromptSpec) -> Self {
        Self { spec }
    }

    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.spec.timeout_sec.map(Duration::from_secs)
    }
}

impl StepHandler for PromptHandler {
    /// Renders the step envelope into user text (for JSON mode, serializes the envelope
    /// directly; for Template mode, extracts top-level fields using `{{key}}`), runs one
    /// LLM call, and uses the output text as the `additional_context` verdict.
    fn handle_step<'a>(
        &'a self,
        envelope: &'a serde_json::Value,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<serde_json::Value>, HookError>> {
        Box::pin(async move {
            let user_text = render_envelope(envelope, &self.spec.render);
            let request = CompletionRequest {
                model: self
                    .spec
                    .model
                    .clone()
                    .unwrap_or_else(|| self.spec.fallback_model.clone()),
                system: Some(Arc::from(self.spec.system.as_str())),
                messages: vec![Message {
                    role: Role::User,
                    content: Arc::from([MessageContent::Text { text: user_text }]),
                }],
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                sampling: SamplingParams::default(),
                hosted_capabilities: Default::default(),
            };
            let stream = self
                .spec
                .provider
                .complete(request, ctx.cancel.clone())
                .await
                .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?;
            let text = collect_text(stream).await?;
            if text.is_empty() {
                return Ok(None);
            }
            Ok(Some(serde_json::json!({ "additional_context": [text] })))
        })
    }
}

/// Renders the envelope: `Json` serializes it; `Template` replaces `{{key}}` with the
/// top-level field value (strings and numbers are converted to text directly).
fn render_envelope(envelope: &serde_json::Value, render: &PromptRender) -> String {
    match render {
        PromptRender::Json => serde_json::to_string(envelope).unwrap_or_default(),
        PromptRender::Template { template } => {
            let mut out = String::with_capacity(template.len());
            let mut rest = template.as_str();
            while let Some(start) = rest.find("{{") {
                let Some((head, tail)) = rest.split_at_checked(start) else {
                    break;
                };
                out.push_str(head);
                let Some(after_open) = tail.get(2..) else {
                    break;
                };
                let Some(close) = after_open.find("}}") else {
                    out.push_str(tail);
                    return out;
                };
                let Some(key) = after_open.get(..close).map(str::trim) else {
                    break;
                };
                match envelope.get(key) {
                    Some(serde_json::Value::String(s)) => out.push_str(s),
                    Some(other) => out.push_str(&other.to_string()),
                    None => {}
                }
                rest = match after_open.get(close + 2..) {
                    Some(s) => s,
                    None => break,
                };
            }
            out.push_str(rest);
            out
        }
    }
}

async fn collect_text(mut stream: crate::llm::ProviderStream) -> Result<String, HookError> {
    let mut out = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(ProviderChunk::TextDelta { text }) => out.push_str(&text),
            Ok(ProviderChunk::Stop { .. }) => break,
            Ok(_) => {} // Ignore thinking, tool_use, usage, etc.
            Err(err) => {
                return Err(HookError::HandlerFailed(BoxError::new(err)));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
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
}
