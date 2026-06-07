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
//! Internal LLM calls must not emit hook events, to avoid
//! infinite recursion. This is guaranteed by the caller (the hook engine) — events
//! entered via `fire` will not trigger hooks again due to LLM calls made inside the
//! handler (there is no back-channel between the hook engine and the LLM provider;
//! `provider.complete` is unaware of the hook system). No additional protection is
//! needed on the handler side.
//!
//! ## Cold-start degradation
//!
//! If the LLM call on `SessionStart` fails, degrade per the degradation table — `SessionStart`
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
mod tests;
