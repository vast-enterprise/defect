//! Prompt hook handler — 把 [`HookEvent`] 喂给一次 LLM 调用。
//!
//! 详见 `docs/internal/hooks.md` §4.3。
//!
//! ## 不在主循环 LLM 调用计数
//!
//! handler 直接拿一份 [`Arc<dyn LlmProvider>`] 跑 [`LlmProvider::complete`]，
//! **不进 history、不计 `turn_request_count`**——避免一个 SessionStart hook
//! 把用户的 `max_turn_requests` 消耗一次。
//!
//! ## 不允许 Prompt handler 套 Prompt handler
//!
//! 设计文档 §4.3.1 第三条：内部 LLM 调用不再 emit hook 事件，避免无限递归。
//! 这条由调用方（hook engine）保证——`fire` 入口的事件不会因为 handler 内
//! 产生的 LLM 调用再次触发 hook（hook engine 与 LLM provider 之间没有反向
//! 通道；provider.complete 不感知 hook 系统）。本 handler 实现侧无需额外
//! 防护。
//!
//! ## 冷启动降级
//!
//! `SessionStart` 上 LLM 失败按 §3.5 表降级——`SessionStart` 不允许 block，
//! 错误降为 warning，pipeline 继续。这条不变量由 [`super::DefaultHookEngine`]
//! 实现，handler 端只负责把错误如实抛上去。
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

/// 模板渲染策略。详见 `docs/internal/hooks.md` §4.3。
///
/// `Template` 形态用极简 `{{key}}` 字符串替换——不引入 handlebars/tera
/// 这类重型依赖。可识别的 key 见 [`render_template`] 的实现：
/// - 全部事件：`{{event}}` / `{{cwd}}` / `{{session_id}}`
/// - PreToolUse / Post*：`{{tool}}` / `{{tool_input}}` / `{{tool_error}}`
/// - UserPromptSubmit：`{{prompt}}`
/// - SessionStart：`{{session_source}}`
///
/// 未识别的 key 替换成空串（保守语义，避免模板写错时把 `{{...}}` 原样发
/// 给模型）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptRender {
    /// 直接喂 [`HookEvent`] 的 JSON 序列化结果。
    Json,
    /// 用 `{{key}}` 字符串替换从 event 字段取值。
    Template { template: String },
}

/// Prompt handler 的配置。
#[derive(Clone)]
pub struct PromptSpec {
    pub provider: Arc<dyn LlmProvider>,
    /// `None` = 用 [`Self::fallback_model`]（session 默认 model）。
    pub model: Option<String>,
    /// `model` 为 `None` 时用这个——CLI 装配期把 `TurnConfig::model` 喂进来。
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

/// `Prompt` handler 实现。
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
    /// Step 模型：把 step 信封渲染成 user 文本（JSON 形态直接序列化信封；Template 形态按
    /// `{{key}}` 取信封顶层字段），跑一次 LLM，输出文本作为 `additional_context` verdict。
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

/// Step 信封渲染：JSON = 序列化信封；Template = `{{key}}` 取信封顶层字段（字符串/数字直接转文本）。
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
            Ok(_) => {} // 忽略 thinking / tool_use / usage 等
            Err(err) => {
                return Err(HookError::HandlerFailed(BoxError::new(err)));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod test {
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

    /// fake provider：固定返回一条 TextDelta + Stop。
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

    /// fake provider：complete() 直接 Err。
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

    /// Step 模型：LLM 输出文本 → additional_context verdict。
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

    /// provider 出错 → HandlerFailed。
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

    // ----- render_envelope（信封模板渲染）-----

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

