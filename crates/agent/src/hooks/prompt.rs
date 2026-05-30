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

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::ContentBlock;
use futures::StreamExt;
use futures::future::BoxFuture;

use crate::error::BoxError;
use crate::llm::{
    CompletionRequest, LlmProvider, Message, MessageContent, ProviderChunk, Role, SamplingParams,
    ToolChoice,
};

use super::{HookCapability, HookCtx, HookError, HookEvent, HookHandler, HookOutcome};

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

impl HookHandler for PromptHandler {
    fn capability(&self) -> HookCapability {
        HookCapability::Intercept
    }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        Box::pin(async move {
            let user_text = render_event(event, &ctx, &self.spec.render);
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
                return Ok(HookOutcome::default());
            }
            Ok(HookOutcome {
                append: vec![ContentBlock::from(text)],
                ..Default::default()
            })
        })
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

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

fn render_event(event: &HookEvent<'_>, ctx: &HookCtx<'_>, render: &PromptRender) -> String {
    match render {
        PromptRender::Json => {
            let envelope = super::command::CommandEventEnvelope::from_event(event);
            serde_json::to_string(&envelope).unwrap_or_default()
        }
        PromptRender::Template { template } => render_template(template, event, ctx),
    }
}

fn render_template(template: &str, event: &HookEvent<'_>, ctx: &HookCtx<'_>) -> String {
    let vars = template_vars(event, ctx);
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let (head, tail) = match rest.split_at_checked(start) {
            Some(parts) => parts,
            None => break,
        };
        out.push_str(head);
        // tail 以 "{{" 开头
        let after_open = match tail.get(2..) {
            Some(s) => s,
            None => break,
        };
        let Some(close) = after_open.find("}}") else {
            // 没闭合——按字面输出 "{{" 后剩余部分
            out.push_str(tail);
            return out;
        };
        let key = match after_open.get(..close) {
            Some(k) => k.trim(),
            None => break,
        };
        if let Some(value) = vars.get(key) {
            out.push_str(value);
        }
        // 否则未识别 key 替换成空串
        rest = match after_open.get(close + 2..) {
            Some(s) => s,
            None => break,
        };
    }
    out.push_str(rest);
    out
}

fn template_vars(event: &HookEvent<'_>, ctx: &HookCtx<'_>) -> BTreeMap<&'static str, String> {
    let mut vars: BTreeMap<&'static str, String> = BTreeMap::new();
    vars.insert("event", event.kind_str().to_string());
    vars.insert("cwd", ctx.cwd.to_string_lossy().into_owned());
    vars.insert("session_id", ctx.session_id.0.to_string());

    match event {
        HookEvent::SessionStart { source, .. } => {
            let label = match source {
                super::SessionSource::New => "new",
                super::SessionSource::Resume { .. } => "resume",
            };
            vars.insert("session_source", label.to_string());
        }
        HookEvent::UserPromptSubmit { content } => {
            let prompt_text = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            vars.insert("prompt", prompt_text);
        }
        HookEvent::PreToolUse { name, args, .. } => {
            vars.insert("tool", (*name).to_string());
            vars.insert("tool_input", args.to_string());
        }
        HookEvent::PostToolUse { name, fields, .. } => {
            vars.insert("tool", (*name).to_string());
            vars.insert(
                "tool_input",
                serde_json::to_string(fields).unwrap_or_default(),
            );
        }
        HookEvent::PostToolUseFailure { name, error, .. } => {
            vars.insert("tool", (*name).to_string());
            vars.insert("tool_error", (*error).to_string());
        }
        _ => {}
    }
    vars
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hooks::SessionSource;
    use crate::llm::{
        Capabilities, FeatureSupport, ModelInfo, ProtocolId, ProviderChunk, ProviderError,
        ProviderErrorKind, ProviderInfo, ProviderStream, StopReason, ThinkingEcho,
    };
    use agent_client_protocol_schema::{SessionId, ToolCallId};
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

    #[tokio::test]
    async fn prompt_handler_appends_text_block() {
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
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert_eq!(outcome.append.len(), 1);
    }

    #[tokio::test]
    async fn prompt_handler_propagates_provider_error() {
        let provider = Arc::new(FailingProvider);
        let h = PromptHandler::new(PromptSpec {
            provider,
            model: None,
            fallback_model: "fake-1".into(),
            system: "x".into(),
            render: PromptRender::Json,
            timeout_sec: None,
        });
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let err = h
            .handle(&ev, ctx(&session_id, cwd))
            .await
            .expect_err("expected error");
        assert!(matches!(err, HookError::HandlerFailed(_)));
    }

    #[test]
    fn template_replaces_known_keys() {
        let session_id = SessionId::new("s9");
        let cwd = Path::new("/repo");
        let ev = HookEvent::SessionStart {
            source: SessionSource::New,
            cwd,
        };
        let rendered = render_template(
            "[event={{event}}] cwd={{cwd}} src={{session_source}}",
            &ev,
            &ctx(&session_id, cwd),
        );
        assert_eq!(rendered, "[event=session_start] cwd=/repo src=new");
    }

    #[test]
    fn template_missing_key_becomes_empty() {
        let session_id = SessionId::new("s9");
        let cwd = Path::new("/");
        let ev = HookEvent::UserPromptSubmit { content: &[] };
        // tool_error 在 UserPromptSubmit 上不存在
        let rendered = render_template("before/{{tool_error}}/after", &ev, &ctx(&session_id, cwd));
        assert_eq!(rendered, "before//after");
    }

    #[test]
    fn template_unclosed_passes_literally() {
        let session_id = SessionId::new("s9");
        let cwd = Path::new("/");
        let ev = HookEvent::UserPromptSubmit { content: &[] };
        let rendered = render_template("hello {{ unclosed", &ev, &ctx(&session_id, cwd));
        assert_eq!(rendered, "hello {{ unclosed");
    }

    #[test]
    fn template_user_prompt_text_extraction() {
        let session_id = SessionId::new("s9");
        let cwd = Path::new("/");
        let blocks = vec![ContentBlock::from("hello "), ContentBlock::from("world")];
        let ev = HookEvent::UserPromptSubmit { content: &blocks };
        let rendered = render_template("Q: {{prompt}}", &ev, &ctx(&session_id, cwd));
        assert_eq!(rendered, "Q: hello world");
    }

    #[test]
    fn json_render_matches_envelope() {
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"k": 1});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: crate::tool::SafetyClass::ReadOnly,
        };
        let rendered = render_event(&ev, &ctx(&session_id, cwd), &PromptRender::Json);
        let parsed: serde_json::Value = serde_json::from_str(&rendered).expect("valid json");
        assert_eq!(parsed["type"], "pre_tool_use");
    }
}
