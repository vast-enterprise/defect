//! v0 内嵌的占位 LLM provider：把用户最近一条消息原样回送。
//!
//! 仅用于让 stdio ACP 链路能在无外部 LLM 凭据的环境下跑通；真实
//! provider 在 `defect-llm` 里。

use std::pin::Pin;

use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, LlmProvider, ModelInfo, ProtocolId,
    ProviderChunk, ProviderError, ProviderInfo, ProviderStream, StopReason, ThinkingEcho,
};
use futures::future::BoxFuture;
use futures::stream;
use tokio_util::sync::CancellationToken;

/// 把最近一条 `MessageContent::Text` 原样回送的 stub。
pub struct EchoProvider;

impl EchoProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EchoProvider {
    fn default() -> Self {
        Self
    }
}

impl LlmProvider for EchoProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            // provider 身份是 `defect`（内置兜底）；model id 仍是 `echo`——回送是它的
            // 功能。选择对即 `defect::echo`。
            vendor: "defect".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Defect built-in echo".to_string(),
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
        Box::pin(async { Ok(vec![echo_model_info()]) })
    }

    fn model_info(&self, model_id: &str) -> Option<ModelInfo> {
        (model_id == "echo").then(echo_model_info)
    }

    fn complete(
        &self,
        req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, ProviderError>> {
        let echo = last_user_text(&req).unwrap_or_else(|| String::from("(empty prompt)"));
        let model = req.model.clone();
        Box::pin(async move {
            let chunks: Vec<Result<ProviderChunk, ProviderError>> = vec![
                Ok(ProviderChunk::MessageStart {
                    id: "echo-0".to_string(),
                    model,
                }),
                Ok(ProviderChunk::TextDelta {
                    text: format!("echo: {echo}"),
                }),
                Ok(ProviderChunk::Stop {
                    reason: StopReason::EndTurn,
                }),
            ];
            let s: Pin<
                Box<dyn futures::Stream<Item = Result<ProviderChunk, ProviderError>> + Send>,
            > = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

fn echo_model_info() -> ModelInfo {
    ModelInfo {
        id: "echo".to_string(),
        display_name: Some("Echo".to_string()),
        context_window: None,
        max_output_tokens: None,
        deprecated: false,
        capabilities_overrides: Default::default(),
    }
}

fn last_user_text(req: &CompletionRequest) -> Option<String> {
    use defect_agent::llm::{MessageContent, Role};
    req.messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, Role::User))
        .and_then(|m| {
            m.content.iter().find_map(|c| match c {
                MessageContent::Text { text } => Some(text.clone()),
                _ => None,
            })
        })
}
