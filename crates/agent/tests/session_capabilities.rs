//! End-to-end tests for capability resolution during session startup.
//!
//! Complementary to the pure-function unit tests in `session::capabilities::tests`: this
//! exercises the real path through `DefaultAgentCore::create_session`, verifying that
//! `(SessionCapabilitiesConfig, hosted_capabilities)` is read during assembly and that
//! the failure path produces `AgentError::Init(SessionInitError::CapabilityUnsatisfied)`.

use std::sync::Arc;

use agent_client_protocol_schema::SessionId;
use defect_agent::fs::{FsBackend, NoopFsBackend};
use defect_agent::llm::{
    Capabilities, CompletionRequest, FeatureSupport, HostedCapabilities, LlmProvider, ModelInfo,
    ProtocolId, ProviderError, ProviderInfo, ProviderStream, ThinkingEcho,
};
use defect_agent::session::{
    AgentCore, AgentError, DefaultAgentCore, Frontend, SessionCapabilitiesConfig, SessionInitError,
    TurnConfig, WebSearchCapabilityConfig, WebSearchCapabilityMode, new_session_id,
};
use defect_agent::shell::{NoopShellBackend, ShellBackend};
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

fn unsupported_caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Supported,
        parallel_tool_calls: FeatureSupport::Supported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

/// A provider that does not support hosted web search.
struct NoHostedProvider;
impl LlmProvider for NoHostedProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "no-hosted".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "No Hosted Provider".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
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
        unimplemented!("not exercised")
    }
}

/// Provider that supports hosted web search.
struct HostedSearchProvider;
impl LlmProvider for HostedSearchProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: "hosted".to_string(),
            protocol: ProtocolId::AnthropicMessages,
            display_name: "Hosted Web Search Provider".to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        unsupported_caps()
    }
    fn hosted_capabilities(&self) -> HostedCapabilities {
        HostedCapabilities::with_web_search(true)
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
        unimplemented!("not exercised")
    }
}

fn build_core(
    provider: Arc<dyn LlmProvider>,
    capabilities: SessionCapabilitiesConfig,
) -> DefaultAgentCore {
    DefaultAgentCore::builder()
        .provider(provider)
        .config(TurnConfig {
            model: "test-001".to_string(),
            ..TurnConfig::default()
        })
        .capabilities(capabilities)
        .build()
}

#[tokio::test]
async fn delegate_with_unsupported_provider_fails_session_init() {
    let core = build_core(
        Arc::new(NoHostedProvider) as Arc<dyn LlmProvider>,
        SessionCapabilitiesConfig::with_web_search(WebSearchCapabilityConfig::new(
            WebSearchCapabilityMode::Delegate,
        )),
    );

    let cwd = std::env::current_dir().expect("cwd");
    let result = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await;

    match result {
        Ok(_) => panic!("expected CapabilityUnsatisfied; got Ok"),
        Err(AgentError::Init(SessionInitError::CapabilityUnsatisfied {
            capability,
            provider,
        })) => {
            assert_eq!(capability, "web_search");
            assert_eq!(provider, "no-hosted");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn delegate_with_supported_provider_creates_session() {
    let core = build_core(
        Arc::new(HostedSearchProvider) as Arc<dyn LlmProvider>,
        SessionCapabilitiesConfig::with_web_search(WebSearchCapabilityConfig::new(
            WebSearchCapabilityMode::Delegate,
        )),
    );

    let cwd = std::env::current_dir().expect("cwd");
    let session = core
        .create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("create session");

    // Indirectly verify that the returned session is valid via `Session::id`; the
    // `hosted_capabilities` field is internal to `DefaultSession` with no public getter,
    // so it will be validated later on the turn path
    // (`CompletionRequest::hosted_capabilities`).
    let _ = session.id();
}

#[tokio::test]
async fn disabled_mode_succeeds_regardless_of_provider() {
    for provider in [
        Arc::new(NoHostedProvider) as Arc<dyn LlmProvider>,
        Arc::new(HostedSearchProvider) as Arc<dyn LlmProvider>,
    ] {
        let core = build_core(
            provider,
            SessionCapabilitiesConfig::with_web_search(WebSearchCapabilityConfig::new(
                WebSearchCapabilityMode::Disabled,
            )),
        );
        let cwd = std::env::current_dir().expect("cwd");
        core.create_session(
            SessionId::new(new_session_id()),
            cwd,
            vec![],
            Arc::new(NoopFsBackend) as Arc<dyn FsBackend>,
            Arc::new(NoopShellBackend) as Arc<dyn ShellBackend>,
            Frontend::Headless,
        )
        .await
        .expect("disabled mode should always succeed");
    }
}
