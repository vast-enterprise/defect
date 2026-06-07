use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::llm::capability::{Capabilities, FeatureSupport, ThinkingEcho};
use crate::llm::model::ProtocolId;
use crate::llm::provider::ProviderStream;
use crate::llm::request::CompletionRequest;

fn stub_caps() -> Capabilities {
    Capabilities {
        tool_calls: FeatureSupport::Unsupported,
        parallel_tool_calls: FeatureSupport::Unsupported,
        thinking: FeatureSupport::Unsupported,
        vision: FeatureSupport::Unsupported,
        prompt_cache: FeatureSupport::Unsupported,
        thinking_echo: ThinkingEcho::Forbidden,
    }
}

/// A stub provider that carries only a vendor identity and does not actually generate
/// anything — used by the registry to assemble a read-only `info()`.
struct StubProvider {
    vendor: &'static str,
}

impl LlmProvider for StubProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            vendor: self.vendor.to_string(),
            protocol: ProtocolId::OpenAiChat,
            display_name: self.vendor.to_string(),
        }
    }
    fn capabilities(&self) -> Capabilities {
        stub_caps()
    }
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>, super::super::ProviderError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn model_info(&self, _model_id: &str) -> Option<ModelInfo> {
        None
    }
    fn complete(
        &self,
        _req: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ProviderStream, super::super::ProviderError>> {
        unreachable!("registry tests never drive completion")
    }
}

fn model(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        display_name: None,
        context_window: None,
        max_output_tokens: None,
        deprecated: false,
        capabilities_overrides: Default::default(),
    }
}

fn entry(vendor: &'static str, models: &[&str]) -> ProviderEntry {
    ProviderEntry::new(
        Arc::new(StubProvider { vendor }),
        models.iter().map(|m| model(m)).collect(),
        SessionCapabilitiesConfig::default(),
    )
}

#[test]
fn same_model_id_across_distinct_vendors_resolves_per_vendor() {
    // Two gateways with different vendors both declare the same model id `gpt-4o` —
    // assembly should succeed, and each (vendor, model) pair should resolve to its
    // correct entry.
    let registry = ProviderRegistry::new(
        vec![entry("gw_a", &["gpt-4o"]), entry("gw_b", &["gpt-4o"])],
        "gw_a",
        "gpt-4o",
    )
    .expect("distinct vendors with same model id must assemble");

    let a = registry
        .entry_for("gw_a", "gpt-4o")
        .expect("gw_a/gpt-4o present");
    let b = registry
        .entry_for("gw_b", "gpt-4o")
        .expect("gw_b/gpt-4o present");
    assert_eq!(a.provider().info().vendor, "gw_a");
    assert_eq!(b.provider().info().vendor, "gw_b");
    // The default entry resolves to the one that declared it.
    assert_eq!(registry.default_entry().provider().info().vendor, "gw_a");
    assert_eq!(registry.default_model(), "gpt-4o");
}

#[test]
fn duplicate_vendor_model_pair_errors() {
    // Only a duplicate (vendor, model) pair is a real error.
    let err = ProviderRegistry::new(
        vec![entry("gw_a", &["gpt-4o"]), entry("gw_a", &["gpt-4o"])],
        "gw_a",
        "gpt-4o",
    )
    .expect_err("duplicate (vendor, model) pair must error");
    assert!(matches!(
        err,
        ProviderRegistryError::DuplicateSelection { .. }
    ));
}

#[test]
fn unknown_default_pair_errors() {
    let err = ProviderRegistry::new(vec![entry("gw_a", &["gpt-4o"])], "gw_b", "gpt-4o")
        .expect_err("default pair not present must error");
    assert!(matches!(
        err,
        ProviderRegistryError::UnknownDefaultModel { .. }
    ));
}

#[test]
fn first_entry_for_model_picks_assembly_order() {
    let registry = ProviderRegistry::new(
        vec![entry("gw_a", &["gpt-4o"]), entry("gw_b", &["gpt-4o"])],
        "gw_a",
        "gpt-4o",
    )
    .unwrap();
    // Without a vendor dimension, the old path picks the first entry that declares it
    // (assembly order).
    assert_eq!(
        registry
            .first_entry_for_model("gpt-4o")
            .unwrap()
            .provider()
            .info()
            .vendor,
        "gw_a"
    );
}
