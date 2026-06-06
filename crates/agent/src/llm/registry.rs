//! `ProviderRegistry`: catalog of configured providers + their model candidates.
//!
//! 用于 ACP 层向客户端暴露 `(provider, model)` 候选列表，并按 `(vendor, model)`
//! 对解析当前 turn 应该走哪个真实 provider。registry 本身**不实现**
//! [`LlmProvider`]——它是装配期落地的一份只读目录，session 在每次
//! `set_model` / `run_turn` 时按这对取出对应的真实 provider 跑。
//!
//! 设计要点：
//! - 每个 [`ProviderEntry`] 一份显式 `Vec<ModelInfo>`：CLI 装配期把
//!   `providers.<p>.default_model` 与 `providers.<p>.models` 翻成模型表，
//!   保证 ACP `list_models` 不依赖具体 adapter 的 `list_models` 网络调用。
//! - 选择键是 `(vendor, model id)` 对：同一 model id 可被多个 vendor 不同的
//!   provider 声明（多网关同模型）。ACP `set_model` 按这对切。
//! - 每个 entry 还携带 [`SessionCapabilitiesConfig`]——跨 provider 切换
//!   model 时 session 需要重新 resolve hosted capabilities。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::model::{ModelInfo, ProviderInfo};
use super::provider::LlmProvider;
use crate::session::SessionCapabilitiesConfig;

/// 一组 provider + 它公开的模型 id + 该 provider 的 session capability 配置。
#[derive(Clone)]
pub struct ProviderEntry {
    provider: Arc<dyn LlmProvider>,
    models: Vec<ModelInfo>,
    capabilities: SessionCapabilitiesConfig,
}

impl std::fmt::Debug for ProviderEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderEntry")
            .field("provider", &self.provider.info())
            .field("models", &self.models)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl ProviderEntry {
    #[must_use]
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        models: Vec<ModelInfo>,
        capabilities: SessionCapabilitiesConfig,
    ) -> Self {
        Self {
            provider,
            models,
            capabilities,
        }
    }

    #[must_use]
    pub fn provider(&self) -> &Arc<dyn LlmProvider> {
        &self.provider
    }

    #[must_use]
    pub fn models(&self) -> &[ModelInfo] {
        &self.models
    }

    #[must_use]
    pub fn capabilities(&self) -> SessionCapabilitiesConfig {
        self.capabilities
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderRegistryError {
    #[error("provider registry requires at least one entry")]
    Empty,
    #[error(
        "duplicate model id `{model}` declared twice by provider `{provider}`; \
         the same (provider, model) pair must be unique within a build"
    )]
    DuplicateSelection { provider: String, model: String },
    #[error(
        "default model `{model}` is not declared by provider `{provider}`; \
         add it under that provider, or point `default.provider` at the one that has it"
    )]
    UnknownDefaultModel { provider: String, model: String },
}

/// 装配期落地的"provider 目录"。session 持有 `Arc<ProviderRegistry>`。
#[derive(Debug)]
pub struct ProviderRegistry {
    entries: Vec<ProviderEntry>,
    /// (vendor, model id) → entries 索引。同一 model id 可由多个 provider
    /// （vendor 不同）声明——选择键是这对 (vendor, model)，不再是裸 model id。
    model_index: HashMap<(String, String), usize>,
    /// 默认 (provider, model) 对应的 entries 索引 + entry.models 索引。
    default: (usize, usize),
}

impl ProviderRegistry {
    /// 单 provider 单 model 的便捷构造。测试 / EchoProvider / `provider()`
    /// builder 入口走此路径——保持 `ProviderRegistry::new` 的不变量校验
    /// （非空 + default_model 必须落在某 entry）成立的最小形态。
    #[must_use]
    pub fn single(provider: Arc<dyn LlmProvider>, default_model: ModelInfo) -> Arc<Self> {
        let vendor = provider.info().vendor;
        let model_id = default_model.id.clone();
        let entries = vec![ProviderEntry::new(
            provider,
            vec![default_model],
            SessionCapabilitiesConfig::default(),
        )];
        Arc::new(
            Self::new(entries, &vendor, &model_id)
                .expect("single-entry registry with matching default model is always valid"),
        )
    }

    /// 用一组 entries + 默认 `(provider vendor, model id)` 对装配。该对必须
    /// 出现在某个 entry 的 `(vendor, models)` 里。
    ///
    /// 同一 model id 可被多个 vendor 不同的 entry 声明（多网关同模型）——选择键
    /// 是 `(vendor, model)`。只有**同一** `(vendor, model)` 对重复出现才是配置
    /// 错误。
    ///
    /// # Errors
    ///
    /// - [`ProviderRegistryError::Empty`]：entries 为空
    /// - [`ProviderRegistryError::DuplicateSelection`]：同一 `(vendor, model)`
    ///   对出现两次
    /// - [`ProviderRegistryError::UnknownDefaultModel`]：默认 `(vendor, model)`
    ///   对不在任何 entry 里
    pub fn new(
        entries: Vec<ProviderEntry>,
        default_provider: &str,
        default_model: &str,
    ) -> Result<Self, ProviderRegistryError> {
        if entries.is_empty() {
            return Err(ProviderRegistryError::Empty);
        }

        let mut model_index = HashMap::new();
        let mut default_pos = None;
        for (entry_idx, entry) in entries.iter().enumerate() {
            let provider_vendor = entry.provider.info().vendor;
            let mut seen_in_entry = HashSet::new();
            for (model_idx, model) in entry.models.iter().enumerate() {
                if !seen_in_entry.insert(model.id.clone()) {
                    continue;
                }
                let key = (provider_vendor.clone(), model.id.clone());
                if model_index.insert(key, entry_idx).is_some() {
                    return Err(ProviderRegistryError::DuplicateSelection {
                        provider: provider_vendor,
                        model: model.id.clone(),
                    });
                }
                if provider_vendor == default_provider
                    && model.id == default_model
                    && default_pos.is_none()
                {
                    default_pos = Some((entry_idx, model_idx));
                }
            }
        }

        let default = default_pos.ok_or_else(|| ProviderRegistryError::UnknownDefaultModel {
            provider: default_provider.to_string(),
            model: default_model.to_string(),
        })?;

        Ok(Self {
            entries,
            model_index,
            default,
        })
    }

    /// 默认 entry——session 启动时用来初始化当前 provider/model。
    #[must_use]
    pub fn default_entry(&self) -> &ProviderEntry {
        let (entry_idx, _) = self.default;
        self.entries
            .get(entry_idx)
            .expect("default index validated in `new`")
    }

    /// 默认 model id。
    #[must_use]
    pub fn default_model(&self) -> &str {
        let (entry_idx, model_idx) = self.default;
        let entry = self
            .entries
            .get(entry_idx)
            .expect("default index validated in `new`");
        entry
            .models
            .get(model_idx)
            .map(|m| m.id.as_str())
            .expect("default model index validated in `new`")
    }

    /// 按 `(vendor, model id)` 对查找对应 entry。`None` 表示当前 registry 没有
    /// 声明此对。
    #[must_use]
    pub fn entry_for(&self, vendor: &str, model_id: &str) -> Option<&ProviderEntry> {
        self.model_index
            .get(&(vendor.to_string(), model_id.to_string()))
            .and_then(|idx| self.entries.get(*idx))
    }

    /// 按裸 model id 查找首个声明它的 entry（装配顺序）。供没有 vendor 维度的
    /// 旧路径（如 prompt hook 的 `model` 字段）用——有歧义时取第一个。
    #[must_use]
    pub fn first_entry_for_model(&self, model_id: &str) -> Option<&ProviderEntry> {
        self.entries
            .iter()
            .find(|entry| entry.models.iter().any(|m| m.id == model_id))
    }

    /// 列出所有 entry（按装配顺序）。
    #[must_use]
    pub fn entries(&self) -> &[ProviderEntry] {
        &self.entries
    }

    /// 平铺出所有 (provider_info, model) 对。ACP `list_models` 用此构造
    /// `SessionModelState::available_models`。
    #[must_use]
    pub fn list_candidates(&self) -> Vec<ModelCandidate> {
        let mut out = Vec::new();
        for entry in &self.entries {
            let info = entry.provider.info();
            for model in &entry.models {
                out.push(ModelCandidate {
                    provider: info.clone(),
                    model: model.clone(),
                });
            }
        }
        out
    }

    /// 按 model id 查 candidate；用于 ACP 层渲染 description。
    #[must_use]
    pub fn candidate_for(&self, vendor: &str, model_id: &str) -> Option<ModelCandidate> {
        let entry = self.entry_for(vendor, model_id)?;
        let model = entry.models.iter().find(|m| m.id == model_id)?.clone();
        Some(ModelCandidate {
            provider: entry.provider.info(),
            model,
        })
    }
}

/// `(provider, model)` 平铺一对——ACP `list_models` 的最小投影单元。
#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub provider: ProviderInfo,
    pub model: ModelInfo,
}

#[cfg(test)]
mod tests {
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

    /// 只带 vendor 身份、不真生成的占位 provider——registry 装配只读 `info()`。
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
        fn list_models(
            &self,
        ) -> BoxFuture<'_, Result<Vec<ModelInfo>, super::super::ProviderError>> {
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
        // 两个网关（vendor 不同）都声明同一个 model id `gpt-4o`——应能装配，
        // 且按 (vendor, model) 对各自解析到正确的 entry。
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
        // default 对解析到声明它的那家。
        assert_eq!(registry.default_entry().provider().info().vendor, "gw_a");
        assert_eq!(registry.default_model(), "gpt-4o");
    }

    #[test]
    fn duplicate_vendor_model_pair_errors() {
        // 同一 (vendor, model) 对重复出现才是真错误。
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
        // 无 vendor 维度的旧路径取首个声明它的 entry（装配顺序）。
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
}
