//! `ProviderRegistry`: catalog of configured providers + their model candidates.
//!
//! Exposes the `(provider, model)` candidate list to the ACP layer, and resolves which
//! real provider should handle the current turn based on the `(vendor, model)` pair. The
//! registry itself does **not** implement [`LlmProvider`] — it is a read-only directory
//! assembled at configuration time. The session calls `set_model` / `run_turn` to look up
//! the corresponding real provider using this pair.
//!
//! Design notes:
//! - Each [`ProviderEntry`] carries an explicit `Vec<ModelInfo>`: during CLI assembly,
//!   `providers.<p>.default_model` and `providers.<p>.models` are flattened into a model
//!   table, so that ACP `list_models` does not require a network call to the adapter's
//!   own `list_models`.
//! - The selection key is the `(vendor, model id)` pair: the same model id may be
//!   declared by multiple providers with different vendors (multi-gateway, same model).
//!   ACP `set_model` switches on this pair.
//! - Each entry also carries [`SessionCapabilitiesConfig`] — when switching models across
//!   providers, the session must re-resolve hosted capabilities.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use defect_core::llm::{LlmProvider, ModelInfo, ProviderInfo};

use crate::session::SessionCapabilitiesConfig;

/// A provider, the model IDs it exposes, and its session capability configuration.
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

/// A "provider directory" that is materialized at assembly time. The session holds an
/// `Arc<ProviderRegistry>`.
#[derive(Debug)]
pub struct ProviderRegistry {
    entries: Vec<ProviderEntry>,
    /// (vendor, model id) → entries index. Multiple providers (with different vendors)
    /// may declare the same model id — the lookup key is the pair (vendor, model), not
    /// the bare model id.
    model_index: HashMap<(String, String), usize>,
    /// Index into `entries` for the default (provider, model), plus index into that
    /// entry's `models`.
    default: (usize, usize),
}

impl ProviderRegistry {
    /// A convenience constructor for a single provider with a single model.
    /// Used by tests, `EchoProvider`, and the `provider()` builder entry point.
    /// This is the minimal form that satisfies the invariants checked by
    /// `ProviderRegistry::new` (non-empty + `default_model` must belong to an entry).
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

    /// Constructs a registry from a list of entries and a default `(provider vendor,
    /// model id)` pair. The pair must appear in the `(vendor, models)` list of some
    /// entry.
    ///
    /// The same model id may be declared by multiple entries with different vendors
    /// (multiple gateways sharing a model) — the selection key is `(vendor, model)`. Only
    /// a duplicate `(vendor, model)` pair is a configuration error.
    ///
    /// # Errors
    ///
    /// - [`ProviderRegistryError::Empty`]: entries is empty
    /// - [`ProviderRegistryError::DuplicateSelection`]: the same `(vendor, model)` pair
    ///   appears twice
    /// - [`ProviderRegistryError::UnknownDefaultModel`]: the default `(vendor, model)`
    ///   pair is not present in any entry
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

    /// The default entry used to initialize the current provider/model when a session
    /// starts.
    #[must_use]
    pub fn default_entry(&self) -> &ProviderEntry {
        let (entry_idx, _) = self.default;
        self.entries
            .get(entry_idx)
            .expect("default index validated in `new`")
    }

    /// The default model ID.
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

    /// Look up the entry for a given `(vendor, model id)` pair. Returns `None` if the
    /// registry does not declare this pair.
    #[must_use]
    pub fn entry_for(&self, vendor: &str, model_id: &str) -> Option<&ProviderEntry> {
        self.model_index
            .get(&(vendor.to_string(), model_id.to_string()))
            .and_then(|idx| self.entries.get(*idx))
    }

    /// Look up the first entry that declares the given bare model ID (in assembly order).
    /// Used by legacy paths that lack a vendor dimension, such as the `model` field in
    /// prompt hooks — when there are multiple matches, the first one is returned.
    #[must_use]
    pub fn first_entry_for_model(&self, model_id: &str) -> Option<&ProviderEntry> {
        self.entries
            .iter()
            .find(|entry| entry.models.iter().any(|m| m.id == model_id))
    }

    /// Returns all entries in assembly order.
    #[must_use]
    pub fn entries(&self) -> &[ProviderEntry] {
        &self.entries
    }

    /// Flatten all (provider_info, model) pairs. ACP `list_models` uses this to build
    /// `SessionModelState::available_models`.
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

    /// Look up a candidate by model ID; used by the ACP layer to render the description.
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

/// A flattened `(provider, model)` pair — the smallest projection unit of ACP
/// `list_models`.
#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub provider: ProviderInfo,
    pub model: ModelInfo,
}

#[cfg(test)]
mod tests;
