//! Session-level capability configuration and startup-time decision.
//!
//! Capability management for sessions.
//!
//! `WebSearchCapabilityMode` controls whether this session uses provider-hosted web
//! search:
//! - `Delegate`: use provider-hosted web search (fails at startup if the adapter does not
//!   support it)
//! - `Disabled`: do not expose hosted web search
//!
//! Note: the local grep/glob tool (`search` tool) is **not** managed at the capability
//! layer; it is controlled independently by `[tools.search].enabled` and is completely
//! separate from `web_search`. Both can be enabled simultaneously, and the LLM will see
//! both the hosted `web_search` and the local `search` tools.
//!
//! Decision timing: once at session startup. The `(provider, mode)` pair is fixed for the
//! session lifetime; the turn loop directly reuses the [`HostedCapabilities`] flag stored
//! on the session.

use serde::{Deserialize, Serialize};

use crate::llm::HostedCapabilities;

use super::SessionInitError;

/// Toggle for hosted web search capability.
///
/// TOML representation: `"delegate"` / `"disabled"`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchCapabilityMode {
    /// Delegate to provider-hosted web search. Session startup fails if the provider does
    /// not support it.
    Delegate,
    /// Do not expose hosted web search.
    #[default]
    Disabled,
}

/// Configuration for a single capability. Reserved for future capabilities of the same
/// form, such as `image_generation` / `code_execution`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchCapabilityConfig {
    pub mode: WebSearchCapabilityMode,
}

impl WebSearchCapabilityConfig {
    /// Constructs from a single `mode`. Cross-crate callers need this entry point because
    /// the struct is `#[non_exhaustive]` and cannot be built with a struct literal
    /// directly.
    #[must_use]
    pub const fn new(mode: WebSearchCapabilityMode) -> Self {
        Self { mode }
    }
}

/// Entry point for session-level capability configuration.
///
/// Constructed by `defect-config` on `EffectiveConfig.capabilities`, overlaid with
/// `providers.<p>.capabilities` overrides, and finally passed to the session during
/// assembly in [`AgentCore::create_session`][crate::session::AgentCore::create_session].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCapabilitiesConfig {
    pub web_search: WebSearchCapabilityConfig,
}

impl SessionCapabilitiesConfig {
    /// Construct from a single [`WebSearchCapabilityConfig`]. Cross-crate callers (e.g.
    /// `defect-config`) need this entry point because the struct is `#[non_exhaustive]`
    /// and cannot be built with a struct literal directly.
    #[must_use]
    pub const fn with_web_search(web_search: WebSearchCapabilityConfig) -> Self {
        Self { web_search }
    }
}

/// Runtime capabilities resolved at session startup.
///
/// Distinct from [`SessionCapabilitiesConfig`]: that is the user's configuration
/// (intent), while this is the actual enabled set after intersecting with the provider's
/// [`HostedCapabilities`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolvedSessionCapabilities {
    /// Whether this session uses hosted web search.
    /// `Delegate × supported` → `true`; otherwise → `false`.
    pub hosted: HostedCapabilities,
}

impl ResolvedSessionCapabilities {
    /// Resolve once: map `(mode, provider_hosted)` to the result.
    ///
    /// # Errors
    ///
    /// Returns [`SessionInitError::CapabilityUnsatisfied`] when the mode is `Delegate`
    /// but the provider does not support hosted web search.
    pub fn resolve(
        config: SessionCapabilitiesConfig,
        provider_hosted: HostedCapabilities,
        provider_id: &str,
    ) -> Result<Self, SessionInitError> {
        let mut hosted = HostedCapabilities::default();

        match config.web_search.mode {
            WebSearchCapabilityMode::Delegate => {
                if !provider_hosted.web_search {
                    return Err(SessionInitError::CapabilityUnsatisfied {
                        capability: "web_search",
                        provider: provider_id.to_string(),
                    });
                }
                hosted.web_search = true;
            }
            WebSearchCapabilityMode::Disabled => {}
        }

        Ok(Self { hosted })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosted(web_search: bool) -> HostedCapabilities {
        HostedCapabilities { web_search }
    }

    fn config(mode: WebSearchCapabilityMode) -> SessionCapabilitiesConfig {
        SessionCapabilitiesConfig {
            web_search: WebSearchCapabilityConfig { mode },
        }
    }

    #[test]
    fn delegate_with_supported_provider_enables_hosted() {
        let resolved = ResolvedSessionCapabilities::resolve(
            config(WebSearchCapabilityMode::Delegate),
            hosted(true),
            "anthropic",
        )
        .expect("should succeed");
        assert!(resolved.hosted.web_search);
    }

    #[test]
    fn delegate_with_unsupported_provider_fails() {
        let err = ResolvedSessionCapabilities::resolve(
            config(WebSearchCapabilityMode::Delegate),
            hosted(false),
            "deepseek",
        )
        .expect_err("should reject");
        match err {
            SessionInitError::CapabilityUnsatisfied {
                capability,
                provider,
            } => {
                assert_eq!(capability, "web_search");
                assert_eq!(provider, "deepseek");
            }
        }
    }

    #[test]
    fn disabled_exposes_nothing() {
        for support in [true, false] {
            let resolved = ResolvedSessionCapabilities::resolve(
                config(WebSearchCapabilityMode::Disabled),
                hosted(support),
                "any",
            )
            .expect("should succeed");
            assert!(!resolved.hosted.web_search);
        }
    }

    #[test]
    fn unsatisfied_error_message_includes_actionable_hint() {
        let err = ResolvedSessionCapabilities::resolve(
            config(WebSearchCapabilityMode::Delegate),
            hosted(false),
            "deepseek",
        )
        .expect_err("should reject");
        let msg = err.to_string();
        assert!(msg.contains("provider `deepseek` does not support hosted web_search"));
        assert!(msg.contains("[providers.deepseek.capabilities.web_search]"));
        assert!(msg.contains("mode = \"disabled\""));
        assert!(msg.contains("[capabilities.web_search]"));
    }
}
