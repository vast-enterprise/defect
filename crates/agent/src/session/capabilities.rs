//! session 级能力配置与启动期裁决。
//!
//! Capability management for sessions.
//!
//! `WebSearchCapabilityMode` 表达「这个 session 要不要走 provider-hosted
//! web search」：
//! - `Delegate`：走 provider-hosted web search（adapter 不支持时启动失败）
//! - `Disabled`：不暴露 hosted web search
//!
//! 注意：本地 grep/glob 工具（`search` tool）**不在** capability 层管理，
//! 由 `[tools.search].enabled` 单独决定，与 `web_search` 完全独立。两者可
//! 同时启用，LLM 会同时看到 hosted `web_search` 与本地 `search` 两个工具。
//!
//! 决策时机：session 启动期一次性。`(provider, mode)` 在 session 生命
//! 周期内不变，turn loop 直接复用 session 上记好的 [`HostedCapabilities`]
//! 标记。

use serde::{Deserialize, Serialize};

use crate::llm::HostedCapabilities;

use super::SessionInitError;

/// hosted web search 能力开关。
///
/// TOML 形态：`"delegate"` / `"disabled"`。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchCapabilityMode {
    /// 委托给 provider-hosted web search。provider 不支持时 session 启动失败。
    Delegate,
    /// 不暴露 hosted web search。
    #[default]
    Disabled,
}

/// 单条 capability 的配置。预留出口给后续 `image_generation` /
/// `code_execution` 等同形态 capability。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchCapabilityConfig {
    pub mode: WebSearchCapabilityMode,
}

impl WebSearchCapabilityConfig {
    /// 用单个 mode 构造。跨 crate 调用方需要这个入口，因为本结构体
    /// `#[non_exhaustive]` 后不能直接 struct literal。
    #[must_use]
    pub const fn new(mode: WebSearchCapabilityMode) -> Self {
        Self { mode }
    }
}

/// session 级能力配置入口。
///
/// 由 `defect-config` 在 `EffectiveConfig.capabilities` 上构造，并叠加
/// `providers.<p>.capabilities` 覆写，最后在 [`AgentCore::create_session`][crate::session::AgentCore::create_session]
/// 装配时传给 session。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCapabilitiesConfig {
    pub web_search: WebSearchCapabilityConfig,
}

impl SessionCapabilitiesConfig {
    /// 用单条 [`WebSearchCapabilityConfig`] 构造。跨 crate 调用方（例如
    /// `defect-config`）需要这个入口，因为本结构体 `#[non_exhaustive]`
    /// 后不能直接 struct literal。
    #[must_use]
    pub const fn with_web_search(web_search: WebSearchCapabilityConfig) -> Self {
        Self { web_search }
    }
}

/// session 启动期裁决出的运行时能力。
///
/// 与 [`SessionCapabilitiesConfig`] 区分：前者是用户配置（意图），这里
/// 是与 provider [`HostedCapabilities`] 交叉后得到的实际启用集合。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolvedSessionCapabilities {
    /// 最终决定本 session 是否走 hosted web search。
    /// `Delegate × supported` → `true`；其余情况 → `false`。
    pub hosted: HostedCapabilities,
}

impl ResolvedSessionCapabilities {
    /// 一次性裁决：把 `(mode, provider hosted)` 映射到结果。
    ///
    /// # Errors
    ///
    /// `Delegate` 但 provider 不支持 hosted web search 时返回
    /// [`SessionInitError::CapabilityUnsatisfied`]。
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
