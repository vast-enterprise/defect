//! session 级能力配置与启动期裁决。
//!
//! 设计详见
//! [`docs/proposals/search-capability-and-fetch-tool.md`](../../../../docs/proposals/search-capability-and-fetch-tool.md)
//! §4 / §6.1。
//!
//! `SearchCapabilityMode` 表达「这个 session 的 search 能力来源是什么」：
//! - `Delegate`：走 provider-hosted search
//! - `Local`：走 defect 本地 `search` tool
//! - `Disabled`：完全不暴露 search
//!
//! 决策时机：session 启动期一次性。`(provider, mode)` 在 session 生命
//! 周期内不变，turn loop 直接复用 session 上记好的 [`HostedCapabilities`]
//! 标记。

use serde::{Deserialize, Serialize};

use crate::llm::HostedCapabilities;

use super::SessionInitError;

/// search 能力来源选择。
///
/// TOML 形态：`"delegate"` / `"local"` / `"disabled"`。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchCapabilityMode {
    /// 委托给 provider-hosted search。provider 不支持时 session 启动失败。
    Delegate,
    /// 用 defect 本地 `search` tool。provider 是否支持 hosted 不影响。
    #[default]
    Local,
    /// 既不暴露 hosted，也不暴露本地 `search` tool。
    Disabled,
}

/// 单条 capability 的配置。预留出口给后续 `image_generation` /
/// `code_execution` 等同形态 capability。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCapabilityConfig {
    pub mode: SearchCapabilityMode,
}

impl SearchCapabilityConfig {
    /// 用单个 mode 构造。跨 crate 调用方需要这个入口，因为本结构体
    /// `#[non_exhaustive]` 后不能直接 struct literal。
    #[must_use]
    pub const fn new(mode: SearchCapabilityMode) -> Self {
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
    pub search: SearchCapabilityConfig,
}

impl SessionCapabilitiesConfig {
    /// 用单条 [`SearchCapabilityConfig`] 构造。跨 crate 调用方（例如
    /// `defect-config`）需要这个入口，因为本结构体 `#[non_exhaustive]`
    /// 后不能直接 struct literal。
    #[must_use]
    pub const fn with_search(search: SearchCapabilityConfig) -> Self {
        Self { search }
    }
}

/// session 启动期裁决出的运行时能力。
///
/// 与 [`SessionCapabilitiesConfig`] 区分：前者是用户配置（意图），这里
/// 是与 provider [`HostedCapabilities`] 交叉后得到的实际启用集合。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolvedSessionCapabilities {
    /// 最终决定本 session 是否走 hosted search。
    /// `Delegate × supported` → `true`；其余情况 → `false`。
    pub hosted: HostedCapabilities,
    /// 最终决定本 session 是否要注册本地 `search` tool。
    /// 仅 `Local` 时为 `true`。
    pub register_local_search: bool,
}

impl ResolvedSessionCapabilities {
    /// 一次性裁决：按提案 §6.1 的表把 `(mode, provider)` 映射到结果。
    ///
    /// # Errors
    ///
    /// `Delegate` 但 provider 不支持 hosted search 时返回
    /// [`SessionInitError::CapabilityUnsatisfied`]。
    pub fn resolve(
        config: SessionCapabilitiesConfig,
        provider_hosted: HostedCapabilities,
        provider_id: &str,
    ) -> Result<Self, SessionInitError> {
        let mut hosted = HostedCapabilities::default();
        let mut register_local_search = false;

        match config.search.mode {
            SearchCapabilityMode::Delegate => {
                if !provider_hosted.search {
                    return Err(SessionInitError::CapabilityUnsatisfied {
                        capability: "search",
                        provider: provider_id.to_string(),
                    });
                }
                hosted.search = true;
            }
            SearchCapabilityMode::Local => {
                register_local_search = true;
            }
            SearchCapabilityMode::Disabled => {}
        }

        Ok(Self {
            hosted,
            register_local_search,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosted(search: bool) -> HostedCapabilities {
        HostedCapabilities { search }
    }

    fn config(mode: SearchCapabilityMode) -> SessionCapabilitiesConfig {
        SessionCapabilitiesConfig {
            search: SearchCapabilityConfig { mode },
        }
    }

    #[test]
    fn delegate_with_supported_provider_enables_hosted() {
        let resolved =
            ResolvedSessionCapabilities::resolve(config(SearchCapabilityMode::Delegate), hosted(true), "anthropic")
                .expect("should succeed");
        assert!(resolved.hosted.search);
        assert!(!resolved.register_local_search);
    }

    #[test]
    fn delegate_with_unsupported_provider_fails() {
        let err = ResolvedSessionCapabilities::resolve(
            config(SearchCapabilityMode::Delegate),
            hosted(false),
            "deepseek",
        )
        .expect_err("should reject");
        match err {
            SessionInitError::CapabilityUnsatisfied {
                capability,
                provider,
            } => {
                assert_eq!(capability, "search");
                assert_eq!(provider, "deepseek");
            }
        }
    }

    #[test]
    fn local_registers_local_search_regardless_of_provider() {
        for support in [true, false] {
            let resolved = ResolvedSessionCapabilities::resolve(
                config(SearchCapabilityMode::Local),
                hosted(support),
                "any",
            )
            .expect("should succeed");
            assert!(!resolved.hosted.search);
            assert!(resolved.register_local_search);
        }
    }

    #[test]
    fn disabled_exposes_nothing() {
        for support in [true, false] {
            let resolved = ResolvedSessionCapabilities::resolve(
                config(SearchCapabilityMode::Disabled),
                hosted(support),
                "any",
            )
            .expect("should succeed");
            assert!(!resolved.hosted.search);
            assert!(!resolved.register_local_search);
        }
    }

    #[test]
    fn unsatisfied_error_message_includes_actionable_hint() {
        let err = ResolvedSessionCapabilities::resolve(
            config(SearchCapabilityMode::Delegate),
            hosted(false),
            "deepseek",
        )
        .expect_err("should reject");
        let msg = err.to_string();
        assert!(msg.contains("provider `deepseek` does not support hosted search"));
        assert!(msg.contains("[providers.deepseek.capabilities.search]"));
        assert!(msg.contains("mode = \"local\""));
        assert!(msg.contains("[capabilities.search]"));
    }
}
