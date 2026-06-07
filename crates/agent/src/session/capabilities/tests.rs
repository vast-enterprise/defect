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
