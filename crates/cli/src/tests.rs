use crate::policy::build_policy;
#[cfg(feature = "provider-openai")]
use crate::providers::{LITELLM_API_KEY_ENV, LITELLM_DEFAULT_BASE_URL, ProviderDefaults};

use defect_agent::policy::{PolicyCtx, PolicyDecision};
use defect_agent::tool::SafetyClass;
#[cfg(feature = "provider-openai")]
use defect_config::ProviderConfigFile;
use defect_config::SandboxMode;
use serde_json::json;

mod subagents;

#[test]
fn read_only_policy_denies_mutating_tools() {
    let policy = build_policy(SandboxMode::ReadOnly);
    let args = json!({});
    let cwd = std::path::Path::new("/");

    let decision = policy.classify(PolicyCtx::new(
        "write_file",
        SafetyClass::Mutating,
        &args,
        cwd,
    ));

    assert!(matches!(decision, PolicyDecision::Deny));
}

#[test]
fn open_policy_allows_destructive_tools() {
    let policy = build_policy(SandboxMode::Open);
    let args = json!({});
    let cwd = std::path::Path::new("/");

    let decision = policy.classify(PolicyCtx::new("bash", SafetyClass::Destructive, &args, cwd));

    assert!(matches!(decision, PolicyDecision::Allow));
}

#[cfg(feature = "provider-openai")]
#[test]
fn litellm_defaults_fill_endpoint_and_credential_env() {
    let provider = ProviderDefaults {
        base_url: LITELLM_DEFAULT_BASE_URL,
        api_key_env: LITELLM_API_KEY_ENV,
    }
    .apply(ProviderConfigFile::default());

    assert_eq!(provider.base_url.as_deref(), Some(LITELLM_DEFAULT_BASE_URL));
    assert_eq!(provider.api_key_env.as_deref(), Some(LITELLM_API_KEY_ENV));
}
