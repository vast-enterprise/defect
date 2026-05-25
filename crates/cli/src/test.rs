use super::build_policy;

use defect_agent::policy::{PolicyCtx, PolicyDecision};
use defect_agent::tool::SafetyClass;
use defect_config::SandboxMode;
use serde_json::json;

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
