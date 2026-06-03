use super::*;

use serde_json::json;
use std::path::PathBuf;

fn ctx<'a>(
    name: &'a str,
    hint: SafetyClass,
    args: &'a serde_json::Value,
    cwd: &'a Path,
) -> PolicyCtx<'a> {
    PolicyCtx::new(name, hint, args, cwd)
}

#[test]
fn open_allows_everything() {
    let policy = OpenPolicy;
    let cwd = PathBuf::from("/");
    let args = json!({});
    for hint in [
        SafetyClass::ReadOnly,
        SafetyClass::Mutating,
        SafetyClass::Destructive,
        SafetyClass::Network,
    ] {
        assert!(matches!(
            policy.classify(ctx("t", hint, &args, &cwd)),
            PolicyDecision::Allow
        ));
    }
}

#[test]
fn read_only_denies_writes() {
    let policy = ReadOnlyPolicy;
    let cwd = PathBuf::from("/");
    let args = json!({});
    assert!(matches!(
        policy.classify(ctx("fs.read", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Allow
    ));
    for hint in [
        SafetyClass::Mutating,
        SafetyClass::Destructive,
        SafetyClass::Network,
    ] {
        assert!(matches!(
            policy.classify(ctx("t", hint, &args, &cwd)),
            PolicyDecision::Deny
        ));
    }
}

#[test]
fn ask_writes_allows_read_asks_writes() {
    let policy = AskWritesPolicy::new();
    let cwd = PathBuf::from("/");
    let args = json!({});

    assert!(matches!(
        policy.classify(ctx("fs.read", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Allow
    ));

    let dec = policy.classify(ctx("bash", SafetyClass::Destructive, &args, &cwd));
    let PolicyDecision::Ask(ask) = dec else {
        panic!("expected Ask, got {dec:?}");
    };
    let ids: Vec<_> = ask
        .options
        .iter()
        .map(|o| o.id.0.as_ref().to_string())
        .collect();
    assert_eq!(ids, vec!["allow_once", "allow_always", "reject_once"]);
    assert_eq!(
        ask.options.iter().map(|o| o.allows).collect::<Vec<_>>(),
        vec![true, true, false]
    );
}

#[test]
fn ask_writes_remembers_allow_always() {
    let policy = AskWritesPolicy::new();
    let cwd = PathBuf::from("/");
    let args = json!({});

    // 先来一次 Ask
    assert!(matches!(
        policy.classify(ctx("bash", SafetyClass::Destructive, &args, &cwd)),
        PolicyDecision::Ask(_)
    ));

    // 用户选了 AllowAlways
    policy.record(
        ctx("bash", SafetyClass::Destructive, &args, &cwd),
        RecordedOutcome::Selected {
            option_id: PermissionOptionId::new(ALLOW_ALWAYS_ID),
            allows: true,
        },
    );

    // 再来一次 → 直接 Allow，不再 Ask
    assert!(matches!(
        policy.classify(ctx("bash", SafetyClass::Destructive, &args, &cwd)),
        PolicyDecision::Allow
    ));
}

#[test]
fn ask_writes_does_not_remember_allow_once() {
    let policy = AskWritesPolicy::new();
    let cwd = PathBuf::from("/");
    let args = json!({});

    policy.record(
        ctx("bash", SafetyClass::Destructive, &args, &cwd),
        RecordedOutcome::Selected {
            option_id: PermissionOptionId::new(ALLOW_ONCE_ID),
            allows: true,
        },
    );

    // 仍然 Ask
    assert!(matches!(
        policy.classify(ctx("bash", SafetyClass::Destructive, &args, &cwd)),
        PolicyDecision::Ask(_)
    ));
}

#[test]
fn deny_all_denies() {
    let policy = DenyAllPolicy;
    let cwd = PathBuf::from("/");
    let args = json!({});
    assert!(matches!(
        policy.classify(ctx("fs.read", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Deny
    ));
}

#[test]
fn non_interactive_maps_ask_to_deny_passes_allow_deny() {
    use std::sync::Arc;

    // 包 AskWritesPolicy：ReadOnly 透传 Allow，写类被降级为 Deny（而非 Ask）。
    let policy = NonInteractivePolicy::new(Arc::new(AskWritesPolicy::new()));
    let cwd = PathBuf::from("/");
    let args = json!({});

    assert!(matches!(
        policy.classify(ctx("fs.read", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Allow
    ));
    for hint in [
        SafetyClass::Mutating,
        SafetyClass::Destructive,
        SafetyClass::Network,
    ] {
        assert!(
            matches!(
                policy.classify(ctx("t", hint, &args, &cwd)),
                PolicyDecision::Deny
            ),
            "inner Ask must be downgraded to Deny for {hint:?}"
        );
    }

    // 包 DenyAllPolicy：Deny 原样透传。
    let deny = NonInteractivePolicy::new(Arc::new(DenyAllPolicy));
    assert!(matches!(
        deny.classify(ctx("fs.read", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Deny
    ));
}

#[test]
fn mode_catalog_rejects_empty_or_unknown_current() {
    // 空目录 → None。
    assert!(ModeCatalog::new(vec![], "x").is_none());

    // current 不命中任一条目 → None。
    let modes = vec![PolicyMode {
        id: "open".to_string(),
        name: "Open".to_string(),
        description: None,
        policy: Arc::new(OpenPolicy),
    }];
    assert!(ModeCatalog::new(modes.clone(), "read-only").is_none());

    // current 命中 → Some。
    assert!(ModeCatalog::new(modes, "open").is_some());
}

#[test]
fn mode_catalog_switches_active_policy() {
    let cwd = PathBuf::from("/");
    let args = json!({});

    let mut catalog = ModeCatalog::new(
        vec![
            PolicyMode {
                id: "open".to_string(),
                name: "Open".to_string(),
                description: None,
                policy: Arc::new(OpenPolicy),
            },
            PolicyMode {
                id: "deny-all".to_string(),
                name: "Deny all".to_string(),
                description: None,
                policy: Arc::new(DenyAllPolicy),
            },
        ],
        "open",
    )
    .expect("catalog");

    assert_eq!(catalog.current_id(), "open");
    assert!(matches!(
        catalog
            .current_policy()
            .classify(ctx("t", SafetyClass::Mutating, &args, &cwd)),
        PolicyDecision::Allow
    ));

    // 切到 deny-all：active policy 随之变。
    assert!(catalog.set_current("deny-all"));
    assert_eq!(catalog.current_id(), "deny-all");
    assert!(matches!(
        catalog
            .current_policy()
            .classify(ctx("t", SafetyClass::ReadOnly, &args, &cwd)),
        PolicyDecision::Deny
    ));

    // 未知 id：set 失败，current 不变。
    assert!(!catalog.set_current("bogus"));
    assert_eq!(catalog.current_id(), "deny-all");
}
