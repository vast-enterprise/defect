use super::*;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

fn ctx<'a>(
    session_id: &'a agent_client_protocol_schema::SessionId,
    cwd: &'a std::path::Path,
) -> HookCtx<'a> {
    HookCtx::new(session_id, cwd, CancellationToken::new())
}

#[test]
fn registry_defaults_have_two_builtins() {
    let reg = BuiltinRegistry::defaults();
    let names: Vec<_> = reg.names().collect();
    assert!(names.contains(&"tracing-audit"));
    assert!(names.contains(&"redact-secrets"));
}

#[test]
fn registry_lookup_unknown_returns_none() {
    let reg = BuiltinRegistry::defaults();
    assert!(reg.lookup_step("does-not-exist").is_none());
}

#[test]
fn registry_step_factories_match_event_factories() {
    let reg = BuiltinRegistry::defaults();
    assert!(reg.lookup_step("tracing-audit").is_some());
    assert!(reg.lookup_step("redact-secrets").is_some());
    assert!(reg.lookup_step("does-not-exist").is_none());
}

#[tokio::test]
async fn redact_secrets_step_redacts_args() {
    let h = RedactSecretsHook;
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({
        "tool": "login",
        "args": {"user": "alice", "password": "hunter2"},
    });
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    assert_eq!(verdict["args"]["password"], "***");
    assert_eq!(verdict["args"]["user"], "alice");
}

#[tokio::test]
async fn redact_secrets_step_no_secrets_no_verdict() {
    let h = RedactSecretsHook;
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({"tool": "ls", "args": {"path": "/tmp"}});
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok");
    assert!(verdict.is_none());
}

/// Create a `SkillEntry` with customizable `description`, `body`, `always`,
/// `keywords`, and `globs`.
fn skill(
    description: &str,
    body: &str,
    always: bool,
    keywords: &[&str],
    globs: &[&str],
) -> SkillEntry {
    let compiled = if globs.is_empty() {
        None
    } else {
        let mut b = globset::GlobSetBuilder::new();
        for g in globs {
            b.add(globset::Glob::new(g).expect("valid glob"));
        }
        Some(b.build().expect("glob set"))
    };
    SkillEntry {
        description: description.to_string(),
        body: body.to_string(),
        dir: std::path::PathBuf::from("/skills/x"),
        always,
        triggers: crate::tool::SkillTriggers {
            globs: compiled,
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
        },
    }
}

#[tokio::test]
async fn skill_manifest_step_injects_context() {
    let mut skills = BTreeMap::new();
    skills.insert(
        "deploy".to_string(),
        skill("deploy the app", "", false, &[], &[]),
    );
    let h = SkillManifestHook::new(Arc::new(skills));
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({"cwd": "/", "source": "new"});
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    let ctx_arr = verdict["additional_context"].as_array().expect("array");
    assert_eq!(ctx_arr.len(), 1);
    assert!(ctx_arr[0].as_str().unwrap().contains("deploy"));
}

#[test]
fn manifest_includes_always_on_body() {
    let mut skills = BTreeMap::new();
    skills.insert(
        "style".to_string(),
        skill("coding style", "ALWAYS USE TABS", true, &[], &[]),
    );
    skills.insert(
        "deploy".to_string(),
        skill("deploy", "deploy body", false, &[], &[]),
    );
    let out = render_skill_manifest(&skills).expect("some");
    // The L1 manifest contains both; the always-on body only includes style.
    assert!(out.contains("**style**"));
    assert!(out.contains("**deploy**"));
    assert!(out.contains("ALWAYS USE TABS"));
    assert!(!out.contains("deploy body"));
}

fn triggers_envelope(prompt: &str) -> Value {
    serde_json::json!({ "source": "user", "input": prompt, "input_len": 1 })
}

#[tokio::test]
async fn triggers_keyword_hit() {
    let mut skills = BTreeMap::new();
    skills.insert(
        "db".to_string(),
        skill("database", "", false, &["migration"], &[]),
    );
    let h = SkillTriggersHook::new(Arc::new(skills));
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    // Case-insensitive substring match.
    let verdict = h
        .handle_step(
            &triggers_envelope("please run the MIGRATION now"),
            ctx(&session_id, cwd),
        )
        .await
        .expect("ok")
        .expect("verdict");
    let arr = verdict["prepend_input"].as_array().expect("array");
    assert_eq!(arr.len(), 1);
    assert!(arr[0].as_str().unwrap().contains("`db`"));
}

#[tokio::test]
async fn triggers_glob_hit_on_path_token() {
    let mut skills = BTreeMap::new();
    skills.insert(
        "sql".to_string(),
        skill("sql files", "", false, &[], &["**/*.sql"]),
    );
    let h = SkillTriggersHook::new(Arc::new(skills));
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let verdict = h
        .handle_step(
            &triggers_envelope("edit migrations/0001.sql to add a column"),
            ctx(&session_id, cwd),
        )
        .await
        .expect("ok")
        .expect("verdict");
    assert_eq!(verdict["prepend_input"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn triggers_no_hit_returns_none() {
    let mut skills = BTreeMap::new();
    skills.insert(
        "db".to_string(),
        skill("database", "", false, &["migration"], &["**/*.sql"]),
    );
    let h = SkillTriggersHook::new(Arc::new(skills));
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let verdict = h
        .handle_step(
            &triggers_envelope("write some rust code"),
            ctx(&session_id, cwd),
        )
        .await
        .expect("ok");
    assert!(verdict.is_none());
}

#[tokio::test]
async fn triggers_excludes_always_on_skill() {
    let mut skills = BTreeMap::new();
    // Skills marked as always-on are not suggested even when keywords match (the
    // entire segment has already been injected).
    skills.insert(
        "style".to_string(),
        skill("style", "body", true, &["rust"], &[]),
    );
    let h = SkillTriggersHook::new(Arc::new(skills));
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let verdict = h
        .handle_step(&triggers_envelope("write rust"), ctx(&session_id, cwd))
        .await
        .expect("ok");
    assert!(verdict.is_none());
}

#[test]
fn path_token_extraction() {
    let toks = extract_path_tokens("look at `crates/agent/src/foo.rs` and Cargo.toml please");
    assert!(toks.contains(&"crates/agent/src/foo.rs".to_string()));
    assert!(toks.contains(&"Cargo.toml".to_string()));
    // Plain words are not paths.
    assert!(!toks.contains(&"please".to_string()));
    assert!(!toks.contains(&"look".to_string()));
}

// ----- goal-gate -----

#[tokio::test]
async fn goal_gate_briefs_at_session_enter() {
    let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
    let h = GoalGate::new(goal);
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({ "hook_event": "after_session_enter" });
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    // Inject system prompt suffix without control (no control flow intervention).
    assert!(verdict.get("control").is_none());
    let ctxs = verdict["additional_context"].as_array().expect("array");
    let briefing = ctxs[0].as_str().expect("str");
    assert!(briefing.contains("ship the feature"));
    assert!(briefing.contains("goal_done"));
}

#[tokio::test]
async fn goal_gate_not_reached_continues_with_feedback() {
    let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
    let h = GoalGate::new(goal);
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({
        "hook_event": "before_turn_end",
        "stop_reason": "end_turn", "continues_so_far": 0, "voluntary": true,
    });
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    assert_eq!(verdict["control"], "continue");
    let ctxs = verdict["additional_context"].as_array().expect("array");
    assert_eq!(ctxs.len(), 1);
    assert!(ctxs[0].as_str().expect("str").contains("ship the feature"));
}

#[tokio::test]
async fn goal_gate_reached_proceeds() {
    let goal = Arc::new(crate::session::GoalState::new("ship the feature"));
    goal.mark_reached();
    let h = GoalGate::new(goal);
    let session_id = agent_client_protocol_schema::SessionId::new("s1");
    let cwd = std::path::Path::new("/");
    let envelope = serde_json::json!({
        "hook_event": "before_turn_end",
        "stop_reason": "end_turn", "continues_so_far": 1, "voluntary": true,
    });
    let verdict = h
        .handle_step(&envelope, ctx(&session_id, cwd))
        .await
        .expect("ok")
        .expect("verdict");
    assert_eq!(verdict["control"], "proceed");
}
